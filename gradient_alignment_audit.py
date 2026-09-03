#!/usr/bin/env python3
"""Export exact gradient-alignment and GRAFT ranking quantities.

This is an analysis-only program: it loads (or, when explicitly allowed, trains)
the repository's clean surrogate checkpoints and never crafts poisons or trains a
victim.  For every target, surrogate, and eligible clean base candidate it saves

    M_i                         logit margin toward y_adv
    R_i = h_i^T h_t             raw feature inner product
    A_i = <grad_phi l_i,
           grad_phi L_t>        exact non-classifier/backbone gradient dot
    <g_i, g_t>                  exact all-parameter gradient dot

The full dot is decomposed exactly as

    <g_i,g_t> = A_i
              + (r_i^T r_t)(h_i^T h_t)       [classifier weight]
              + (r_i^T r_t)                  [classifier bias, if present].

Thus the files directly expose what Algorithm 1 omits from A_i.  Candidate and
target residual vectors are saved as well, preserving their classwise signs.

Notation for the Jacobian fields
--------------------------------
``JiT_Jt_backbone_loss`` is the exact scalar loss-Jacobian dot A_i.  Equivalently
it is the residual-weighted contraction

    r_i^T W [J_h(x_i) J_h(x_t)^T] W^T r_t.

This is the scalar from the representation-NTK matrix that enters Equation (3).
The uncontracted d x d matrix can be enormous.  ``--representation-ntk-mode``
therefore makes its treatment explicit:

* contracted (default): save the exact contraction above;
* trace-exact: additionally save tr(J_h(x_i) J_h(x_t)^T) exactly;
* full-exact: additionally save every d x d matrix in a separate .npy file.

The exact trace/full modes can be very expensive (d reverse/JVP sweeps per
target) and full-exact can require many GiB per shard.  No stochastic estimator
is used: requested quantities are either exact or the program fails.

On-disk layout
--------------

    OUTPUT/DATASET/MODEL/PAIR/
      manifest.json
      ensemble_candidates.csv
      target_TARGET/
        surrogate_000.npz
        surrogate_001.npz
        ...
        ensemble.npz
        surrogate_000_representation_ntk.npy   # full-exact only

Per-surrogate shards contain raw values, standardized values, stable ranks, the
checkpoint identity, logits/residuals, and decomposition checks.  ``ensemble``
first averages raw M/R/A over surrogates and then standardizes across candidates,
matching Algorithm 1.  Writes are atomic and valid completed shards are skipped.
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import final_update as poison


SCHEMA_VERSION = 1


def die(message: str) -> "NoReturn":
    raise SystemExit("ERROR: " + message)


def safe_slug(value: str) -> str:
    return value.replace("/", "_").replace(" ", "_")


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(temporary, path)


def atomic_npz(path: Path, **arrays: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp.npz")
    np.savez_compressed(temporary, **arrays)
    os.replace(temporary, path)


def atomic_csv(path: Path, fieldnames: Sequence[str], rows: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def zscore(values: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Repository-compatible sample-standardization (torch.std default)."""
    x = np.asarray(values, dtype=np.float64)
    ddof = 1 if len(x) > 1 else 0
    scale = float(x.std(ddof=ddof))
    return ((x - float(x.mean())) / (scale + eps)).astype(np.float32)


def stable_rank(values: np.ndarray, largest: bool) -> np.ndarray:
    """One-based rank; ties are resolved stably by candidate order."""
    x = np.asarray(values)
    order = np.argsort(-x if largest else x, kind="mergesort")
    rank = np.empty(len(x), dtype=np.int32)
    rank[order] = np.arange(1, len(x) + 1, dtype=np.int32)
    return rank


def parse_int_list(value: str) -> list[int]:
    try:
        result = sorted({int(item) for item in value.replace(" ", "").split(",")
                         if item != ""})
    except ValueError:
        die(f"expected comma-separated integers, got {value!r}")
    if not result or result[0] < 0:
        die(f"expected nonnegative comma-separated integers, got {value!r}")
    return result


def resolve_device(spec: str) -> torch.device:
    if spec == "auto":
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device = torch.device(spec)
    if device.type == "cuda" and not torch.cuda.is_available():
        die(f"{spec} requested but CUDA is unavailable")
    if device.type == "cuda":
        torch.cuda.set_device(device)
    return device


def load_target_indices(path: Path, class_pair: str) -> list[int]:
    try:
        with path.open() as handle:
            payload = json.load(handle)
        values = (payload["pairs"][class_pair]["indices"]
                  if "pairs" in payload else payload[class_pair])
        indices = [int(value) for value in values]
    except (OSError, KeyError, TypeError, ValueError) as exc:
        die(f"cannot read targets for {class_pair!r} from {path}: {exc}")
    if not indices:
        die(f"{path} contains no target indices for {class_pair}")
    return indices


def production_args(cli: argparse.Namespace) -> argparse.Namespace:
    argv = [
        "--dataset", cli.dataset,
        "--data_path", str(cli.data_path),
        "--model", cli.model,
        "--seed", str(cli.seed),
        "--cache_dir", str(cli.cache_dir),
        "--out_dir", str(cli.output_root / "unused_attack_output"),
        "--class_pair", cli.class_pair,
        "--pair_order", cli.pair_order,
        "--num_surrogates", str(cli.num_surrogates),
        "--surrogate_epochs", str(cli.surrogate_epochs),
        "--surrogate_lr", str(cli.surrogate_lr),
        "--surrogate_bs", str(cli.surrogate_bs),
        "--surrogate_decay", *[str(value) for value in cli.surrogate_decay],
        "--surrogate_wd", str(cli.surrogate_wd),
        "--gpus", "none",
    ]
    if cli.surrogate_aug:
        argv.append("--surrogate_aug")
    return poison.parse_args(argv)


def checkpoint_paths(parsed: argparse.Namespace,
                     surrogate_ids: Sequence[int]) -> list[Path]:
    directory = Path(poison.surrogate_dir(parsed))
    return [directory / f"net_{model_id}.pt" for model_id in surrogate_ids]


def ensure_checkpoints(cli: argparse.Namespace, parsed: argparse.Namespace,
                       ctx: dict, surrogate_ids: Sequence[int]) -> list[Path]:
    paths = checkpoint_paths(parsed, surrogate_ids)
    missing = [(model_id, path) for model_id, path in zip(surrogate_ids, paths)
               if not path.is_file()]
    if missing and not cli.train_missing:
        listing = "\n".join(str(path) for _, path in missing)
        die("missing surrogate checkpoints; rerun with --train-missing or train "
            f"them separately:\n{listing}")
    for model_id, path in missing:
        print(f"training missing surrogate {model_id}: {path}", flush=True)
        nets = poison.get_surrogates(
            parsed,
            ctx["train_imgs"], ctx["train_labs"],
            ctx["test_imgs"], ctx["test_labs"],
            ctx["channel"], ctx["num_classes"], ctx["im_size"],
            str(ctx["device"]), ctx["dsa_param"], only_id=model_id,
        )
        del nets
        gc.collect()
        if ctx["device"].type == "cuda":
            torch.cuda.empty_cache()
        if not path.is_file():
            die(f"training returned without creating {path}")
    return paths


def load_net(cli: argparse.Namespace, ctx: dict, checkpoint: Path,
             model_id: int) -> nn.Module:
    net = poison.build_network(
        cli.model, ctx["channel"], ctx["num_classes"], ctx["im_size"],
        str(ctx["device"]), seed=cli.seed + 1000 + model_id,
    )
    state = torch.load(checkpoint, map_location=ctx["device"])
    net.load_state_dict(state)
    net.eval()
    return net


@torch.no_grad()
def forward_candidates(net: nn.Module, candidates: torch.Tensor, y_adv: int,
                       batch_size: int) -> dict[str, torch.Tensor]:
    embed = poison.embed_of(net)
    logits, features = [], []
    for start in range(0, len(candidates), batch_size):
        batch = candidates[start:start + batch_size]
        logits.append(net(batch).detach())
        features.append(embed(batch).detach().flatten(1))
    z = torch.cat(logits)
    h = torch.cat(features)
    residual = F.softmax(z, dim=1)
    residual[:, y_adv] -= 1.0
    other = z.clone()
    other[:, y_adv] = float("-inf")
    margin = z[:, y_adv] - other.max(dim=1).values
    return {"logits": z, "features": h, "residual": residual,
            "margin": margin, "feature_norm": h.norm(dim=1)}


@torch.no_grad()
def target_and_head_quantities(net: nn.Module, static: dict[str, torch.Tensor],
                               target: torch.Tensor, y_adv: int) -> dict[str, torch.Tensor]:
    core = net.module if isinstance(net, nn.DataParallel) else net
    if not hasattr(core, "classifier") or not isinstance(core.classifier, nn.Linear):
        die(f"{type(core).__name__} must expose its final nn.Linear as .classifier")

    target_batch = target.unsqueeze(0)
    target_logits = net(target_batch).detach()
    target_feature = poison.embed_of(net)(target_batch).detach().flatten(1)
    target_residual = F.softmax(target_logits, dim=1)
    target_residual[:, y_adv] -= 1.0

    features = static["features"]
    residuals = static["residual"]
    feature_dot = (features * target_feature).sum(dim=1)
    target_feature_norm = target_feature.norm(dim=1)[0]
    feature_cosine = feature_dot / (
        static["feature_norm"] * target_feature_norm + 1e-12)
    feature_l2_sq = ((features - target_feature) ** 2).sum(dim=1)
    residual_dot = (residuals * target_residual).sum(dim=1)
    classifier_weight = residual_dot * feature_dot
    classifier_bias = (residual_dot if core.classifier.bias is not None
                       else torch.zeros_like(residual_dot))
    return {
        "target_logits": target_logits[0],
        "target_feature": target_feature[0],
        "target_residual": target_residual[0],
        "target_feature_norm": target_feature_norm,
        "feature_dot": feature_dot,
        "feature_cosine": feature_cosine,
        "feature_l2_sq": feature_l2_sq,
        "residual_dot": residual_dot,
        "classifier_weight_grad_dot": classifier_weight,
        "classifier_bias_grad_dot": classifier_bias,
        "classifier_grad_dot": classifier_weight + classifier_bias,
    }


def direct_full_gradient_dot(net: nn.Module, candidate: torch.Tensor,
                             target: torch.Tensor, y_adv: int) -> float:
    """Slow independent check of one all-parameter gradient inner product."""
    core = net.module if isinstance(net, nn.DataParallel) else net
    parameters = tuple(parameter for parameter in core.parameters()
                       if parameter.requires_grad)
    label = torch.tensor([int(y_adv)], dtype=torch.long, device=target.device)
    with torch.enable_grad():
        target_loss = F.cross_entropy(core(target.unsqueeze(0)), label)
        target_grads = torch.autograd.grad(target_loss, parameters)
        candidate_loss = F.cross_entropy(core(candidate.unsqueeze(0)), label)
        candidate_grads = torch.autograd.grad(candidate_loss, parameters)
        value = sum((left * right).sum()
                    for left, right in zip(candidate_grads, target_grads))
    return float(value.detach())


class EmbeddingView(nn.Module):
    """Functional-callable view whose forward is the model representation."""

    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model.embed(x).flatten(1)


def representation_ntk_exact(net: nn.Module, candidates: torch.Tensor,
                              target: torch.Tensor, batch_size: int, mode: str,
                              full_output: Path | None,
                              max_full_gb: float, trace_probes: int = 32,
                              random_seed: int = 0) -> tuple[np.ndarray | None, str | None]:
    """Compute the pure representation-Jacobian kernel without loss residuals.

    ``trace-exact`` returns tr(J_i J_t^T) exactly. ``trace-hutchinson``
    estimates that same scalar with Rademacher probes, and is normally much
    faster when only its candidate-wise correlation/ranking is needed.
    """
    if mode == "contracted":
        return None, None
    if not all(hasattr(torch.func, name) for name in ("functional_call", "grad", "jvp")):
        die("trace-exact/full-exact requires torch.func functional_call/grad/jvp")

    core = net.module if isinstance(net, nn.DataParallel) else net
    view = EmbeddingView(core)
    named_parameters = dict(view.named_parameters())
    head_ids = {id(parameter) for parameter in core.classifier.parameters()}
    backbone = {name: parameter for name, parameter in named_parameters.items()
                if id(parameter) not in head_ids}
    fixed = {name: parameter for name, parameter in named_parameters.items()
             if id(parameter) in head_ids}
    buffers = dict(view.named_buffers())
    target_batch = target.unsqueeze(0)

    def functional_embed(backbone_parameters: dict[str, torch.Tensor],
                         images: torch.Tensor) -> torch.Tensor:
        parameters = dict(fixed)
        parameters.update(backbone_parameters)
        return torch.func.functional_call(view, (parameters, buffers), (images,))

    with torch.no_grad():
        feature_dim = int(view(target_batch).shape[1])
    candidate_count = len(candidates)
    trace = (np.zeros(candidate_count, dtype=np.float64)
             if mode in ("trace-exact", "trace-hutchinson") else None)
    matrix = None
    temporary = None
    if mode == "full-exact":
        assert full_output is not None
        required_gb = candidate_count * feature_dim * feature_dim * 4 / 1e9
        if required_gb > max_full_gb:
            die(f"full representation NTK needs about {required_gb:.2f} GB for one "
                f"shard, above --max-full-ntk-gb={max_full_gb:g}; raise the limit "
                "deliberately or use trace-exact/contracted")
        full_output.parent.mkdir(parents=True, exist_ok=True)
        temporary = full_output.with_name(full_output.name + ".tmp.npy")
        matrix = np.lib.format.open_memmap(
            temporary, mode="w+", dtype=np.float32,
            shape=(candidate_count, feature_dim, feature_dim),
        )

    states = [(module, module.training) for module in view.modules()]
    view.eval()
    try:
        with torch.enable_grad():
            if mode == "trace-hutchinson":
                rng = np.random.default_rng(random_seed)
                for probe_index in range(trace_probes):
                    probe = torch.as_tensor(
                        rng.choice((-1.0, 1.0), size=feature_dim),
                        device=target_batch.device,
                        dtype=target_batch.dtype,
                    )

                    def target_projection(parameters: dict[str, torch.Tensor]) -> torch.Tensor:
                        return (functional_embed(parameters, target_batch)[0] * probe).sum()

                    target_tangent = torch.func.grad(target_projection)(backbone)
                    for start in range(0, candidate_count, batch_size):
                        batch = candidates[start:start + batch_size]

                        def candidate_features(parameters: dict[str, torch.Tensor]) -> torch.Tensor:
                            return functional_embed(parameters, batch)

                        _, tangent = torch.func.jvp(
                            candidate_features, (backbone,), (target_tangent,))
                        values = (tangent * probe).sum(dim=1)
                        stop = start + len(batch)
                        assert trace is not None
                        trace[start:stop] += values.detach().float().cpu().numpy()
                    if (probe_index == 0 or (probe_index + 1) % 8 == 0 or
                            probe_index + 1 == trace_probes):
                        print(f"      representation NTK probe "
                              f"{probe_index + 1}/{trace_probes}", flush=True)
                assert trace is not None
                trace /= trace_probes
            else:
                for column in range(feature_dim):
                    def target_coordinate(parameters: dict[str, torch.Tensor]) -> torch.Tensor:
                        return functional_embed(parameters, target_batch)[0, column]

                    target_tangent = torch.func.grad(target_coordinate)(backbone)
                    for start in range(0, candidate_count, batch_size):
                        batch = candidates[start:start + batch_size]

                        def candidate_features(parameters: dict[str, torch.Tensor]) -> torch.Tensor:
                            return functional_embed(parameters, batch)

                        _, tangent = torch.func.jvp(
                            candidate_features, (backbone,), (target_tangent,))
                        values = tangent.detach().float().cpu().numpy()
                        stop = start + len(batch)
                        if trace is not None:
                            trace[start:stop] += values[:, column]
                        else:
                            matrix[start:stop, :, column] = values
                    if (column == 0 or (column + 1) % 64 == 0 or
                            column + 1 == feature_dim):
                        print(f"      representation NTK column "
                              f"{column + 1}/{feature_dim}", flush=True)
    except Exception:
        if matrix is not None:
            del matrix
        if temporary is not None and temporary.exists():
            temporary.unlink()
        raise
    finally:
        poison._restore_training_states(states)

    if matrix is not None:
        matrix.flush()
        del matrix
        os.replace(temporary, full_output)
        return None, str(full_output)
    return trace.astype(np.float32), None


def tensor_numpy(value: torch.Tensor, dtype=np.float32) -> np.ndarray:
    return value.detach().cpu().numpy().astype(dtype, copy=False)


def shard_valid(path: Path, cli: argparse.Namespace, target_idx: int,
                model_id: int, candidate_indices: np.ndarray,
                checkpoint: Path) -> bool:
    try:
        stat = checkpoint.stat()
        with np.load(path, allow_pickle=False) as data:
            return bool(
                int(data["schema_version"]) == SCHEMA_VERSION and
                str(data["dataset"]) == cli.dataset and
                str(data["model"]) == cli.model and
                str(data["class_pair"]) == cli.class_pair and
                int(data["target_idx"]) == target_idx and
                int(data["surrogate_id"]) == model_id and
                int(data["checkpoint_size"]) == stat.st_size and
                int(data["checkpoint_mtime_ns"]) == stat.st_mtime_ns and
                str(data["representation_ntk_mode"]) ==
                cli.representation_ntk_mode and
                np.array_equal(data["candidate_idx"], candidate_indices) and
                len(data["exact_full_grad_dot"]) == len(candidate_indices) and
                (cli.representation_ntk_mode not in
                 ("trace-exact", "trace-hutchinson") or
                 "representation_ntk_trace_JiJt" in data) and
                (cli.representation_ntk_mode != "trace-hutchinson" or
                 int(data["ntk_trace_probes"]) == cli.ntk_trace_probes) and
                (cli.representation_ntk_mode != "full-exact" or
                 (str(data["representation_ntk_full_path"]) != "" and
                  Path(str(data["representation_ntk_full_path"])).is_file()))
            )
    except (OSError, KeyError, TypeError, ValueError):
        return False


def build_shard(cli: argparse.Namespace, net: nn.Module, static: dict[str, torch.Tensor],
                candidates: torch.Tensor, candidate_indices: np.ndarray,
                candidate_labels: np.ndarray, target: torch.Tensor,
                target_idx: int, target_position: int, y_adv: int,
                target_class: int, model_id: int, checkpoint: Path,
                output: Path) -> None:
    print(f"    target {target_position + 1}: index {target_idx}", flush=True)
    head = target_and_head_quantities(net, static, target, y_adv)
    backbone, backend = poison._backbone_gradient_interactions(
        net, candidates, target, y_adv, cli.jacobian_batch_size)
    exact = backbone + head["classifier_grad_dot"]

    representation_path = output.with_name(
        f"surrogate_{model_id:03d}_representation_ntk.npy")
    trace, full_path = representation_ntk_exact(
        net, candidates, target, cli.jacobian_batch_size,
        cli.representation_ntk_mode, representation_path, cli.max_full_ntk_gb,
        trace_probes=cli.ntk_trace_probes,
        random_seed=cli.seed + 1_000_003 * model_id + 10_007 * target_idx,
    )

    validation_indices: list[int] = []
    validation_direct: list[float] = []
    validation_decomposed: list[float] = []
    if cli.validation_samples:
        positions = np.linspace(
            0, len(candidates) - 1,
            num=min(cli.validation_samples, len(candidates)), dtype=np.int64)
        for position in np.unique(positions):
            direct = direct_full_gradient_dot(
                net, candidates[int(position)], target, y_adv)
            decomposed = float(exact[int(position)].detach())
            scale = max(
                abs(direct), abs(decomposed),
                abs(float(backbone[int(position)].detach())),
                abs(float(head["classifier_grad_dot"][int(position)].detach())),
            )
            tolerance = cli.validation_atol + cli.validation_rtol * scale
            if not math.isfinite(direct) or abs(direct - decomposed) > tolerance:
                die("direct all-parameter gradient check failed for "
                    f"target={target_idx}, surrogate={model_id}, "
                    f"candidate={candidate_indices[position]}: direct={direct:.9g}, "
                    f"decomposition={decomposed:.9g}, tolerance={tolerance:.3g}")
            validation_indices.append(int(candidate_indices[position]))
            validation_direct.append(direct)
            validation_decomposed.append(decomposed)

    margin = tensor_numpy(static["margin"])
    feature_dot = tensor_numpy(head["feature_dot"])
    interaction = tensor_numpy(backbone)
    exact_np = tensor_numpy(exact)
    z_margin = zscore(margin)
    z_feature_dot = zscore(feature_dot)
    z_interaction = zscore(interaction)
    paper_beta0 = z_feature_dot - z_margin
    paper_beta1 = paper_beta0 + z_interaction
    cosine_distance = 1.0 - tensor_numpy(head["feature_cosine"])
    l2_distance = tensor_numpy(head["feature_l2_sq"])
    repo_cosine_beta0 = zscore(cosine_distance) + z_margin
    repo_l2_beta0 = zscore(l2_distance) + z_margin

    checkpoint_stat = checkpoint.stat()
    payload: dict[str, object] = {
        "schema_version": np.array(SCHEMA_VERSION, dtype=np.int16),
        "dataset": np.array(cli.dataset),
        "model": np.array(cli.model),
        "class_pair": np.array(cli.class_pair),
        "pair_order": np.array(cli.pair_order),
        "seed": np.array(cli.seed, dtype=np.int64),
        "target_idx": np.array(target_idx, dtype=np.int64),
        "target_position": np.array(target_position, dtype=np.int32),
        "target_class": np.array(target_class, dtype=np.int16),
        "adversarial_label": np.array(y_adv, dtype=np.int16),
        "surrogate_id": np.array(model_id, dtype=np.int16),
        "surrogate_count_requested": np.array(cli.num_surrogates, dtype=np.int16),
        "checkpoint": np.array(str(checkpoint.resolve())),
        "checkpoint_size": np.array(checkpoint_stat.st_size, dtype=np.int64),
        "checkpoint_mtime_ns": np.array(checkpoint_stat.st_mtime_ns, dtype=np.int64),
        "candidate_scope": np.array(cli.candidate_scope),
        "candidate_count": np.array(len(candidate_indices), dtype=np.int32),
        "candidate_idx": candidate_indices,
        "candidate_label": candidate_labels,
        "num_classes": np.array(static["logits"].shape[1], dtype=np.int16),
        "feature_dim": np.array(static["features"].shape[1], dtype=np.int32),
        "classifier_has_bias": np.array(bool(
            (net.module if isinstance(net, nn.DataParallel) else net).classifier.bias
            is not None)),
        "jacobian_backend": np.array(backend),
        "representation_ntk_mode": np.array(cli.representation_ntk_mode),
        "ntk_trace_probes": np.array(
            cli.ntk_trace_probes
            if cli.representation_ntk_mode == "trace-hutchinson" else 0,
            dtype=np.int32),
        "representation_ntk_full_path": np.array(full_path or ""),
        "candidate_logits": tensor_numpy(static["logits"]),
        "candidate_residual": tensor_numpy(static["residual"]),
        "target_logits": tensor_numpy(head["target_logits"]),
        "target_residual": tensor_numpy(head["target_residual"]),
        "M_logit_margin": margin,
        "R_feature_dot_hiTht": feature_dot,
        "feature_cosine": tensor_numpy(head["feature_cosine"]),
        "feature_l2_sq": l2_distance,
        "candidate_feature_norm": tensor_numpy(static["feature_norm"]),
        "target_feature_norm": tensor_numpy(head["target_feature_norm"]),
        "residual_dot_riTrt": tensor_numpy(head["residual_dot"]),
        "classifier_weight_grad_dot": tensor_numpy(
            head["classifier_weight_grad_dot"]),
        "classifier_bias_grad_dot": tensor_numpy(
            head["classifier_bias_grad_dot"]),
        "classifier_grad_dot": tensor_numpy(head["classifier_grad_dot"]),
        "A_backbone_grad_dot": interaction,
        "JiT_Jt_backbone_loss": interaction,
        "representation_ntk_residual_contraction": interaction,
        "exact_full_grad_dot": exact_np,
        "giT_gt_full": exact_np,
        "z_M": z_margin,
        "z_R": z_feature_dot,
        "z_A": z_interaction,
        "paper_score_beta0": paper_beta0.astype(np.float32),
        "paper_score_beta1": paper_beta1.astype(np.float32),
        "repo_cosine_cost_beta0": repo_cosine_beta0.astype(np.float32),
        "repo_cosine_cost_beta1": (repo_cosine_beta0 - z_interaction).astype(np.float32),
        "repo_l2_cost_beta0": repo_l2_beta0.astype(np.float32),
        "repo_l2_cost_beta1": (repo_l2_beta0 - z_interaction).astype(np.float32),
        "rank_M_low": stable_rank(margin, largest=False),
        "rank_R_high": stable_rank(feature_dot, largest=True),
        "rank_A_high": stable_rank(interaction, largest=True),
        "rank_exact_high": stable_rank(exact_np, largest=True),
        "rank_paper_beta0_high": stable_rank(paper_beta0, largest=True),
        "rank_paper_beta1_high": stable_rank(paper_beta1, largest=True),
        "validation_candidate_idx": np.asarray(validation_indices, dtype=np.int64),
        "validation_direct_full_grad_dot": np.asarray(validation_direct, dtype=np.float64),
        "validation_decomposed_full_grad_dot": np.asarray(
            validation_decomposed, dtype=np.float64),
    }
    if trace is not None:
        payload["representation_ntk_trace_JiJt"] = trace
    atomic_npz(output, **payload)
    print(f"      wrote {output}", flush=True)
    del backbone, exact


def pearson(left: np.ndarray, right: np.ndarray) -> float:
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    if len(left) < 2 or left.std() == 0 or right.std() == 0:
        return float("nan")
    return float(np.corrcoef(left, right)[0, 1])


def spearman(left: np.ndarray, right: np.ndarray) -> float:
    # Candidate values are effectively continuous; stable ranks make tie handling
    # deterministic and avoid a SciPy dependency in the exporter.
    return pearson(stable_rank(left, largest=False),
                   stable_rank(right, largest=False))


def aggregate_target(cli: argparse.Namespace, combo_dir: Path, target_idx: int,
                     target_position: int, surrogate_ids: Sequence[int]) -> dict:
    target_dir = combo_dir / f"target_{target_idx}"
    paths = [target_dir / f"surrogate_{model_id:03d}.npz"
             for model_id in surrogate_ids]
    missing = [path for path in paths if not path.is_file()]
    if missing:
        die("cannot aggregate target with missing shards:\n" +
            "\n".join(str(path) for path in missing))

    names = (
        "M_logit_margin", "R_feature_dot_hiTht", "A_backbone_grad_dot",
        "exact_full_grad_dot", "classifier_weight_grad_dot",
        "classifier_bias_grad_dot", "classifier_grad_dot",
        "residual_dot_riTrt", "feature_cosine", "feature_l2_sq",
    )
    stacks: dict[str, list[np.ndarray]] = {name: [] for name in names}
    candidate_idx = None
    candidate_label = None
    trace_stack: list[np.ndarray] = []
    for path in paths:
        with np.load(path, allow_pickle=False) as data:
            current = np.asarray(data["candidate_idx"], dtype=np.int64)
            if candidate_idx is None:
                candidate_idx = current
                candidate_label = np.asarray(data["candidate_label"], dtype=np.int16)
            elif not np.array_equal(candidate_idx, current):
                die(f"candidate order differs in {path}")
            for name in names:
                stacks[name].append(np.asarray(data[name], dtype=np.float64))
            if "representation_ntk_trace_JiJt" in data:
                trace_stack.append(np.asarray(
                    data["representation_ntk_trace_JiJt"], dtype=np.float64))

    assert candidate_idx is not None and candidate_label is not None
    mean = {name: np.mean(np.stack(values), axis=0)
            for name, values in stacks.items()}
    std = {name: np.std(np.stack(values), axis=0, ddof=0)
           for name, values in stacks.items()}
    z_m = zscore(mean["M_logit_margin"])
    z_r = zscore(mean["R_feature_dot_hiTht"])
    z_a = zscore(mean["A_backbone_grad_dot"])
    paper0 = z_r - z_m
    paper1 = paper0 + z_a
    repo_cos0 = zscore(1.0 - mean["feature_cosine"]) + z_m
    repo_l20 = zscore(mean["feature_l2_sq"]) + z_m
    exact = mean["exact_full_grad_dot"]

    output = target_dir / "ensemble.npz"
    payload: dict[str, object] = {
        "schema_version": np.array(SCHEMA_VERSION, dtype=np.int16),
        "dataset": np.array(cli.dataset),
        "model": np.array(cli.model),
        "class_pair": np.array(cli.class_pair),
        "target_idx": np.array(target_idx, dtype=np.int64),
        "target_position": np.array(target_position, dtype=np.int32),
        "surrogate_ids": np.asarray(surrogate_ids, dtype=np.int16),
        "surrogate_count": np.array(len(surrogate_ids), dtype=np.int16),
        "candidate_idx": candidate_idx,
        "candidate_label": candidate_label,
        "z_M_after_mean": z_m,
        "z_R_after_mean": z_r,
        "z_A_after_mean": z_a,
        "paper_score_beta0": paper0.astype(np.float32),
        "paper_score_beta1": paper1.astype(np.float32),
        "repo_cosine_cost_beta0": repo_cos0.astype(np.float32),
        "repo_cosine_cost_beta1": (repo_cos0 - z_a).astype(np.float32),
        "repo_l2_cost_beta0": repo_l20.astype(np.float32),
        "repo_l2_cost_beta1": (repo_l20 - z_a).astype(np.float32),
        "rank_exact_high": stable_rank(exact, largest=True),
        "rank_paper_beta0_high": stable_rank(paper0, largest=True),
        "rank_paper_beta1_high": stable_rank(paper1, largest=True),
    }
    for name in names:
        payload[f"mean_{name}"] = mean[name].astype(np.float32)
        payload[f"std_{name}"] = std[name].astype(np.float32)
    if trace_stack:
        if len(trace_stack) != len(paths):
            die(f"only {len(trace_stack)}/{len(paths)} shards have representation "
                f"NTK traces for target {target_idx}")
        trace_values = np.stack(trace_stack)
        payload["mean_representation_ntk_trace_JiJt"] = trace_values.mean(0).astype(np.float32)
        payload["std_representation_ntk_trace_JiJt"] = trace_values.std(0).astype(np.float32)
    atomic_npz(output, **payload)

    return {
        "target_idx": target_idx,
        "candidate_idx": candidate_idx,
        "candidate_label": candidate_label,
        "mean": mean,
        "z_m": z_m,
        "z_r": z_r,
        "z_a": z_a,
        "paper0": paper0,
        "paper1": paper1,
        "repo_cos0": repo_cos0,
        "repo_l20": repo_l20,
        "corr": {
            "pearson_paper_beta0_vs_exact": pearson(paper0, exact),
            "spearman_paper_beta0_vs_exact": spearman(paper0, exact),
            "pearson_paper_beta1_vs_exact": pearson(paper1, exact),
            "spearman_paper_beta1_vs_exact": spearman(paper1, exact),
            "pearson_A_vs_exact": pearson(mean["A_backbone_grad_dot"], exact),
            "spearman_A_vs_exact": spearman(mean["A_backbone_grad_dot"], exact),
        },
    }


def write_ensemble_csv(combo_dir: Path, aggregates: Sequence[dict]) -> None:
    fields = [
        "target_idx", "candidate_idx", "candidate_label",
        "mean_M", "mean_R_hiTht", "mean_A", "mean_exact_giTgt",
        "mean_classifier_weight", "mean_classifier_bias",
        "mean_residual_dot", "mean_feature_cosine", "mean_feature_l2_sq",
        "z_M", "z_R", "z_A", "paper_score_beta0", "paper_score_beta1",
        "rank_exact_high", "rank_paper_beta0_high", "rank_paper_beta1_high",
    ]

    def rows() -> Iterable[dict]:
        for aggregate in aggregates:
            exact_rank = stable_rank(
                aggregate["mean"]["exact_full_grad_dot"], largest=True)
            beta0_rank = stable_rank(aggregate["paper0"], largest=True)
            beta1_rank = stable_rank(aggregate["paper1"], largest=True)
            for position, candidate_idx in enumerate(aggregate["candidate_idx"]):
                mean = aggregate["mean"]
                yield {
                    "target_idx": aggregate["target_idx"],
                    "candidate_idx": int(candidate_idx),
                    "candidate_label": int(aggregate["candidate_label"][position]),
                    "mean_M": float(mean["M_logit_margin"][position]),
                    "mean_R_hiTht": float(mean["R_feature_dot_hiTht"][position]),
                    "mean_A": float(mean["A_backbone_grad_dot"][position]),
                    "mean_exact_giTgt": float(mean["exact_full_grad_dot"][position]),
                    "mean_classifier_weight": float(
                        mean["classifier_weight_grad_dot"][position]),
                    "mean_classifier_bias": float(
                        mean["classifier_bias_grad_dot"][position]),
                    "mean_residual_dot": float(mean["residual_dot_riTrt"][position]),
                    "mean_feature_cosine": float(mean["feature_cosine"][position]),
                    "mean_feature_l2_sq": float(mean["feature_l2_sq"][position]),
                    "z_M": float(aggregate["z_m"][position]),
                    "z_R": float(aggregate["z_r"][position]),
                    "z_A": float(aggregate["z_a"][position]),
                    "paper_score_beta0": float(aggregate["paper0"][position]),
                    "paper_score_beta1": float(aggregate["paper1"][position]),
                    "rank_exact_high": int(exact_rank[position]),
                    "rank_paper_beta0_high": int(beta0_rank[position]),
                    "rank_paper_beta1_high": int(beta1_rank[position]),
                }

    atomic_csv(combo_dir / "ensemble_candidates.csv", fields, rows())


def run(cli: argparse.Namespace) -> None:
    device = resolve_device(cli.device)
    poison.set_seed(cli.seed)
    parsed = production_args(cli)
    print(f"loading {cli.dataset} on {device}", flush=True)
    ctx = poison.build_context(parsed, str(device))
    # final_update contexts store the input device string; use a torch.device here
    # for uniform checks while leaving all tensors exactly as production built them.
    ctx["device"] = device

    y_adv, target_class = poison.parse_pair(
        cli.class_pair, ctx["class_names"], cli.pair_order)
    targets = load_target_indices(cli.target_file, cli.class_pair)
    if cli.max_targets:
        targets = targets[:cli.max_targets]
    bad_targets = [index for index in targets
                   if index < 0 or index >= len(ctx["test_labs"])]
    if bad_targets:
        die(f"out-of-range target indices: {bad_targets}")
    wrong_class = [index for index in targets
                   if int(ctx["test_labs"][index]) != target_class]
    if wrong_class:
        die(f"targets {wrong_class} are not class {target_class} "
            f"({ctx['class_names'][target_class]})")

    if cli.candidate_scope == "adversarial-class":
        class_indices = (ctx["train_labs"] == y_adv).nonzero(as_tuple=True)[0]
    else:
        class_indices = torch.arange(len(ctx["train_labs"]), device=device)
    if cli.max_candidates:
        class_indices = class_indices[:cli.max_candidates]
    candidates = ctx["train_imgs"][class_indices]
    candidate_indices = tensor_numpy(class_indices, dtype=np.int64)
    candidate_labels = tensor_numpy(ctx["train_labs"][class_indices], dtype=np.int16)
    if not len(candidates):
        die("candidate pool is empty")

    surrogate_ids = (parse_int_list(cli.surrogate_ids)
                     if cli.surrogate_ids else list(range(cli.num_surrogates)))
    if surrogate_ids[-1] >= cli.num_surrogates:
        die("--surrogate-ids must be smaller than --num-surrogates")
    paths = ensure_checkpoints(cli, parsed, ctx, surrogate_ids)

    combo_dir = (cli.output_root / cli.dataset / cli.model /
                 safe_slug(cli.class_pair))
    combo_dir.mkdir(parents=True, exist_ok=True)
    print(f"candidate pool: {len(candidates)}; targets: {len(targets)}; "
          f"surrogates: {len(surrogate_ids)}", flush=True)

    for model_id, checkpoint in zip(surrogate_ids, paths):
        print(f"  surrogate {model_id}: {checkpoint}", flush=True)
        net = load_net(cli, ctx, checkpoint, model_id)
        static = forward_candidates(
            net, candidates, y_adv, cli.forward_batch_size)
        for target_position, target_idx in enumerate(targets):
            target_dir = combo_dir / f"target_{target_idx}"
            output = target_dir / f"surrogate_{model_id:03d}.npz"
            if output.is_file() and not cli.force:
                if shard_valid(output, cli, target_idx, model_id,
                               candidate_indices, checkpoint):
                    print(f"    complete, skipping {output}", flush=True)
                    continue
                die(f"incompatible/incomplete shard {output}; inspect it or use --force")
            build_shard(
                cli, net, static, candidates, candidate_indices, candidate_labels,
                ctx["test_imgs"][target_idx], target_idx, target_position,
                y_adv, target_class, model_id, checkpoint, output,
            )
            if device.type == "cuda":
                torch.cuda.empty_cache()
        del static, net
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    aggregates = [aggregate_target(cli, combo_dir, target_idx, position, surrogate_ids)
                  for position, target_idx in enumerate(targets)]
    write_ensemble_csv(combo_dir, aggregates)

    checkpoint_metadata = []
    for model_id, path in zip(surrogate_ids, paths):
        stat = path.stat()
        checkpoint_metadata.append({
            "surrogate_id": model_id,
            "path": str(path.resolve()),
            "size": stat.st_size,
            "mtime_ns": stat.st_mtime_ns,
        })
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "generated_by": Path(__file__).name,
        "generated_at_unix": time.time(),
        "dataset": cli.dataset,
        "model": cli.model,
        "class_pair": cli.class_pair,
        "pair_order": cli.pair_order,
        "adversarial_label": y_adv,
        "target_class": target_class,
        "class_names": list(ctx["class_names"]),
        "target_file": str(cli.target_file.resolve()),
        "target_indices": targets,
        "candidate_scope": cli.candidate_scope,
        "candidate_count": len(candidate_indices),
        "surrogate_ids": surrogate_ids,
        "checkpoints": checkpoint_metadata,
        "representation_ntk_mode": cli.representation_ntk_mode,
        "ntk_trace_probes": (
            cli.ntk_trace_probes
            if cli.representation_ntk_mode == "trace-hutchinson" else 0),
        "field_definitions": {
            "M_logit_margin": "z_yadv(x_i) - max_{c != yadv} z_c(x_i)",
            "R_feature_dot_hiTht": "h(x_i)^T h(x_t), raw and unnormalized",
            "A_backbone_grad_dot": "<grad_phi CE(x_i,yadv), grad_phi CE(x_t,yadv)>",
            "JiT_Jt_backbone_loss": "alias of A for scalar loss Jacobians",
            "representation_ntk_residual_contraction":
                "r_i^T W [J_h(x_i)J_h(x_t)^T] W^T r_t; alias of A",
            "classifier_weight_grad_dot": "(r_i^T r_t)(h_i^T h_t)",
            "classifier_bias_grad_dot": "r_i^T r_t when the linear head has bias",
            "exact_full_grad_dot": "A + classifier weight + classifier bias",
            "paper_score_beta0": "z(R) - z(M); higher ranks first",
            "paper_score_beta1": "z(R) - z(M) + z(A); higher ranks first",
            "repo_cosine_cost_beta0": "z(1-cos(h_i,h_t)) + z(M); lower ranks first",
            "repo_cosine_cost_beta1": "repo cosine cost - z(A); lower ranks first",
            "representation_ntk_trace_JiJt":
                "tr(J_h(x_i)J_h(x_t)^T), present in trace-exact or "
                "trace-hutchinson mode; contains no W or loss residuals",
        },
        "target_correlations": [
            {"target_idx": aggregate["target_idx"], **aggregate["corr"]}
            for aggregate in aggregates
        ],
        "notes": [
            "Candidate and target losses both use y_adv, as in Equation (2).",
            "Ensemble files average raw per-surrogate terms before standardization.",
            "Ranks are one-based and stable; high/low direction is in each field name.",
            "No poison perturbations or victims are created by this program.",
        ],
    }
    atomic_json(combo_dir / "manifest.json", manifest)
    print(f"complete: {combo_dir}", flush=True)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export exact gradient alignment and GRAFT ranking quantities.")
    parser.add_argument("--dataset", required=True,
                        choices=["CIFAR10", "CIFAR100", "SVHN", "TinyImageNet"])
    parser.add_argument("--model", required=True, choices=poison.SUPPORTED_MODELS)
    parser.add_argument("--class-pair", required=True,
                        help="pair accepted by final_update, normally adversarial-target")
    parser.add_argument("--pair-order", default="poison-target",
                        choices=["poison-target", "target-poison"])
    parser.add_argument("--target-file", type=Path, required=True)
    parser.add_argument("--data-path", type=Path, default=Path("./data"))
    parser.add_argument("--cache-dir", type=Path, default=Path("./cache"))
    parser.add_argument("--output-root", type=Path,
                        default=Path("./gradient_alignment_outputs"))
    parser.add_argument("--device", default="auto",
                        help="auto, cpu, cuda, cuda:0, ...")
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--num-surrogates", type=int, default=20)
    parser.add_argument("--surrogate-ids", default=None,
                        help="optional comma-separated subset, e.g. 0,1,2")
    parser.add_argument("--train-missing", action="store_true",
                        help="train any requested cached surrogate that is absent")
    parser.add_argument("--surrogate-epochs", type=int, default=60)
    parser.add_argument("--surrogate-lr", type=float, default=0.1)
    parser.add_argument("--surrogate-bs", type=int, default=128)
    parser.add_argument("--surrogate-decay", nargs="*", type=int, default=[35, 45])
    parser.add_argument("--surrogate-wd", type=float, default=0.0)
    parser.add_argument("--surrogate-aug", action="store_true")

    parser.add_argument("--candidate-scope", default="adversarial-class",
                        choices=["adversarial-class", "all"],
                        help="Algorithm 1 uses adversarial-class; all is diagnostic")
    parser.add_argument("--forward-batch-size", type=int, default=512)
    parser.add_argument("--jacobian-batch-size", type=int, default=64)
    parser.add_argument("--representation-ntk-mode", default="contracted",
                        choices=["contracted", "trace-hutchinson",
                                 "trace-exact", "full-exact"])
    parser.add_argument("--ntk-trace-probes", type=int, default=32,
                        help="Rademacher probes for trace-hutchinson")
    parser.add_argument("--max-full-ntk-gb", type=float, default=50.0)
    parser.add_argument("--validation-samples", type=int, default=1,
                        help="independent direct-gradient checks per shard (0 disables)")
    parser.add_argument("--validation-rtol", type=float, default=5e-4)
    parser.add_argument("--validation-atol", type=float, default=1e-6)
    parser.add_argument("--max-targets", type=int, default=0,
                        help="debug/subset option; 0 means every target in the file")
    parser.add_argument("--max-candidates", type=int, default=0,
                        help="debug/subset option; 0 means the full candidate pool")
    parser.add_argument("--force", action="store_true",
                        help="overwrite existing compatible shards")
    args = parser.parse_args(argv)

    if args.num_surrogates <= 0:
        parser.error("--num-surrogates must be positive")
    if args.forward_batch_size <= 0 or args.jacobian_batch_size <= 0:
        parser.error("batch sizes must be positive")
    if args.ntk_trace_probes <= 0:
        parser.error("--ntk-trace-probes must be positive")
    if args.validation_samples < 0:
        parser.error("--validation-samples must be nonnegative")
    if args.max_targets < 0 or args.max_candidates < 0:
        parser.error("--max-targets/--max-candidates must be nonnegative")
    if args.max_full_ntk_gb <= 0:
        parser.error("--max-full-ntk-gb must be positive")
    args.target_file = args.target_file.expanduser().resolve()
    args.data_path = args.data_path.expanduser().resolve()
    args.cache_dir = args.cache_dir.expanduser().resolve()
    args.output_root = args.output_root.expanduser().resolve()
    if not args.target_file.is_file():
        parser.error(f"target file does not exist: {args.target_file}")
    return args


if __name__ == "__main__":
    run(parse_args())
