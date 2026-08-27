#!/usr/bin/env python3
"""Residual-suppression validation and selector ablation from ``tmp.md``.

GPU modes
---------
``precompute`` trains or loads exactly one cached surrogate/clean victim.
``collect`` computes M, ||r||_2, H, and A for one of the pinned targets.
``ablation`` runs one (attack, selector) poisoning experiment by reusing the
candidate files written by ``collect``.

CPU mode
--------
``analyze`` assigns margin deciles independently inside every (target, model)
pool, writes decile/ablation tables, and makes the requested plots.

The script intentionally imports the production data, model, training, crafting,
and exact-Jacobian code from final_update.py.  This keeps the validation on the
same normalized inputs, checkpoints, adversarial label, and victim protocol as
the rest of PoisonBase.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Iterable, NoReturn, Sequence

import numpy as np
import torch
import torch.nn.functional as F

import final_update as poison


SELECTORS = ("full", "no-margin", "high-margin")
ATTACKS = ("gradmatch", "sapa")
METRICS = ("residual_norm", "abs_head_interaction", "abs_backbone_interaction")
SELECTOR_TAGS = {
    "full": "resfull",
    "no-margin": "resnomargin",
    "high-margin": "reshighmargin",
}
TAG_TO_SELECTOR = {value: key for key, value in SELECTOR_TAGS.items()}


def _die(message: str) -> NoReturn:
    raise SystemExit("ERROR: " + message)


def _atomic_savez(path: Path, **arrays: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp.npz")
    np.savez_compressed(temporary, **arrays)
    os.replace(temporary, path)


def _atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(temporary, path)


def _write_csv(path: Path, fieldnames: Sequence[str], rows: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row.get(name, "") for name in fieldnames})
    os.replace(temporary, path)


def _shared_poison_argv(args: argparse.Namespace) -> list[str]:
    argv = [
        "--dataset", args.dataset,
        "--data_path", args.data_path,
        "--model", args.model,
        "--seed", str(args.seed),
        "--cache_dir", str(Path(args.output_root) / "cache"),
        "--out_dir", str(Path(args.output_root) / "ablation"),
        "--class_pair", args.class_pair,
        "--pair_order", "poison-target",
        "--num_surrogates", str(args.num_surrogates),
        "--surrogate_epochs", str(args.surrogate_epochs),
        "--surrogate_decay", "35", "45",
        "--num_targets", str(args.num_targets),
        "--target_select", str(args.target_select),
        "--num_victims", str(args.num_victims),
        "--victim_epochs", str(args.victim_epochs),
        "--victim_lr", "0.1",
        "--victim_bs", "125",
        "--victim_decay", "40",
        "--victim_wd", "0.0",
        "--rank_on_surrogates",
        "--gpus", args.gpus,
    ]
    if args.target_idx_file:
        argv += ["--target_idx_file", args.target_idx_file]
    return argv


def _poison_args(args: argparse.Namespace, extra: Sequence[str] = ()) -> argparse.Namespace:
    return poison.parse_args(_shared_poison_argv(args) + list(extra))


def run_precompute(args: argparse.Namespace) -> None:
    extra = [
        "--precompute_only",
        "--precompute_part", args.kind,
        "--precompute_id", str(args.cache_id),
    ]
    parsed = _poison_args(args, extra)
    poison.main(parsed)


def _candidate_path(args: argparse.Namespace, target_idx: int) -> Path:
    return (Path(args.output_root) / "candidates" / args.model / args.class_pair /
            f"target_{target_idx}.npz")


def _load_context_and_surrogates(args: argparse.Namespace):
    parsed = _poison_args(args)
    gpus = poison.resolve_gpus(parsed.gpus)
    if not gpus:
        _die("collect/ablation requires a visible CUDA device; use analyze for CPU work")
    device = f"cuda:{gpus[0]}"
    torch.cuda.set_device(gpus[0])
    poison.set_seed(parsed.seed)
    ctx = poison.build_context(parsed, device)
    nets = poison.get_surrogates(
        parsed,
        ctx["train_imgs"], ctx["train_labs"],
        ctx["test_imgs"], ctx["test_labs"],
        ctx["channel"], ctx["num_classes"], ctx["im_size"],
        device, ctx["dsa_param"],
    )
    return parsed, ctx, nets


def _select_targets(parsed: argparse.Namespace, ctx: dict, nets: Sequence[torch.nn.Module]):
    y_adv, target_class = poison.parse_pair(
        parsed.class_pair, ctx["class_names"], parsed.pair_order)
    generator = torch.Generator(device="cpu").manual_seed(parsed.seed)
    targets, scores = poison.select_targets(
        parsed, nets, ctx["test_imgs"], ctx["test_labs"],
        y_adv, target_class, generator)
    if len(targets) != parsed.num_targets:
        _die(f"expected {parsed.num_targets} eligible targets, selected {len(targets)}")
    return y_adv, target_class, targets, scores


@torch.no_grad()
def _head_quantities(net: torch.nn.Module, candidates: torch.Tensor,
                     target: torch.Tensor, y_adv: int, batch_size: int):
    """Return M, ||r||, H, and cosine relevance for one surrogate."""
    net.eval()
    embed = poison.embed_of(net)
    target_batch = target.unsqueeze(0)
    target_logits = net(target_batch)
    target_residual = F.softmax(target_logits, dim=1)
    target_residual[:, y_adv] -= 1.0
    target_feature = embed(target_batch).flatten(1)

    margins, residual_norms, head_terms, relevances = [], [], [], []
    for start in range(0, len(candidates), batch_size):
        batch = candidates[start:start + batch_size]
        logits = net(batch)
        residual = F.softmax(logits, dim=1)
        residual[:, y_adv] -= 1.0

        adversarial_logit = logits[:, y_adv]
        other_logits = logits.clone()
        other_logits[:, y_adv] = float("-inf")
        margins.append(adversarial_logit - other_logits.max(dim=1).values)
        residual_norms.append(residual.norm(dim=1))

        feature = embed(batch).flatten(1)
        residual_dot = (residual * target_residual).sum(dim=1)
        feature_dot = (feature * target_feature).sum(dim=1)
        head_terms.append(residual_dot * feature_dot)
        relevances.append(F.cosine_similarity(
            feature, target_feature.expand(len(feature), -1), dim=1))

    return tuple(torch.cat(parts).detach().cpu().numpy()
                 for parts in (margins, residual_norms, head_terms, relevances))


def run_collect(args: argparse.Namespace) -> None:
    parsed, ctx, nets = _load_context_and_surrogates(args)
    y_adv, _, targets, target_scores = _select_targets(parsed, ctx, nets)
    if not 0 <= args.target_position < len(targets):
        _die(f"--target-position must be in [0, {len(targets) - 1}]")
    target_idx = int(targets[args.target_position])
    output = _candidate_path(args, target_idx)
    if output.exists() and not args.force:
        with np.load(output, allow_pickle=False) as cached:
            if (int(cached["target_idx"]) == target_idx and
                    len(cached["model_id"]) == len(nets) * int(cached["candidate_count"])):
                print(f"candidate cache already complete: {output}")
                return

    class_indices = (ctx["train_labs"] == y_adv).nonzero(as_tuple=True)[0]
    candidates = ctx["train_imgs"][class_indices]
    target = ctx["test_imgs"][target_idx]
    candidate_indices = class_indices.detach().cpu().numpy().astype(np.int64)

    columns: dict[str, list[np.ndarray]] = defaultdict(list)
    backends: list[str] = []
    for model_id, net in enumerate(nets):
        print(f"target {target_idx}: surrogate {model_id + 1}/{len(nets)}", flush=True)
        margin, residual_norm, head_interaction, relevance = _head_quantities(
            net, candidates, target, y_adv, args.forward_batch_size)
        backbone_interaction, backend = poison._backbone_gradient_interactions(
            net, candidates, target, y_adv, args.jacobian_batch_size)
        backends.append(backend)
        count = len(candidate_indices)
        columns["model_id"].append(np.full(count, model_id, dtype=np.int16))
        columns["candidate_idx"].append(candidate_indices)
        columns["margin"].append(margin.astype(np.float32))
        columns["residual_norm"].append(residual_norm.astype(np.float32))
        columns["head_interaction"].append(head_interaction.astype(np.float32))
        columns["backbone_interaction"].append(
            backbone_interaction.detach().cpu().numpy().astype(np.float32))
        columns["relevance"].append(relevance.astype(np.float32))
        del backbone_interaction
        torch.cuda.empty_cache()

    _atomic_savez(
        output,
        schema_version=np.array(1, dtype=np.int16),
        dataset=np.array(args.dataset),
        model=np.array(args.model),
        class_pair=np.array(args.class_pair),
        seed=np.array(args.seed, dtype=np.int64),
        target_idx=np.array(target_idx, dtype=np.int64),
        target_position=np.array(args.target_position, dtype=np.int16),
        target_score=np.array(target_scores[target_idx], dtype=np.float32),
        adversarial_label=np.array(y_adv, dtype=np.int16),
        candidate_count=np.array(len(candidate_indices), dtype=np.int32),
        model_count=np.array(len(nets), dtype=np.int16),
        target_image=target.detach().cpu().numpy().astype(np.float32),
        jacobian_backends=np.asarray(backends),
        **{name: np.concatenate(parts) for name, parts in columns.items()},
    )
    print(f"wrote {len(candidate_indices)} candidates x {len(nets)} models -> {output}")


def _zscore(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    # final_update.standardize uses torch.std's sample standard deviation.
    scale = values.std(ddof=1)
    if not np.isfinite(scale) or scale < 1e-12:
        return np.zeros_like(values)
    return (values - values.mean()) / scale


def _candidate_files(args: argparse.Namespace) -> list[Path]:
    pattern = Path(args.output_root) / "candidates" / args.model / args.class_pair
    return sorted(pattern.glob("target_*.npz"))


def _load_candidate_cache(args: argparse.Namespace) -> dict[int, dict[str, np.ndarray]]:
    caches: dict[int, dict[str, np.ndarray]] = {}
    for path in _candidate_files(args):
        with np.load(path, allow_pickle=False) as loaded:
            cache = {key: loaded[key].copy() for key in loaded.files}
        target_idx = int(cache["target_idx"])
        if int(cache["model_count"]) != args.num_surrogates:
            _die(f"{path} has {int(cache['model_count'])} models, expected {args.num_surrogates}")
        caches[target_idx] = cache
    if len(caches) != args.num_targets:
        _die(f"found {len(caches)} candidate target files, expected {args.num_targets}; "
             "wait for all collect jobs or use the CPU script with ALLOW_PARTIAL=1")
    return caches


def _match_target_cache(caches: dict[int, dict[str, np.ndarray]],
                        target: torch.Tensor) -> dict[str, np.ndarray]:
    target_cpu = target.detach().cpu().numpy()
    matches = [cache for cache in caches.values()
               if cache["target_image"].shape == target_cpu.shape and
               np.allclose(cache["target_image"], target_cpu, rtol=0.0, atol=1e-6)]
    if len(matches) != 1:
        _die(f"could not match current target tensor to exactly one candidate cache "
             f"(matches={len(matches)})")
    return matches[0]


def _selection_from_cache(cache: dict[str, np.ndarray], selector: str,
                          num_poisons: int, gamma: float, beta: float) -> np.ndarray:
    candidate_count = int(cache["candidate_count"])
    model_count = int(cache["model_count"])
    if num_poisons > candidate_count:
        _die(f"poison budget {num_poisons} exceeds candidate pool {candidate_count}")

    def shaped(name: str) -> np.ndarray:
        values = np.asarray(cache[name], dtype=np.float64)
        return values.reshape(model_count, candidate_count)

    relevance = shaped("relevance")
    margin = shaped("margin")
    interaction = shaped("backbone_interaction")
    relevance_z = np.stack([_zscore(row) for row in relevance])
    margin_z = np.stack([_zscore(row) for row in margin])
    interaction_z = np.stack([_zscore(row) for row in interaction])
    no_margin_score = (relevance_z + beta * interaction_z).mean(axis=0)

    if selector == "full":
        score = (relevance_z - gamma * margin_z + beta * interaction_z).mean(axis=0)
        eligible = np.arange(candidate_count)
    elif selector == "no-margin":
        score = no_margin_score
        eligible = np.arange(candidate_count)
    elif selector == "high-margin":
        # Stable rank selection gives exactly ceil(n/4) candidates even if the
        # ensemble margins contain ties.
        ensemble_margin = margin_z.mean(axis=0)
        quartile_size = int(math.ceil(candidate_count / 4.0))
        eligible = np.argsort(ensemble_margin, kind="mergesort")[-quartile_size:]
        score = no_margin_score
    else:
        _die(f"unknown selector {selector!r}")

    order = eligible[np.argsort(-score[eligible], kind="mergesort")]
    return np.asarray(cache["candidate_idx"][:candidate_count], dtype=np.int64)[
        order[:num_poisons]]


def run_ablation(args: argparse.Namespace) -> None:
    caches = _load_candidate_cache(args)
    selector_tag = SELECTOR_TAGS[args.selector]

    def cached_selector(criterion, nets, images_norm, labels, x_t_norm, y_adv,
                        num_poisons, lam, device, base_dist="l2", bs=512):
        del nets, images_norm, labels, y_adv, lam, base_dist, bs
        if criterion != selector_tag:
            _die(f"unexpected selector tag {criterion!r}; expected {selector_tag!r}")
        cache = _match_target_cache(caches, x_t_norm)
        chosen = _selection_from_cache(
            cache, args.selector, num_poisons, args.gamma, args.beta)
        return torch.as_tensor(chosen, dtype=torch.long, device=device)

    poison.select_base_criterion = cached_selector
    extra = [
        "--attack", args.attack,
        "--base", "ours",
        "--budget", str(args.budget),
        "--epsilon", str(args.epsilon),
        "--craft_steps", str(args.craft_steps),
        "--craft_alpha", str(args.craft_alpha),
        "--restarts", str(args.restarts),
        "--craft_ensemble", str(args.craft_ensemble),
        "--base_dist", "cosine",
        "--lambda_margin", "1.0",
        "--clean_baseline",
        "--no_parallel_targets",
    ]
    if args.attack == "sapa":
        extra += ["--sharp_mode", "worst", "--sharp_sigma", str(args.sharp_sigma)]
    parsed = _poison_args(args, extra)
    # A distinct run directory is essential: the three selectors must never share
    # poison caches.  prepare_poisons routes truthy sel_criterion through the
    # monkey-patched selector above.
    parsed.sel_criterion = selector_tag
    parsed.sel_mode = None
    parsed.use_jacobian_score = False
    run_dir = Path(parsed.out_dir) / poison.build_run_name(parsed)
    _atomic_json(run_dir / "residual_selector.json", {
        "attack": args.attack,
        "selector": args.selector,
        "selector_tag": selector_tag,
        "score_direction": "largest",
        "full_score": "z(relevance) - gamma*z(margin) + beta*z(backbone_interaction)",
        "no_margin_score": "z(relevance) + beta*z(backbone_interaction)",
        "high_margin_gate": "top quartile of ensemble-mean z(margin)",
        "gamma": args.gamma,
        "beta": args.beta,
        "candidate_root": str(Path(args.output_root) / "candidates"),
        "num_targets": args.num_targets,
        "num_victims": args.num_victims,
        "num_surrogates": args.num_surrogates,
    })
    poison.main(parsed)


def _decile_ids(margin: np.ndarray) -> np.ndarray:
    order = np.argsort(margin, kind="mergesort")
    decile = np.empty(len(margin), dtype=np.int8)
    decile[order] = np.minimum(9, np.arange(len(margin)) * 10 // len(margin)) + 1
    return decile


def _candidate_decile_rows(files: Sequence[Path]):
    pool_rows: list[dict] = []
    pooled_values: dict[tuple[int, str], list[np.ndarray]] = defaultdict(list)
    for path in files:
        with np.load(path, allow_pickle=False) as data:
            model = str(data["model"])
            class_pair = str(data["class_pair"])
            target_idx = int(data["target_idx"])
            model_count = int(data["model_count"])
            candidate_count = int(data["candidate_count"])
            for model_id in range(model_count):
                sl = slice(model_id * candidate_count, (model_id + 1) * candidate_count)
                values = {
                    "residual_norm": np.asarray(data["residual_norm"][sl]),
                    "abs_head_interaction": np.abs(data["head_interaction"][sl]),
                    "abs_backbone_interaction": np.abs(data["backbone_interaction"][sl]),
                }
                deciles = _decile_ids(np.asarray(data["margin"][sl]))
                for decile in range(1, 11):
                    take = deciles == decile
                    row = {
                        "model": model,
                        "class_pair": class_pair,
                        "target_idx": target_idx,
                        "model_id": model_id,
                        "margin_decile": decile,
                        "candidate_count": int(take.sum()),
                    }
                    for metric, metric_values in values.items():
                        chosen = metric_values[take]
                        row[f"{metric}_median"] = float(np.median(chosen))
                        row[f"{metric}_p90"] = float(np.percentile(chosen, 90))
                        pooled_values[(decile, metric)].append(chosen)
                    pool_rows.append(row)

    pooled_rows = []
    for decile in range(1, 11):
        row = {"margin_decile": decile}
        count = 0
        for metric in METRICS:
            values = np.concatenate(pooled_values[(decile, metric)])
            count = len(values)
            row[f"{metric}_median"] = float(np.median(values))
            row[f"{metric}_p90"] = float(np.percentile(values, 90))
        row["candidate_count"] = count
        pooled_rows.append(row)
    return pool_rows, pooled_rows


def _decile_fieldnames() -> list[str]:
    return (["model", "class_pair", "target_idx", "model_id", "margin_decile",
             "candidate_count"] +
            [f"{metric}_{stat}" for metric in METRICS for stat in ("median", "p90")])


def _plot_deciles(output: Path, pooled_rows: Sequence[dict]) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = {
        "residual_norm": r"$||r_i||_2$",
        "abs_head_interaction": r"$|H_i|$",
        "abs_backbone_interaction": r"$|A_i|$",
    }
    x = np.arange(1, 11)
    figure, axes = plt.subplots(1, 3, figsize=(13.5, 3.8), constrained_layout=True)
    for axis, metric in zip(axes, METRICS):
        median = [row[f"{metric}_median"] for row in pooled_rows]
        p90 = [row[f"{metric}_p90"] for row in pooled_rows]
        axis.plot(x, median, marker="o", label="median")
        axis.plot(x, p90, marker="s", label="90th percentile")
        axis.set_xlabel("Signed-margin decile (low to high)")
        axis.set_ylabel(labels[metric])
        axis.grid(alpha=0.25)
    axes[0].legend(frameon=False)
    figure.savefig(output, dpi=220)
    plt.close(figure)


def _all_result_rows(run_dir: Path) -> dict[tuple[int, int], dict]:
    rows: dict[tuple[int, int], dict] = {}
    for path in sorted(run_dir.glob("results*.csv")):
        try:
            with path.open(newline="") as handle:
                for row in csv.DictReader(handle):
                    if row.get("target_idx", "").isdigit() and row.get("victim_id", "").isdigit():
                        rows[(int(row["target_idx"]), int(row["victim_id"]))] = row
        except OSError:
            continue
    return rows


def _identify_run(run_dir: Path):
    selector = next((TAG_TO_SELECTOR[tag] for tag in TAG_TO_SELECTOR
                     if f"_sel{tag}" in run_dir.name), None)
    attack = next((attack for attack in ATTACKS if f"_{attack}_" in run_dir.name), None)
    return attack, selector


def _selected_metrics(cache: dict[str, np.ndarray], selected: set[int]) -> dict[str, float]:
    take = np.isin(cache["candidate_idx"], np.fromiter(selected, dtype=np.int64))
    if not take.any():
        _die("selected base indices do not occur in the matching candidate cache")
    return {
        "selected_mean_margin": float(np.mean(cache["margin"][take])),
        "selected_mean_residual_norm": float(np.mean(cache["residual_norm"][take])),
        "selected_mean_abs_head_interaction": float(np.mean(np.abs(cache["head_interaction"][take]))),
        "selected_mean_abs_backbone_interaction": float(np.mean(np.abs(cache["backbone_interaction"][take]))),
    }


def _ablation_rows(args: argparse.Namespace, caches: dict[int, dict[str, np.ndarray]],
                   allow_partial: bool) -> tuple[list[dict], list[dict]]:
    root = Path(args.output_root) / "ablation"
    found: dict[tuple[str, str], Path] = {}
    for run_dir in root.iterdir() if root.exists() else ():
        if not run_dir.is_dir():
            continue
        attack, selector = _identify_run(run_dir)
        if attack and selector:
            found[(attack, selector)] = run_dir

    rows = []
    target_rows = []
    missing = []
    for attack in ATTACKS:
        for selector in SELECTORS:
            run_dir = found.get((attack, selector))
            if run_dir is None:
                missing.append(f"{attack}/{selector}")
                continue
            trials = _all_result_rows(run_dir)
            expected_trials = args.num_targets * args.num_victims
            if len(trials) < expected_trials and not allow_partial:
                missing.append(f"{attack}/{selector} ({len(trials)}/{expected_trials} trials)")
                continue

            per_target = []
            for target_idx, cache in caches.items():
                base_path = run_dir / "poison_cache" / f"base_{target_idx}.json"
                if not base_path.exists():
                    continue
                with base_path.open() as handle:
                    selected = {int(value) for value in json.load(handle)}
                metrics = _selected_metrics(cache, selected)
                target_trials = [row for (trial_target, _), row in trials.items()
                                 if trial_target == target_idx]
                target_row = {
                    "attack": attack,
                    "selector": selector,
                    "target_idx": target_idx,
                    "selected_count": len(selected),
                    "completed_victims": len(target_trials),
                    "expected_victims": args.num_victims,
                    "asr": (float(np.mean([float(row["success"]) for row in target_trials]))
                            if target_trials else float("nan")),
                    **metrics,
                }
                per_target.append(target_row)
                target_rows.append(target_row)
            if len(per_target) < args.num_targets and not allow_partial:
                missing.append(
                    f"{attack}/{selector} ({len(per_target)}/{args.num_targets} base caches)")
                continue
            summary = {
                "attack": attack,
                "selector": selector,
                "targets_with_bases": len(per_target),
                "completed_trials": len(trials),
                "expected_trials": expected_trials,
                "asr": (float(np.mean([float(row["success"]) for row in trials.values()]))
                        if trials else float("nan")),
                "run_dir": str(run_dir),
            }
            for metric in (
                "selected_mean_margin", "selected_mean_residual_norm",
                "selected_mean_abs_head_interaction", "selected_mean_abs_backbone_interaction",
            ):
                summary[metric] = (float(np.mean([row[metric] for row in per_target]))
                                   if per_target else float("nan"))
            rows.append(summary)

    if missing and not allow_partial:
        _die("ablation is incomplete: " + ", ".join(missing))
    if missing:
        print("warning: partial ablation: " + ", ".join(missing), file=sys.stderr)
    return rows, target_rows


def _plot_ablation(output: Path, rows: Sequence[dict]) -> None:
    if not rows:
        return
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    lookup = {(row["attack"], row["selector"]): row for row in rows}
    figure, axes = plt.subplots(1, 2, figsize=(10, 3.8), constrained_layout=True)
    x = np.arange(len(SELECTORS))
    width = 0.36
    for attack_id, attack in enumerate(ATTACKS):
        values = [lookup.get((attack, selector), {}).get("asr", np.nan)
                  for selector in SELECTORS]
        axes[0].bar(x + (attack_id - 0.5) * width, values, width, label=attack)
    axes[0].set_xticks(x, ["Full", "No margin", "High margin"])
    axes[0].set_ylabel("Attack success rate")
    axes[0].set_ylim(0, 1)
    axes[0].legend(frameon=False)
    axes[0].grid(axis="y", alpha=0.25)

    # Selection is attack-independent in definition. Average the two realized
    # cache summaries when both attacks have completed.
    for selector in SELECTORS:
        selected = [lookup[(attack, selector)]["selected_mean_residual_norm"]
                    for attack in ATTACKS if (attack, selector) in lookup]
        axes[1].bar(selector, np.mean(selected) if selected else np.nan)
    axes[1].set_xticks(x, ["Full", "No margin", "High margin"])
    axes[1].set_ylabel(r"Selected mean $||r_i||_2$")
    axes[1].grid(axis="y", alpha=0.25)
    figure.savefig(output, dpi=220)
    plt.close(figure)


def run_analyze(args: argparse.Namespace) -> None:
    files = _candidate_files(args)
    if not files:
        _die(f"no candidate files under {Path(args.output_root) / 'candidates'}")
    if len(files) != args.num_targets and not args.allow_partial:
        _die(f"found {len(files)} target files, expected {args.num_targets}")

    tables = Path(args.output_root) / "tables"
    figures = Path(args.output_root) / "figures"
    tables.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)
    pool_rows, pooled_rows = _candidate_decile_rows(files)
    _write_csv(tables / "deciles_by_target_model.csv", _decile_fieldnames(), pool_rows)
    pooled_fields = [name for name in _decile_fieldnames()
                     if name not in ("model", "class_pair", "target_idx", "model_id")]
    _write_csv(tables / "deciles_pooled.csv", pooled_fields, pooled_rows)
    _plot_deciles(figures / "candidate_deciles.png", pooled_rows)

    caches = {}
    for path in files:
        with np.load(path, allow_pickle=False) as loaded:
            cache = {key: loaded[key].copy() for key in loaded.files}
        caches[int(cache["target_idx"])] = cache
    ablation_rows, target_rows = _ablation_rows(args, caches, args.allow_partial)
    ablation_fields = [
        "attack", "selector", "targets_with_bases", "completed_trials",
        "expected_trials", "asr", "selected_mean_margin",
        "selected_mean_residual_norm", "selected_mean_abs_head_interaction",
        "selected_mean_abs_backbone_interaction", "run_dir",
    ]
    _write_csv(tables / "selector_ablation.csv", ablation_fields, ablation_rows)
    target_fields = [
        "attack", "selector", "target_idx", "selected_count",
        "completed_victims", "expected_victims", "asr",
        "selected_mean_margin", "selected_mean_residual_norm",
        "selected_mean_abs_head_interaction", "selected_mean_abs_backbone_interaction",
    ]
    _write_csv(tables / "selector_ablation_by_target.csv", target_fields, target_rows)
    _plot_ablation(figures / "selector_ablation.png", ablation_rows)
    print(f"analysis complete: {tables} and {figures}")


def _add_shared(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--output-root", default="logs-proposiot")
    parser.add_argument("--dataset", default="CIFAR10")
    parser.add_argument("--data-path", default="./data")
    parser.add_argument("--model", default="ConvNetBN", choices=poison.SUPPORTED_MODELS)
    parser.add_argument("--class-pair", default="dog-bird")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-targets", type=int, default=5)
    parser.add_argument("--num-victims", type=int, default=5)
    parser.add_argument("--num-surrogates", type=int, default=5)
    parser.add_argument("--surrogate-epochs", type=int, default=60)
    parser.add_argument("--victim-epochs", type=int, default=50)
    parser.add_argument("--target-select", type=poison.target_select_arg, default=70)
    parser.add_argument("--target-idx-file")
    parser.add_argument("--gpus", default="0")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="mode", required=True)

    precompute = subparsers.add_parser("precompute", help="cache one model on one GPU")
    _add_shared(precompute)
    precompute.add_argument("--kind", required=True, choices=("surrogate", "victim"))
    precompute.add_argument("--cache-id", required=True, type=int)
    precompute.set_defaults(func=run_precompute)

    collect = subparsers.add_parser("collect", help="collect one target's candidate metrics")
    _add_shared(collect)
    collect.add_argument("--target-position", required=True, type=int)
    collect.add_argument("--forward-batch-size", type=int, default=512)
    collect.add_argument("--jacobian-batch-size", type=int, default=64)
    collect.add_argument("--force", action="store_true")
    collect.set_defaults(func=run_collect)

    ablation = subparsers.add_parser("ablation", help="run one attack/selector cell")
    _add_shared(ablation)
    ablation.add_argument("--attack", required=True, choices=ATTACKS)
    ablation.add_argument("--selector", required=True, choices=SELECTORS)
    ablation.add_argument("--gamma", type=float, default=1.0)
    ablation.add_argument("--beta", type=float, default=1.0)
    ablation.add_argument("--budget", type=float, default=0.002)
    ablation.add_argument("--epsilon", type=float, default=8.0 / 255.0)
    ablation.add_argument("--craft-steps", type=int, default=250)
    ablation.add_argument("--craft-alpha", type=float, default=1.0 / 255.0)
    ablation.add_argument("--restarts", type=int, default=8)
    ablation.add_argument("--craft-ensemble", type=int, default=5)
    ablation.add_argument("--sharp-sigma", type=float, default=0.05)
    ablation.set_defaults(func=run_ablation)

    analyze = subparsers.add_parser("analyze", help="CPU-only tables and plots")
    _add_shared(analyze)
    analyze.add_argument("--allow-partial", action="store_true")
    analyze.set_defaults(func=run_analyze)

    args = parser.parse_args(argv)
    if args.num_targets <= 0 or args.num_victims <= 0 or args.num_surrogates <= 0:
        parser.error("target, victim, and surrogate counts must be positive")
    if args.mode == "precompute" and not 0 <= args.cache_id < (
            args.num_surrogates if args.kind == "surrogate" else args.num_victims):
        parser.error("--cache-id is outside the requested cache pool")
    if args.mode == "collect" and (
            args.forward_batch_size <= 0 or args.jacobian_batch_size <= 0):
        parser.error("candidate batch sizes must be positive")
    if args.mode == "ablation":
        if not math.isfinite(args.gamma) or args.gamma < 0:
            parser.error("--gamma must be finite and nonnegative")
        if not math.isfinite(args.beta) or args.beta < 0:
            parser.error("--beta must be finite and nonnegative")
        if args.budget < 0 or args.craft_steps <= 0 or args.restarts <= 0:
            parser.error("budget must be nonnegative; craft steps and restarts must be positive")
        if not 0 < args.craft_ensemble <= args.num_surrogates:
            parser.error("--craft-ensemble must be in [1, --num-surrogates]")
    return args


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    Path(args.output_root).mkdir(parents=True, exist_ok=True)
    args.func(args)


if __name__ == "__main__":
    main()
