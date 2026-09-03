#!/usr/bin/env python3
"""Additive vs multiplicative leave-one-surrogate-out ranking transfer."""

from __future__ import annotations

import argparse
import csv
import gc
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import final_update as poison
from utils import get_dataset


PAPER_RHOS = (0.001, 0.002, 0.005, 0.01, 0.02, 0.04)


@dataclass(frozen=True)
class Job:
    dataset: str
    model: str
    class_pair: str
    target_file: Path


def die(message: str) -> "NoReturn":
    raise SystemExit("ERROR: " + message)


def parse_rhos(value: str) -> tuple[float, ...]:
    try:
        rhos = tuple(sorted({float(item) for item in value.split(",")
                             if item.strip()}))
    except ValueError as exc:
        raise argparse.ArgumentTypeError("rhos must be comma-separated numbers") from exc
    if not rhos or rhos[0] <= 0 or rhos[-1] > 1:
        raise argparse.ArgumentTypeError("rhos must lie in (0,1]")
    return rhos


def rho_tag(rho: float) -> str:
    return ("%g" % rho).replace(".", "p")


def safe_slug(value: str) -> str:
    return value.replace("/", "_").replace(" ", "_")


def atomic_csv(path: Path, fieldnames: Sequence[str], rows: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def atomic_npz(path: Path, **arrays: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp.npz")
    np.savez_compressed(temporary, **arrays)
    os.replace(temporary, path)


def atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(text)
    os.replace(temporary, path)


def save_figure(fig: plt.Figure, path: Path, dpi: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.stem + ".tmp" + path.suffix)
    kwargs = {"bbox_inches": "tight"}
    if path.suffix.lower() == ".png":
        kwargs["dpi"] = dpi
    fig.savefig(temporary, **kwargs)
    os.replace(temporary, path)


def load_jobs(path: Path) -> list[Job]:
    jobs: list[Job] = []
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        expected = {"dataset", "model", "class_pair", "target_file"}
        if set(reader.fieldnames or ()) != expected:
            die(f"{path} must have columns {sorted(expected)}")
        for row in reader:
            target_file = Path(row["target_file"])
            if not target_file.is_absolute():
                target_file = path.parent / target_file
            jobs.append(Job(row["dataset"], row["model"], row["class_pair"],
                            target_file.resolve()))
    if not jobs:
        die(f"no jobs in {path}")
    identities = [(job.dataset, job.model, job.class_pair) for job in jobs]
    if len(identities) != len(set(identities)):
        die("jobs must have unique dataset/model/class_pair identities")
    return jobs


def load_target_ids(path: Path, class_pair: str) -> list[int]:
    import json

    try:
        with path.open() as handle:
            payload = json.load(handle)
        values = (payload["pairs"][class_pair]["indices"]
                  if "pairs" in payload else payload[class_pair])
        target_ids = [int(value) for value in values]
    except (OSError, KeyError, TypeError, ValueError) as exc:
        die(f"cannot load {class_pair} targets from {path}: {exc}")
    if not target_ids or len(target_ids) != len(set(target_ids)):
        die(f"target IDs in {path} must be nonempty and unique")
    return target_ids


def labels_of(dataset: object) -> torch.Tensor:
    for attribute in ("targets", "labels"):
        if hasattr(dataset, attribute):
            values = getattr(dataset, attribute)
            if isinstance(values, torch.Tensor):
                return values.detach().cpu().long()
            return torch.as_tensor(np.asarray(values), dtype=torch.long)
    return torch.tensor([int(dataset[index][1]) for index in range(len(dataset))],
                        dtype=torch.long)


def images_at(dataset: object, indices: Sequence[int], device: torch.device,
              chunk_size: int = 512) -> torch.Tensor:
    ids = [int(index) for index in indices]
    if not ids:
        die("cannot load an empty image index set")
    result = None
    for start in range(0, len(ids), chunk_size):
        batch = torch.stack([dataset[index][0]
                             for index in ids[start:start + chunk_size]]).to(device)
        if result is None:
            result = torch.empty((len(ids),) + tuple(batch.shape[1:]),
                                 dtype=batch.dtype, device=device)
        result[start:start + len(batch)] = batch
    assert result is not None
    return result


def checkpoint_directory(args: argparse.Namespace, dataset: str,
                         model: str) -> Path:
    production = argparse.Namespace(
        dataset=dataset,
        model=model,
        cache_dir=str(args.cache_dir),
        surrogate_epochs=args.surrogate_epochs,
        surrogate_lr=args.surrogate_lr,
        surrogate_bs=args.surrogate_bs,
        seed=args.seed,
    )
    return Path(poison.surrogate_dir(production))


def saved_checkpoints(args: argparse.Namespace, dataset: str,
                      model: str) -> list[tuple[int, Path]]:
    directory = checkpoint_directory(args, dataset, model)
    checkpoints: list[tuple[int, Path]] = []
    for path in directory.glob("net_*.pt"):
        match = re.fullmatch(r"net_(\d+)\.pt", path.name)
        if match:
            checkpoints.append((int(match.group(1)), path))
    checkpoints.sort()
    if len(checkpoints) < 2:
        die(f"need at least two saved surrogates in {directory}; found "
            f"{len(checkpoints)}. This program never trains missing checkpoints")
    ids = [model_id for model_id, _ in checkpoints]
    if len(ids) != len(set(ids)):
        die(f"duplicate surrogate IDs in {directory}")
    return checkpoints


def load_net(job: Job, checkpoint: Path, model_id: int, channel: int,
             num_classes: int, image_size: tuple[int, int],
             device: torch.device, seed: int) -> nn.Module:
    net = poison.build_network(job.model, channel, num_classes, image_size,
                               str(device), seed=seed + 1000 + model_id)
    net.load_state_dict(torch.load(checkpoint, map_location=device))
    net.eval()
    for parameter in net.parameters():
        parameter.requires_grad_(False)
    return net


@torch.no_grad()
def candidate_forward(net: nn.Module, candidates: torch.Tensor, y_adv: int,
                      batch_size: int) -> dict[str, torch.Tensor]:
    logits, features = [], []
    embed = poison.embed_of(net)
    for start in range(0, len(candidates), batch_size):
        batch = candidates[start:start + batch_size]
        features.append(embed(batch).detach().flatten(1))
        logits.append(net(batch).detach())
    z = torch.cat(logits)
    h = torch.cat(features)
    p = F.softmax(z, dim=1)
    p[:, y_adv] -= 1.0
    other = z.clone()
    other[:, y_adv] = float("-inf")
    margin = z[:, y_adv] - other.max(dim=1).values
    return {"feature": h, "residual": p, "margin": margin}


@torch.no_grad()
def target_scores(net: nn.Module, static: dict[str, torch.Tensor],
                  target: torch.Tensor, y_adv: int) -> tuple[np.ndarray, ...]:
    target_batch = target.unsqueeze(0)
    target_feature = poison.embed_of(net)(target_batch).detach().flatten(1)
    target_logits = net(target_batch).detach()
    target_residual = F.softmax(target_logits, dim=1)
    target_residual[:, y_adv] -= 1.0

    feature = static["feature"]
    feature_dot = (feature * target_feature).sum(dim=1)
    feature_cosine = F.cosine_similarity(
        feature, target_feature.expand_as(feature), dim=1)
    residual_dot = (static["residual"] * target_residual).sum(dim=1)
    exact_first_term = residual_dot * feature_dot

    # This is exactly the production GRAFT beta=0 component from
    # final_update._ours_pointwise_score, negated so every score ranks high first.
    production_cost = (
        poison.standardize(1.0 - feature_cosine) +
        poison.standardize(static["margin"])
    )
    return tuple(value.detach().cpu().numpy().astype(np.float64, copy=False)
                 for value in (feature_dot, exact_first_term, -production_cost))


def average_percentile_rank(values: np.ndarray) -> np.ndarray:
    """Ascending percentile ranks in [0,1], with exact ties averaged."""
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1 or len(values) < 2 or not np.isfinite(values).all():
        die("rank inputs must be a finite one-dimensional pool with N >= 2")
    order = np.argsort(values, kind="mergesort")
    sorted_values = values[order]
    ranks = np.empty(len(values), dtype=np.float64)
    start = 0
    while start < len(values):
        stop = start + 1
        while stop < len(values) and sorted_values[stop] == sorted_values[start]:
            stop += 1
        average_one_based_rank = 0.5 * ((start + 1) + stop)
        ranks[order[start:stop]] = average_one_based_rank
        start = stop
    result = (ranks - 1.0) / (len(values) - 1.0)
    if result.min() < 0.0 or result.max() > 1.0:
        die("percentile rank escaped [0,1]")
    return result


def spearman(left: np.ndarray, right: np.ndarray) -> float:
    q_left = average_percentile_rank(left)
    q_right = average_percentile_rank(right)
    if np.ptp(q_left) == 0 or np.ptp(q_right) == 0:
        return float("nan")
    return float(np.corrcoef(q_left, q_right)[0, 1])


def top_m_set(values: np.ndarray, candidate_ids: np.ndarray, count: int) -> set[int]:
    values = np.asarray(values, dtype=np.float64)
    candidate_ids = np.asarray(candidate_ids, dtype=np.int64)
    if len(values) != len(candidate_ids) or len(set(candidate_ids.tolist())) != len(values):
        die("candidate IDs must be aligned with scores and unique")
    if count < 1 or count > len(values):
        die(f"invalid top-m request m={count} for N={len(values)}")
    order = np.lexsort((candidate_ids, -values))
    selected = set(candidate_ids[order[:count]].tolist())
    if len(selected) != count:
        die(f"Top_{count} did not contain exactly {count} unique samples")
    return selected


def overlap(left: set[int], right: set[int], count: int) -> float:
    value = len(left & right) / count
    if not 0.0 <= value <= 1.0:
        die(f"overlap outside [0,1]: {value}")
    return value


def cache_path(root: Path, job: Job, target_id: int,
               heldout: int) -> Path:
    return (root / "results" / "add_vs_mult_leave_one_out_candidates" /
            job.dataset / job.model / safe_slug(job.class_pair) /
            f"target_{target_id}" / f"heldout_{heldout:03d}.npz")


def evaluate_target(args: argparse.Namespace, job: Job, target_id: int,
                    candidate_ids: np.ndarray, surrogate_ids: np.ndarray,
                    margins: np.ndarray, relevance: np.ndarray,
                    exact_first: np.ndarray,
                    graft_high: np.ndarray,
                    training_set_size: int) -> list[dict]:
    surrogate_count, candidate_count = margins.shape
    expected_shape = (surrogate_count, candidate_count)
    for name, values in (("R", relevance), ("T", exact_first),
                         ("GRAFT", graft_high)):
        if values.shape != expected_shape:
            die(f"{name} shape {values.shape} != {expected_shape}")
    if len(candidate_ids) != candidate_count:
        die("candidate IDs differ from surrogate score matrices")

    q_d_each = np.stack([average_percentile_rank(-row) for row in margins])
    q_r_each = np.stack([average_percentile_rank(row) for row in relevance])
    rows: list[dict] = []

    for heldout_position, heldout_id in enumerate(surrogate_ids.tolist()):
        keep = np.arange(surrogate_count) != heldout_position
        if keep[heldout_position] or int(keep.sum()) != surrogate_count - 1:
            die("held-out surrogate leaked into the training-side aggregation")

        q_r = q_r_each[keep].mean(axis=0)
        q_d = q_d_each[keep].mean(axis=0)
        if q_r.min() < 0 or q_r.max() > 1 or q_d.min() < 0 or q_d.max() > 1:
            die("aggregated percentile ranks escaped [0,1]")
        score_add = q_r + q_d
        score_mult = q_r * q_d
        score_graft = graft_high[keep].mean(axis=0)
        target_score = exact_first[heldout_position].copy()
        r_bar = relevance[keep].mean(axis=0)
        m_bar = margins[keep].mean(axis=0)

        output = cache_path(args.output_root, job, target_id, heldout_id)
        atomic_npz(
            output,
            dataset=np.array(job.dataset), model=np.array(job.model),
            class_pair=np.array(job.class_pair), target_id=np.int64(target_id),
            heldout_surrogate=np.int64(heldout_id),
            nonheldout_surrogates=surrogate_ids[keep].astype(np.int32),
            candidate_id=candidate_ids.astype(np.int64),
            T=target_score.astype(np.float32),
            S_add=score_add.astype(np.float32),
            S_mult=score_mult.astype(np.float32),
            S_graft=score_graft.astype(np.float32),
            qR=q_r.astype(np.float32), qD=q_d.astype(np.float32),
            Rbar=r_bar.astype(np.float32), Mbar=m_bar.astype(np.float32),
        )

        row = {
            "dataset": job.dataset,
            "model": job.model,
            "class_pair": job.class_pair,
            "target_id": target_id,
            "heldout_surrogate": heldout_id,
            "n_candidates": candidate_count,
            "spearman_add": spearman(score_add, target_score),
            "spearman_mult": spearman(score_mult, target_score),
            "spearman_graft": spearman(score_graft, target_score),
            "candidate_cache": str(output),
        }
        for rho in args.rhos:
            count = poison.rho_to_m(rho, training_set_size)
            if count != int(round(rho * training_set_size)):
                die("rho-to-m differs from the production attack expression")
            if count < 1 or count > candidate_count:
                die(f"rho={rho:g} maps to m={count}, invalid for candidate "
                    f"pool N={candidate_count} in {job.dataset}/{job.model}/"
                    f"{job.class_pair}")
            exact_top = top_m_set(target_score, candidate_ids, count)
            add_top = top_m_set(score_add, candidate_ids, count)
            mult_top = top_m_set(score_mult, candidate_ids, count)
            graft_top = top_m_set(score_graft, candidate_ids, count)
            tag = rho_tag(rho)
            row[f"m_rho_{tag}"] = count
            row[f"overlap_add_exact_rho_{tag}"] = overlap(add_top, exact_top, count)
            row[f"overlap_mult_exact_rho_{tag}"] = overlap(mult_top, exact_top, count)
            row[f"overlap_graft_exact_rho_{tag}"] = overlap(graft_top, exact_top, count)
            row[f"overlap_add_mult_rho_{tag}"] = overlap(add_top, mult_top, count)
        rows.append(row)
    return rows


def collect_job(args: argparse.Namespace, job: Job,
                device: torch.device) -> list[dict]:
    print(f"{job.dataset} / {job.model} / {job.class_pair}", flush=True)
    channel, image_size, num_classes, class_names, _, _, train_set, test_set, _ = \
        get_dataset(job.dataset, str(args.data_path))
    y_adv, target_class = poison.parse_pair(job.class_pair, class_names,
                                             "poison-target")
    train_labels = labels_of(train_set)
    test_labels = labels_of(test_set)
    candidate_positions = (train_labels == y_adv).nonzero(as_tuple=True)[0]
    candidate_ids = candidate_positions.numpy().astype(np.int64, copy=False)
    if len(candidate_ids) < 2 or len(set(candidate_ids.tolist())) != len(candidate_ids):
        die("candidate pool must have at least two unique IDs")
    candidates = images_at(train_set, candidate_ids.tolist(), device,
                           args.image_load_batch_size)

    target_ids = load_target_ids(job.target_file, job.class_pair)
    wrong_targets = [target_id for target_id in target_ids
                     if target_id < 0 or target_id >= len(test_set) or
                     int(test_labels[target_id]) != target_class]
    if wrong_targets:
        die(f"invalid or wrong-class target IDs in {job.target_file}: {wrong_targets}")
    target_images = images_at(test_set, target_ids, device,
                              args.image_load_batch_size)

    checkpoints = saved_checkpoints(args, job.dataset, job.model)
    surrogate_ids = np.asarray([model_id for model_id, _ in checkpoints],
                               dtype=np.int32)
    surrogate_count = len(checkpoints)
    candidate_count = len(candidate_ids)
    margins = np.empty((surrogate_count, candidate_count), dtype=np.float64)
    relevance = {target_id: np.empty((surrogate_count, candidate_count),
                                     dtype=np.float64)
                 for target_id in target_ids}
    exact_first = {target_id: np.empty((surrogate_count, candidate_count),
                                      dtype=np.float64)
                   for target_id in target_ids}
    graft_high = {target_id: np.empty((surrogate_count, candidate_count),
                                     dtype=np.float64)
                  for target_id in target_ids}

    print(f"  candidates={candidate_count}, targets={len(target_ids)}, "
          f"saved_surrogates={surrogate_count}", flush=True)
    for position, (model_id, checkpoint) in enumerate(checkpoints):
        print(f"  loading surrogate {model_id}: {checkpoint}", flush=True)
        net = load_net(job, checkpoint, model_id, channel, num_classes,
                       image_size, device, args.seed)
        static = candidate_forward(net, candidates, y_adv,
                                   args.forward_batch_size)
        margins[position] = static["margin"].detach().cpu().numpy()
        for target_position, target_id in enumerate(target_ids):
            r_value, t_value, graft_value = target_scores(
                net, static, target_images[target_position], y_adv)
            relevance[target_id][position] = r_value
            exact_first[target_id][position] = t_value
            graft_high[target_id][position] = graft_value
        del static, net
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    if not (np.isfinite(margins).all() and
            all(np.isfinite(values).all() for mapping in
                (relevance, exact_first, graft_high) for values in mapping.values())):
        die(f"non-finite candidate output in {job}")

    rows: list[dict] = []
    for target_id in target_ids:
        rows.extend(evaluate_target(
            args, job, target_id, candidate_ids, surrogate_ids, margins,
            relevance[target_id], exact_first[target_id], graft_high[target_id],
            len(train_set)))
    del candidates, target_images
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return rows


def detailed_fields(rhos: Sequence[float]) -> list[str]:
    fields = [
        "dataset", "model", "class_pair", "target_id", "heldout_surrogate",
        "n_candidates", "spearman_add", "spearman_mult", "spearman_graft",
        "candidate_cache",
    ]
    for rho in rhos:
        tag = rho_tag(rho)
        fields.extend([
            f"m_rho_{tag}",
            f"overlap_add_exact_rho_{tag}",
            f"overlap_mult_exact_rho_{tag}",
            f"overlap_graft_exact_rho_{tag}",
            f"overlap_add_mult_rho_{tag}",
        ])
    return fields


def summary_rows(rows: Sequence[dict], rhos: Sequence[float]) -> list[dict]:
    result: list[dict] = []
    for rho in rhos:
        tag = rho_tag(rho)
        output = {"rho": rho, "n_units": len(rows)}
        counts = np.asarray([row[f"m_rho_{tag}"] for row in rows], dtype=int)
        output["m_min"] = int(counts.min())
        output["m_max"] = int(counts.max())
        for key in ("add_exact", "mult_exact", "graft_exact", "add_mult"):
            values = np.asarray([row[f"overlap_{key}_rho_{tag}"] for row in rows],
                                dtype=np.float64)
            output[f"mean_overlap_{key}"] = float(values.mean())
            output[f"std_overlap_{key}"] = float(values.std(ddof=1))
        for key in ("add", "mult", "graft"):
            values = np.asarray([row[f"spearman_{key}"] for row in rows],
                                dtype=np.float64)
            finite = values[np.isfinite(values)]
            output[f"mean_spearman_{key}"] = (float(finite.mean())
                                               if len(finite) else float("nan"))
            output[f"std_spearman_{key}"] = (float(finite.std(ddof=1))
                                              if len(finite) > 1 else float("nan"))
        result.append(output)
    return result


def write_table(path: Path, summaries: Sequence[dict]) -> None:
    lines = [
        r"\begin{tabular}{rcccc}",
        r"\toprule",
        r"$\rho$ & Additive--exact & Multiplicative--exact & GRAFT--exact & Additive--multiplicative \\",
        r"\midrule",
    ]
    for row in summaries:
        lines.append(
            f"{row['rho']:g} & {row['mean_overlap_add_exact']:.3f} & "
            f"{row['mean_overlap_mult_exact']:.3f} & "
            f"{row['mean_overlap_graft_exact']:.3f} & "
            f"{row['mean_overlap_add_mult']:.3f} " + r"\\")
    spearman = summaries[0]
    lines.extend([
        r"\bottomrule", r"\end{tabular}", "",
        r"\begin{tabular}{lccc}", r"\toprule",
        r"Metric & Additive & Multiplicative & GRAFT R--M \\", r"\midrule",
        "Mean Spearman with held-out $T$ & "
        f"{spearman['mean_spearman_add']:.3f} & "
        f"{spearman['mean_spearman_mult']:.3f} & "
        f"{spearman['mean_spearman_graft']:.3f} " + r"\\",
        r"\bottomrule", r"\end{tabular}", "",
    ])
    atomic_text(path, "\n".join(lines))


def configure_plot_style() -> None:
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 10,
        "axes.spines.top": False, "axes.spines.right": False,
        "pdf.fonttype": 42, "ps.fonttype": 42,
    })


def plot_overlap(output_root: Path, summaries: Sequence[dict], dpi: int) -> None:
    configure_plot_style()
    x = np.asarray([row["rho"] for row in summaries], dtype=np.float64)
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2), constrained_layout=True)
    for key, label, color in (
            ("add_exact", "Additive", "#0072B2"),
            ("mult_exact", "Multiplicative", "#D55E00"),
            ("graft_exact", "GRAFT R-M", "#009E73")):
        y = [row[f"mean_overlap_{key}"] for row in summaries]
        axes[0].plot(x, y, marker="o", linewidth=2, label=label, color=color)
    axes[1].plot(x, [row["mean_overlap_add_mult"] for row in summaries],
                 marker="o", linewidth=2, color="#7B2CBF")
    axes[0].set_title("A. Transfer to held-out exact $T$")
    axes[1].set_title("B. Additive vs multiplicative")
    axes[0].set_ylabel("Mean top-$m$ overlap")
    axes[1].set_ylabel("Mean top-$m$ overlap")
    for axis in axes:
        axis.set_xlabel(r"Poison ratio $\rho$")
        axis.set_ylim(0, 1)
        axis.set_xticks(x)
        axis.set_xticklabels([f"{value:g}" for value in x], rotation=35)
        axis.grid(axis="y", alpha=0.25)
    axes[0].legend(frameon=False)
    for extension in ("pdf", "png"):
        save_figure(fig, output_root / "figures" /
                    f"add_vs_mult_topm_overlap.{extension}", dpi)
    plt.close(fig)


def plot_spearman(output_root: Path, rows: Sequence[dict], dpi: int) -> None:
    configure_plot_style()
    x = np.asarray([row["spearman_mult"] for row in rows], dtype=np.float64)
    y = np.asarray([row["spearman_add"] for row in rows], dtype=np.float64)
    finite = np.isfinite(x) & np.isfinite(y)
    if not finite.any():
        die("no finite paired Spearman values to plot")
    fig, axis = plt.subplots(figsize=(5.4, 5.4), constrained_layout=True)
    axis.scatter(x[finite], y[finite], s=20, alpha=0.65, color="#0072B2",
                 edgecolors="none")
    axis.plot([-1, 1], [-1, 1], linestyle="--", linewidth=1.3,
              color="#333333")
    axis.set_xlim(-1, 1)
    axis.set_ylim(-1, 1)
    axis.set_aspect("equal", adjustable="box")
    axis.set_xlabel(r"Spearman($S_{mult}$, $T$)")
    axis.set_ylabel(r"Spearman($S_{add}$, $T$)")
    axis.grid(alpha=0.2)
    for extension in ("pdf", "png"):
        save_figure(fig, output_root / "figures" /
                    f"add_vs_mult_spearman_scatter.{extension}", dpi)
    plt.close(fig)


def resolve_device(spec: str) -> torch.device:
    if spec == "auto":
        spec = "cuda:0" if torch.cuda.is_available() else "cpu"
    device = torch.device(spec)
    if device.type == "cuda" and not torch.cuda.is_available():
        die(f"{spec} requested but CUDA is unavailable")
    if device.type == "cuda":
        torch.cuda.set_device(device)
    return device


def run(args: argparse.Namespace) -> None:
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.use_deterministic_algorithms(True)
    poison.set_seed(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = resolve_device(args.device)
    jobs = load_jobs(args.jobs_file)
    rows: list[dict] = []
    for job in jobs:
        rows.extend(collect_job(args, job, device))
    if not rows:
        die("no leave-one-out units were produced")

    detailed = args.output_root / "results" / \
        "add_vs_mult_leave_one_out_detailed.csv"
    atomic_csv(detailed, detailed_fields(args.rhos), rows)
    summaries = summary_rows(rows, args.rhos)
    summary = args.output_root / "results" / \
        "add_vs_mult_leave_one_out_summary.csv"
    atomic_csv(summary, list(summaries[0]), summaries)
    write_table(args.output_root / "tables" /
                "add_vs_mult_leave_one_out.tex", summaries)
    plot_overlap(args.output_root, summaries, args.dpi)
    plot_spearman(args.output_root, rows, args.dpi)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--jobs-file", type=Path,
                        default=Path("add_vs_mult_leave_one_out_jobs.tsv"))
    parser.add_argument("--data-path", type=Path,
                        default=Path("/home/mmoslem3/scratch/data"))
    parser.add_argument("--cache-dir", type=Path, default=Path("./cache"))
    parser.add_argument("--output-root", type=Path, default=Path("."))
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--rhos", type=parse_rhos,
                        default=tuple(PAPER_RHOS))
    parser.add_argument("--surrogate-epochs", type=int, default=60)
    parser.add_argument("--surrogate-lr", type=float, default=0.1)
    parser.add_argument("--surrogate-bs", type=int, default=128)
    parser.add_argument("--forward-batch-size", type=int, default=512)
    parser.add_argument("--image-load-batch-size", type=int, default=512)
    parser.add_argument("--dpi", type=int, default=220)
    args = parser.parse_args(argv)
    if args.forward_batch_size <= 0 or args.image_load_batch_size <= 0:
        parser.error("batch sizes must be positive")
    args.jobs_file = args.jobs_file.resolve()
    args.data_path = args.data_path.resolve()
    args.cache_dir = args.cache_dir.resolve()
    args.output_root = args.output_root.resolve()
    return args


if __name__ == "__main__":
    run(parse_args())
