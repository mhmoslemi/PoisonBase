#!/usr/bin/env python3
"""Checkpoint-only diagnostic for the margin -> residual-mass -> residual-dot chain."""

from __future__ import annotations

import argparse
import csv
import gc
import json
import math
import os
import re
import sys
import zlib
from collections import defaultdict
from pathlib import Path
from typing import Iterable, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import final_update as poison  # noqa: E402
from utils import get_dataset  # noqa: E402


SCHEMA_VERSION = 1
SUMMARY_FIELDS = [
    "dataset", "model", "class_pair", "target_id", "surrogate_id",
    "y_adv", "num_classes", "n_candidates", "u_t",
    "spearman_neg_margin_vs_u",
    "spearman_u_vs_normalized_residual_dot",
    "spearman_neg_margin_vs_residual_dot",
    "fraction_margin_bounds_satisfied",
    "fraction_residual_dot_bounds_satisfied",
    "margin_bound_violation_count", "margin_bound_max_violation",
    "residual_dot_bound_violation_count",
    "residual_dot_bound_max_violation", "checkpoint",
]


def die(message: str) -> "NoReturn":
    raise SystemExit("ERROR: " + message)


def safe_slug(value: str) -> str:
    return value.replace("/", "_").replace(" ", "_")


def atomic_npz(path: Path, **arrays: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp.npz")
    np.savez_compressed(temporary, **arrays)
    os.replace(temporary, path)


def atomic_csv(path: Path, fields: Sequence[str], rows: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def resolve_device(spec: str) -> torch.device:
    if spec == "auto":
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device = torch.device(spec)
    if device.type == "cuda" and not torch.cuda.is_available():
        die(f"{spec} requested but CUDA is unavailable")
    if device.type == "cuda":
        torch.cuda.set_device(device)
    return device


def read_jobs(path: Path) -> list[dict[str, str]]:
    jobs: list[dict[str, str]] = []
    try:
        with path.open() as handle:
            for line_number, raw in enumerate(handle, 1):
                line = raw.split("#", 1)[0].strip()
                if not line:
                    continue
                fields = line.split("|")
                if len(fields) != 4:
                    die(f"{path}:{line_number}: expected dataset|model|pair|targets")
                dataset, model, pair, target_file = (field.strip() for field in fields)
                target_path = Path(target_file)
                if not target_path.is_absolute():
                    target_path = REPO_ROOT / target_path
                jobs.append({
                    "dataset": dataset, "model": model, "class_pair": pair,
                    "target_file": str(target_path),
                })
    except OSError as exc:
        die(f"cannot read jobs file {path}: {exc}")
    if not jobs:
        die(f"jobs file contains no jobs: {path}")
    return jobs


def load_target_indices(path: Path, class_pair: str) -> list[int]:
    try:
        with path.open() as handle:
            payload = json.load(handle)
        values = (payload["pairs"][class_pair]["indices"]
                  if "pairs" in payload else payload[class_pair])
        result = [int(value) for value in values]
    except (OSError, KeyError, TypeError, ValueError) as exc:
        die(f"cannot read targets for {class_pair!r} from {path}: {exc}")
    if not result or len(result) != len(set(result)):
        die(f"target IDs must be nonempty and unique in {path}")
    return result


def labels_numpy(dataset: object) -> np.ndarray:
    for name in ("targets", "labels"):
        if hasattr(dataset, name):
            values = getattr(dataset, name)
            if torch.is_tensor(values):
                return values.detach().cpu().numpy().astype(np.int64, copy=False)
            return np.asarray(values, dtype=np.int64)
    return np.asarray([int(dataset[index][1]) for index in range(len(dataset))],
                      dtype=np.int64)


def image_batch(dataset: object, indices: np.ndarray,
                device: torch.device) -> torch.Tensor:
    images = [dataset[int(index)][0] for index in indices]
    return torch.stack(images).to(device, non_blocking=device.type == "cuda")


def production_args(cli: argparse.Namespace, job: dict[str, str]) -> argparse.Namespace:
    argv = [
        "--dataset", job["dataset"], "--data_path", str(cli.data_path),
        "--model", job["model"], "--seed", str(cli.seed),
        "--cache_dir", str(cli.cache_dir),
        "--out_dir", str(cli.output_root / "unused_attack_output"),
        "--class_pair", job["class_pair"], "--pair_order", cli.pair_order,
        "--surrogate_epochs", str(cli.surrogate_epochs),
        "--surrogate_lr", str(cli.surrogate_lr),
        "--surrogate_bs", str(cli.surrogate_bs),
        "--surrogate_decay", *[str(value) for value in cli.surrogate_decay],
        "--surrogate_wd", str(cli.surrogate_wd), "--gpus", "none",
    ]
    if cli.surrogate_aug:
        argv.append("--surrogate_aug")
    return poison.parse_args(argv)


def saved_checkpoints(parsed: argparse.Namespace) -> list[tuple[int, Path]]:
    directory = Path(poison.surrogate_dir(parsed))
    if not directory.is_dir():
        die(f"saved surrogate directory does not exist: {directory}")
    found: list[tuple[int, Path]] = []
    for path in directory.glob("net_*.pt"):
        match = re.fullmatch(r"net_(\d+)\.pt", path.name)
        if match:
            found.append((int(match.group(1)), path))
    found.sort(key=lambda item: item[0])
    if not found:
        die(f"no saved net_*.pt surrogate checkpoints in {directory}")
    return found


def load_net(job: dict[str, str], channel: int, num_classes: int,
             im_size: tuple[int, int], device: torch.device, seed: int,
             surrogate_id: int, checkpoint: Path) -> nn.Module:
    net = poison.build_network(
        job["model"], channel, num_classes, im_size, str(device),
        seed=seed + 1000 + surrogate_id,
    )
    state = torch.load(checkpoint, map_location=device)
    net.load_state_dict(state)
    net.eval()
    return net


@torch.no_grad()
def forward_logits(net: nn.Module, dataset: object, indices: np.ndarray,
                   device: torch.device, batch_size: int) -> torch.Tensor:
    chunks: list[torch.Tensor] = []
    for start in range(0, len(indices), batch_size):
        batch_ids = indices[start:start + batch_size]
        chunks.append(net(image_batch(dataset, batch_ids, device)).detach().cpu())
    if not chunks:
        die("empty forward-pass index set")
    return torch.cat(chunks, dim=0).to(torch.float64)


def candidate_quantities(logits: torch.Tensor, y_adv: int) -> dict[str, np.ndarray]:
    if logits.ndim != 2 or not (0 <= y_adv < logits.shape[1]):
        die("invalid logits or adversarial class")
    num_classes = int(logits.shape[1])
    if num_classes < 2:
        die("model must have at least two output classes")

    # M is deliberately computed from the raw float64 logits before softmax.
    other_logits = logits.clone()
    other_logits[:, y_adv] = -torch.inf
    margin = logits[:, y_adv] - other_logits.max(dim=1).values
    probabilities = torch.exp(F.log_softmax(logits, dim=1))
    other_mask = torch.ones(num_classes, dtype=torch.bool)
    other_mask[y_adv] = False
    residual_mass = probabilities[:, other_mask].sum(dim=1)
    lower = torch.sigmoid(-margin)
    upper = torch.sigmoid(math.log(num_classes - 1.0) - margin)
    return {
        "M": margin.numpy(),
        "probabilities": probabilities.numpy(),
        "u_i": residual_mass.numpy(),
        "margin_lower_bound": lower.numpy(),
        "margin_upper_bound": upper.numpy(),
    }


def bound_diagnostics(values: np.ndarray, lower: np.ndarray, upper: np.ndarray,
                      atol: float, rtol: float) -> tuple[np.ndarray, np.ndarray]:
    violation = np.maximum(np.maximum(lower - values, values - upper), 0.0)
    scale = np.maximum.reduce((np.abs(values), np.abs(lower), np.abs(upper)))
    satisfied = violation <= atol + rtol * scale
    return satisfied, violation


def average_ranks(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    sorted_values = values[order]
    start = 0
    while start < len(values):
        stop = start + 1
        while stop < len(values) and sorted_values[stop] == sorted_values[start]:
            stop += 1
        ranks[order[start:stop]] = 0.5 * (start + stop - 1) + 1.0
        start = stop
    return ranks


def spearman(left: np.ndarray, right: np.ndarray) -> float:
    if len(left) != len(right) or len(left) < 2:
        return float("nan")
    a, b = average_ranks(left), average_ranks(right)
    a -= a.mean()
    b -= b.mean()
    denominator = float(np.linalg.norm(a) * np.linalg.norm(b))
    return float(np.dot(a, b) / denominator) if denominator else float("nan")


def scalar_string(value: str) -> np.ndarray:
    return np.asarray(value)


def write_panel_a_cache(path: Path, job: dict[str, str], candidate_ids: np.ndarray,
                        y_adv: int, num_classes: int, surrogate_id: int,
                        checkpoint: Path, values: dict[str, np.ndarray],
                        satisfied: np.ndarray, violation: np.ndarray) -> None:
    atomic_npz(
        path,
        schema_version=np.int16(SCHEMA_VERSION),
        dataset=scalar_string(job["dataset"]), model=scalar_string(job["model"]),
        class_pair=scalar_string(job["class_pair"]), y_adv=np.int32(y_adv),
        num_classes=np.int32(num_classes), surrogate_id=np.int32(surrogate_id),
        checkpoint=scalar_string(str(checkpoint.resolve())),
        candidate_id=candidate_ids.astype(np.int64, copy=False),
        M=values["M"], u_i=values["u_i"],
        margin_lower_bound=values["margin_lower_bound"],
        margin_upper_bound=values["margin_upper_bound"],
        margin_bound_satisfied=satisfied,
        margin_bound_violation=violation,
    )


def write_panel_b_cache(path: Path, job: dict[str, str], candidate_ids: np.ndarray,
                        target_id: int, y_adv: int, num_classes: int,
                        surrogate_id: int, checkpoint: Path,
                        candidate: dict[str, np.ndarray], u_t: float,
                        residual_dot: np.ndarray, normalized: np.ndarray,
                        direct_error: np.ndarray, satisfied: np.ndarray,
                        violation: np.ndarray) -> None:
    atomic_npz(
        path,
        schema_version=np.int16(SCHEMA_VERSION),
        dataset=scalar_string(job["dataset"]), model=scalar_string(job["model"]),
        class_pair=scalar_string(job["class_pair"]), target_id=np.int64(target_id),
        y_adv=np.int32(y_adv), num_classes=np.int32(num_classes),
        surrogate_id=np.int32(surrogate_id),
        checkpoint=scalar_string(str(checkpoint.resolve())),
        candidate_id=candidate_ids.astype(np.int64, copy=False),
        M=candidate["M"], u_i=candidate["u_i"], u_t=np.float64(u_t),
        residual_dot=residual_dot,
        normalized_residual_dot=normalized,
        lower_residual_dot_bound=candidate["u_i"],
        upper_residual_dot_bound=2.0 * candidate["u_i"],
        residual_dot_direct_equivalence_error=direct_error,
        residual_dot_bound_satisfied=satisfied,
        residual_dot_bound_violation=violation,
    )


def quantile_trend(x: np.ndarray, y: np.ndarray,
                   bins: int) -> tuple[np.ndarray, ...]:
    order = np.argsort(x, kind="mergesort")
    groups = [group for group in np.array_split(order, min(bins, len(order)))
              if len(group)]
    return tuple(np.asarray([np.quantile(values[group], quantile) for group in groups])
                 for values, quantile in ((x, 0.5), (y, 0.5),
                                          (y, 0.1), (y, 0.9)))


def deterministic_subsample(length: int, maximum: int, seed: int) -> np.ndarray:
    if length <= maximum:
        return np.arange(length)
    return np.sort(np.random.default_rng(seed).choice(length, maximum, replace=False))


def load_plot_arrays(paths: Sequence[Path], x_name: str,
                     y_name: str) -> tuple[np.ndarray, np.ndarray]:
    xs, ys = [], []
    for path in paths:
        with np.load(path, allow_pickle=False) as shard:
            xs.append(np.asarray(shard[x_name], dtype=np.float64))
            ys.append(np.asarray(shard[y_name], dtype=np.float64))
    if not xs:
        die(f"no cache shards available for {x_name}, {y_name}")
    return np.concatenate(xs), np.concatenate(ys)


def style_axis(axis: plt.Axes) -> None:
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.grid(alpha=0.16, linewidth=0.6)


def plot_dataset(dataset: str, num_classes: int, panel_a_paths: Sequence[Path],
                 panel_b_paths: Sequence[Path], figure_dir: Path,
                 bins: int, max_scatter: int, seed: int) -> list[Path]:
    margin, u_a = load_plot_arrays(panel_a_paths, "M", "u_i")
    u_b, normalized = load_plot_arrays(
        panel_b_paths, "u_i", "normalized_residual_dot")
    dataset_seed = seed + zlib.crc32(dataset.encode("utf-8"))
    a_sample = deterministic_subsample(len(margin), max_scatter, dataset_seed)
    b_sample = deterministic_subsample(len(u_b), max_scatter, dataset_seed + 1)

    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.25), constrained_layout=True)
    axes[0].scatter(margin[a_sample], u_a[a_sample], s=4, alpha=0.08,
                    color="#506784", linewidths=0, rasterized=True)
    bx, by, blo, bhi = quantile_trend(margin, u_a, bins)
    axes[0].fill_between(bx, blo, bhi, color="#2474B5", alpha=0.18, linewidth=0)
    axes[0].plot(bx, by, color="#1261A0", marker="o", markersize=3,
                 linewidth=1.7, label="Empirical median")
    grid = np.linspace(float(margin.min()), float(margin.max()), 500)
    lower = 1.0 / (1.0 + np.exp(np.clip(grid, -745.0, 709.0)))
    upper_argument = math.log(num_classes - 1.0) - grid
    upper = np.empty_like(upper_argument)
    positive = upper_argument >= 0
    upper[positive] = 1.0 / (1.0 + np.exp(-upper_argument[positive]))
    exp_value = np.exp(upper_argument[~positive])
    upper[~positive] = exp_value / (1.0 + exp_value)
    axes[0].plot(grid, lower, "--", color="#238B45", linewidth=1.5,
                 label="Lower bound")
    axes[0].plot(grid, upper, "-.", color="#CB3A36", linewidth=1.5,
                 label="Upper bound")
    axes[0].set(xlabel=r"Raw logit margin $M_i$",
                ylabel=r"Candidate residual mass $u_i$", ylim=(-0.02, 1.02))
    axes[0].set_title("(a) Margin to residual mass", loc="left")
    axes[0].legend(frameon=False, fontsize=8)
    style_axis(axes[0])

    axes[1].scatter(u_b[b_sample], normalized[b_sample], s=4, alpha=0.08,
                    color="#506784", linewidths=0, rasterized=True)
    bx, by, blo, bhi = quantile_trend(u_b, normalized, bins)
    axes[1].fill_between(bx, blo, bhi, color="#A55D18", alpha=0.18, linewidth=0)
    axes[1].plot(bx, by, color="#9A4D0A", marker="o", markersize=3,
                 linewidth=1.7, label="Empirical median")
    xmax = float(u_b.max())
    line_x = np.linspace(0.0, xmax, 300)
    axes[1].plot(line_x, line_x, "--", color="#238B45", linewidth=1.5,
                 label=r"$y=x$")
    axes[1].plot(line_x, 2.0 * line_x, "-.", color="#CB3A36", linewidth=1.5,
                 label=r"$y=2x$")
    axes[1].set(xlabel=r"Candidate residual mass $u_i$",
                ylabel=r"Normalized residual dot $(r_i^\top r_t)/u_t$",
                xlim=(-0.01 * xmax, 1.01 * xmax),
                ylim=(-0.02 * xmax, 2.02 * xmax))
    axes[1].set_title("(b) Residual mass to residual inner product", loc="left")
    axes[1].legend(frameon=False, fontsize=8)
    style_axis(axes[1])

    figure_dir.mkdir(parents=True, exist_ok=True)
    stem = figure_dir / f"margin_residual_chain_{safe_slug(dataset)}"
    outputs = [stem.with_suffix(".pdf"), stem.with_suffix(".png")]
    fig.savefig(outputs[0], bbox_inches="tight")
    fig.savefig(outputs[1], dpi=300, bbox_inches="tight")
    plt.close(fig)

    positive_u = u_b[u_b > 0]
    if len(positive_u) >= 2:
        log_span = float(np.quantile(np.log10(positive_u), 0.99)
                         - np.quantile(np.log10(positive_u), 0.01))
    else:
        log_span = 0.0
    if log_span >= 3.0:
        valid = (u_b > 0) & (normalized > 0)
        log_ids = deterministic_subsample(int(valid.sum()), max_scatter,
                                          dataset_seed + 2)
        x_log, y_log = u_b[valid], normalized[valid]
        fig, axis = plt.subplots(figsize=(5.1, 4.5), constrained_layout=True)
        axis.scatter(x_log[log_ids], y_log[log_ids], s=4, alpha=0.08,
                     color="#506784", linewidths=0, rasterized=True)
        xmin, xmax_log = float(x_log.min()), float(x_log.max())
        line_x = np.geomspace(xmin, xmax_log, 400)
        axis.plot(line_x, line_x, "--", color="#238B45", linewidth=1.5,
                  label=r"$y=x$")
        axis.plot(line_x, 2.0 * line_x, "-.", color="#CB3A36", linewidth=1.5,
                  label=r"$y=2x$")
        axis.set_xscale("log")
        axis.set_yscale("log")
        axis.set(xlabel=r"Candidate residual mass $u_i$",
                 ylabel=r"Normalized residual dot $(r_i^\top r_t)/u_t$")
        axis.legend(frameon=False)
        style_axis(axis)
        log_stem = figure_dir / f"margin_residual_chain_log_{safe_slug(dataset)}"
        log_outputs = [log_stem.with_suffix(".pdf"), log_stem.with_suffix(".png")]
        fig.savefig(log_outputs[0], bbox_inches="tight")
        fig.savefig(log_outputs[1], dpi=300, bbox_inches="tight")
        plt.close(fig)
        outputs.extend(log_outputs)
    return outputs


def latex_escape(value: str) -> str:
    return (value.replace("\\", r"\textbackslash{}")
            .replace("_", r"\_").replace("%", r"\%")
            .replace("&", r"\&").replace("#", r"\#"))


def write_latex(path: Path, rows: Sequence[dict]) -> None:
    metrics = [
        "spearman_neg_margin_vs_u",
        "spearman_u_vs_normalized_residual_dot",
        "spearman_neg_margin_vs_residual_dot",
        "fraction_margin_bounds_satisfied",
        "fraction_residual_dot_bounds_satisfied",
    ]
    grouped: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["dataset"]), str(row["class_pair"]))].append(row)

    table_rows: list[tuple[str, str, list[float]]] = []
    for (dataset, pair), members in sorted(grouped.items()):
        values = [float(np.nanmean([float(member[metric]) for member in members]))
                  for metric in metrics]
        table_rows.append((dataset, pair, values))
    overall = [float(np.nanmean([float(row[metric]) for row in rows]))
               for metric in metrics]
    table_rows.append(("Overall", "--", overall))

    lines = [
        r"\begin{tabular}{llccccc}", r"\toprule",
        r"Dataset & Class pair & $\rho_s(-M,u)$ & $\rho_s(u,\bar r)$ & "
        r"$\rho_s(-M,r_i^\top r_t)$ & Margin bounds & Residual bounds \\",
        r"\midrule",
    ]
    for dataset, pair, values in table_rows:
        lines.append("{} & {} & {} \\\\".format(
            latex_escape(dataset), latex_escape(pair),
            " & ".join(f"{value:.4f}" for value in values)))
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text("\n".join(lines) + "\n")
    os.replace(temporary, path)


def collect(cli: argparse.Namespace) -> None:
    device = resolve_device(cli.device)
    poison.set_seed(cli.seed)
    jobs = read_jobs(cli.jobs_file)
    cache_root = cli.output_root / "results" / "margin_residual_chain_candidates"
    panel_a_by_dataset: dict[str, list[Path]] = defaultdict(list)
    panel_b_by_dataset: dict[str, list[Path]] = defaultdict(list)
    num_classes_by_dataset: dict[str, int] = {}
    summary_rows: list[dict] = []
    unique_surrogates: set[tuple[str, str, int, str]] = set()
    unique_targets: set[tuple[str, str, str, int]] = set()
    panel_a_observations = panel_b_observations = 0
    margin_violations = residual_violations = 0
    max_margin_violation = max_residual_violation = 0.0
    skipped_units: list[tuple[str, str, str, int, int, float]] = []

    for job_index, job in enumerate(jobs):
        print(f"[{job_index + 1}/{len(jobs)}] {job['dataset']} "
              f"{job['model']} {job['class_pair']}", flush=True)
        (channel, im_size, num_classes, class_names, _mean, _std,
         train_dataset, test_dataset, _testloader) = get_dataset(
             job["dataset"], str(cli.data_path))
        previous_classes = num_classes_by_dataset.setdefault(job["dataset"], num_classes)
        if previous_classes != num_classes:
            die(f"inconsistent output dimensions for dataset {job['dataset']}")
        y_adv, target_class = poison.parse_pair(
            job["class_pair"], class_names, cli.pair_order)
        if class_names[y_adv] != job["class_pair"].split("-")[0]:
            die("y_adv does not match the attack code's poison-target convention")

        train_labels = labels_numpy(train_dataset)
        test_labels = labels_numpy(test_dataset)
        candidate_ids = np.flatnonzero(train_labels == y_adv).astype(np.int64)
        if not len(candidate_ids):
            die(f"empty adversarial-class candidate pool for {job['class_pair']}")
        targets = load_target_indices(Path(job["target_file"]), job["class_pair"])
        bad = [target for target in targets
               if target < 0 or target >= len(test_labels)
               or int(test_labels[target]) != target_class]
        if bad:
            die(f"invalid/wrong-class target IDs for {job['class_pair']}: {bad}")
        parsed = production_args(cli, job)
        checkpoints = saved_checkpoints(parsed)
        print(f"  candidates={len(candidate_ids)} targets={len(targets)} "
              f"saved_surrogates={len(checkpoints)}", flush=True)

        combo = (cache_root / safe_slug(job["dataset"]) / safe_slug(job["model"])
                 / safe_slug(job["class_pair"]))
        for surrogate_id, checkpoint in checkpoints:
            unique_surrogates.add((job["dataset"], job["model"], surrogate_id,
                                   str(checkpoint.resolve())))
            net = load_net(job, channel, num_classes, im_size, device, cli.seed,
                           surrogate_id, checkpoint)
            candidate_logits = forward_logits(
                net, train_dataset, candidate_ids, device, cli.forward_batch_size)
            if int(candidate_logits.shape[1]) != num_classes:
                die(f"checkpoint output dimension {candidate_logits.shape[1]} "
                    f"does not match dataset C={num_classes}: {checkpoint}")
            candidate = candidate_quantities(candidate_logits, y_adv)
            u_i = candidate["u_i"]
            if (not np.all(np.isfinite(u_i)) or u_i.min() < -cli.bound_atol
                    or u_i.max() > 1.0 + cli.bound_atol):
                die(f"candidate residual mass outside [0,1]: {checkpoint}")
            margin_ok, margin_violation = bound_diagnostics(
                u_i, candidate["margin_lower_bound"],
                candidate["margin_upper_bound"], cli.bound_atol, cli.bound_rtol)
            panel_a_path = combo / f"panel_a_surrogate_{surrogate_id:03d}.npz"
            write_panel_a_cache(
                panel_a_path, job, candidate_ids, y_adv, num_classes,
                surrogate_id, checkpoint, candidate, margin_ok, margin_violation)
            panel_a_by_dataset[job["dataset"]].append(panel_a_path)
            panel_a_observations += len(candidate_ids)
            margin_violations += int((~margin_ok).sum())
            max_margin_violation = max(max_margin_violation,
                                       float(margin_violation.max(initial=0.0)))
            rho_margin_u = spearman(-candidate["M"], u_i)

            probabilities = candidate["probabilities"]
            residuals = probabilities.copy()
            residuals[:, y_adv] -= 1.0
            other_mask = np.ones(num_classes, dtype=bool)
            other_mask[y_adv] = False
            for target_id in targets:
                unique_targets.add((job["dataset"], job["model"],
                                    job["class_pair"], target_id))
                target_logits = forward_logits(
                    net, test_dataset, np.asarray([target_id], dtype=np.int64),
                    device, 1)[0]
                target_probabilities = torch.exp(
                    F.log_softmax(target_logits, dim=0)).numpy()
                u_t = float(target_probabilities[other_mask].sum())
                if not np.isfinite(u_t) or not (-cli.bound_atol <= u_t <= 1 + cli.bound_atol):
                    die(f"target residual mass outside [0,1]: target={target_id}, "
                        f"checkpoint={checkpoint}")
                if u_t <= cli.u_t_threshold:
                    skipped_units.append((job["dataset"], job["model"],
                                          job["class_pair"], target_id,
                                          surrogate_id, u_t))
                    print(f"    SKIP target={target_id} surrogate={surrogate_id}: "
                          f"u_t={u_t:.3e} <= {cli.u_t_threshold:.3e}", flush=True)
                    continue

                target_residual = target_probabilities.copy()
                target_residual[y_adv] -= 1.0
                direct_dot = residuals @ target_residual
                stable_dot = (u_i * u_t
                              + probabilities[:, other_mask]
                              @ target_probabilities[other_mask])
                direct_error = np.abs(direct_dot - stable_dot)
                direct_tol = (cli.bound_atol + cli.bound_rtol
                              * np.maximum(np.abs(direct_dot), np.abs(stable_dot)))
                if np.any(direct_error > direct_tol):
                    die(f"residual-dot definitions disagree for target={target_id}, "
                        f"surrogate={surrogate_id}; max={direct_error.max():.3e}")
                normalized = stable_dot / u_t
                residual_ok, residual_violation = bound_diagnostics(
                    normalized, u_i, 2.0 * u_i,
                    cli.bound_atol, cli.bound_rtol)
                panel_b_path = (combo / f"target_{target_id}"
                                / f"panel_b_surrogate_{surrogate_id:03d}.npz")
                write_panel_b_cache(
                    panel_b_path, job, candidate_ids, target_id, y_adv,
                    num_classes, surrogate_id, checkpoint, candidate, u_t,
                    stable_dot, normalized, direct_error, residual_ok,
                    residual_violation)
                panel_b_by_dataset[job["dataset"]].append(panel_b_path)
                panel_b_observations += len(candidate_ids)
                residual_violations += int((~residual_ok).sum())
                max_residual_violation = max(
                    max_residual_violation,
                    float(residual_violation.max(initial=0.0)))
                summary_rows.append({
                    "dataset": job["dataset"], "model": job["model"],
                    "class_pair": job["class_pair"], "target_id": target_id,
                    "surrogate_id": surrogate_id, "y_adv": y_adv,
                    "num_classes": num_classes,
                    "n_candidates": len(candidate_ids), "u_t": f"{u_t:.17g}",
                    "spearman_neg_margin_vs_u": f"{rho_margin_u:.17g}",
                    "spearman_u_vs_normalized_residual_dot":
                        f"{spearman(u_i, normalized):.17g}",
                    "spearman_neg_margin_vs_residual_dot":
                        f"{spearman(-candidate['M'], stable_dot):.17g}",
                    "fraction_margin_bounds_satisfied":
                        f"{float(margin_ok.mean()):.17g}",
                    "fraction_residual_dot_bounds_satisfied":
                        f"{float(residual_ok.mean()):.17g}",
                    "margin_bound_violation_count": int((~margin_ok).sum()),
                    "margin_bound_max_violation":
                        f"{float(margin_violation.max(initial=0.0)):.17g}",
                    "residual_dot_bound_violation_count": int((~residual_ok).sum()),
                    "residual_dot_bound_max_violation":
                        f"{float(residual_violation.max(initial=0.0)):.17g}",
                    "checkpoint": str(checkpoint.resolve()),
                })

            del candidate_logits, candidate, probabilities, residuals, net
            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()
        del train_dataset, test_dataset
        gc.collect()

    if not summary_rows:
        die("all target/surrogate units were skipped")
    summary_path = cli.output_root / "results" / "margin_residual_chain_summary.csv"
    table_path = cli.output_root / "tables" / "margin_residual_chain.tex"
    atomic_csv(summary_path, SUMMARY_FIELDS, summary_rows)
    write_latex(table_path, summary_rows)

    figure_outputs: list[Path] = []
    for dataset in sorted(panel_a_by_dataset):
        figure_outputs.extend(plot_dataset(
            dataset, num_classes_by_dataset[dataset], panel_a_by_dataset[dataset],
            panel_b_by_dataset[dataset], cli.output_root / "figures",
            cli.bins, cli.max_scatter, cli.seed))

    print("\nSanity-check totals", flush=True)
    print(f"  margin-bound violations: {margin_violations}; "
          f"max magnitude: {max_margin_violation:.17g}", flush=True)
    print(f"  residual-dot-bound violations: {residual_violations}; "
          f"max magnitude: {max_residual_violation:.17g}", flush=True)
    print(f"  skipped target/surrogate units (u_t threshold): "
          f"{len(skipped_units)}", flush=True)
    print("\nData totals", flush=True)
    print(f"  datasets: {len({job['dataset'] for job in jobs})}", flush=True)
    print(f"  class pairs: "
          f"{len({(job['dataset'], job['class_pair']) for job in jobs})}", flush=True)
    print(f"  targets: {len(unique_targets)}", flush=True)
    print(f"  saved surrogates: {len(unique_surrogates)}", flush=True)
    print(f"  unique Panel A candidate observations: {panel_a_observations}", flush=True)
    print(f"  Panel B candidate-target observations: {panel_b_observations}", flush=True)
    print(f"  summary: {summary_path}", flush=True)
    print(f"  table: {table_path}", flush=True)
    for path in figure_outputs:
        print(f"  figure: {path}", flush=True)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze saved surrogate logits without training or crafting.")
    parser.add_argument("--jobs-file", type=Path, required=True)
    parser.add_argument("--data-path", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--pair-order", choices=["poison-target", "target-poison"],
                        default="poison-target")
    parser.add_argument("--forward-batch-size", type=int, default=512)
    parser.add_argument("--bins", type=int, default=40)
    parser.add_argument("--max-scatter", type=int, default=100000)
    parser.add_argument("--u-t-threshold", type=float, default=1e-12)
    parser.add_argument("--bound-atol", type=float, default=1e-10)
    parser.add_argument("--bound-rtol", type=float, default=1e-8)
    parser.add_argument("--surrogate-epochs", type=int, default=60)
    parser.add_argument("--surrogate-lr", type=float, default=0.1)
    parser.add_argument("--surrogate-bs", type=int, default=128)
    parser.add_argument("--surrogate-decay", nargs="*", type=int, default=[35, 45])
    parser.add_argument("--surrogate-wd", type=float, default=0.0)
    parser.add_argument("--surrogate-aug", action="store_true", default=False)
    args = parser.parse_args(argv)
    if args.forward_batch_size <= 0 or not (30 <= args.bins <= 50):
        parser.error("--forward-batch-size must be positive and --bins must be 30..50")
    if args.max_scatter <= 0 or args.u_t_threshold < 0:
        parser.error("--max-scatter must be positive and --u-t-threshold nonnegative")
    return args


if __name__ == "__main__":
    collect(parse_args())
