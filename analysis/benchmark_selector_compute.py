#!/usr/bin/env python3
"""CUDA compute/memory benchmark for CIFAR-10 candidate selectors."""

from __future__ import annotations

import argparse
import csv
import gc
import json
import math
import os
import platform
import subprocess
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Iterable, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import final_update as poison  # noqa: E402


METHODS = ("RAND", "M", "R", "GRAFT", "A", "GRAFT+", "EXACT")
DISPLAY_METHOD = {"EXACT": "Exact"}
RAW_FIELDS = [
    "dataset", "architecture", "checkpoint_model", "class_pair", "target_id",
    "method", "K", "n_candidates", "repeat", "wall_time_sec",
    "baseline_gpu_memory_mib", "peak_gpu_memory_mib",
    "incremental_peak_gpu_memory_mib", "peak_reserved_memory_mib",
    "candidates_per_sec",
]
SUMMARY_FIELDS = [
    "dataset", "architecture", "checkpoint_model", "method", "K", "n_units",
    "mean_wall_time_sec", "std_wall_time_sec", "mean_peak_gpu_memory_mib",
    "mean_incremental_peak_gpu_memory_mib", "mean_peak_reserved_memory_mib",
    "mean_candidates_per_sec", "time_relative_to_GRAFT",
    "memory_relative_to_GRAFT",
]
COMPONENT_FIELDS = [
    "dataset", "architecture", "checkpoint_model", "component", "K",
    "n_units", "mean_wall_time_sec", "std_wall_time_sec",
    "mean_peak_gpu_memory_mib", "mean_incremental_peak_gpu_memory_mib",
    "mean_peak_reserved_memory_mib", "mean_candidates_per_sec",
]


def die(message: str) -> "NoReturn":
    raise SystemExit("ERROR: " + message)


def atomic_csv(path: Path, fields: Sequence[str], rows: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(temporary, path)


def atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(text)
    os.replace(temporary, path)


def safe_slug(value: str) -> str:
    return value.replace("/", "_").replace(" ", "_")


def read_jobs(path: Path) -> list[dict[str, str]]:
    jobs: list[dict[str, str]] = []
    try:
        with path.open() as handle:
            for line_number, raw in enumerate(handle, 1):
                line = raw.split("#", 1)[0].strip()
                if not line:
                    continue
                fields = [field.strip() for field in line.split("|")]
                if len(fields) != 4:
                    die(f"{path}:{line_number}: expected label|model|pair|targets")
                architecture, model, class_pair, target_file = fields
                target_path = Path(target_file)
                if not target_path.is_absolute():
                    target_path = REPO_ROOT / target_path
                jobs.append({
                    "architecture": architecture,
                    "model": model,
                    "class_pair": class_pair,
                    "target_file": str(target_path),
                })
    except OSError as exc:
        die(f"cannot read jobs file {path}: {exc}")
    if not jobs:
        die(f"jobs file contains no jobs: {path}")
    expected = {"ConvNet", "ResNet20", "VGG13"}
    present = {job["architecture"] for job in jobs}
    if present != expected:
        die(f"jobs architectures must be {sorted(expected)}, got {sorted(present)}")
    return jobs


def load_target_ids(path: Path, class_pair: str) -> list[int]:
    try:
        with path.open() as handle:
            payload = json.load(handle)
        values = (payload["pairs"][class_pair]["indices"]
                  if "pairs" in payload else payload[class_pair])
        targets = [int(value) for value in values]
    except (OSError, KeyError, TypeError, ValueError) as exc:
        die(f"cannot load {class_pair!r} targets from {path}: {exc}")
    if not targets or len(targets) != len(set(targets)):
        die(f"target IDs must be nonempty and unique in {path}")
    return targets


def checkpoint_namespace(cli: argparse.Namespace, model: str) -> argparse.Namespace:
    return argparse.Namespace(
        dataset="CIFAR10", model=model, cache_dir=str(cli.cache_dir),
        surrogate_epochs=cli.surrogate_epochs,
        surrogate_lr=cli.surrogate_lr,
        surrogate_bs=cli.surrogate_bs,
        surrogate_wd=cli.surrogate_wd,
        surrogate_decay=cli.surrogate_decay,
        surrogate_aug=cli.surrogate_aug,
        seed=cli.seed,
    )


def checkpoint_paths(cli: argparse.Namespace, model: str) -> list[Path]:
    directory = Path(poison.surrogate_dir(checkpoint_namespace(cli, model)))
    paths = [directory / f"net_{index}.pt" for index in range(cli.k)]
    missing = [path for path in paths if not path.is_file()]
    if missing:
        die("missing saved surrogate checkpoints; no training is allowed:\n"
            + "\n".join(str(path) for path in missing))
    return paths


def load_cpu_state(path: Path) -> dict[str, torch.Tensor]:
    try:
        state = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        state = torch.load(path, map_location="cpu")
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    if not isinstance(state, dict):
        die(f"checkpoint is not a state dictionary: {path}")
    return state


def load_models(
    model_name: str,
    states: Sequence[dict[str, torch.Tensor]],
    channel: int,
    num_classes: int,
    im_size: tuple[int, int],
    device: torch.device,
    seed: int,
) -> list[nn.Module]:
    models: list[nn.Module] = []
    for surrogate_id, state in enumerate(states):
        net = poison.build_network(
            model_name, channel, num_classes, im_size, str(device),
            seed=seed + 1000 + surrogate_id)
        net.load_state_dict(state)
        net.eval()
        models.append(net)
    return models


def core_model(net: nn.Module) -> nn.Module:
    return net.module if isinstance(net, nn.DataParallel) else net


def classifier_logits(net: nn.Module, features: torch.Tensor) -> torch.Tensor:
    core = core_model(net)
    if not hasattr(core, "classifier") or not isinstance(core.classifier, nn.Linear):
        die(f"{type(core).__name__} does not expose the production linear classifier")
    return core.classifier(features)


@torch.no_grad()
def forward_components(
    nets: Sequence[nn.Module],
    candidates: torch.Tensor,
    target: torch.Tensor,
    y_adv: int,
    batch_size: int,
    need_margin: bool,
    need_relevance: bool,
) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    """Production M/R definitions, raw-ensemble mean then repo standardization."""
    margin_sum = (torch.zeros(len(candidates), device=candidates.device)
                  if need_margin else None)
    relevance_sum = (torch.zeros(len(candidates), device=candidates.device)
                     if need_relevance else None)
    for net in nets:
        net.eval()
        embed = poison.embed_of(net)
        target_feature = (embed(target.unsqueeze(0)).flatten(1)
                          if need_relevance else None)
        margins: list[torch.Tensor] = []
        relevances: list[torch.Tensor] = []
        for start in range(0, len(candidates), batch_size):
            batch = candidates[start:start + batch_size]
            if need_relevance:
                features = embed(batch).flatten(1)
                relevances.append((features * target_feature).sum(dim=1))
                # GRAFT and GRAFT+ share this backbone pass for R and M.
                logits = classifier_logits(net, features) if need_margin else None
            else:
                logits = net(batch) if need_margin else None
            if need_margin:
                other = logits.clone()
                other[:, y_adv] = -torch.inf
                margins.append(logits[:, y_adv] - other.max(dim=1).values)
        if need_margin:
            margin_sum += torch.cat(margins)
        if need_relevance:
            relevance_sum += torch.cat(relevances)
    margin = (poison.standardize(margin_sum / len(nets))
              if need_margin else None)
    relevance = (poison.standardize(relevance_sum / len(nets))
                 if need_relevance else None)
    return margin, relevance


def backbone_interaction_score(
    nets: Sequence[nn.Module],
    candidates: torch.Tensor,
    target: torch.Tensor,
    y_adv: int,
    batch_size: int,
) -> torch.Tensor:
    """Exact production GRAFT+ A, raw-ensemble mean then standardization."""
    total = torch.zeros(len(candidates), device=candidates.device)
    for net in nets:
        interaction, _backend = poison._backbone_gradient_interactions(
            net, candidates, target, y_adv, batch_size)
        total += interaction
        del interaction
    return poison.standardize(total / len(nets))


def exact_alignment_score(
    nets: Sequence[nn.Module],
    candidates: torch.Tensor,
    target: torch.Tensor,
    y_adv: int,
    batch_size: int,
) -> torch.Tensor:
    """Call the repo's exact full-gradient selector implementation."""
    total = torch.zeros(len(candidates), device=candidates.device)
    for net in nets:
        interaction, _backend = poison._full_gradient_interactions(
            net, candidates, target, y_adv, batch_size)
        # Match select_base_exact_alignment: standardize each surrogate first.
        total += poison.standardize(interaction)
        del interaction
    return total / len(nets)


def rank_scores(scores: torch.Tensor) -> torch.Tensor:
    return torch.argsort(scores, descending=True, stable=True)


def run_selector(
    method: str,
    nets: Sequence[nn.Module],
    candidates: torch.Tensor,
    target: torch.Tensor,
    y_adv: int,
    forward_batch_size: int,
    gradient_batch_size: int,
    random_seed: int,
    rand_inner_repeats: int,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    if method == "RAND":
        generator = torch.Generator(device=candidates.device)
        generator.manual_seed(random_seed)
        scores = order = None
        for _ in range(rand_inner_repeats):
            scores = torch.rand(
                len(candidates), generator=generator, device=candidates.device)
            order = rank_scores(scores)
        return scores, order, rand_inner_repeats

    if not nets:
        die(f"{method} requires loaded surrogate models")
    if method == "M":
        margin, _ = forward_components(
            nets, candidates, target, y_adv, forward_batch_size, True, False)
        scores = -margin
    elif method == "R":
        _, relevance = forward_components(
            nets, candidates, target, y_adv, forward_batch_size, False, True)
        scores = relevance
    elif method == "GRAFT":
        margin, relevance = forward_components(
            nets, candidates, target, y_adv, forward_batch_size, True, True)
        scores = relevance - margin
    elif method == "A":
        scores = backbone_interaction_score(
            nets, candidates, target, y_adv, gradient_batch_size)
    elif method == "GRAFT+":
        margin, relevance = forward_components(
            nets, candidates, target, y_adv, forward_batch_size, True, True)
        interaction = backbone_interaction_score(
            nets, candidates, target, y_adv, gradient_batch_size)
        scores = relevance - margin + interaction
    elif method == "EXACT":
        scores = exact_alignment_score(
            nets, candidates, target, y_adv, gradient_batch_size)
    else:
        die(f"unknown method {method}")
    return scores, rank_scores(scores), 1


def validate_ranking(
    scores: torch.Tensor,
    order: torch.Tensor,
    candidate_ids: torch.Tensor,
    method: str,
) -> None:
    count = len(candidate_ids)
    if scores.ndim != 1 or len(scores) != count:
        die(f"{method} returned {len(scores)} scores for {count} candidates")
    if order.ndim != 1 or len(order) != count:
        die(f"{method} returned an incomplete full ranking")
    if not bool(torch.isfinite(scores).all()):
        die(f"{method} produced non-finite scores")
    expected = torch.arange(count, device=order.device)
    if not torch.equal(torch.sort(order).values, expected):
        die(f"{method} ranking is not a unique permutation of the candidate pool")
    ranked_ids = candidate_ids[order]
    if not torch.equal(torch.sort(ranked_ids).values,
                       torch.sort(candidate_ids).values):
        die(f"{method} did not rank exactly the shared candidate IDs")


def warmup(
    callback: Callable[[], tuple[torch.Tensor, torch.Tensor, int]],
    candidate_ids: torch.Tensor,
    method: str,
) -> None:
    gc.collect()
    torch.cuda.empty_cache()
    scores, order, _inner = callback()
    torch.cuda.synchronize()
    validate_ranking(scores, order, candidate_ids, method)
    del scores, order
    gc.collect()
    torch.cuda.empty_cache()


def measure(
    callback: Callable[[], tuple[torch.Tensor, torch.Tensor, int]],
    candidate_ids: torch.Tensor,
    method: str,
) -> dict[str, float]:
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    baseline = torch.cuda.memory_allocated()
    torch.cuda.synchronize()
    start = time.perf_counter()
    scores, order, inner_repeats = callback()
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    peak = torch.cuda.max_memory_allocated()
    peak_reserved = torch.cuda.max_memory_reserved()
    validate_ranking(scores, order, candidate_ids, method)
    del scores, order
    return {
        "wall_time_sec": elapsed / inner_repeats,
        "baseline_gpu_memory_mib": baseline / (1024.0 ** 2),
        "peak_gpu_memory_mib": peak / (1024.0 ** 2),
        "incremental_peak_gpu_memory_mib": (peak - baseline) / (1024.0 ** 2),
        "peak_reserved_memory_mib": peak_reserved / (1024.0 ** 2),
    }


def mean_std(values: Sequence[float]) -> tuple[float, float]:
    array = np.asarray(values, dtype=np.float64)
    return float(array.mean()), float(array.std(ddof=1)) if len(array) > 1 else 0.0


def summarize(raw_rows: Sequence[dict], k: int) -> list[dict]:
    groups: dict[tuple[str, str], list[dict]] = defaultdict(list)
    checkpoint_models: dict[str, str] = {}
    for row in raw_rows:
        groups[(row["architecture"], row["method"])].append(row)
        checkpoint_models[row["architecture"]] = row["checkpoint_model"]

    base: dict[tuple[str, str], dict] = {}
    for architecture in ("ConvNet", "ResNet20", "VGG13"):
        for method in METHODS:
            members = groups[(architecture, method)]
            if not members:
                die(f"no raw rows for {architecture}/{method}")
            mean_time, std_time = mean_std(
                [float(row["wall_time_sec"]) for row in members])
            base[(architecture, method)] = {
                "dataset": "CIFAR10", "architecture": architecture,
                "checkpoint_model": checkpoint_models[architecture],
                "method": method, "K": k, "n_units": len(members),
                "mean_wall_time_sec": mean_time,
                "std_wall_time_sec": std_time,
                "mean_peak_gpu_memory_mib": float(np.mean([
                    float(row["peak_gpu_memory_mib"]) for row in members])),
                "mean_incremental_peak_gpu_memory_mib": float(np.mean([
                    float(row["incremental_peak_gpu_memory_mib"])
                    for row in members])),
                "mean_peak_reserved_memory_mib": float(np.mean([
                    float(row["peak_reserved_memory_mib"]) for row in members])),
                "mean_candidates_per_sec": float(np.mean([
                    float(row["candidates_per_sec"]) for row in members])),
            }

    rows: list[dict] = []
    for architecture in ("ConvNet", "ResNet20", "VGG13"):
        graft = base[(architecture, "GRAFT")]
        graft_time = float(graft["mean_wall_time_sec"])
        graft_memory = float(graft["mean_incremental_peak_gpu_memory_mib"])
        if graft_time <= 0 or graft_memory <= 0:
            die(f"nonpositive GRAFT reference cost for {architecture}")
        for method in METHODS:
            row = dict(base[(architecture, method)])
            row["time_relative_to_GRAFT"] = (
                float(row["mean_wall_time_sec"]) / graft_time)
            row["memory_relative_to_GRAFT"] = (
                float(row["mean_incremental_peak_gpu_memory_mib"])
                / graft_memory)
            rows.append(row)
    return rows


def component_rows(summary_rows: Sequence[dict]) -> list[dict]:
    result: list[dict] = []
    for row in summary_rows:
        if row["method"] not in ("M", "R", "A", "EXACT"):
            continue
        result.append({
            "dataset": row["dataset"], "architecture": row["architecture"],
            "checkpoint_model": row["checkpoint_model"],
            "component": row["method"], "K": row["K"],
            "n_units": row["n_units"],
            "mean_wall_time_sec": row["mean_wall_time_sec"],
            "std_wall_time_sec": row["std_wall_time_sec"],
            "mean_peak_gpu_memory_mib": row["mean_peak_gpu_memory_mib"],
            "mean_incremental_peak_gpu_memory_mib":
                row["mean_incremental_peak_gpu_memory_mib"],
            "mean_peak_reserved_memory_mib":
                row["mean_peak_reserved_memory_mib"],
            "mean_candidates_per_sec": row["mean_candidates_per_sec"],
        })
    return result


def summary_lookup(rows: Sequence[dict]) -> dict[tuple[str, str], dict]:
    return {(row["architecture"], row["method"]): row for row in rows}


def plot_costs(summary_rows: Sequence[dict], figure_dir: Path) -> None:
    lookup = summary_lookup(summary_rows)
    architectures = ("ConvNet", "ResNet20", "VGG13")
    colors = ("#0072B2", "#D55E00", "#009E73")
    x = np.arange(len(METHODS), dtype=np.float64)
    width = 0.24
    fig, axes = plt.subplots(1, 2, figsize=(12.2, 4.4), constrained_layout=True)
    for arch_index, (architecture, color) in enumerate(zip(architectures, colors)):
        offset = (arch_index - 1) * width
        times = np.asarray([
            lookup[(architecture, method)]["mean_wall_time_sec"]
            for method in METHODS], dtype=np.float64)
        stds = np.asarray([
            lookup[(architecture, method)]["std_wall_time_sec"]
            for method in METHODS], dtype=np.float64)
        lower = np.minimum(stds, 0.9 * times)
        axes[0].bar(
            x + offset, times, width, yerr=np.vstack((lower, stds)),
            capsize=2.5, color=color, alpha=0.86, label=architecture)
        memory_gib = np.asarray([
            lookup[(architecture, method)][
                "mean_incremental_peak_gpu_memory_mib"] / 1024.0
            for method in METHODS], dtype=np.float64)
        axes[1].bar(
            x + offset, memory_gib, width, color=color, alpha=0.86,
            label=architecture)

    labels = [DISPLAY_METHOD.get(method, method) for method in METHODS]
    axes[0].set_yscale("log")
    axes[0].set_ylabel("Selection time (s, log scale)")
    axes[0].set_title("(a) Selection runtime", loc="left")
    axes[1].set_ylabel("Incremental peak GPU memory (GiB)")
    axes[1].set_title("(b) GPU memory above model/input baseline", loc="left")
    for axis in axes:
        axis.set_xticks(x, labels, rotation=28, ha="right")
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
        axis.grid(axis="y", alpha=0.2, linewidth=0.6)
    axes[0].legend(frameon=False, ncol=3, fontsize=8)
    figure_dir.mkdir(parents=True, exist_ok=True)
    pdf = figure_dir / "cifar10_selector_compute.pdf"
    png = figure_dir / "cifar10_selector_compute.png"
    fig.savefig(pdf, bbox_inches="tight")
    fig.savefig(png, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_relative(summary_rows: Sequence[dict], figure_dir: Path) -> None:
    lookup = summary_lookup(summary_rows)
    architectures = ("ConvNet", "ResNet20", "VGG13")
    colors = ("#0072B2", "#D55E00", "#009E73")
    x = np.arange(len(METHODS), dtype=np.float64)
    width = 0.24
    fig, axis = plt.subplots(figsize=(8.2, 4.1), constrained_layout=True)
    for arch_index, (architecture, color) in enumerate(zip(architectures, colors)):
        relative = [lookup[(architecture, method)]["time_relative_to_GRAFT"]
                    for method in METHODS]
        axis.bar(x + (arch_index - 1) * width, relative, width,
                 color=color, alpha=0.86, label=architecture)
    axis.axhline(1.0, color="black", linestyle="--", linewidth=1.0)
    axis.set_yscale("log")
    axis.set_ylabel("Runtime / GRAFT")
    axis.set_xticks(x, [DISPLAY_METHOD.get(method, method) for method in METHODS],
                    rotation=28, ha="right")
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.grid(axis="y", alpha=0.2, linewidth=0.6)
    axis.legend(frameon=False, ncol=3, fontsize=8)
    pdf = figure_dir / "cifar10_selector_compute_relative.pdf"
    png = figure_dir / "cifar10_selector_compute_relative.png"
    fig.savefig(pdf, bbox_inches="tight")
    fig.savefig(png, dpi=300, bbox_inches="tight")
    plt.close(fig)


def write_latex(summary_rows: Sequence[dict], path: Path) -> None:
    lookup = summary_lookup(summary_rows)
    architectures = ("ConvNet", "ResNet20", "VGG13")
    lines = [
        r"\begin{tabular}{l" + "ccc" * len(architectures) + "}",
        r"\toprule",
        "Method & " + " & ".join(
            rf"\multicolumn{{3}}{{c}}{{{architecture}}}"
            for architecture in architectures) + r" \\",
        " & " + " & ".join(
            [r"Time (s)", r"Time/GRAFT", r"Peak Mem. (GiB)"]
            * len(architectures)) + r" \\",
        r"\midrule",
    ]
    for method in METHODS:
        cells = [DISPLAY_METHOD.get(method, method)]
        for architecture in architectures:
            row = lookup[(architecture, method)]
            mean_time = float(row["mean_wall_time_sec"])
            std_time = float(row["std_wall_time_sec"])
            relative = float(row["time_relative_to_GRAFT"])
            memory = float(row["mean_incremental_peak_gpu_memory_mib"]) / 1024.0
            cells.extend([
                rf"${mean_time:.3g}\pm{std_time:.2g}$",
                f"{relative:.2f}$\\times$", f"{memory:.3f}"])
        lines.append(" & ".join(cells) + r" \\")
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    atomic_text(path, "\n".join(lines) + "\n")


def git_commit() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT,
            stderr=subprocess.DEVNULL, text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def run(cli: argparse.Namespace) -> None:
    if not torch.cuda.is_available():
        die("CUDA is required for this GPU benchmark")
    device = torch.device(cli.device)
    if device.type != "cuda":
        die("--device must be a CUDA device")
    torch.cuda.set_device(device)
    poison.set_seed(cli.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    jobs = read_jobs(cli.jobs_file)
    by_architecture: dict[str, list[dict[str, str]]] = defaultdict(list)
    for job in jobs:
        by_architecture[job["architecture"]].append(job)

    context_args = argparse.Namespace(dataset="CIFAR10", data_path=str(cli.data_path))
    print("Loading CIFAR-10 and moving normalized tensors to the GPU (untimed).",
          flush=True)
    ctx = poison.build_context(context_args, str(device))
    if ctx["num_classes"] != 10:
        die(f"expected CIFAR-10 output dimension 10, got {ctx['num_classes']}")
    train_images, train_labels = ctx["train_imgs"], ctx["train_labs"]
    test_images, test_labels = ctx["test_imgs"], ctx["test_labs"]

    print(f"GPU: {torch.cuda.get_device_name(device)}", flush=True)
    print(f"PyTorch: {torch.__version__}", flush=True)
    print(f"CUDA runtime: {torch.version.cuda}", flush=True)
    print(f"K={cli.k}; measured repeats={cli.repeats}; "
          f"forward batch={cli.forward_batch_size}; "
          f"gradient batch={cli.gradient_batch_size}", flush=True)

    raw_rows: list[dict] = []
    metadata_pools: dict[str, dict] = {}
    metadata_checkpoints: dict[str, list[str]] = {}
    total_target_units = 0

    for architecture in ("ConvNet", "ResNet20", "VGG13"):
        arch_jobs = by_architecture[architecture]
        models = {job["model"] for job in arch_jobs}
        if len(models) != 1:
            die(f"{architecture} maps to multiple checkpoint model names: {models}")
        model_name = next(iter(models))
        paths = checkpoint_paths(cli, model_name)
        print(f"\n{architecture}: reading {len(paths)} checkpoints into CPU memory "
              "(untimed).", flush=True)
        states = [load_cpu_state(path) for path in paths]
        metadata_checkpoints[architecture] = [str(path.resolve()) for path in paths]

        prepared: list[dict] = []
        for job in arch_jobs:
            y_adv, target_class = poison.parse_pair(
                job["class_pair"], ctx["class_names"], "poison-target")
            candidate_ids = (train_labels == y_adv).nonzero(as_tuple=True)[0]
            candidates = train_images[candidate_ids]
            target_ids = load_target_ids(
                Path(job["target_file"]), job["class_pair"])
            invalid = [target_id for target_id in target_ids
                       if target_id < 0 or target_id >= len(test_labels)
                       or int(test_labels[target_id]) != target_class]
            if invalid:
                die(f"invalid target IDs for {architecture}/{job['class_pair']}: "
                    f"{invalid}")
            prepared.append({
                **job, "y_adv": y_adv, "target_class": target_class,
                "candidate_ids": candidate_ids, "candidates": candidates,
                "target_ids": target_ids,
            })
            metadata_pools[f"{architecture}/{job['class_pair']}"] = {
                "n_candidates": len(candidate_ids), "candidate_class": int(y_adv),
                "target_ids": target_ids,
                "target_file": str(Path(job["target_file"]).resolve()),
            }
            total_target_units += len(target_ids)
            print(f"  {job['class_pair']}: targets={len(target_ids)}, "
                  f"candidates={len(candidate_ids)}", flush=True)

        warmed: set[str] = set()

        # RAND is benchmarked with no model resident on the GPU.
        for item in prepared:
            for target_id in item["target_ids"]:
                target = test_images[target_id]
                method = "RAND"
                callback = lambda m=method, t=target, it=item: run_selector(
                    m, (), it["candidates"], t, it["y_adv"],
                    cli.forward_batch_size, cli.gradient_batch_size,
                    cli.seed + target_id, cli.rand_inner_repeats)
                if method not in warmed:
                    print(f"  warmup {architecture}/{method}", flush=True)
                    warmup(callback, item["candidate_ids"], method)
                    warmed.add(method)
                for repeat in range(cli.repeats):
                    measured_callback = lambda m=method, t=target, it=item, r=repeat: \
                        run_selector(
                            m, (), it["candidates"], t, it["y_adv"],
                            cli.forward_batch_size, cli.gradient_batch_size,
                            cli.seed + 100000 * repeat + target_id,
                            cli.rand_inner_repeats)
                    measurement = measure(
                        measured_callback, item["candidate_ids"], method)
                    raw_rows.append(make_raw_row(
                        architecture, model_name, item, target_id, method,
                        repeat, cli.k, measurement))

        print(f"  loading {cli.k} {model_name} models onto {device} (untimed).",
              flush=True)
        nets = load_models(
            model_name, states, ctx["channel"], ctx["num_classes"],
            ctx["im_size"], device, cli.seed)
        del states
        gc.collect()
        torch.cuda.empty_cache()

        for method in METHODS[1:]:
            for item in prepared:
                for target_id in item["target_ids"]:
                    target = test_images[target_id]
                    callback = lambda m=method, t=target, it=item: run_selector(
                        m, nets, it["candidates"], t, it["y_adv"],
                        cli.forward_batch_size, cli.gradient_batch_size,
                        cli.seed + target_id, cli.rand_inner_repeats)
                    if method not in warmed:
                        print(f"  warmup {architecture}/{method}", flush=True)
                        warmup(callback, item["candidate_ids"], method)
                        warmed.add(method)
                    for repeat in range(cli.repeats):
                        measurement = measure(
                            callback, item["candidate_ids"], method)
                        raw_rows.append(make_raw_row(
                            architecture, model_name, item, target_id,
                            method, repeat, cli.k, measurement))
                    print(f"  complete {architecture}/{item['class_pair']}/"
                          f"target={target_id}/{method}", flush=True)

        del nets
        for item in prepared:
            del item["candidates"]
        gc.collect()
        torch.cuda.empty_cache()

    summary_rows = summarize(raw_rows, cli.k)
    components = component_rows(summary_rows)
    results_dir = cli.output_root / "results"
    figures_dir = cli.output_root / "figures"
    tables_dir = cli.output_root / "tables"
    atomic_csv(results_dir / "cifar10_selector_compute_raw.csv",
               RAW_FIELDS, raw_rows)
    atomic_csv(results_dir / "cifar10_selector_compute_summary.csv",
               SUMMARY_FIELDS, summary_rows)
    atomic_csv(results_dir / "cifar10_selector_component_timing.csv",
               COMPONENT_FIELDS, components)
    write_latex(summary_rows, tables_dir / "cifar10_selector_compute.tex")
    plot_costs(summary_rows, figures_dir)
    plot_relative(summary_rows, figures_dir)

    metadata = {
        "dataset": "CIFAR10",
        "gpu_name": torch.cuda.get_device_name(device),
        "cuda_version": torch.version.cuda,
        "pytorch_version": torch.__version__,
        "python_version": platform.python_version(),
        "device": str(device),
        "forward_batch_size": cli.forward_batch_size,
        "gradient_batch_size": cli.gradient_batch_size,
        "rand_inner_repeats": cli.rand_inner_repeats,
        "measured_repeats": cli.repeats,
        "K": cli.k,
        "number_of_target_units": total_target_units,
        "candidate_pools": metadata_pools,
        "architecture_names": ["ConvNet", "ResNet20", "VGG13"],
        "checkpoint_model_names": {
            architecture: by_architecture[architecture][0]["model"]
            for architecture in ("ConvNet", "ResNet20", "VGG13")},
        "checkpoint_paths": metadata_checkpoints,
        "datetime_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": git_commit(),
        "seed": cli.seed,
    }
    atomic_json(results_dir / "cifar10_selector_compute_metadata.json", metadata)
    print("\nBenchmark complete.", flush=True)


def make_raw_row(
    architecture: str,
    model_name: str,
    item: dict,
    target_id: int,
    method: str,
    repeat: int,
    k: int,
    measurement: dict[str, float],
) -> dict:
    elapsed = measurement["wall_time_sec"]
    return {
        "dataset": "CIFAR10", "architecture": architecture,
        "checkpoint_model": model_name, "class_pair": item["class_pair"],
        "target_id": target_id, "method": method, "K": k,
        "n_candidates": len(item["candidate_ids"]), "repeat": repeat,
        "wall_time_sec": elapsed,
        "baseline_gpu_memory_mib": measurement["baseline_gpu_memory_mib"],
        "peak_gpu_memory_mib": measurement["peak_gpu_memory_mib"],
        "incremental_peak_gpu_memory_mib":
            measurement["incremental_peak_gpu_memory_mib"],
        "peak_reserved_memory_mib": measurement["peak_reserved_memory_mib"],
        "candidates_per_sec": len(item["candidate_ids"]) / elapsed,
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark CIFAR-10 selector compute and CUDA memory only.")
    parser.add_argument("--jobs-file", type=Path, required=True)
    parser.add_argument("--data-path", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--k", type=int, default=20)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--rand-inner-repeats", type=int, default=100)
    parser.add_argument("--forward-batch-size", type=int, default=512)
    parser.add_argument("--gradient-batch-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--surrogate-epochs", type=int, default=60)
    parser.add_argument("--surrogate-lr", type=float, default=0.1)
    parser.add_argument("--surrogate-bs", type=int, default=128)
    parser.add_argument("--surrogate-decay", nargs="*", type=int,
                        default=[35, 45])
    parser.add_argument("--surrogate-wd", type=float, default=0.0)
    parser.add_argument("--surrogate-aug", action="store_true", default=False)
    args = parser.parse_args(argv)
    if args.k != 20:
        parser.error("--k must be 20 to match the main GRAFT experiments")
    if args.repeats < 3:
        parser.error("--repeats must be at least 3")
    if min(args.rand_inner_repeats, args.forward_batch_size,
           args.gradient_batch_size) <= 0:
        parser.error("repeat and batch-size arguments must be positive")
    return args


if __name__ == "__main__":
    run(parse_args())
