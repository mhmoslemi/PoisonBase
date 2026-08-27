#!/usr/bin/env python3
"""Validate residual suppression on 10 targets and four image datasets.

This file extends the proposition-only experiment.  It does not craft poisons or
train victims.  Collection still requires a CUDA GPU because ``A_i`` is the exact
candidate/target gradient inner product over all non-head parameters.

Default study
-------------

================  ==========  =============================
Dataset           Model       adversarial--target pair
================  ==========  =============================
CIFAR10           ConvNetBN   dog--bird
CIFAR100          ResNet18BN  sea--willow_tree
TinyImageNet      ResNet18BN  n01443537--n01629819
SVHN              ConvNetBN   9--2
================  ==========  =============================

Each dataset uses 10 targets, 5 seed-matched surrogates, and 30 equal-count
signed-margin groups formed separately inside every target--surrogate pool.
Target indices are frozen under OUTPUT_ROOT/targets.  Collection is resumable.

Run inside a GPU allocation, one dataset at a time
-------------------------------------------------

    python residual_suppression_multidataset.py collect --dataset CIFAR10
    python residual_suppression_multidataset.py collect --dataset CIFAR100
    python residual_suppression_multidataset.py collect --dataset TinyImageNet
    python residual_suppression_multidataset.py collect --dataset SVHN

If an allocation expires, repeat the same command.  Completed targets are
validated and skipped.  To distribute targets over independent jobs, use for
example ``--positions 0,1,2`` and ``--positions 3,4,5``.

After all collections finish, analysis is CPU-only
--------------------------------------------------

    python residual_suppression_multidataset.py analyze

The analysis writes audit tables and a vector PDF with one row per dataset and
one column for ||r_i||_2, |H_i|, and |A_i|.
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import math
import os
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, NoReturn, Sequence

import numpy as np
import torch
import torch.nn.functional as F

import final_update as poison


@dataclass(frozen=True)
class DatasetSpec:
    dataset: str
    model: str
    class_pair: str
    display_name: str
    initial_target_file: str | None = None


SPECS = {
    "CIFAR10": DatasetSpec(
        dataset="CIFAR10",
        model="ConvNetBN",
        class_pair="dog-bird",
        display_name="CIFAR-10",
        # This file already contains the ten targets used by the main sweep.  Its
        # first five entries are the targets in the original proposition run.
        initial_target_file="target_sets/ConvNetBN_gradmatch_dog-bird.json",
    ),
    "CIFAR100": DatasetSpec(
        dataset="CIFAR100",
        model="ResNet18BN",
        class_pair="sea-willow_tree",
        display_name="CIFAR-100",
    ),
    "TinyImageNet": DatasetSpec(
        dataset="TinyImageNet",
        model="ResNet18BN",
        class_pair="n01443537-n01629819",
        display_name="Tiny-ImageNet",
    ),
    "SVHN": DatasetSpec(
        dataset="SVHN",
        model="ConvNetBN",
        class_pair="9-2",
        display_name="SVHN",
    ),
}

METRICS = (
    "residual_norm",
    "abs_head_interaction",
    "abs_backbone_interaction",
)


def die(message: str) -> NoReturn:
    raise SystemExit("ERROR: " + message)


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


def write_csv(path: Path, fieldnames: Sequence[str], rows: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})
    os.replace(temporary, path)


def safe_slug(value: str) -> str:
    return value.replace("/", "_").replace(" ", "_")


def candidate_dir(output_root: Path, spec: DatasetSpec) -> Path:
    return (output_root / "candidates" / spec.dataset / spec.model /
            safe_slug(spec.class_pair))


def target_manifest_path(output_root: Path, spec: DatasetSpec,
                         num_targets: int) -> Path:
    return (output_root / "targets" /
            f"{spec.dataset}_{spec.model}_{safe_slug(spec.class_pair)}_"
            f"{num_targets}.json")


def production_args(cli: argparse.Namespace, spec: DatasetSpec) -> argparse.Namespace:
    argv = [
        "--dataset", spec.dataset,
        "--data_path", str(cli.data_path),
        "--model", spec.model,
        "--seed", str(cli.seed),
        "--cache_dir", str(cli.cache_dir),
        "--out_dir", str(cli.output_root / "unused_attack_output"),
        "--class_pair", spec.class_pair,
        "--pair_order", "poison-target",
        "--num_surrogates", str(cli.num_surrogates),
        "--surrogate_epochs", str(cli.surrogate_epochs),
        "--surrogate_decay", "35", "45",
        "--num_targets", str(cli.num_targets),
        "--target_select", str(cli.target_select),
        "--num_victims", "1",
        "--rank_on_surrogates",
        "--gpus", cli.gpus,
    ]
    return poison.parse_args(argv)


def read_target_count(path: Path, class_pair: str) -> int:
    try:
        with path.open() as handle:
            payload = json.load(handle)
        values = (payload["pairs"][class_pair]["indices"]
                  if "pairs" in payload else payload[class_pair])
        return len(values)
    except (OSError, KeyError, TypeError, ValueError):
        return 0


def freeze_targets(parsed: argparse.Namespace, cli: argparse.Namespace,
                   spec: DatasetSpec, ctx: dict,
                   nets: Sequence[torch.nn.Module]) -> tuple[int, list[int], dict[int, float]]:
    """Select targets once and persist the exact indices used by every rerun."""
    manifest = target_manifest_path(cli.output_root, spec, cli.num_targets)
    parsed.target_idx_file = None
    source = None

    if manifest.is_file():
        source = manifest
    elif cli.target_file:
        requested = Path(cli.target_file).expanduser().resolve()
        if read_target_count(requested, spec.class_pair) < cli.num_targets:
            die(f"{requested} does not contain {cli.num_targets} targets for "
                f"{spec.class_pair}")
        source = requested
    elif spec.initial_target_file:
        requested = (cli.source_root / spec.initial_target_file).resolve()
        if read_target_count(requested, spec.class_pair) >= cli.num_targets:
            source = requested

    if source is not None:
        parsed.target_idx_file = str(source)

    y_adv, target_class = poison.parse_pair(
        spec.class_pair, ctx["class_names"], parsed.pair_order)
    generator = torch.Generator(device="cpu").manual_seed(cli.seed)
    targets, scores = poison.select_targets(
        parsed, nets, ctx["test_imgs"], ctx["test_labs"],
        y_adv, target_class,
        generator,
    )
    targets = [int(value) for value in targets]
    if len(targets) != cli.num_targets:
        die(f"{spec.dataset} produced {len(targets)} eligible targets; expected "
            f"{cli.num_targets}")

    payload = {
        "_generated_by": "residual_suppression_multidataset.py",
        "dataset": spec.dataset,
        "model": spec.model,
        "class_pair": spec.class_pair,
        "num_targets": cli.num_targets,
        "target_select": cli.target_select,
        "seed": cli.seed,
        "source_target_file": str(source) if source is not None else None,
        "pairs": {spec.class_pair: {"indices": targets}},
        "target_scores": {str(key): float(value) for key, value in scores.items()},
    }
    # Identical concurrent writers produce identical content; os.replace keeps the
    # manifest complete even if several target-position jobs start together.
    atomic_json(manifest, payload)
    print(f"targets {spec.dataset}: {targets}", flush=True)
    return y_adv, targets, scores


@torch.no_grad()
def head_quantities(net: torch.nn.Module, candidates: torch.Tensor,
                    target: torch.Tensor, y_adv: int, batch_size: int):
    """Compute M_i, ||r_i||_2, H_i, and R_i for one surrogate."""
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
        head_terms.append(
            (residual * target_residual).sum(dim=1) *
            (feature * target_feature).sum(dim=1)
        )
        relevances.append(F.cosine_similarity(
            feature, target_feature.expand(len(feature), -1), dim=1))

    return tuple(torch.cat(parts).detach().cpu().numpy() for parts in (
        margins, residual_norms, head_terms, relevances))


def validate_candidate_file(path: Path, spec: DatasetSpec, target_idx: int,
                            num_surrogates: int) -> bool:
    try:
        with np.load(path, allow_pickle=False) as data:
            count = int(data["candidate_count"])
            return bool(
                str(data["dataset"]) == spec.dataset and
                str(data["model"]) == spec.model and
                str(data["class_pair"]) == spec.class_pair and
                int(data["target_idx"]) == target_idx and
                int(data["model_count"]) == num_surrogates and
                len(data["model_id"]) == count * num_surrogates and
                all(len(data[name]) == count * num_surrogates for name in (
                    "candidate_idx", "margin", "residual_norm",
                    "head_interaction", "backbone_interaction", "relevance"))
            )
    except (OSError, KeyError, TypeError, ValueError):
        return False


def parse_positions(value: str, num_targets: int) -> list[int]:
    if value.strip().lower() == "all":
        return list(range(num_targets))
    try:
        positions = sorted({int(item) for item in value.replace(" ", "").split(",")
                            if item != ""})
    except ValueError:
        die("--positions must be 'all' or comma-separated integers")
    if not positions or positions[0] < 0 or positions[-1] >= num_targets:
        die(f"--positions must lie in [0, {num_targets - 1}]")
    return positions


def ensure_surrogates_cached(parsed: argparse.Namespace, num_surrogates: int,
                             allow_train_missing: bool) -> None:
    directory = Path(poison.surrogate_dir(parsed))
    missing = [directory / f"net_{model_id}.pt" for model_id in range(num_surrogates)
               if not (directory / f"net_{model_id}.pt").is_file()]
    if missing and not allow_train_missing:
        formatted = "\n".join(str(path) for path in missing)
        die("missing surrogate checkpoints; precompute them first or pass "
            f"--allow-train-missing\n{formatted}")


def collect_dataset(cli: argparse.Namespace, spec: DatasetSpec) -> None:
    if not torch.cuda.is_available():
        die("collection requires CUDA; run 'analyze' on CPU only after collection")
    parsed = production_args(cli, spec)
    gpus = poison.resolve_gpus(parsed.gpus)
    if len(gpus) != 1:
        die("collection expects exactly one visible/requested GPU")
    device = f"cuda:{gpus[0]}"
    torch.cuda.set_device(gpus[0])
    poison.set_seed(cli.seed)

    ensure_surrogates_cached(parsed, cli.num_surrogates, cli.allow_train_missing)
    print(f"loading {spec.dataset} on {device}", flush=True)
    ctx = poison.build_context(parsed, device)
    nets = poison.get_surrogates(
        parsed,
        ctx["train_imgs"], ctx["train_labs"],
        ctx["test_imgs"], ctx["test_labs"],
        ctx["channel"], ctx["num_classes"], ctx["im_size"],
        device, ctx["dsa_param"],
    )
    y_adv, targets, target_scores = freeze_targets(parsed, cli, spec, ctx, nets)
    positions = parse_positions(cli.positions, cli.num_targets)
    class_indices = (ctx["train_labs"] == y_adv).nonzero(as_tuple=True)[0]
    candidates = ctx["train_imgs"][class_indices]
    candidate_indices = class_indices.detach().cpu().numpy().astype(np.int64)
    output_dir = candidate_dir(cli.output_root, spec)

    for position in positions:
        target_idx = targets[position]
        output = output_dir / f"target_{target_idx}.npz"
        if output.exists() and not cli.force:
            if validate_candidate_file(
                    output, spec, target_idx, cli.num_surrogates):
                print(f"complete, skipping {output}", flush=True)
                continue
            die(f"incomplete or incompatible cache {output}; inspect it or rerun "
                "with --force")

        target = ctx["test_imgs"][target_idx]
        columns: dict[str, list[np.ndarray]] = defaultdict(list)
        backends: list[str] = []
        print(
            f"{spec.dataset} target {position + 1}/{cli.num_targets} "
            f"(index {target_idx}, candidates {len(candidate_indices)})",
            flush=True,
        )
        for model_id, net in enumerate(nets):
            print(f"  surrogate {model_id + 1}/{len(nets)}", flush=True)
            margin, residual, head, relevance = head_quantities(
                net, candidates, target, y_adv, cli.forward_batch_size)
            backbone, backend = poison._backbone_gradient_interactions(
                net, candidates, target, y_adv, cli.jacobian_batch_size)
            count = len(candidate_indices)
            columns["model_id"].append(np.full(count, model_id, dtype=np.int16))
            columns["candidate_idx"].append(candidate_indices)
            columns["margin"].append(margin.astype(np.float32))
            columns["residual_norm"].append(residual.astype(np.float32))
            columns["head_interaction"].append(head.astype(np.float32))
            columns["backbone_interaction"].append(
                backbone.detach().cpu().numpy().astype(np.float32))
            columns["relevance"].append(relevance.astype(np.float32))
            backends.append(backend)
            del backbone
            torch.cuda.empty_cache()

        atomic_npz(
            output,
            schema_version=np.array(2, dtype=np.int16),
            dataset=np.array(spec.dataset),
            model=np.array(spec.model),
            class_pair=np.array(spec.class_pair),
            seed=np.array(cli.seed, dtype=np.int64),
            target_idx=np.array(target_idx, dtype=np.int64),
            target_position=np.array(position, dtype=np.int16),
            target_score=np.array(target_scores[target_idx], dtype=np.float32),
            adversarial_label=np.array(y_adv, dtype=np.int16),
            num_classes=np.array(ctx["num_classes"], dtype=np.int16),
            candidate_count=np.array(len(candidate_indices), dtype=np.int32),
            model_count=np.array(len(nets), dtype=np.int16),
            target_image=target.detach().cpu().numpy().astype(np.float32),
            jacobian_backends=np.asarray(backends),
            **{name: np.concatenate(parts) for name, parts in columns.items()},
        )
        print(f"wrote {output}", flush=True)

    del nets, ctx, candidates
    gc.collect()
    torch.cuda.empty_cache()


def rank_groups(values: np.ndarray, number: int) -> np.ndarray:
    """Assign equal-count groups by stable rank, independently within a pool."""
    order = np.argsort(values, kind="mergesort")
    groups = np.empty(len(values), dtype=np.int16)
    groups[order] = np.minimum(
        number - 1, np.arange(len(values)) * number // len(values)) + 1
    return groups


def candidate_files(output_root: Path, dataset: str | None = None) -> list[Path]:
    root = output_root / "candidates"
    if dataset:
        return sorted((root / dataset).glob("*/*/target_*.npz"))
    return sorted(root.glob("*/*/*/target_*.npz"))


def analyze_candidates(cli: argparse.Namespace) -> None:
    files = candidate_files(cli.output_root)
    if not files:
        die(f"no candidate files under {cli.output_root / 'candidates'}")

    by_dataset: dict[str, list[Path]] = defaultdict(list)
    for path in files:
        with np.load(path, allow_pickle=False) as data:
            by_dataset[str(data["dataset"])].append(path)

    requested = list(SPECS)
    if not cli.allow_partial:
        problems = []
        for dataset in requested:
            count = len(by_dataset.get(dataset, []))
            if count != cli.num_targets:
                problems.append(f"{dataset} has {count}/{cli.num_targets} targets")
        if problems:
            die("collection incomplete: " + "; ".join(problems))

    pool_rows: list[dict] = []
    grouped_values: dict[tuple[str, int, str], list[np.ndarray]] = defaultdict(list)
    audit_rows: list[dict] = []

    for dataset in requested:
        dataset_files = by_dataset.get(dataset, [])
        if not dataset_files:
            continue
        target_ids = set()
        evaluations = 0
        violations = 0
        pool_count = 0
        candidate_counts = set()

        for path in dataset_files:
            with np.load(path, allow_pickle=False) as data:
                model = str(data["model"])
                class_pair = str(data["class_pair"])
                target_idx = int(data["target_idx"])
                model_count = int(data["model_count"])
                count = int(data["candidate_count"])
                num_classes = int(data["num_classes"])
                target_ids.add(target_idx)
                candidate_counts.add(count)
                evaluations += count * model_count

                margin_all = np.asarray(data["margin"], dtype=np.float64)
                residual_all = np.asarray(data["residual_norm"], dtype=np.float64)
                upper = (math.sqrt(2.0) * (num_classes - 1) /
                         (np.exp(np.clip(margin_all, -700, 700)) + num_classes - 1))
                violations += int(np.sum(
                    residual_all > upper * (1.0 + 1e-4) + 1e-8))

                for model_id in range(model_count):
                    pool_count += 1
                    section = slice(model_id * count, (model_id + 1) * count)
                    margins = margin_all[section]
                    groups = rank_groups(margins, cli.num_groups)
                    values = {
                        "residual_norm": residual_all[section],
                        "abs_head_interaction": np.abs(np.asarray(
                            data["head_interaction"][section], dtype=np.float64)),
                        "abs_backbone_interaction": np.abs(np.asarray(
                            data["backbone_interaction"][section], dtype=np.float64)),
                    }
                    for group in range(1, cli.num_groups + 1):
                        take = groups == group
                        row = {
                            "dataset": dataset,
                            "model": model,
                            "class_pair": class_pair,
                            "target_idx": target_idx,
                            "model_id": model_id,
                            "margin_group": group,
                            "candidate_count": int(take.sum()),
                        }
                        for metric, metric_values in values.items():
                            selected = metric_values[take]
                            row[f"{metric}_median"] = float(np.median(selected))
                            grouped_values[(dataset, group, metric)].append(selected)
                        pool_rows.append(row)

        audit_rows.append({
            "dataset": dataset,
            "targets": len(target_ids),
            "target_surrogate_pools": pool_count,
            "candidate_surrogate_evaluations": evaluations,
            "candidate_count_per_target": ";".join(map(str, sorted(candidate_counts))),
            "margin_groups": cli.num_groups,
            "residual_bound_violations": violations,
        })

    dataset_rows: list[dict] = []
    for dataset in requested:
        if dataset not in by_dataset:
            continue
        for group in range(1, cli.num_groups + 1):
            row = {"dataset": dataset, "margin_group": group}
            count = 0
            for metric in METRICS:
                values = np.concatenate(grouped_values[(dataset, group, metric)])
                count = len(values)
                row[f"{metric}_median"] = float(np.median(values))
            row["candidate_count"] = count
            dataset_rows.append(row)

    table_dir = cli.output_root / "tables"
    pool_fields = [
        "dataset", "model", "class_pair", "target_idx", "model_id",
        "margin_group", "candidate_count",
    ] + [f"{metric}_median" for metric in METRICS]
    dataset_fields = ["dataset", "margin_group", "candidate_count"] + [
        f"{metric}_median" for metric in METRICS]
    write_csv(table_dir / "margin_groups_by_target_model.csv", pool_fields, pool_rows)
    write_csv(table_dir / "margin_groups_by_dataset.csv", dataset_fields, dataset_rows)
    write_csv(table_dir / "residual_suppression_audit.csv", [
        "dataset", "targets", "target_surrogate_pools",
        "candidate_surrogate_evaluations", "candidate_count_per_target",
        "margin_groups", "residual_bound_violations",
    ], audit_rows)

    make_figure(cli, pool_rows, dataset_rows)
    print(f"analysis complete under {cli.output_root}")


def make_figure(cli: argparse.Namespace, pool_rows: Sequence[dict],
                dataset_rows: Sequence[dict]) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    navy = "#004481"
    grey = "#7B8794"
    dark = "#18212B"
    grid = "#D9DEE3"

    matplotlib.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "mathtext.fontset": "dejavuserif",
        "font.size": 9.5,
        "axes.titlesize": 11.5,
        "axes.titleweight": "bold",
        "axes.labelsize": 10.5,
        "axes.labelweight": "bold",
        "xtick.labelsize": 8.5,
        "ytick.labelsize": 8.5,
        "legend.fontsize": 9.5,
        "axes.edgecolor": dark,
        "axes.linewidth": 0.9,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })

    specs = [
        ("residual_norm", r"$\|r_i\|_2$", "Residual norm"),
        ("abs_head_interaction", r"$|H_i|$", "Classifier-head interaction"),
        ("abs_backbone_interaction", r"$|A_i|$", "Backbone interaction"),
    ]
    present = [dataset for dataset in SPECS
               if any(row["dataset"] == dataset for row in dataset_rows)]
    figure, axes = plt.subplots(
        len(present), 3, figsize=(14.2, 3.15 * len(present)),
        squeeze=False, constrained_layout=True,
    )

    for row_id, dataset in enumerate(present):
        pooled = sorted(
            (row for row in dataset_rows if row["dataset"] == dataset),
            key=lambda row: row["margin_group"],
        )
        pools = [row for row in pool_rows if row["dataset"] == dataset]
        pool_keys = sorted({(row["target_idx"], row["model_id"]) for row in pools})
        x = np.array([row["margin_group"] for row in pooled], dtype=float)

        for column_id, (metric, ylabel, title) in enumerate(specs):
            axis = axes[row_id, column_id]
            median_key = f"{metric}_median"

            for target_idx, model_id in pool_keys:
                frame = sorted(
                    (row for row in pools
                     if row["target_idx"] == target_idx and row["model_id"] == model_id),
                    key=lambda row: row["margin_group"],
                )
                axis.plot(
                    [row["margin_group"] for row in frame],
                    [float(row[median_key]) if float(row[median_key]) > 0 else np.nan
                     for row in frame],
                    color=grey, lw=0.45, alpha=0.12, zorder=1,
                )

            median_raw = np.array([row[median_key] for row in pooled], dtype=float)
            median = np.where(median_raw > 0, median_raw, np.nan)
            axis.plot(
                x, median, color=navy, lw=2.1, marker="o", markersize=2.8,
                markeredgecolor="white", markeredgewidth=0.35, zorder=4,
            )
            if row_id == 0:
                axis.set_title(f"({chr(97 + column_id)})  {title}", pad=7)
            if column_id == 0:
                axis.text(
                    -0.20, 0.5, SPECS[dataset].display_name,
                    transform=axis.transAxes, rotation=90, ha="center", va="center",
                    fontsize=11.5, fontweight="bold",
                )
            axis.set_ylabel(ylabel)
            if row_id == len(present) - 1:
                axis.set_xlabel("Signed-margin group  (low → high)")
            axis.set_xlim(1, cli.num_groups)
            ticks = sorted({1, cli.num_groups, *range(5, cli.num_groups + 1, 5)})
            axis.set_xticks(ticks)
            axis.set_yscale("log")
            axis.grid(True, axis="y", which="major", color=grid,
                      lw=0.55, ls="--", alpha=0.85)
            axis.grid(True, axis="y", which="minor", color=grid,
                      lw=0.3, ls=":", alpha=0.4)
            axis.spines["top"].set_visible(False)
            axis.spines["right"].set_visible(False)

    figure.legend(handles=[
        Line2D([0], [0], color=grey, lw=1.0, alpha=0.5,
               label="Each target–surrogate pool"),
        Line2D([0], [0], color=navy, lw=2.1, marker="o", ms=4,
               label="Overall median"),
    ], loc="upper center", bbox_to_anchor=(0.5, 1.025), ncol=2,
       frameon=True, fancybox=False, edgecolor=grid)

    output = cli.output_root / "figures"
    output.mkdir(parents=True, exist_ok=True)
    pdf = output / "residual_suppression_30_groups_multidataset.pdf"
    png = output / "residual_suppression_30_groups_multidataset.png"
    figure.savefig(pdf, bbox_inches="tight")
    figure.savefig(png, dpi=500, bbox_inches="tight")
    plt.close(figure)
    print(f"wrote {pdf}")
    print(f"wrote {png}")


def add_common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--source-root", type=Path, default=Path(__file__).resolve().parent,
        help="PoisonBase project root",
    )
    parser.add_argument("--data-path", type=Path, default=Path("./data"))
    parser.add_argument("--cache-dir", type=Path, default=Path("./cache"))
    parser.add_argument(
        "--output-root", type=Path,
        default=Path("./logs-proposition-multidataset"),
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-targets", type=int, default=10)
    parser.add_argument("--num-surrogates", type=int, default=5)
    parser.add_argument("--num-groups", type=int, default=30)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="mode", required=True)

    collect = subparsers.add_parser("collect", help="GPU collection")
    add_common(collect)
    collect.add_argument("--dataset", choices=list(SPECS) + ["all"], required=True)
    collect.add_argument("--positions", default="all")
    collect.add_argument("--gpus", default="0")
    collect.add_argument("--target-select", default="70")
    collect.add_argument(
        "--target-file",
        help="optional pinned target JSON; valid only when collecting one dataset",
    )
    collect.add_argument("--surrogate-epochs", type=int, default=60)
    collect.add_argument("--forward-batch-size", type=int, default=512)
    collect.add_argument("--jacobian-batch-size", type=int, default=64)
    collect.add_argument("--allow-train-missing", action="store_true")
    collect.add_argument("--force", action="store_true")

    analyze = subparsers.add_parser("analyze", help="CPU tables and figure")
    add_common(analyze)
    analyze.add_argument("--allow-partial", action="store_true")

    args = parser.parse_args(argv)
    args.source_root = args.source_root.expanduser().resolve()
    args.data_path = args.data_path.expanduser().resolve()
    args.cache_dir = args.cache_dir.expanduser().resolve()
    args.output_root = args.output_root.expanduser().resolve()
    if args.num_targets <= 0 or args.num_surrogates <= 0:
        parser.error("--num-targets and --num-surrogates must be positive")
    if args.num_groups < 2:
        parser.error("--num-groups must be at least 2")
    if args.mode == "collect":
        args.target_select = poison.target_select_arg(args.target_select)
        if args.dataset == "all" and args.target_file:
            parser.error("--target-file cannot be combined with --dataset all")
        if args.forward_batch_size <= 0 or args.jacobian_batch_size <= 0:
            parser.error("batch sizes must be positive")
    return args


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    args.output_root.mkdir(parents=True, exist_ok=True)
    if args.mode == "analyze":
        analyze_candidates(args)
        return

    datasets = list(SPECS) if args.dataset == "all" else [args.dataset]
    for dataset in datasets:
        collect_dataset(args, SPECS[dataset])


if __name__ == "__main__":
    main()
