#!/usr/bin/env python3
"""Plot pairwise overlap of samples selected by five ranking rules.

For every target and poison ratio rho, this script selects m=rho*n candidates
with each method and computes

    |S_a intersection S_b| / m.

The methods are exact <g_i,g_t>, a hybrid A+(-M)R, GRAFT, GRAFT+, and the
expected overlap of a uniformly random size-m selection.  The default hybrid
uses the already standardized ensemble terms:

    z(A) + (-z(M))*z(R) = z(A) - z(M)z(R).

Use ``--hybrid-scale raw`` to instead evaluate raw A-MR.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import PercentFormatter


TRAIN_SET_SIZES = {
    "CIFAR10": 50_000,
    "CIFAR100": 50_000,
    "SVHN": 73_257,
    "TinyImageNet": 100_000,
}

METHODS = (
    ("exact", r"Exact $\langle g_i,g_t\rangle$", "#000000"),
    ("hybrid", r"Hybrid $A+(-M)R$", "#CC79A7"),
    ("graft", r"GRAFT", "#0072B2"),
    ("graft_plus", r"GRAFT+", "#D55E00"),
    ("random", r"Random (expected)", "#777777"),
)


@dataclass
class TargetScores:
    dataset: str
    model: str
    class_pair: str
    target_idx: int
    candidate_count: int
    scores: dict[str, np.ndarray]


def die(message: str) -> "NoReturn":
    raise SystemExit("ERROR: " + message)


def parse_rhos(value: str) -> list[float]:
    try:
        result = sorted({float(item) for item in value.split(",") if item.strip()})
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "rhos must be comma-separated numbers") from exc
    if not result or result[0] <= 0 or result[-1] > 1:
        raise argparse.ArgumentTypeError("rhos must lie in (0,1]")
    return result


def atomic_csv(path: Path, fieldnames: Sequence[str], rows: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def top_set(values: np.ndarray, count: int) -> set[int]:
    order = np.argsort(-np.asarray(values), kind="stable")
    return set(order[:count].tolist())


def load_records(root: Path, hybrid_scale: str) -> tuple[list[TargetScores], list[str]]:
    records: list[TargetScores] = []
    skipped: list[str] = []
    for manifest_path in sorted(root.glob("*/*/*/manifest.json")):
        combo = manifest_path.parent
        try:
            manifest = json.loads(manifest_path.read_text())
            targets = [int(value) for value in manifest["target_indices"]]
        except (OSError, KeyError, TypeError, ValueError,
                json.JSONDecodeError) as exc:
            skipped.append(f"{combo}: unreadable manifest ({exc})")
            continue
        missing = [target for target in targets
                   if not (combo / f"target_{target}" / "ensemble.npz").is_file()]
        if missing:
            skipped.append(f"{combo}: missing target outputs {missing}")
            continue
        for target in targets:
            path = combo / f"target_{target}" / "ensemble.npz"
            with np.load(path, allow_pickle=False) as data:
                required = (
                    "mean_exact_full_grad_dot",
                    "mean_A_backbone_grad_dot",
                    "mean_M_logit_margin",
                    "mean_R_feature_dot_hiTht",
                    "z_A_after_mean",
                    "z_M_after_mean",
                    "z_R_after_mean",
                    "paper_score_beta0",
                    "paper_score_beta1",
                )
                absent = [name for name in required if name not in data]
                if absent:
                    die(f"{path} is missing {', '.join(absent)}")
                exact = np.asarray(data["mean_exact_full_grad_dot"],
                                   dtype=np.float64)
                if hybrid_scale == "standardized":
                    hybrid = (
                        np.asarray(data["z_A_after_mean"], dtype=np.float64) -
                        np.asarray(data["z_M_after_mean"], dtype=np.float64) *
                        np.asarray(data["z_R_after_mean"], dtype=np.float64)
                    )
                else:
                    hybrid = (
                        np.asarray(data["mean_A_backbone_grad_dot"],
                                   dtype=np.float64) -
                        np.asarray(data["mean_M_logit_margin"],
                                   dtype=np.float64) *
                        np.asarray(data["mean_R_feature_dot_hiTht"],
                                   dtype=np.float64)
                    )
                scores = {
                    "exact": exact,
                    "hybrid": hybrid,
                    "graft": np.asarray(data["paper_score_beta0"],
                                         dtype=np.float64),
                    "graft_plus": np.asarray(data["paper_score_beta1"],
                                              dtype=np.float64),
                }
            lengths = {len(values) for values in scores.values()}
            if len(lengths) != 1:
                die(f"score lengths disagree in {path}: {sorted(lengths)}")
            if not all(np.isfinite(values).all() for values in scores.values()):
                die(f"non-finite score found in {path}")
            records.append(TargetScores(
                dataset=str(manifest["dataset"]),
                model=str(manifest["model"]),
                class_pair=str(manifest["class_pair"]),
                target_idx=target,
                candidate_count=len(exact),
                scores=scores,
            ))
    if not records:
        die("no complete ensemble outputs found")
    return records, skipped


def compute_rows(records: Sequence[TargetScores], rhos: Sequence[float],
                 hybrid_scale: str) -> list[dict]:
    rows: list[dict] = []
    method_keys = [key for key, _, _ in METHODS]
    for record in records:
        if record.dataset not in TRAIN_SET_SIZES:
            die(f"unknown full training-set size for {record.dataset}")
        train_size = TRAIN_SET_SIZES[record.dataset]
        candidate_count = record.candidate_count
        for rho in rhos:
            count = int(round(rho * train_size))
            valid = 1 <= count <= candidate_count
            selected = ({key: top_set(values, count)
                         for key, values in record.scores.items()}
                        if valid else {})
            for left_index, left in enumerate(method_keys):
                for right in method_keys[left_index:]:
                    if not valid:
                        overlap = float("nan")
                        intersection = float("nan")
                        jaccard = float("nan")
                        reason = (f"m={count} exceeds candidate pool "
                                  f"N={candidate_count}" if count > candidate_count
                                  else f"m={count} is below one")
                    elif left == right:
                        overlap = 1.0
                        intersection = float(count)
                        jaccard = 1.0
                        reason = ""
                    elif "random" in (left, right):
                        # For a uniformly random size-m set, expected overlap
                        # with any fixed size-m set is m^2/N.
                        overlap = count / candidate_count
                        intersection = count * overlap
                        jaccard = intersection / (2 * count - intersection)
                        reason = ""
                    else:
                        intersection = len(selected[left] & selected[right])
                        overlap = intersection / count
                        jaccard = intersection / (2 * count - intersection)
                        reason = ""
                    rows.append({
                        "dataset": record.dataset,
                        "model": record.model,
                        "class_pair": record.class_pair,
                        "target_idx": record.target_idx,
                        "poison_rho": rho,
                        "train_set_size": train_size,
                        "candidate_count": candidate_count,
                        "m": count,
                        "candidate_fraction": count / candidate_count,
                        "budget_valid": valid,
                        "invalid_reason": reason,
                        "hybrid_scale": hybrid_scale,
                        "method_left": left,
                        "method_right": right,
                        "intersection": intersection,
                        "overlap_fraction": overlap,
                        "jaccard": jaccard,
                    })
    return rows


def summarize_rows(rows: Sequence[dict]) -> list[dict]:
    keys = sorted({(
        str(row["dataset"]), float(row["poison_rho"]),
        str(row["method_left"]), str(row["method_right"]),
    ) for row in rows})
    result = []
    for dataset, rho, left, right in keys:
        group = [row for row in rows
                 if row["dataset"] == dataset and row["poison_rho"] == rho and
                 row["method_left"] == left and row["method_right"] == right]
        valid = [row for row in group if row["budget_valid"]]
        base = group[0]
        values = np.asarray([float(row["overlap_fraction"]) for row in valid])
        jaccard = np.asarray([float(row["jaccard"]) for row in valid])
        result.append({
            "dataset": dataset,
            "poison_rho": rho,
            "train_set_size": base["train_set_size"],
            "candidate_count": base["candidate_count"],
            "m": base["m"],
            "candidate_fraction": base["candidate_fraction"],
            "hybrid_scale": base["hybrid_scale"],
            "method_left": left,
            "method_right": right,
            "target_pools": len(valid),
            "budget_valid": bool(valid),
            "mean_overlap_fraction": (float(np.mean(values)) if len(values)
                                      else float("nan")),
            "median_overlap_fraction": (float(np.median(values)) if len(values)
                                        else float("nan")),
            "q25_overlap_fraction": (float(np.percentile(values, 25)) if len(values)
                                     else float("nan")),
            "q75_overlap_fraction": (float(np.percentile(values, 75)) if len(values)
                                     else float("nan")),
            "mean_jaccard": (float(np.mean(jaccard)) if len(jaccard)
                              else float("nan")),
        })
    return result


def configure_style() -> None:
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 9,
        "axes.titlesize": 11,
        "figure.titlesize": 15,
        "pdf.fonttype": 42,
    })


def save_figure(fig: plt.Figure, output: Path, stem: str,
                formats: Sequence[str], dpi: int) -> None:
    for extension in formats:
        path = output / f"{stem}.{extension}"
        kwargs = {"bbox_inches": "tight"}
        if extension == "png":
            kwargs["dpi"] = dpi
        fig.savefig(path, **kwargs)
        print(f"wrote {path}", flush=True)
    plt.close(fig)


def matrix_for(summary: Sequence[dict], dataset: str,
               rho: float) -> tuple[np.ndarray, int]:
    keys = [key for key, _, _ in METHODS]
    matrix = np.eye(len(keys), dtype=np.float64)
    matching = [row for row in summary
                if row["dataset"] == dataset and row["poison_rho"] == rho and
                row["budget_valid"]]
    if not matching:
        return np.full_like(matrix, np.nan), 0
    for row in matching:
        left = keys.index(str(row["method_left"]))
        right = keys.index(str(row["method_right"]))
        value = float(row["mean_overlap_fraction"])
        matrix[left, right] = value
        matrix[right, left] = value
    return matrix, int(matching[0]["m"])


def plot_heatmaps(summary: Sequence[dict], output: Path,
                  formats: Sequence[str], dpi: int, hybrid_scale: str) -> None:
    configure_style()
    method_labels = [label for _, label, _ in METHODS]
    datasets = sorted({str(row["dataset"]) for row in summary})
    for dataset in datasets:
        rhos = sorted({float(row["poison_rho"]) for row in summary
                       if row["dataset"] == dataset and row["budget_valid"]})
        if not rhos:
            continue
        columns = min(3, len(rhos))
        rows_count = math.ceil(len(rhos) / columns)
        fig, axes = plt.subplots(rows_count, columns,
                                 figsize=(5.0 * columns, 4.5 * rows_count),
                                 constrained_layout=True, squeeze=False)
        image = None
        for ax, rho in zip(axes.flat, rhos):
            matrix, count = matrix_for(summary, dataset, rho)
            image = ax.imshow(matrix, vmin=0, vmax=1, cmap="Blues")
            ax.set_xticks(np.arange(len(METHODS)))
            ax.set_xticklabels(method_labels, rotation=35, ha="right")
            ax.set_yticks(np.arange(len(METHODS)))
            ax.set_yticklabels(method_labels)
            for row in range(len(METHODS)):
                for column in range(len(METHODS)):
                    value = matrix[row, column]
                    color = "white" if value > .58 else "#222222"
                    ax.text(column, row, f"{value:.1%}", ha="center",
                            va="center", color=color, fontsize=9,
                            fontweight="bold")
            ax.set_title(rf"$\rho={rho:g}$, $m={count}$")
        for ax in axes.flat[len(rhos):]:
            ax.axis("off")
        if image is not None:
            colorbar = fig.colorbar(image, ax=axes, shrink=.85)
            colorbar.ax.yaxis.set_major_formatter(PercentFormatter(1.0))
            colorbar.set_label(r"Mean mutual overlap $|S_a\cap S_b|/m$")
        hybrid_text = (r"$\widetilde A-\widetilde M\widetilde R$"
                       if hybrid_scale == "standardized" else r"$A-MR$")
        fig.suptitle(
            f"{dataset}: pairwise overlap of selected candidates\n"
            f"Hybrid score is {hybrid_text}; random cells are expectations")
        safe_dataset = dataset.replace("/", "_").replace(" ", "_")
        save_figure(fig, output,
                    f"mutual_selection_overlap_{safe_dataset}", formats, dpi)


def plot_exact_overlap_curves(summary: Sequence[dict], output: Path,
                              formats: Sequence[str], dpi: int,
                              hybrid_scale: str) -> None:
    """Plot each method's selected-set overlap with exact <g_i,g_t>."""
    configure_style()
    datasets = [name for name in TRAIN_SET_SIZES
                if any(row["dataset"] == name and row["budget_valid"]
                       for row in summary)]
    fig, axes = plt.subplots(
        1, len(datasets), figsize=(5.4 * len(datasets), 5.3),
        constrained_layout=True, sharey=True, squeeze=False,
    )
    axes = axes[0]
    method_lookup = {key: (label, color) for key, label, color in METHODS}
    curve_order = ("exact", "hybrid", "graft", "graft_plus", "random")

    for ax, dataset in zip(axes, datasets):
        rhos = sorted({float(row["poison_rho"]) for row in summary
                       if row["dataset"] == dataset and row["budget_valid"]})
        for method in curve_order:
            means, lower, upper = [], [], []
            for rho in rhos:
                row = next(
                    item for item in summary
                    if item["dataset"] == dataset and
                    item["poison_rho"] == rho and
                    item["method_left"] == "exact" and
                    item["method_right"] == method and
                    item["budget_valid"]
                )
                means.append(float(row["mean_overlap_fraction"]))
                lower.append(float(row["q25_overlap_fraction"]))
                upper.append(float(row["q75_overlap_fraction"]))
            label_text, color = method_lookup[method]
            linestyle = "--" if method in ("exact", "random") else "-"
            ax.plot(rhos, means, marker="o", markersize=5, linewidth=2.2,
                    linestyle=linestyle, color=color, label=label_text)
            if method not in ("exact", "random"):
                ax.fill_between(rhos, lower, upper, color=color,
                                alpha=.14, linewidth=0)

        tick_labels = []
        for rho in rhos:
            matching = next(
                row for row in summary
                if row["dataset"] == dataset and
                row["poison_rho"] == rho and row["budget_valid"]
            )
            tick_labels.append(f"{rho:g}\n$m$={matching['m']}")
        ax.set_xscale("log")
        ax.set_xticks(rhos)
        ax.set_xticklabels(tick_labels)
        ax.set_ylim(0, 1.03)
        ax.yaxis.set_major_formatter(PercentFormatter(1.0))
        ax.grid(axis="y", color="#dddddd", linewidth=.8)
        ax.set_xlabel(r"Poison ratio $\rho=m/n$ and absolute budget")
        title = {"CIFAR10": "CIFAR-10", "CIFAR100": "CIFAR-100",
                 "TinyImageNet": "Tiny ImageNet"}.get(dataset, dataset)
        ax.set_title(title)

    axes[0].set_ylabel(
        r"Overlap with exact $\langle g_i,g_t\rangle$ selected set")
    axes[-1].legend(frameon=False, loc="best")
    hybrid_text = (r"$\widetilde A-\widetilde M\widetilde R$"
                   if hybrid_scale == "standardized" else r"$A-MR$")
    fig.suptitle(
        "How much do the selected samples agree with exact gradient alignment?\n"
        f"Hybrid is {hybrid_text}; lines are target means and bands are target IQRs")
    save_figure(fig, output, "selection_overlap_with_exact_curves",
                formats, dpi)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path,
                        default=Path("./gradient_alignment_outputs"))
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--rhos", type=parse_rhos,
                        default=parse_rhos("0.001,0.002,0.005,0.01,0.02,0.04"))
    parser.add_argument("--hybrid-scale", choices=("standardized", "raw"),
                        default="standardized")
    parser.add_argument("--formats", default="pdf,png")
    parser.add_argument("--dpi", type=int, default=300)
    args = parser.parse_args(argv)
    args.input_root = args.input_root.expanduser().resolve()
    args.output_dir = ((args.input_root / "figures" / "mutual_selection")
                       if args.output_dir is None
                       else args.output_dir.expanduser().resolve())
    args.formats = [item.strip().lower() for item in args.formats.split(",")
                    if item.strip()]
    if not args.input_root.is_dir():
        parser.error(f"input root does not exist: {args.input_root}")
    if not args.formats:
        parser.error("--formats cannot be empty")
    return args


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    records, skipped = load_records(args.input_root, args.hybrid_scale)
    if skipped:
        print("skipping incomplete combinations:", flush=True)
        for item in skipped:
            print(f"  {item}", flush=True)
    print(f"loaded {len(records)} target candidate pools", flush=True)
    rows = compute_rows(records, args.rhos, args.hybrid_scale)
    summary = summarize_rows(rows)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    atomic_csv(args.output_dir / "mutual_overlap_by_target.csv",
               list(rows[0]), rows)
    atomic_csv(args.output_dir / "mutual_overlap_summary.csv",
               list(summary[0]), summary)
    plot_exact_overlap_curves(summary, args.output_dir, args.formats, args.dpi,
                              args.hybrid_scale)
    print(f"analysis complete: {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
