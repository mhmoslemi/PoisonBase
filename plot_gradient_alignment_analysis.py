#!/usr/bin/env python3
"""Create reviewer-facing figures from ``gradient_alignment_audit.py`` outputs.

The script is CPU-only.  It reads the per-target ``ensemble.npz`` files after
the expensive GPU collection and produces six complementary views:

1. rank-density plots: does GRAFT order candidates like exact <g_i, g_t>?
2. poison-budget recovery: at each paper budget rho=m/n, do the candidates
   selected by the heuristic overlap those preferred by exact alignment?
3. exact-gradient decomposition: how much absolute alignment magnitude comes
   from the backbone, classifier weights, and classifier bias?
4. a correlation heatmap: which individual terms and combined scores track the
   exact objective across datasets, architectures, class pairs, and targets?
5. budget-local rank correlation: within the candidates selected at each
   paper poison budget, do the heuristic and exact alignment order them alike?
6. a multiplicative ablation comparing GRAFT+ with both possible ranking
   directions of the proposed raw product (-M)RA.

All correlations are calculated *within a target candidate pool* first.  This
avoids a misleading pooled correlation caused by different gradient scales for
different targets or architectures.  Figures use candidate rank percentiles for
the same reason.  Raw Pearson and rank-based Spearman values are both written to
CSV for further analysis.

Normal use after jobs 0..7 finish::

    /home/mmoslem3/ENV/bin/python plot_gradient_alignment_analysis.py

Outputs default to ``gradient_alignment_outputs/figures`` as both PDF and PNG.
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
from scipy.stats import pearsonr, rankdata, spearmanr


REQUIRED_FIELDS = (
    "candidate_idx",
    "paper_score_beta0",
    "paper_score_beta1",
    "mean_M_logit_margin",
    "mean_R_feature_dot_hiTht",
    "mean_A_backbone_grad_dot",
    "mean_exact_full_grad_dot",
    "mean_classifier_weight_grad_dot",
    "mean_classifier_bias_grad_dot",
    "mean_classifier_grad_dot",
)

SELECTORS = (
    ("beta0", "GRAFT ($\\beta=0$)", "#0072B2"),
    ("beta1", "GRAFT+ ($\\beta=1$)", "#D55E00"),
    ("A", "Backbone alignment $A$", "#009E73"),
)

PRODUCT_SELECTORS = (
    ("product_neg_MRA_high", "Product $(-M)RA$ · high first", "#CC79A7"),
    ("product_neg_MRA_low", "Product $(-M)RA$ · low first", "#6F4E7C"),
)

BUDGET_SELECTORS = SELECTORS + PRODUCT_SELECTORS

CORRELATION_METRICS = (
    ("neg_M", "$-M$"),
    ("R", "$R=h_i^\\top h_t$"),
    ("A", "$A$"),
    ("head", "Classifier"),
    ("beta0", "GRAFT\n$\\beta=0$"),
    ("beta1", "GRAFT+\n$\\beta=1$"),
)

# Standard training-set sizes used by the datasets collected by the audit.
# Poison budgets in the paper use rho=m/n with this full training-set n, not
# the size of the adversarial-class candidate pool.
TRAIN_SET_SIZES = {
    "CIFAR10": 50_000,
    "CIFAR100": 50_000,
    "SVHN": 73_257,
    "TinyImageNet": 100_000,
}


@dataclass
class TargetRecord:
    dataset: str
    model: str
    class_pair: str
    target_idx: int
    path: Path
    candidate_idx: np.ndarray
    values: dict[str, np.ndarray]

    @property
    def combo_key(self) -> tuple[str, str, str]:
        return self.dataset, self.model, self.class_pair

    @property
    def combo_label(self) -> str:
        return combo_label(*self.combo_key)


def die(message: str) -> "NoReturn":
    raise SystemExit("ERROR: " + message)


def combo_label(dataset: str, model: str, class_pair: str) -> str:
    dataset_name = {
        "CIFAR10": "CIFAR-10",
        "CIFAR100": "CIFAR-100",
        "TinyImageNet": "Tiny ImageNet",
        "SVHN": "SVHN",
    }.get(dataset, dataset)
    # Repository convention is adversarial-target, while the paper reports
    # target -> adversarial.
    pieces = class_pair.split("-", 1)
    direction = f"{pieces[1]}$\\rightarrow${pieces[0]}" if len(pieces) == 2 else class_pair
    return f"{dataset_name} · {model} · {direction}"


def atomic_csv(path: Path, fieldnames: Sequence[str], rows: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def finite_pair(left: np.ndarray, right: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    keep = np.isfinite(left) & np.isfinite(right)
    return left[keep], right[keep]


def safe_spearman(left: np.ndarray, right: np.ndarray) -> float:
    left, right = finite_pair(left, right)
    if len(left) < 3 or np.ptp(left) == 0 or np.ptp(right) == 0:
        return float("nan")
    return float(spearmanr(left, right).statistic)


def safe_pearson(left: np.ndarray, right: np.ndarray) -> float:
    left, right = finite_pair(left, right)
    if len(left) < 3 or np.ptp(left) == 0 or np.ptp(right) == 0:
        return float("nan")
    return float(pearsonr(left, right).statistic)


def high_rank_percentile(values: np.ndarray) -> np.ndarray:
    """0 is the lowest value and 100 is the highest, with average tie ranks."""
    values = np.asarray(values, dtype=np.float64)
    if not np.isfinite(values).all():
        die("rank percentile received non-finite values")
    if len(values) == 1:
        return np.array([50.0])
    return 100.0 * (rankdata(values, method="average") - 1.0) / (len(values) - 1.0)


def top_indices(values: np.ndarray, count: int) -> np.ndarray:
    values = np.asarray(values)
    count = min(max(1, int(count)), len(values))
    if count == len(values):
        return np.arange(len(values))
    return np.argpartition(values, -count)[-count:]


def configure_style() -> None:
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "axes.titlesize": 12,
        "axes.labelsize": 11,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "figure.titlesize": 15,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


def discover_combinations(input_root: Path) -> tuple[list[Path], list[str]]:
    complete: list[Path] = []
    incomplete: list[str] = []
    for combo_dir in sorted(path for path in input_root.glob("*/*/*") if path.is_dir()):
        target_dirs = list(combo_dir.glob("target_*"))
        manifest_path = combo_dir / "manifest.json"
        if not target_dirs and not manifest_path.exists():
            continue
        if not manifest_path.is_file():
            incomplete.append(f"{combo_dir}: collection still has no manifest")
            continue
        try:
            manifest = json.loads(manifest_path.read_text())
            targets = [int(value) for value in manifest["target_indices"]]
        except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            incomplete.append(f"{combo_dir}: unreadable manifest ({exc})")
            continue
        missing = [value for value in targets
                   if not (combo_dir / f"target_{value}" / "ensemble.npz").is_file()]
        if missing:
            incomplete.append(f"{combo_dir}: missing ensemble outputs for {missing}")
            continue
        complete.append(combo_dir)
    return complete, incomplete


def load_records(combo_dirs: Sequence[Path]) -> list[TargetRecord]:
    records: list[TargetRecord] = []
    for combo_dir in combo_dirs:
        dataset, model, class_pair = combo_dir.parts[-3:]
        manifest = json.loads((combo_dir / "manifest.json").read_text())
        for target_idx in manifest["target_indices"]:
            path = combo_dir / f"target_{int(target_idx)}" / "ensemble.npz"
            with np.load(path, allow_pickle=False) as data:
                missing = [field for field in REQUIRED_FIELDS if field not in data]
                if missing:
                    die(f"{path} is missing fields: {', '.join(missing)}")
                arrays = {field: np.asarray(data[field], dtype=np.float64)
                          for field in REQUIRED_FIELDS if field != "candidate_idx"}
                candidate_idx = np.asarray(data["candidate_idx"], dtype=np.int64)
            lengths = {len(value) for value in arrays.values()} | {len(candidate_idx)}
            if len(lengths) != 1:
                die(f"array lengths disagree in {path}: {sorted(lengths)}")
            exact = arrays["mean_exact_full_grad_dot"]
            decomposed = (arrays["mean_A_backbone_grad_dot"] +
                          arrays["mean_classifier_grad_dot"])
            error = float(np.max(np.abs(exact - decomposed)))
            tolerance = 2e-5 * max(1.0, float(np.max(np.abs(exact))))
            if not math.isfinite(error) or error > tolerance:
                die(f"decomposition check failed in {path}: max error {error:g}, "
                    f"tolerance {tolerance:g}")
            values = {
                "neg_M": -arrays["mean_M_logit_margin"],
                "R": arrays["mean_R_feature_dot_hiTht"],
                "A": arrays["mean_A_backbone_grad_dot"],
                "head": arrays["mean_classifier_grad_dot"],
                "head_weight": arrays["mean_classifier_weight_grad_dot"],
                "head_bias": arrays["mean_classifier_bias_grad_dot"],
                "exact": exact,
                "beta0": arrays["paper_score_beta0"],
                "beta1": arrays["paper_score_beta1"],
            }
            product = values["neg_M"] * values["R"] * values["A"]
            values["product_neg_MRA_high"] = product
            # Every internal selector is represented as "larger ranks first".
            # Negating the product therefore evaluates selecting its low tail.
            values["product_neg_MRA_low"] = -product
            records.append(TargetRecord(
                dataset=dataset, model=model, class_pair=class_pair,
                target_idx=int(target_idx), path=path,
                candidate_idx=candidate_idx, values=values,
            ))
    if not records:
        die("no complete target ensemble files were found")
    return records


def target_correlation_rows(records: Sequence[TargetRecord]) -> list[dict]:
    rows: list[dict] = []
    for record in records:
        row = {
            "dataset": record.dataset,
            "model": record.model,
            "class_pair": record.class_pair,
            "target_idx": record.target_idx,
            "candidate_count": len(record.candidate_idx),
        }
        for key, _ in CORRELATION_METRICS:
            row[f"spearman_{key}_vs_exact"] = safe_spearman(
                record.values[key], record.values["exact"])
            row[f"pearson_{key}_vs_exact"] = safe_pearson(
                record.values[key], record.values["exact"])
        rows.append(row)
    return rows


def grouped_records(records: Sequence[TargetRecord]) -> dict[tuple[str, str, str], list[TargetRecord]]:
    groups: dict[tuple[str, str, str], list[TargetRecord]] = {}
    for record in records:
        groups.setdefault(record.combo_key, []).append(record)
    return dict(sorted(groups.items()))


def save_figure(fig: plt.Figure, output_dir: Path, stem: str,
                formats: Sequence[str], dpi: int) -> None:
    for extension in formats:
        path = output_dir / f"{stem}.{extension}"
        kwargs = {"bbox_inches": "tight"}
        if extension.lower() == "png":
            kwargs["dpi"] = dpi
        fig.savefig(path, **kwargs)
        print(f"wrote {path}", flush=True)
    plt.close(fig)


def plot_rank_agreement(records: Sequence[TargetRecord], output_dir: Path,
                        formats: Sequence[str], dpi: int) -> None:
    configure_style()
    fig, axes = plt.subplots(1, 3, figsize=(14.2, 4.5), constrained_layout=True,
                             sharex=True, sharey=True)
    first_hexbin = None
    for ax, (key, label, _) in zip(axes, SELECTORS):
        x_parts, y_parts, target_rhos = [], [], []
        for record in records:
            x_parts.append(high_rank_percentile(record.values[key]))
            y_parts.append(high_rank_percentile(record.values["exact"]))
            target_rhos.append(safe_spearman(
                record.values[key], record.values["exact"]))
        x = np.concatenate(x_parts)
        y = np.concatenate(y_parts)
        density = ax.hexbin(x, y, gridsize=55, extent=(0, 100, 0, 100),
                            mincnt=1, bins="log", cmap="viridis")
        if first_hexbin is None:
            first_hexbin = density
        ax.plot([0, 100], [0, 100], color="white", linewidth=2.5, alpha=0.9)
        ax.plot([0, 100], [0, 100], color="#333333", linestyle="--",
                linewidth=1.0, alpha=0.9)
        rho = np.asarray(target_rhos, dtype=float)
        median = float(np.nanmedian(rho))
        lower, upper = np.nanpercentile(rho, [25, 75])
        ax.text(0.04, 0.96,
                f"Target-wise Spearman\nmedian {median:.2f}  "
                f"[IQR {lower:.2f}, {upper:.2f}]",
                transform=ax.transAxes, ha="left", va="top",
                bbox={"facecolor": "white", "alpha": 0.9,
                      "edgecolor": "#cccccc", "boxstyle": "round,pad=0.35"})
        ax.set_title(label)
        ax.set_xlim(0, 100)
        ax.set_ylim(0, 100)
        ax.grid(False)
    for ax in axes:
        ax.set_xlabel("Candidate rank by selector (%)\nhigher = selected earlier")
    axes[0].set_ylabel("Candidate rank by exact $\\langle g_i,g_t\\rangle$ (%)\nhigher = better aligned")
    fig.suptitle(
        "Does the heuristic reproduce the exact gradient-alignment ordering?\n"
        "Density is pooled only after converting every target pool to rank percentiles",
        y=1.08,
    )
    if first_hexbin is not None:
        colorbar = fig.colorbar(first_hexbin, ax=axes, shrink=0.88, pad=0.015)
        colorbar.set_label("Candidate density (log count)")
    save_figure(fig, output_dir, "01_exact_rank_agreement", formats, dpi)


def compute_budget_rows(records: Sequence[TargetRecord],
                        poison_rhos: Sequence[float]) -> list[dict]:
    """Evaluate selectors at the paper's full-training-set poison budgets.

    ``overlap_at_m`` is the fraction of the exact top-m set recovered by the
    selector.  Because both sets have size m, it is also precision and recall.

    The two local correlations answer different budget-dependent questions:
    ``spearman_exact_top_m`` measures ordering within the exact top-m set, while
    ``spearman_selected_union`` measures ordering on the union of the exact and
    selector top-m sets.  Global all-candidate Spearman is intentionally kept in
    ``correlations_by_target.csv`` because it does not depend on rho.
    """
    rows: list[dict] = []
    for record in records:
        exact = record.values["exact"]
        candidate_count = len(exact)
        if record.dataset not in TRAIN_SET_SIZES:
            die(f"no training-set size registered for dataset {record.dataset}")
        train_size = TRAIN_SET_SIZES[record.dataset]
        for poison_rho in poison_rhos:
            count = int(round(poison_rho * train_size))
            valid = 1 <= count <= candidate_count
            exact_positions = (top_indices(exact, count) if valid
                               else np.asarray([], dtype=np.int64))
            exact_top = set(exact_positions.tolist())
            for key, label, _ in BUDGET_SELECTORS:
                if valid:
                    selected_positions = top_indices(record.values[key], count)
                    selected = set(selected_positions.tolist())
                    overlap = len(exact_top & selected)
                    union_positions = np.asarray(
                        sorted(exact_top | selected), dtype=np.int64)
                    exact_top_spearman = safe_spearman(
                        record.values[key][exact_positions], exact[exact_positions])
                    union_spearman = safe_spearman(
                        record.values[key][union_positions], exact[union_positions])
                    overlap_at_m = overlap / count
                    jaccard = overlap / (2 * count - overlap)
                    random_overlap = count / candidate_count
                    invalid_reason = ""
                else:
                    overlap = 0
                    overlap_at_m = float("nan")
                    jaccard = float("nan")
                    exact_top_spearman = float("nan")
                    union_spearman = float("nan")
                    random_overlap = float("nan")
                    invalid_reason = (
                        f"requested m={count} exceeds candidate pool "
                        f"N={candidate_count}" if count > candidate_count
                        else f"requested m={count} is below one")
                rows.append({
                    "dataset": record.dataset,
                    "model": record.model,
                    "class_pair": record.class_pair,
                    "target_idx": record.target_idx,
                    "selector": key,
                    "selector_label": label.replace("$", ""),
                    "poison_rho": poison_rho,
                    "train_set_size": train_size,
                    "candidate_count": candidate_count,
                    "requested_m": count,
                    "candidate_fraction": count / candidate_count,
                    "budget_valid": valid,
                    "invalid_reason": invalid_reason,
                    "overlap": overlap,
                    "overlap_at_m": overlap_at_m,
                    "jaccard_at_m": jaccard,
                    "spearman_exact_top_m": exact_top_spearman,
                    "spearman_selected_union": union_spearman,
                    "random_expected_overlap_at_m": random_overlap,
                })
    return rows


def budget_summary_rows(rows: Sequence[dict]) -> list[dict]:
    """Aggregate target-level budget measurements by experimental setting."""
    keys = sorted({(
        str(row["dataset"]), str(row["model"]), str(row["class_pair"]),
        float(row["poison_rho"]), str(row["selector"]),
    ) for row in rows})
    summaries: list[dict] = []
    for dataset, model, class_pair, poison_rho, selector in keys:
        group = [row for row in rows
                 if row["dataset"] == dataset and row["model"] == model and
                 row["class_pair"] == class_pair and
                 row["poison_rho"] == poison_rho and
                 row["selector"] == selector]
        valid = bool(group[0]["budget_valid"])
        summary = {
            "dataset": dataset,
            "model": model,
            "class_pair": class_pair,
            "poison_rho": poison_rho,
            "train_set_size": group[0]["train_set_size"],
            "candidate_count": group[0]["candidate_count"],
            "requested_m": group[0]["requested_m"],
            "candidate_fraction": group[0]["candidate_fraction"],
            "selector": selector,
            "selector_label": group[0]["selector_label"],
            "targets": len(group),
            "budget_valid": valid,
            "invalid_reason": group[0]["invalid_reason"],
        }
        for metric in ("overlap_at_m", "jaccard_at_m",
                       "spearman_exact_top_m", "spearman_selected_union"):
            values = np.asarray([float(row[metric]) for row in group],
                                dtype=np.float64)
            summary[f"mean_{metric}"] = (float(np.nanmean(values)) if valid
                                          else float("nan"))
            summary[f"median_{metric}"] = (float(np.nanmedian(values)) if valid
                                            else float("nan"))
            summary[f"q25_{metric}"] = (float(np.nanpercentile(values, 25)) if valid
                                         else float("nan"))
            summary[f"q75_{metric}"] = (float(np.nanpercentile(values, 75)) if valid
                                         else float("nan"))
        summaries.append(summary)
    return summaries


def mean_ci(values: Sequence[float]) -> tuple[float, float, float]:
    array = np.asarray(values, dtype=np.float64)
    mean = float(np.mean(array))
    if len(array) < 2:
        return mean, mean, mean
    half = 1.96 * float(np.std(array, ddof=1)) / math.sqrt(len(array))
    return mean, max(0.0, mean - half), min(1.0, mean + half)


def plot_budget_recovery(budget_rows: Sequence[dict], output_dir: Path,
                         formats: Sequence[str], dpi: int) -> None:
    configure_style()
    datasets = [name for name in TRAIN_SET_SIZES
                if any(row["dataset"] == name for row in budget_rows)]
    fig, axes = plt.subplots(1, len(datasets),
                             figsize=(5.2 * len(datasets), 5.2),
                             constrained_layout=True, sharey=True)
    axes = np.atleast_1d(axes)
    for ax, dataset in zip(axes, datasets):
        dataset_rows = [row for row in budget_rows
                        if row["dataset"] == dataset and row["budget_valid"]]
        rhos = sorted({float(row["poison_rho"]) for row in dataset_rows})
        for key, label, color in SELECTORS:
            means, lowers, uppers = [], [], []
            for poison_rho in rhos:
                values = [float(row["overlap_at_m"]) for row in dataset_rows
                          if row["selector"] == key and
                          row["poison_rho"] == poison_rho]
                mean, lower, upper = mean_ci(values)
                means.append(mean)
                lowers.append(lower)
                uppers.append(upper)
            ax.plot(rhos, means, marker="o", markersize=4.5, linewidth=2,
                    color=color, label=label)
            ax.fill_between(rhos, lowers, uppers, color=color,
                            alpha=0.15, linewidth=0)
        random_means = [np.mean([
            float(row["random_expected_overlap_at_m"])
            for row in dataset_rows
            if row["selector"] == "beta0" and row["poison_rho"] == poison_rho
        ]) for poison_rho in rhos]
        ax.plot(rhos, random_means, color="#555555", linestyle="--",
                linewidth=1.4, label="Random ranking (expected)")
        ax.set_xscale("log")
        ax.set_xticks(rhos)
        tick_labels = []
        for poison_rho in rhos:
            matching = next(row for row in dataset_rows
                            if row["poison_rho"] == poison_rho)
            tick_labels.append(f"{poison_rho:g}\n$m$={matching['requested_m']}")
        ax.set_xticklabels(tick_labels)
        ax.set_ylim(0, 1.02)
        ax.yaxis.set_major_formatter(PercentFormatter(1.0))
        ax.set_xlabel("Poison ratio $\\rho=m/n$ and absolute budget")
        ax.set_title(dataset.replace("CIFAR10", "CIFAR-10").replace(
            "CIFAR100", "CIFAR-100"))
        ax.grid(axis="y", color="#dddddd", linewidth=0.8)
    axes[0].set_ylabel("Exact top-$m$ candidates recovered")
    axes[-1].legend(frameon=False, loc="best")
    fig.suptitle(
        "Candidate-selection agreement at the actual poisoning budgets\n"
        "Lines are target means; bands are 95% target-level CIs")
    save_figure(fig, output_dir, "02_exact_budget_recovery", formats, dpi)


def plot_budget_local_correlation(budget_rows: Sequence[dict], output_dir: Path,
                                  formats: Sequence[str], dpi: int) -> None:
    """Plot rank agreement specifically on candidates relevant at each budget."""
    configure_style()
    datasets = [name for name in TRAIN_SET_SIZES
                if any(row["dataset"] == name for row in budget_rows)]
    fig, axes = plt.subplots(1, len(datasets),
                             figsize=(5.2 * len(datasets), 5.2),
                             constrained_layout=True, sharey=True)
    axes = np.atleast_1d(axes)
    for ax, dataset in zip(axes, datasets):
        dataset_rows = [row for row in budget_rows
                        if row["dataset"] == dataset and row["budget_valid"]]
        rhos = sorted({float(row["poison_rho"]) for row in dataset_rows})
        for key, label, color in SELECTORS:
            medians, lowers, uppers = [], [], []
            for poison_rho in rhos:
                values = np.asarray([
                    float(row["spearman_selected_union"])
                    for row in dataset_rows
                    if row["selector"] == key and
                    row["poison_rho"] == poison_rho
                ], dtype=np.float64)
                medians.append(float(np.nanmedian(values)))
                lowers.append(float(np.nanpercentile(values, 25)))
                uppers.append(float(np.nanpercentile(values, 75)))
            ax.plot(rhos, medians, marker="o", markersize=4.5, linewidth=2,
                    color=color, label=label)
            ax.fill_between(rhos, lowers, uppers, color=color,
                            alpha=0.15, linewidth=0)
        ax.axhline(0, color="#777777", linewidth=1, linestyle="--")
        ax.set_xscale("log")
        ax.set_xticks(rhos)
        tick_labels = []
        for poison_rho in rhos:
            matching = next(row for row in dataset_rows
                            if row["poison_rho"] == poison_rho)
            tick_labels.append(f"{poison_rho:g}\n$m$={matching['requested_m']}")
        ax.set_xticklabels(tick_labels)
        ax.set_ylim(-1.02, 1.02)
        ax.set_xlabel("Poison ratio $\\rho=m/n$ and absolute budget")
        ax.set_title(dataset.replace("CIFAR10", "CIFAR-10").replace(
            "CIFAR100", "CIFAR-100"))
        ax.grid(axis="y", color="#dddddd", linewidth=0.8)
    axes[0].set_ylabel("Spearman on union of selector/exact top-$m$ sets")
    axes[-1].legend(frameon=False, loc="best")
    fig.suptitle(
        "Budget-local rank agreement with exact gradient alignment\n"
        "Lines are target medians; bands are target IQRs")
    save_figure(fig, output_dir, "05_budget_local_rank_correlation", formats, dpi)


def plot_product_budget_ablation(budget_rows: Sequence[dict], output_dir: Path,
                                 formats: Sequence[str], dpi: int) -> None:
    """Compare additive GRAFT+ with both directions of the proposed product."""
    configure_style()
    rows = [row for row in budget_rows
            if row["dataset"] == "CIFAR10" and row["budget_valid"]]
    if not rows:
        return
    rhos = sorted({float(row["poison_rho"]) for row in rows})
    comparison = (SELECTORS[1], SELECTORS[2]) + PRODUCT_SELECTORS
    fig, axes = plt.subplots(1, 2, figsize=(12.8, 5.2),
                             constrained_layout=True)
    for key, label, color in comparison:
        overlap_means = []
        local_medians = []
        for poison_rho in rhos:
            selected = [row for row in rows
                        if row["selector"] == key and
                        row["poison_rho"] == poison_rho]
            overlap_means.append(float(np.mean([
                float(row["overlap_at_m"]) for row in selected])))
            local_medians.append(float(np.nanmedian([
                float(row["spearman_selected_union"]) for row in selected])))
        axes[0].plot(rhos, overlap_means, marker="o", linewidth=2,
                     color=color, label=label)
        axes[1].plot(rhos, local_medians, marker="o", linewidth=2,
                     color=color, label=label)
    tick_labels = []
    for poison_rho in rhos:
        matching = next(row for row in rows if row["poison_rho"] == poison_rho)
        tick_labels.append(f"{poison_rho:g}\n$m$={matching['requested_m']}")
    for ax in axes:
        ax.set_xscale("log")
        ax.set_xticks(rhos)
        ax.set_xticklabels(tick_labels)
        ax.set_xlabel("Poison ratio $\\rho=m/n$ and absolute budget")
        ax.grid(axis="y", color="#dddddd", linewidth=0.8)
    axes[0].set_ylim(0, 1.02)
    axes[0].yaxis.set_major_formatter(PercentFormatter(1.0))
    axes[0].set_ylabel("Exact top-$m$ candidates recovered")
    axes[0].set_title("Selected-set overlap")
    axes[1].set_ylim(-1.02, 1.02)
    axes[1].axhline(0, color="#777777", linewidth=1, linestyle="--")
    axes[1].set_ylabel("Spearman on union of selector/exact top-$m$ sets")
    axes[1].set_title("Budget-local rank agreement")
    axes[1].legend(frameon=False, loc="best")
    fig.suptitle(
        "CIFAR-10 multiplicative-score ablation across all architectures, "
        "class pairs, and targets")
    save_figure(fig, output_dir, "06_multiplicative_budget_ablation",
                formats, dpi)


def decomposition_for_record(record: TargetRecord) -> tuple[float, float, float]:
    magnitudes = np.array([
        np.sum(np.abs(record.values["A"])),
        np.sum(np.abs(record.values["head_weight"])),
        np.sum(np.abs(record.values["head_bias"])),
    ], dtype=np.float64)
    total = float(magnitudes.sum())
    if total == 0:
        return float("nan"), float("nan"), float("nan")
    return tuple((magnitudes / total).tolist())


def plot_decomposition(records: Sequence[TargetRecord], output_dir: Path,
                       formats: Sequence[str], dpi: int) -> None:
    configure_style()
    groups = grouped_records(records)
    keys = list(groups)
    labels = [combo_label(*key) for key in keys]
    shares = []
    correlation_values = []
    for key in keys:
        per_target = np.asarray([decomposition_for_record(record)
                                 for record in groups[key]], dtype=np.float64)
        shares.append(np.nanmean(per_target, axis=0))
        correlation_values.append([
            safe_spearman(record.values["A"], record.values["exact"])
            for record in groups[key]
        ])
    shares_array = np.asarray(shares)

    height = max(5.2, 0.58 * len(keys) + 2.0)
    fig, (left, right) = plt.subplots(1, 2, figsize=(14.0, height),
                                     constrained_layout=True)
    y = np.arange(len(keys))
    colors = ("#009E73", "#CC79A7", "#F0E442")
    component_labels = ("Backbone $A$", "Classifier weights", "Classifier bias")
    running = np.zeros(len(keys))
    for column, (color, label) in enumerate(zip(colors, component_labels)):
        values = shares_array[:, column]
        left.barh(y, values, left=running, color=color, edgecolor="white",
                  linewidth=0.6, label=label)
        running += values
    left.set_yticks(y)
    left.set_yticklabels(labels)
    left.invert_yaxis()
    left.set_xlim(0, 1)
    left.xaxis.set_major_formatter(PercentFormatter(1.0))
    left.set_xlabel("Share of total absolute alignment magnitude")
    left.set_title("What part of the exact gradient dot does $A$ omit?\n"
                   "Shares use $|A|$, classifier-weight, and classifier-bias magnitudes")
    left.legend(frameon=False, loc="lower right")
    left.grid(axis="x", color="#e2e2e2", linewidth=0.8)

    rng = np.random.default_rng(42)
    for row, values in enumerate(correlation_values):
        values_array = np.asarray(values, dtype=np.float64)
        jitter = rng.uniform(-0.13, 0.13, size=len(values_array))
        right.scatter(values_array, row + jitter, s=24, alpha=0.65,
                      color="#0072B2", edgecolor="none")
        median = float(np.nanmedian(values_array))
        right.plot([median, median], [row - 0.23, row + 0.23],
                   color="#D55E00", linewidth=3)
    right.set_yticks(y)
    right.set_yticklabels(labels)
    right.invert_yaxis()
    right.set_xlim(-1.02, 1.02)
    right.set_xlabel("Spearman correlation: backbone $A$ vs exact full alignment")
    right.set_title("Does omitting the classifier change the ordering?\n"
                    "Blue points are targets; orange bars are target medians")
    right.axvline(0, color="#777777", linewidth=1, linestyle="--")
    right.grid(axis="x", color="#e2e2e2", linewidth=0.8)
    fig.suptitle("Exact gradient-alignment decomposition", y=1.02)
    save_figure(fig, output_dir, "03_gradient_decomposition", formats, dpi)


def plot_correlation_heatmap(records: Sequence[TargetRecord], output_dir: Path,
                             formats: Sequence[str], dpi: int) -> None:
    configure_style()
    groups = grouped_records(records)
    keys = list(groups)
    labels = [combo_label(*key) for key in keys]
    matrix = np.zeros((len(keys), len(CORRELATION_METRICS)), dtype=np.float64)
    for row, key in enumerate(keys):
        for column, (metric, _) in enumerate(CORRELATION_METRICS):
            values = [safe_spearman(record.values[metric], record.values["exact"])
                      for record in groups[key]]
            matrix[row, column] = np.nanmedian(values)

    width = max(8.4, 1.15 * len(CORRELATION_METRICS) + 3.0)
    height = max(4.2, 0.62 * len(keys) + 2.0)
    fig, ax = plt.subplots(figsize=(width, height), constrained_layout=True)
    image = ax.imshow(matrix, vmin=-1, vmax=1, cmap="coolwarm", aspect="auto")
    ax.set_xticks(np.arange(len(CORRELATION_METRICS)))
    ax.set_xticklabels([label for _, label in CORRELATION_METRICS])
    ax.set_yticks(np.arange(len(keys)))
    ax.set_yticklabels(labels)
    ax.tick_params(axis="x", pad=8)
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            value = matrix[row, column]
            color = "white" if abs(value) > 0.55 else "#222222"
            ax.text(column, row, f"{value:.2f}", ha="center", va="center",
                    color=color, fontsize=9, fontweight="bold")
    colorbar = fig.colorbar(image, ax=ax, shrink=0.88)
    colorbar.set_label("Median target-wise Spearman correlation with exact $\\langle g_i,g_t\\rangle$")
    ax.set_title(
        "Which terms actually track the exact gradient-alignment objective?\n"
        "Every cell is computed within target first, then summarized by the median")
    save_figure(fig, output_dir, "04_component_correlation_heatmap", formats, dpi)


def combination_summary_rows(records: Sequence[TargetRecord]) -> list[dict]:
    rows: list[dict] = []
    for key, group in grouped_records(records).items():
        correlations = {
            metric: np.asarray([
                safe_spearman(record.values[metric], record.values["exact"])
                for record in group
            ], dtype=np.float64)
            for metric, _ in CORRELATION_METRICS
        }
        shares = np.asarray([decomposition_for_record(record) for record in group])
        row = {
            "dataset": key[0],
            "model": key[1],
            "class_pair": key[2],
            "targets": len(group),
            "candidates_per_target": len(group[0].candidate_idx),
            "median_backbone_abs_share": float(np.nanmedian(shares[:, 0])),
            "median_classifier_weight_abs_share": float(np.nanmedian(shares[:, 1])),
            "median_classifier_bias_abs_share": float(np.nanmedian(shares[:, 2])),
        }
        for metric in correlations:
            row[f"median_spearman_{metric}_vs_exact"] = float(
                np.nanmedian(correlations[metric]))
            row[f"q25_spearman_{metric}_vs_exact"] = float(
                np.nanpercentile(correlations[metric], 25))
            row[f"q75_spearman_{metric}_vs_exact"] = float(
                np.nanpercentile(correlations[metric], 75))
        rows.append(row)
    return rows


def write_text_summary(path: Path, records: Sequence[TargetRecord],
                       budget_rows: Sequence[dict]) -> None:
    beta0 = np.asarray([safe_spearman(record.values["beta0"], record.values["exact"])
                        for record in records])
    beta1 = np.asarray([safe_spearman(record.values["beta1"], record.values["exact"])
                        for record in records])
    a_corr = np.asarray([safe_spearman(record.values["A"], record.values["exact"])
                         for record in records])
    shares = np.asarray([decomposition_for_record(record) for record in records])
    lines = [
        "Gradient-alignment analysis summary",
        "===================================",
        f"Complete combinations analyzed: {len(grouped_records(records))}",
        f"Target candidate pools analyzed: {len(records)}",
        "",
        "Target-wise rank correlation with exact <g_i,g_t>",
        f"  GRAFT beta=0 median Spearman: {np.nanmedian(beta0):.4f}",
        f"  GRAFT+ beta=1 median Spearman: {np.nanmedian(beta1):.4f}",
        f"  Backbone A alone median Spearman: {np.nanmedian(a_corr):.4f}",
        f"  beta=1 improves over beta=0 on {int(np.sum(beta1 > beta0))}/{len(records)} targets",
        "",
        "CIFAR-10 agreement at the paper poison budgets",
    ]
    cifar_rows = [row for row in budget_rows
                  if row["dataset"] == "CIFAR10" and row["budget_valid"]]
    for poison_rho in sorted({float(row["poison_rho"]) for row in cifar_rows}):
        matching = [row for row in cifar_rows
                    if row["poison_rho"] == poison_rho]
        count = int(matching[0]["requested_m"])
        overlap_values: dict[str, float] = {}
        local_values: dict[str, float] = {}
        for key, _, _ in SELECTORS:
            selected = [row for row in matching if row["selector"] == key]
            overlap_values[key] = float(np.mean([
                float(row["overlap_at_m"]) for row in selected]))
            local_values[key] = float(np.nanmedian([
                float(row["spearman_selected_union"]) for row in selected]))
        lines.append(
            f"  rho={poison_rho:g}, m={count}: exact-set recovery "
            f"beta0={overlap_values['beta0']:.2%}, "
            f"beta1={overlap_values['beta1']:.2%}, A={overlap_values['A']:.2%}; "
            f"selected-union Spearman beta0={local_values['beta0']:.3f}, "
            f"beta1={local_values['beta1']:.3f}, A={local_values['A']:.3f}")
    lines.extend(["", "CIFAR-10 multiplicative score at the same budgets"])
    for poison_rho in sorted({float(row["poison_rho"]) for row in cifar_rows}):
        matching = [row for row in cifar_rows
                    if row["poison_rho"] == poison_rho]
        count = int(matching[0]["requested_m"])
        values = {}
        local = {}
        for key, _, _ in PRODUCT_SELECTORS:
            selected = [row for row in matching if row["selector"] == key]
            values[key] = float(np.mean([
                float(row["overlap_at_m"]) for row in selected]))
            local[key] = float(np.nanmedian([
                float(row["spearman_selected_union"]) for row in selected]))
        lines.append(
            f"  rho={poison_rho:g}, m={count}: high-first recovery="
            f"{values['product_neg_MRA_high']:.2%}, local Spearman="
            f"{local['product_neg_MRA_high']:.3f}; low-first recovery="
            f"{values['product_neg_MRA_low']:.2%}, local Spearman="
            f"{local['product_neg_MRA_low']:.3f}")
    lines.extend([
        "",
        "Median absolute-magnitude decomposition",
        f"  Backbone A: {np.nanmedian(shares[:, 0]):.2%}",
        f"  Classifier weights: {np.nanmedian(shares[:, 1]):.2%}",
        f"  Classifier bias: {np.nanmedian(shares[:, 2]):.2%}",
        "",
        "Interpretation",
        "  Figure 1 tests global rank agreement without mixing target scales.",
        "  Figure 2 measures exact-set recovery at rho=m/n from the paper.",
        "  Figure 3 quantifies the contribution omitted by backbone-only A.",
        "  Figure 4 separates the evidence supplied by M, R, A, and their sums.",
        "  Figure 5 measures budget-local Spearman on the union of selected sets.",
        "  Figure 6 compares additive GRAFT+ with both ranking directions of (-M)RA.",
        "  These are descriptive correlations; they do not establish an approximation error bound.",
    ])
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text("\n".join(lines) + "\n")
    os.replace(temporary, path)


def parse_rhos(value: str) -> list[float]:
    try:
        poison_rhos = sorted({float(item) for item in value.split(",") if item.strip()})
    except ValueError as exc:
        raise argparse.ArgumentTypeError("poison rhos must be comma-separated numbers") from exc
    if not poison_rhos or poison_rhos[0] <= 0 or poison_rhos[-1] > 1:
        raise argparse.ArgumentTypeError("poison rhos must lie in (0,1]")
    return poison_rhos


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path,
                        default=Path("./gradient_alignment_outputs"))
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--expected-combinations", type=int, default=0,
                        help="optional required number of complete combinations; default 0 plots whatever is ready")
    parser.add_argument("--formats", default="pdf,png",
                        help="comma-separated output formats")
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument(
        "--poison-rhos", type=parse_rhos,
        default=parse_rhos("0.001,0.002,0.005,0.01,0.02,0.04"),
        help="comma-separated full-training-set poison ratios rho=m/n",
    )
    args = parser.parse_args(argv)
    args.input_root = args.input_root.expanduser().resolve()
    args.output_dir = ((args.input_root / "figures") if args.output_dir is None
                       else args.output_dir.expanduser().resolve())
    args.formats = [item.strip().lower() for item in args.formats.split(",")
                    if item.strip()]
    if not args.input_root.is_dir():
        parser.error(f"input root does not exist: {args.input_root}")
    if args.expected_combinations < 0:
        parser.error("--expected-combinations must be nonnegative")
    if args.dpi <= 0:
        parser.error("--dpi must be positive")
    if not args.formats:
        parser.error("--formats cannot be empty")
    return args


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    complete, incomplete = discover_combinations(args.input_root)
    if incomplete:
        print("skipping incomplete combinations:", flush=True)
        for message in incomplete:
            print(f"  {message}", flush=True)
    if args.expected_combinations and len(complete) != args.expected_combinations:
        die(f"found {len(complete)}/{args.expected_combinations} complete combinations; "
            "wait for jobs 0..7 before plotting")
    if args.expected_combinations and len(complete) > args.expected_combinations:
        print(f"warning: found {len(complete)} combinations, more than expected "
              f"{args.expected_combinations}; plotting all", flush=True)
    print(f"loading {len(complete)} complete combinations", flush=True)
    for path in complete:
        print(f"  {path.relative_to(args.input_root)}", flush=True)
    records = load_records(complete)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    correlation_rows = target_correlation_rows(records)
    correlation_fields = list(correlation_rows[0])
    atomic_csv(args.output_dir / "correlations_by_target.csv",
               correlation_fields, correlation_rows)

    budget_rows = compute_budget_rows(records, args.poison_rhos)
    atomic_csv(args.output_dir / "budget_analysis_by_target.csv",
               list(budget_rows[0]), budget_rows)
    budget_summaries = budget_summary_rows(budget_rows)
    atomic_csv(args.output_dir / "budget_summary_by_combination.csv",
               list(budget_summaries[0]), budget_summaries)

    summary_rows = combination_summary_rows(records)
    atomic_csv(args.output_dir / "summary_by_combination.csv",
               list(summary_rows[0]), summary_rows)

    plot_rank_agreement(records, args.output_dir, args.formats, args.dpi)
    plot_budget_recovery(budget_rows, args.output_dir, args.formats, args.dpi)
    plot_decomposition(records, args.output_dir, args.formats, args.dpi)
    plot_correlation_heatmap(records, args.output_dir, args.formats, args.dpi)
    plot_budget_local_correlation(budget_rows, args.output_dir,
                                  args.formats, args.dpi)
    plot_product_budget_ablation(budget_rows, args.output_dir,
                                 args.formats, args.dpi)
    write_text_summary(args.output_dir / "analysis_summary.txt", records,
                       budget_rows)
    print(f"analysis complete: {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
