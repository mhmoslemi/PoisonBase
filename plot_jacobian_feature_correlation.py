#!/usr/bin/env python3
"""Plot pure Jacobian similarity versus feature similarity as raw dots.

The two compared quantities are

    T_i = tr(J_h(x_i) J_h(x_t)^T) = <J_h(x_i), J_h(x_t)>_F
    R_i = h(x_i)^T h(x_t)

Neither score contains W, r_i, r_t, a loss, or a loss-gradient contraction.
Every candidate-target pair is drawn as one dot using the raw ensemble-mean
values; there is no percentile conversion, binning, or density aggregation.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import pearsonr, rankdata, spearmanr


@dataclass
class Record:
    dataset: str
    model: str
    class_pair: str
    target_idx: int
    estimator: str
    probes: int
    feature_dot: np.ndarray
    jacobian_trace: np.ndarray

    @property
    def key(self) -> tuple[str, str, str]:
        return self.dataset, self.model, self.class_pair


def die(message: str) -> "NoReturn":
    raise SystemExit("ERROR: " + message)


def atomic_csv(path: Path, fieldnames: Sequence[str], rows: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def correlations(left: np.ndarray, right: np.ndarray) -> tuple[float, float]:
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    keep = np.isfinite(left) & np.isfinite(right)
    left, right = left[keep], right[keep]
    if len(left) < 3 or np.ptp(left) == 0 or np.ptp(right) == 0:
        return float("nan"), float("nan")
    return (float(spearmanr(left, right).statistic),
            float(pearsonr(left, right).statistic))


def percentile(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    if len(values) == 1:
        return np.array([50.0])
    return 100.0 * (rankdata(values, method="average") - 1) / (len(values) - 1)


def label(key: tuple[str, str, str]) -> str:
    dataset, model, class_pair = key
    dataset = {"CIFAR10": "CIFAR-10", "CIFAR100": "CIFAR-100",
               "TinyImageNet": "Tiny ImageNet"}.get(dataset, dataset)
    return f"{dataset} · {model} · {class_pair}"


def load_records(root: Path) -> tuple[list[Record], list[str]]:
    records: list[Record] = []
    skipped: list[str] = []
    for manifest_path in sorted(root.glob("*/*/*/manifest.json")):
        combo = manifest_path.parent
        try:
            manifest = json.loads(manifest_path.read_text())
            mode = str(manifest["representation_ntk_mode"])
            targets = [int(value) for value in manifest["target_indices"]]
        except (OSError, KeyError, TypeError, ValueError,
                json.JSONDecodeError) as exc:
            skipped.append(f"{combo}: unreadable manifest ({exc})")
            continue
        if mode not in ("trace-hutchinson", "trace-exact"):
            skipped.append(f"{combo}: mode is {mode}, not a pure-trace mode")
            continue
        missing = [target for target in targets
                   if not (combo / f"target_{target}" / "ensemble.npz").is_file()]
        if missing:
            skipped.append(f"{combo}: missing ensemble targets {missing}")
            continue
        for target in targets:
            path = combo / f"target_{target}" / "ensemble.npz"
            with np.load(path, allow_pickle=False) as data:
                required = ("mean_R_feature_dot_hiTht",
                            "mean_representation_ntk_trace_JiJt")
                absent = [name for name in required if name not in data]
                if absent:
                    die(f"{path} is missing {', '.join(absent)}")
                feature_dot = np.asarray(
                    data["mean_R_feature_dot_hiTht"], dtype=np.float64)
                jacobian_trace = np.asarray(
                    data["mean_representation_ntk_trace_JiJt"], dtype=np.float64)
            if feature_dot.shape != jacobian_trace.shape:
                die(f"shape mismatch in {path}: {feature_dot.shape} versus "
                    f"{jacobian_trace.shape}")
            records.append(Record(
                dataset=str(manifest["dataset"]),
                model=str(manifest["model"]),
                class_pair=str(manifest["class_pair"]),
                target_idx=target,
                estimator=mode,
                probes=int(manifest.get("ntk_trace_probes", 0)),
                feature_dot=feature_dot,
                jacobian_trace=jacobian_trace,
            ))
    if not records:
        details = "\n".join(f"  {item}" for item in skipped)
        die("no completed pure Jacobian-trace outputs found" +
            (f":\n{details}" if details else ""))
    return records, skipped


def configure_style() -> None:
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "axes.titlesize": 12,
        "axes.labelsize": 11,
        "figure.titlesize": 15,
        "axes.spines.top": False,
        "axes.spines.right": False,
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


def plot_raw_scatter(records: Sequence[Record], output: Path,
                     formats: Sequence[str], dpi: int) -> None:
    """Draw every candidate-target pair as one raw (hiTht, JiJt) dot."""
    configure_style()
    x = np.concatenate([record.feature_dot for record in records]).astype(
        np.float64, copy=False)
    y = np.concatenate([record.jacobian_trace for record in records]).astype(
        np.float64, copy=False)
    keep = np.isfinite(x) & np.isfinite(y)
    x, y = x[keep], y[keep]
    if len(x) < 3:
        die("fewer than three finite sample pairs are available")
    pooled_spearman = float(spearmanr(x, y).statistic)
    pooled_pearson = float(pearsonr(x, y).statistic)

    fig, ax = plt.subplots(figsize=(7.4, 6.1), constrained_layout=True)
    ax.scatter(x, y, s=4, alpha=0.16, color="#0072B2",
               edgecolors="none", rasterized=True)
    ax.set_xlabel(r"Feature similarity $h_i^\top h_t$")
    ax.set_ylabel(r"Pure Jacobian similarity $\mathrm{tr}(J_iJ_t^\top)$")
    ax.set_title(r"$h_i^\top h_t$ versus $\mathrm{tr}(J_iJ_t^\top)$"
                 "\nEvery candidate-target sample is one raw dot")
    ax.text(
        0.03, 0.97,
        f"points = {len(x):,}\n"
        f"pooled Spearman = {pooled_spearman:.4f}\n"
        f"pooled Pearson = {pooled_pearson:.4f}",
        transform=ax.transAxes, ha="left", va="top",
        bbox={"facecolor": "white", "alpha": 0.9,
              "edgecolor": "#cccccc", "boxstyle": "round,pad=.35"},
    )
    ax.grid(color="#e2e2e2", linewidth=.7)
    save_figure(fig, output, "hiht_vs_jijt_all_samples", formats, dpi)


def plot_rank_density(records: Sequence[Record], output: Path,
                      formats: Sequence[str], dpi: int) -> None:
    configure_style()
    x = np.concatenate([percentile(record.feature_dot) for record in records])
    y = np.concatenate([percentile(record.jacobian_trace) for record in records])
    target_rhos = np.asarray([
        correlations(record.feature_dot, record.jacobian_trace)[0]
        for record in records
    ])
    fig, ax = plt.subplots(figsize=(6.8, 5.7), constrained_layout=True)
    density = ax.hexbin(x, y, gridsize=60, extent=(0, 100, 0, 100),
                        mincnt=1, bins="log", cmap="viridis")
    ax.plot([0, 100], [0, 100], color="white", linewidth=2.5)
    ax.plot([0, 100], [0, 100], color="#333333", linestyle="--", linewidth=1)
    median = float(np.nanmedian(target_rhos))
    low, high = np.nanpercentile(target_rhos, [25, 75])
    ax.text(0.04, 0.96,
            f"Target-wise Spearman\nmedian {median:.3f} "
            f"[IQR {low:.3f}, {high:.3f}]",
            transform=ax.transAxes, ha="left", va="top",
            bbox={"facecolor": "white", "alpha": .92,
                  "edgecolor": "#cccccc", "boxstyle": "round,pad=.35"})
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.set_xlabel("Rank percentile of $h_i^\\top h_t$")
    ax.set_ylabel("Rank percentile of $\\mathrm{tr}(J_iJ_t^\\top)$")
    ax.set_title("Does feature similarity track pure Jacobian similarity?\n"
                 "Each target pool is rank-normalized before pooling")
    colorbar = fig.colorbar(density, ax=ax)
    colorbar.set_label("Candidate density (log count)")
    save_figure(fig, output, "01_JiJt_trace_vs_hiTht_rank_density",
                formats, dpi)


def plot_by_setting(records: Sequence[Record], output: Path,
                    formats: Sequence[str], dpi: int) -> None:
    configure_style()
    groups: dict[tuple[str, str, str], list[Record]] = {}
    for record in records:
        groups.setdefault(record.key, []).append(record)
    keys = sorted(groups)
    height = max(4.5, 0.62 * len(keys) + 1.8)
    fig, ax = plt.subplots(figsize=(9.5, height), constrained_layout=True)
    rng = np.random.default_rng(42)
    for row, key in enumerate(keys):
        values = np.asarray([
            correlations(record.feature_dot, record.jacobian_trace)[0]
            for record in groups[key]
        ])
        jitter = rng.uniform(-.14, .14, size=len(values))
        ax.scatter(values, row + jitter, s=28, alpha=.68,
                   color="#0072B2", edgecolor="none")
        median = float(np.nanmedian(values))
        ax.plot([median, median], [row - .24, row + .24],
                color="#D55E00", linewidth=3)
        ax.text(min(1.02, median + .025), row, f"{median:.3f}", va="center")
    ax.set_yticks(np.arange(len(keys)))
    ax.set_yticklabels([label(key) for key in keys])
    ax.invert_yaxis()
    ax.set_xlim(-1.05, 1.10)
    ax.axvline(0, color="#777777", linestyle="--", linewidth=1)
    ax.grid(axis="x", color="#dddddd", linewidth=.8)
    ax.set_xlabel("Spearman correlation: "
                  "$h_i^\\top h_t$ versus $\\mathrm{tr}(J_iJ_t^\\top)$")
    ax.set_title("Pure Jacobian/feature similarity correlation by setting\n"
                 "Blue points are targets; orange bars are target medians")
    save_figure(fig, output, "02_JiJt_trace_vs_hiTht_by_setting",
                formats, dpi)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path,
                        default=Path("./jacobian_feature_trace_outputs"))
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--formats", default="pdf,png")
    parser.add_argument("--dpi", type=int, default=300)
    args = parser.parse_args(argv)
    args.input_root = args.input_root.expanduser().resolve()
    args.output_dir = ((args.input_root / "figures") if args.output_dir is None
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
    records, skipped = load_records(args.input_root)
    if skipped:
        print("skipped incomplete/non-trace combinations:", flush=True)
        for item in skipped:
            print(f"  {item}", flush=True)
    print(f"loaded {len(records)} target pools", flush=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    target_rows = []
    for record in records:
        spearman, pearson = correlations(record.feature_dot,
                                         record.jacobian_trace)
        target_rows.append({
            "dataset": record.dataset,
            "model": record.model,
            "class_pair": record.class_pair,
            "target_idx": record.target_idx,
            "candidate_count": len(record.feature_dot),
            "trace_estimator": record.estimator,
            "hutchinson_probes": record.probes,
            "spearman_JiJt_trace_vs_hiTht": spearman,
            "pearson_JiJt_trace_vs_hiTht": pearson,
        })
    atomic_csv(args.output_dir / "JiJt_trace_vs_hiTht_by_target.csv",
               list(target_rows[0]), target_rows)

    summary_rows = []
    for key in sorted({record.key for record in records}):
        group = [record for record in records if record.key == key]
        spearman_values = np.asarray([
            correlations(record.feature_dot, record.jacobian_trace)[0]
            for record in group
        ])
        pearson_values = np.asarray([
            correlations(record.feature_dot, record.jacobian_trace)[1]
            for record in group
        ])
        summary_rows.append({
            "dataset": key[0], "model": key[1], "class_pair": key[2],
            "targets": len(group),
            "candidate_count": len(group[0].feature_dot),
            "trace_estimator": group[0].estimator,
            "hutchinson_probes": group[0].probes,
            "median_spearman": float(np.nanmedian(spearman_values)),
            "q25_spearman": float(np.nanpercentile(spearman_values, 25)),
            "q75_spearman": float(np.nanpercentile(spearman_values, 75)),
            "median_pearson": float(np.nanmedian(pearson_values)),
            "q25_pearson": float(np.nanpercentile(pearson_values, 25)),
            "q75_pearson": float(np.nanpercentile(pearson_values, 75)),
        })
    atomic_csv(args.output_dir / "JiJt_trace_vs_hiTht_by_setting.csv",
               list(summary_rows[0]), summary_rows)

    plot_raw_scatter(records, args.output_dir, args.formats, args.dpi)

    all_spearman = np.asarray([
        correlations(record.feature_dot, record.jacobian_trace)[0]
        for record in records
    ])
    summary = [
        "Pure Jacobian similarity versus feature similarity",
        "===================================================",
        "T_i = tr(J_h(x_i) J_h(x_t)^T)",
        "R_i = h(x_i)^T h(x_t)",
        "No W, r_i, r_t, loss, or loss-gradient contraction is used.",
        "",
        f"Complete settings: {len(summary_rows)}",
        f"Target pools: {len(records)}",
        f"Median target-wise Spearman: {np.nanmedian(all_spearman):.6f}",
        f"Target-wise Spearman IQR: "
        f"[{np.nanpercentile(all_spearman, 25):.6f}, "
        f"{np.nanpercentile(all_spearman, 75):.6f}]",
    ]
    temporary = args.output_dir / "JiJt_trace_vs_hiTht_summary.txt.tmp"
    temporary.write_text("\n".join(summary) + "\n")
    os.replace(temporary, args.output_dir / "JiJt_trace_vs_hiTht_summary.txt")
    print(f"analysis complete: {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
