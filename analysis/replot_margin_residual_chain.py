# #!/usr/bin/env python3
# """Replot the margin/residual diagnostic exclusively from cached NPZ shards."""

# from __future__ import annotations

# import argparse
# import math
# import os
# import sys
# import zlib
# from collections import defaultdict
# from pathlib import Path
# from typing import Sequence

# import matplotlib

# matplotlib.use("Agg")
# import matplotlib.pyplot as plt
# import numpy as np


# DATASET_ORDER = ("CIFAR10", "CIFAR100", "SVHN", "TinyImageNet")
# DATASET_CLASS_FALLBACK = {
#     "CIFAR10": 10,
#     "CIFAR100": 100,
#     "SVHN": 10,
#     "TinyImageNet": 200,
# }


# def die(message: str) -> "NoReturn":
#     raise SystemExit("ERROR: " + message)


# def scalar_text(value: np.ndarray) -> str:
#     item = np.asarray(value).reshape(()).item()
#     return item.decode("utf-8") if isinstance(item, bytes) else str(item)


# def scalar_int(value: np.ndarray) -> int:
#     return int(np.asarray(value).reshape(()).item())


# def broadcast_scalar_or_array(value: np.ndarray, length: int,
#                               field: str, path: Path) -> np.ndarray:
#     array = np.asarray(value, dtype=np.float64)
#     if array.ndim == 0:
#         return np.full(length, float(array), dtype=np.float64)
#     array = array.reshape(-1)
#     if len(array) != length:
#         die(f"{path}: {field} has {len(array)} values, expected {length}")
#     return array


# def average_ranks(values: np.ndarray) -> np.ndarray:
#     values = np.asarray(values, dtype=np.float64)
#     order = np.argsort(values, kind="mergesort")
#     sorted_values = values[order]
#     ranks = np.empty(len(values), dtype=np.float64)
#     start = 0
#     while start < len(values):
#         stop = start + 1
#         while stop < len(values) and sorted_values[stop] == sorted_values[start]:
#             stop += 1
#         ranks[order[start:stop]] = 0.5 * (start + stop - 1) + 1.0
#         start = stop
#     return ranks


# def spearman(left: np.ndarray, right: np.ndarray) -> float:
#     if len(left) != len(right) or len(left) < 2:
#         return float("nan")
#     a = average_ranks(left)
#     b = average_ranks(right)
#     a -= a.mean()
#     b -= b.mean()
#     denominator = float(np.linalg.norm(a) * np.linalg.norm(b))
#     return float(np.dot(a, b) / denominator) if denominator else float("nan")


# def stable_sigmoid(values: np.ndarray) -> np.ndarray:
#     values = np.asarray(values, dtype=np.float64)
#     result = np.empty_like(values)
#     positive = values >= 0
#     result[positive] = 1.0 / (1.0 + np.exp(-values[positive]))
#     exp_values = np.exp(values[~positive])
#     result[~positive] = exp_values / (1.0 + exp_values)
#     return result


# def quantile_trend(x: np.ndarray, y: np.ndarray,
#                    bins: int) -> tuple[np.ndarray, ...]:
#     order = np.argsort(x, kind="mergesort")
#     groups = [group for group in np.array_split(order, min(bins, len(order)))
#               if len(group)]
#     return tuple(
#         np.asarray([np.quantile(values[group], quantile) for group in groups])
#         for values, quantile in ((x, 0.5), (y, 0.5), (y, 0.1), (y, 0.9))
#     )


# def deterministic_sample(length: int, maximum: int, seed: int) -> np.ndarray:
#     if length <= maximum:
#         return np.arange(length)
#     return np.sort(np.random.default_rng(seed).choice(
#         length, size=maximum, replace=False))


# def robust_limits(values: np.ndarray, low: float = 0.005,
#                   high: float = 0.995, padding: float = 0.035) -> tuple[float, float]:
#     left, right = np.quantile(values, [low, high])
#     if not np.isfinite(left) or not np.isfinite(right):
#         die("cannot determine finite plotting limits")
#     if right <= left:
#         delta = max(abs(float(left)) * 0.05, 1e-6)
#         return float(left - delta), float(right + delta)
#     delta = float(right - left) * padding
#     return float(left - delta), float(right + delta)


# def discover_shards(root: Path) -> tuple[list[Path], list[Path]]:
#     if not root.is_dir():
#         die(f"candidate cache directory does not exist: {root}")
#     panel_a = sorted(root.rglob("panel_a_surrogate_*.npz"))
#     panel_b = sorted(root.rglob("panel_b_surrogate_*.npz"))
#     if not panel_a:
#         die(f"no panel_a_surrogate_*.npz files below {root}")
#     if not panel_b:
#         die(f"no panel_b_surrogate_*.npz files below {root}")
#     return panel_a, panel_b


# def empty_dataset_store() -> dict:
#     return {
#         "M": [], "u_margin": [], "margin_lower": [], "margin_upper": [],
#         "u_dot": [], "ratio": [], "num_classes": set(),
#         "panel_a_files": 0, "panel_b_files": 0,
#         "panel_a_excluded_nonpositive": 0,
#         "panel_b_excluded_nonpositive": 0,
#         "panel_b_total": 0,
#     }


# def load_cached_values(root: Path) -> dict[str, dict]:
#     panel_a_paths, panel_b_paths = discover_shards(root)
#     print(f"Panel A cache shards: {len(panel_a_paths)}", flush=True)
#     print(f"Panel B cache shards: {len(panel_b_paths)}", flush=True)
#     stores: dict[str, dict] = defaultdict(empty_dataset_store)

#     for path in panel_a_paths:
#         print(f"READ {path}", flush=True)
#         with np.load(path, allow_pickle=False) as shard:
#             required = ("dataset", "M", "u_i", "margin_lower_bound",
#                         "margin_upper_bound")
#             missing = [name for name in required if name not in shard]
#             if missing:
#                 die(f"{path}: missing cached fields {missing}")
#             dataset = scalar_text(shard["dataset"])
#             margin = np.asarray(shard["M"], dtype=np.float64).reshape(-1)
#             residual_mass = np.asarray(shard["u_i"], dtype=np.float64).reshape(-1)
#             lower = np.asarray(
#                 shard["margin_lower_bound"], dtype=np.float64).reshape(-1)
#             upper = np.asarray(
#                 shard["margin_upper_bound"], dtype=np.float64).reshape(-1)
#             if not (len(margin) == len(residual_mass) == len(lower) == len(upper)):
#                 die(f"{path}: inconsistent Panel A array lengths")
#             if not all(np.all(np.isfinite(values))
#                        for values in (margin, residual_mass, lower, upper)):
#                 die(f"{path}: non-finite Panel A values")
#             store = stores[dataset]
#             store["M"].append(margin)
#             store["u_margin"].append(residual_mass)
#             store["margin_lower"].append(lower)
#             store["margin_upper"].append(upper)
#             store["panel_a_files"] += 1
#             store["panel_a_excluded_nonpositive"] += int(
#                 np.count_nonzero(residual_mass <= 0))
#             if "num_classes" in shard:
#                 store["num_classes"].add(scalar_int(shard["num_classes"]))

#     for path in panel_b_paths:
#         print(f"READ {path}", flush=True)
#         with np.load(path, allow_pickle=False) as shard:
#             required = ("dataset", "u_i", "u_t", "residual_dot",
#                         "normalized_residual_dot")
#             missing = [name for name in required if name not in shard]
#             if missing:
#                 die(f"{path}: missing cached fields {missing}")
#             dataset = scalar_text(shard["dataset"])
#             residual_mass = np.asarray(shard["u_i"], dtype=np.float64).reshape(-1)
#             residual_dot = np.asarray(
#                 shard["residual_dot"], dtype=np.float64).reshape(-1)
#             normalized = np.asarray(
#                 shard["normalized_residual_dot"], dtype=np.float64).reshape(-1)
#             if not (len(residual_mass) == len(residual_dot) == len(normalized)):
#                 die(f"{path}: inconsistent Panel B array lengths")
#             target_mass = broadcast_scalar_or_array(
#                 shard["u_t"], len(residual_mass), "u_t", path)
#             valid = (residual_mass > 0) & (target_mass > 0)
#             store = stores[dataset]
#             store["panel_b_files"] += 1
#             store["panel_b_total"] += len(residual_mass)
#             store["panel_b_excluded_nonpositive"] += int(np.count_nonzero(~valid))
#             if not np.any(valid):
#                 continue
#             if not all(np.all(np.isfinite(values[valid])) for values in
#                        (residual_mass, target_mass, residual_dot, normalized)):
#                 die(f"{path}: non-finite values among valid Panel B observations")
#             ratio_direct = residual_dot[valid] / (
#                 residual_mass[valid] * target_mass[valid])
#             ratio_equivalent = normalized[valid] / residual_mass[valid]
#             if not np.allclose(
#                     ratio_direct, ratio_equivalent, rtol=1e-8, atol=1e-10):
#                 difference = float(np.max(np.abs(ratio_direct - ratio_equivalent)))
#                 die(f"{path}: cached ratio definitions disagree; max={difference:g}")
#             store["u_dot"].append(residual_mass[valid])
#             store["ratio"].append(ratio_direct)
#             if "num_classes" in shard:
#                 store["num_classes"].add(scalar_int(shard["num_classes"]))

#     unexpected = sorted(set(stores) - set(DATASET_ORDER))
#     missing_datasets = sorted(set(DATASET_ORDER) - set(stores))
#     if unexpected:
#         die(f"unexpected datasets in cache: {unexpected}")
#     if missing_datasets:
#         die(f"missing required datasets in cache: {missing_datasets}")

#     loaded: dict[str, dict] = {}
#     for dataset in DATASET_ORDER:
#         store = stores[dataset]
#         if not store["M"] or not store["u_dot"]:
#             die(f"{dataset}: no usable cached observations")
#         classes = store["num_classes"]
#         if len(classes) > 1:
#             die(f"{dataset}: inconsistent cached output dimensions: {sorted(classes)}")
#         num_classes = (next(iter(classes)) if classes
#                        else DATASET_CLASS_FALLBACK[dataset])
#         loaded[dataset] = {
#             "M": np.concatenate(store["M"]),
#             "u_margin": np.concatenate(store["u_margin"]),
#             "margin_lower": np.concatenate(store["margin_lower"]),
#             "margin_upper": np.concatenate(store["margin_upper"]),
#             "u_dot": np.concatenate(store["u_dot"]),
#             "ratio": np.concatenate(store["ratio"]),
#             "num_classes": num_classes,
#             "panel_a_files": store["panel_a_files"],
#             "panel_b_files": store["panel_b_files"],
#             "panel_a_excluded_nonpositive":
#                 store["panel_a_excluded_nonpositive"],
#             "panel_b_excluded_nonpositive":
#                 store["panel_b_excluded_nonpositive"],
#             "panel_b_total": store["panel_b_total"],
#         }
#     return loaded


# def prepare_diagnostics(data: dict, tolerance: float) -> dict:
#     positive_margin = data["u_margin"] > 0
#     margin = data["M"][positive_margin]
#     residual_mass = data["u_margin"][positive_margin]
#     lower = data["margin_lower"][positive_margin]
#     upper = data["margin_upper"][positive_margin]
#     margin_ok = ((residual_mass >= lower - tolerance)
#                  & (residual_mass <= upper + tolerance))
#     ratio = data["ratio"]
#     ratio_ok = (ratio >= 1.0) & (ratio <= 2.0)
#     result = dict(data)
#     result.update({
#         "positive_margin_mask": positive_margin,
#         "spearman_margin": spearman(-margin, residual_mass),
#         "fraction_margin_bounds": float(margin_ok.mean()),
#         "fraction_ratio_bounds": float(ratio_ok.mean()),
#         "ratio_min": float(ratio.min()),
#         "ratio_median": float(np.median(ratio)),
#         "ratio_max": float(ratio.max()),
#     })
#     return result


# def draw_panel_a(axis: plt.Axes, dataset: str, data: dict, bins: int,
#                  max_scatter: int, seed: int, show_legend: bool) -> None:
#     valid = data["positive_margin_mask"]
#     margin = data["M"][valid]
#     residual_mass = data["u_margin"][valid]
#     x_min, x_max = robust_limits(margin)
#     in_view = (margin >= x_min) & (margin <= x_max)
#     view_margin = margin[in_view]
#     view_mass = residual_mass[in_view]
#     if not len(view_margin):
#         die(f"{dataset}: robust Panel A range contains no observations")
#     dataset_seed = seed + zlib.crc32(dataset.encode("utf-8"))
#     sample = deterministic_sample(len(view_margin), max_scatter, dataset_seed)
#     axis.scatter(
#         view_margin[sample], view_mass[sample], s=3.2, color="#777777",
#         alpha=0.055, edgecolors="none", rasterized=True, zorder=1)

#     bin_x, median, low, high = quantile_trend(margin, residual_mass, bins)
#     trend_view = (bin_x >= x_min) & (bin_x <= x_max)
#     axis.fill_between(
#         bin_x[trend_view], low[trend_view], high[trend_view],
#         color="#2878B5", alpha=0.22, linewidth=0, zorder=2,
#         label="10th–90th percentile")
#     axis.plot(
#         bin_x[trend_view], median[trend_view], color="#075B9A",
#         linewidth=2.7, marker="o", markersize=3.8, zorder=4,
#         label="Empirical median")

#     grid = np.linspace(x_min, x_max, 600)
#     theoretical_lower = stable_sigmoid(-grid)
#     theoretical_upper = stable_sigmoid(
#         math.log(data["num_classes"] - 1.0) - grid)
#     axis.plot(
#         grid, theoretical_lower, linestyle="--", color="#178344",
#         linewidth=1.8, zorder=3, label="Lower bound")
#     axis.plot(
#         grid, theoretical_upper, linestyle="-.", color="#C43B32",
#         linewidth=1.8, zorder=3, label="Upper bound")

#     positive_for_limits = view_mass[view_mass > 0]
#     y_low, y_high = np.quantile(positive_for_limits, [0.001, 0.999])
#     curve_low = min(float(theoretical_lower.min()),
#                     float(theoretical_upper.min()))
#     curve_high = max(float(theoretical_lower.max()),
#                      float(theoretical_upper.max()))
#     y_low = max(np.finfo(np.float64).tiny,
#                 min(float(y_low), curve_low) / 1.35)
#     y_high = min(1.05, max(float(y_high), curve_high) * 1.25)
#     if y_high <= y_low:
#         y_high = min(1.05, y_low * 10.0)

#     axis.set_yscale("log")
#     axis.set_xlim(x_min, x_max)
#     axis.set_ylim(y_low, y_high)
#     axis.set_xlabel(r"Logit margin $M_i$")
#     axis.set_ylabel(r"Candidate residual mass $1-p_{i,y_{\mathrm{adv}}}$")
#     axis.set_title("(a) Margin predicts residual magnitude", loc="left")
#     axis.text(
#         0.03, 0.055, "higher margin $\\rightarrow$ smaller residual mass",
#         transform=axis.transAxes, fontsize=9, color="#444444")
#     axis.text(
#         0.97, 0.96,
#         (rf"$\rho_s(-M,u_i)={data['spearman_margin']:.3f}$" + "\n"
#          + f"bounds satisfied: {100.0 * data['fraction_margin_bounds']:.2f}%"),
#         transform=axis.transAxes, ha="right", va="top", fontsize=8.5,
#         bbox={"boxstyle": "round,pad=0.3", "facecolor": "white",
#               "edgecolor": "#BBBBBB", "alpha": 0.86})
#     if show_legend:
#         axis.legend(frameon=False, fontsize=8, loc="upper right",
#                     bbox_to_anchor=(1.0, 0.77))


# def draw_panel_b(axis: plt.Axes, dataset: str, data: dict, bins: int,
#                  max_scatter: int, seed: int, show_legend: bool) -> None:
#     residual_mass = data["u_dot"]
#     ratio = data["ratio"]
#     dataset_seed = seed + zlib.crc32(dataset.encode("utf-8")) + 1
#     sample = deterministic_sample(len(residual_mass), max_scatter, dataset_seed)
#     axis.scatter(
#         residual_mass[sample], ratio[sample], s=3.2, color="#777777",
#         alpha=0.055, edgecolors="none", rasterized=True, zorder=1)

#     bin_x, median, low, high = quantile_trend(residual_mass, ratio, bins)
#     axis.fill_between(
#         bin_x, low, high, color="#A9611B", alpha=0.22,
#         linewidth=0, zorder=2, label="10th–90th percentile")
#     axis.plot(
#         bin_x, median, color="#914C08", linewidth=2.7,
#         marker="o", markersize=3.8, zorder=4, label="Empirical median")
#     axis.axhline(
#         1.0, linestyle="--", color="#178344", linewidth=1.8,
#         zorder=3, label="Lower bound: 1")
#     axis.axhline(
#         2.0, linestyle="-.", color="#C43B32", linewidth=1.8,
#         zorder=3, label="Upper bound: 2")

#     x_low = max(np.finfo(np.float64).tiny, float(residual_mass.min()) / 1.2)
#     x_high = float(residual_mass.max()) * 1.2
#     ratio_min = data["ratio_min"]
#     ratio_max = data["ratio_max"]
#     ratio_span = max(ratio_max - ratio_min, 0.1)
#     y_low = min(0.9, ratio_min - 0.025 * ratio_span)
#     y_high = max(2.1, ratio_max + 0.025 * ratio_span)

#     axis.set_xscale("log")
#     axis.set_xlim(x_low, x_high)
#     axis.set_ylim(y_low, y_high)
#     axis.set_xlabel(r"Candidate residual mass $1-p_{i,y_{\mathrm{adv}}}$")
#     axis.set_ylabel("Normalized residual alignment")
#     axis.set_title(
#         "(b) Residual alignment remains within the theoretical bound",
#         loc="left")
#     axis.text(
#         0.03, 0.055,
#         r"ratio $=\frac{r_i^\top r_t}{(1-p_{i,y_{\rm adv}})(1-p_{t,y_{\rm adv}})}$",
#         transform=axis.transAxes, fontsize=9, color="#444444")
#     axis.text(
#         0.97, 0.96,
#         ("theoretical range: [1, 2]\n"
#          + f"inside range: {100.0 * data['fraction_ratio_bounds']:.2f}%"),
#         transform=axis.transAxes, ha="right", va="top", fontsize=8.5,
#         bbox={"boxstyle": "round,pad=0.3", "facecolor": "white",
#               "edgecolor": "#BBBBBB", "alpha": 0.86})
#     if show_legend:
#         axis.legend(frameon=False, fontsize=8, loc="lower right",
#                     bbox_to_anchor=(1.0, 0.19))


# def style_axis(axis: plt.Axes) -> None:
#     axis.spines["top"].set_visible(False)
#     axis.spines["right"].set_visible(False)
#     axis.grid(True, which="major", alpha=0.16, linewidth=0.65)
#     axis.tick_params(labelsize=9)


# def write_figure(fig: plt.Figure, stem: Path, dpi: int) -> None:
#     stem.parent.mkdir(parents=True, exist_ok=True)
#     fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
#     fig.savefig(stem.with_suffix(".png"), dpi=dpi, bbox_inches="tight")
#     plt.close(fig)


# def make_figures(data_by_dataset: dict[str, dict], output_dir: Path,
#                  bins: int, max_scatter: int, seed: int, dpi: int) -> None:
#     for dataset in DATASET_ORDER:
#         fig, axes = plt.subplots(
#             1, 2, figsize=(11.8, 4.65), constrained_layout=True)
#         draw_panel_a(axes[0], dataset, data_by_dataset[dataset], bins,
#                      max_scatter, seed, True)
#         draw_panel_b(axes[1], dataset, data_by_dataset[dataset], bins,
#                      max_scatter, seed, True)
#         for axis in axes:
#             style_axis(axis)
#         fig.suptitle(dataset, fontsize=14, fontweight="semibold")
#         write_figure(
#             fig, output_dir / f"margin_residual_chain_clear_{dataset}", dpi)

#     fig, axes = plt.subplots(
#         len(DATASET_ORDER), 2, figsize=(12.0, 17.2), constrained_layout=True)
#     for row, dataset in enumerate(DATASET_ORDER):
#         draw_panel_a(axes[row, 0], dataset, data_by_dataset[dataset], bins,
#                      max_scatter, seed, row == 0)
#         draw_panel_b(axes[row, 1], dataset, data_by_dataset[dataset], bins,
#                      max_scatter, seed, row == 0)
#         for column in range(2):
#             style_axis(axes[row, column])
#             axes[row, column].text(
#                 -0.13, 1.04, dataset, transform=axes[row, column].transAxes,
#                 fontsize=11, fontweight="semibold", va="bottom")
#     write_figure(
#         fig, output_dir / "margin_residual_chain_clear_all", dpi)


# def main(cli: argparse.Namespace) -> None:
#     input_root = cli.input.resolve()
#     output_dir = cli.output_dir.resolve()
#     print(f"Cache root: {input_root}", flush=True)
#     loaded = load_cached_values(input_root)
#     diagnostics = {
#         dataset: prepare_diagnostics(loaded[dataset], cli.bound_tolerance)
#         for dataset in DATASET_ORDER
#     }

#     for dataset in DATASET_ORDER:
#         data = diagnostics[dataset]
#         print(f"\n{dataset}", flush=True)
#         print(f"  Panel A observations: {len(data['M'])}", flush=True)
#         print(f"  Panel B observations: {data['panel_b_total']}", flush=True)
#         print("  Panel A excluded from log-y (u_i <= 0): "
#               f"{data['panel_a_excluded_nonpositive']}", flush=True)
#         print("  Panel B excluded from ratio/log-x (u_i <= 0 or u_t <= 0): "
#               f"{data['panel_b_excluded_nonpositive']}", flush=True)
#         print(f"  ratio min: {data['ratio_min']:.17g}", flush=True)
#         print(f"  ratio median: {data['ratio_median']:.17g}", flush=True)
#         print(f"  ratio max: {data['ratio_max']:.17g}", flush=True)
#         print("  fraction ratio in [1,2]: "
#               f"{data['fraction_ratio_bounds']:.17g}", flush=True)

#     make_figures(
#         diagnostics, output_dir, cli.bins, cli.max_scatter,
#         cli.seed, cli.dpi)


# def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
#     parser = argparse.ArgumentParser(
#         description="Replot cached margin/residual-chain values; no model code.")
#     parser.add_argument("--input", type=Path, required=True,
#                         help="margin_residual_chain_candidates directory")
#     parser.add_argument("--output-dir", type=Path, required=True)
#     parser.add_argument("--bins", type=int, default=35)
#     parser.add_argument("--max-scatter", type=int, default=60000)
#     parser.add_argument("--seed", type=int, default=42)
#     parser.add_argument("--dpi", type=int, default=350)
#     parser.add_argument("--bound-tolerance", type=float, default=1e-8)
#     args = parser.parse_args(argv)
#     if not 30 <= args.bins <= 40:
#         parser.error("--bins must be between 30 and 40")
#     if args.max_scatter <= 0 or args.dpi <= 0:
#         parser.error("--max-scatter and --dpi must be positive")
#     if args.bound_tolerance < 0:
#         parser.error("--bound-tolerance must be nonnegative")
#     return args


# if __name__ == "__main__":
#     main(parse_args())

#!/usr/bin/env python3
"""Make a simple paper-facing margin/residual figure from cached NPZ shards.

No model loading, inference, training, or poison optimization is performed.

Panel (a): margin percentile -> median candidate residual mass, one curve per dataset.
Panel (b): per target/surrogate Spearman correlation between -M_i and r_i^T r_t.
"""

from __future__ import annotations

import argparse
import csv
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


DATASET_ORDER = ("CIFAR10", "CIFAR100", "SVHN", "TinyImageNet")
DISPLAY_NAME = {
    "CIFAR10": "CIFAR-10",
    "CIFAR100": "CIFAR-100",
    "SVHN": "SVHN",
    "TinyImageNet": "Tiny ImageNet",
}


def die(message: str) -> "NoReturn":
    raise SystemExit("ERROR: " + message)


def scalar_text(value: np.ndarray) -> str:
    item = np.asarray(value).reshape(()).item()
    return item.decode("utf-8") if isinstance(item, bytes) else str(item)


def normalize_text_item(value: Any) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if isinstance(value, np.generic):
        value = value.item()
    return str(value)


def average_ranks(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    order = np.argsort(values, kind="mergesort")
    sorted_values = values[order]
    ranks = np.empty(len(values), dtype=np.float64)
    start = 0
    while start < len(values):
        stop = start + 1
        while stop < len(values) and sorted_values[stop] == sorted_values[start]:
            stop += 1
        ranks[order[start:stop]] = 0.5 * (start + stop - 1) + 1.0
        start = stop
    return ranks


def spearman(left: np.ndarray, right: np.ndarray) -> float:
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    valid = np.isfinite(left) & np.isfinite(right)
    left, right = left[valid], right[valid]
    if len(left) < 3:
        return float("nan")
    a = average_ranks(left)
    b = average_ranks(right)
    a -= a.mean()
    b -= b.mean()
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    return float(np.dot(a, b) / denom) if denom else float("nan")


def discover_shards(root: Path) -> tuple[list[Path], list[Path]]:
    if not root.is_dir():
        die(f"cache directory does not exist: {root}")
    panel_a = sorted(root.rglob("panel_a_surrogate_*.npz"))
    panel_b = sorted(root.rglob("panel_b_surrogate_*.npz"))
    if not panel_a:
        die(f"no panel_a_surrogate_*.npz files below {root}")
    if not panel_b:
        die(f"no panel_b_surrogate_*.npz files below {root}")
    return panel_a, panel_b


def surrogate_key(shard: np.lib.npyio.NpzFile, path: Path) -> str:
    for key in ("surrogate_id", "surrogate_idx", "surrogate_index", "model_id"):
        if key in shard and np.asarray(shard[key]).size == 1:
            return normalize_text_item(np.asarray(shard[key]).reshape(()).item())
    match = re.search(r"panel_[ab]_surrogate_(.+)$", path.stem)
    return match.group(1) if match else path.stem


def optional_id_array(
    shard: np.lib.npyio.NpzFile,
    keys: tuple[str, ...],
    length: int,
) -> np.ndarray | None:
    for key in keys:
        if key not in shard:
            continue
        arr = np.asarray(shard[key])
        if arr.ndim == 0 or arr.size == 1:
            item = normalize_text_item(arr.reshape(()).item())
            return np.full(length, item, dtype=object)
        arr = arr.reshape(-1)
        if len(arr) != length:
            continue
        return np.asarray([normalize_text_item(v) for v in arr], dtype=object)
    return None


def optional_candidate_ids(
    shard: np.lib.npyio.NpzFile, length: int
) -> np.ndarray | None:
    for key in ("candidate_id", "candidate_idx", "candidate_index", "candidate_ids"):
        if key not in shard:
            continue
        arr = np.asarray(shard[key]).reshape(-1)
        if len(arr) != length:
            continue
        return np.asarray([normalize_text_item(v) for v in arr], dtype=object)
    return None


def percentile_curve(margin: np.ndarray, residual: np.ndarray, bins: int) -> tuple[np.ndarray, np.ndarray]:
    valid = np.isfinite(margin) & np.isfinite(residual) & (residual > 0)
    margin = np.asarray(margin[valid], dtype=np.float64)
    residual = np.asarray(residual[valid], dtype=np.float64)
    if len(margin) < bins:
        die(f"only {len(margin)} positive observations; cannot make {bins} percentile bins")

    order = np.argsort(margin, kind="mergesort")
    groups = [g for g in np.array_split(order, bins) if len(g)]
    x = 100.0 * (np.arange(len(groups), dtype=np.float64) + 0.5) / len(groups)
    y = np.asarray([np.median(residual[g]) for g in groups], dtype=np.float64)
    return x, y


def load_panel_a(
    paths: list[Path],
) -> tuple[dict[str, dict[str, np.ndarray]], dict[tuple[str, str], list[dict[str, Any]]]]:
    """Load Panel-A values and build an index usable to recover M in Panel B."""
    aggregate: dict[str, dict[str, list[np.ndarray]]] = defaultdict(lambda: {"M": [], "u": []})
    by_surrogate: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)

    for path in paths:
        with np.load(path, allow_pickle=False) as shard:
            for field in ("dataset", "M", "u_i"):
                if field not in shard:
                    die(f"{path}: missing {field}")
            dataset = scalar_text(shard["dataset"])
            margin = np.asarray(shard["M"], dtype=np.float64).reshape(-1)
            residual = np.asarray(shard["u_i"], dtype=np.float64).reshape(-1)
            if len(margin) != len(residual):
                die(f"{path}: M and u_i lengths differ")
            sid = surrogate_key(shard, path)
            candidate_ids = optional_candidate_ids(shard, len(margin))

            aggregate[dataset]["M"].append(margin)
            aggregate[dataset]["u"].append(residual)
            by_surrogate[(dataset, sid)].append({
                "M": margin,
                "candidate_ids": candidate_ids,
                "path": path,
            })

    loaded: dict[str, dict[str, np.ndarray]] = {}
    for dataset in DATASET_ORDER:
        if dataset not in aggregate:
            die(f"missing Panel-A observations for {dataset}")
        loaded[dataset] = {
            "M": np.concatenate(aggregate[dataset]["M"]),
            "u": np.concatenate(aggregate[dataset]["u"]),
        }
    return loaded, by_surrogate


def recover_panel_b_margin(
    shard: np.lib.npyio.NpzFile,
    path: Path,
    dataset: str,
    sid: str,
    length: int,
    panel_a_index: dict[tuple[str, str], list[dict[str, Any]]],
) -> tuple[np.ndarray, str]:
    # Best case: the previous experiment cached M directly in Panel B.
    if "M" in shard:
        margin = np.asarray(shard["M"], dtype=np.float64).reshape(-1)
        if len(margin) != length:
            die(f"{path}: Panel-B M has {len(margin)} values, expected {length}")
        return margin, "direct"

    candidates_b = optional_candidate_ids(shard, length)
    entries = panel_a_index.get((dataset, sid), [])
    if not entries:
        die(f"{path}: no matching Panel-A cache for dataset={dataset}, surrogate={sid}")

    # Preferred fallback: exact candidate-ID join.
    if candidates_b is not None:
        for entry in entries:
            candidates_a = entry["candidate_ids"]
            if candidates_a is None:
                continue
            mapping = {cid: value for cid, value in zip(candidates_a, entry["M"])}
            if all(cid in mapping for cid in candidates_b):
                return np.asarray([mapping[cid] for cid in candidates_b], dtype=np.float64), "candidate-id join"

    # Last fallback: same surrogate, identical candidate-array length/order.
    same_length = [entry for entry in entries if len(entry["M"]) == length]
    if same_length:
        first = same_length[0]["M"]
        if all(np.array_equal(first, entry["M"]) for entry in same_length[1:]):
            return np.asarray(first, dtype=np.float64), "same-order fallback"
        if len(same_length) == 1:
            return np.asarray(first, dtype=np.float64), "same-order fallback"

    die(
        f"{path}: cannot recover candidate margins for Panel B. "
        "Cache M in panel_b shards or cache candidate_id in both panel_a and panel_b."
    )


def load_panel_b_correlations(
    root: Path,
    paths: list[Path],
    panel_a_index: dict[tuple[str, str], list[dict[str, Any]]],
) -> tuple[dict[str, list[float]], list[dict[str, Any]]]:
    correlations: dict[str, list[float]] = defaultdict(list)
    rows: list[dict[str, Any]] = []
    recovery_modes: dict[str, int] = defaultdict(int)

    for path in paths:
        with np.load(path, allow_pickle=False) as shard:
            for field in ("dataset", "residual_dot"):
                if field not in shard:
                    die(f"{path}: missing {field}")
            dataset = scalar_text(shard["dataset"])
            residual_dot = np.asarray(shard["residual_dot"], dtype=np.float64).reshape(-1)
            n = len(residual_dot)
            sid = surrogate_key(shard, path)
            margin, mode = recover_panel_b_margin(
                shard, path, dataset, sid, n, panel_a_index
            )
            recovery_modes[mode] += 1

            target = optional_id_array(
                shard,
                ("target_id", "target_idx", "target_index", "target_ids"),
                n,
            )
            class_pair = optional_id_array(
                shard,
                ("class_pair", "class_pairs", "pair"),
                n,
            )

            # If target metadata was not cached, each shard is treated as one
            # target/surrogate unit. The relative parent path keeps units distinct.
            if target is None:
                try:
                    fallback_target = str(path.relative_to(root).parent)
                except ValueError:
                    fallback_target = str(path.parent)
                target = np.full(n, fallback_target, dtype=object)
            if class_pair is None:
                class_pair = np.full(n, "unknown", dtype=object)
            surrogate = np.full(n, sid, dtype=object)

            groups: dict[tuple[str, str, str], list[int]] = defaultdict(list)
            for idx, key in enumerate(zip(target, class_pair, surrogate)):
                groups[(str(key[0]), str(key[1]), str(key[2]))].append(idx)

            for (target_id, pair, surrogate_id), indices in groups.items():
                idx = np.asarray(indices, dtype=np.int64)
                valid = np.isfinite(margin[idx]) & np.isfinite(residual_dot[idx])
                idx = idx[valid]
                rho = spearman(-margin[idx], residual_dot[idx])
                if not np.isfinite(rho):
                    continue
                correlations[dataset].append(rho)
                rows.append({
                    "dataset": dataset,
                    "class_pair": pair,
                    "target_id": target_id,
                    "surrogate_id": surrogate_id,
                    "n_candidates": int(len(idx)),
                    "spearman_neg_margin_residual_dot": float(rho),
                    "margin_source": mode,
                    "cache_shard": str(path),
                })

    for dataset in DATASET_ORDER:
        if not correlations.get(dataset):
            die(f"no finite target/surrogate correlations found for {dataset}")

    print("Panel-B margin recovery:", flush=True)
    for mode, count in sorted(recovery_modes.items()):
        print(f"  {mode}: {count} shard(s)", flush=True)
    return correlations, rows


def style_axis(axis: plt.Axes) -> None:
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.grid(True, axis="y", alpha=0.18, linewidth=0.7)
    axis.tick_params(labelsize=10)


def make_main_figure(
    panel_a: dict[str, dict[str, np.ndarray]],
    correlations: dict[str, list[float]],
    output_dir: Path,
    bins: int,
    dpi: int,
    seed: int,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11.4, 4.35), constrained_layout=True)
    ax_a, ax_b = axes

    # Panel A: one simple monotone curve per dataset.
    for dataset in DATASET_ORDER:
        margin = panel_a[dataset]["M"]
        residual = panel_a[dataset]["u"]
        x, y = percentile_curve(margin, residual, bins)
        rho = spearman(-margin[residual > 0], residual[residual > 0])
        ax_a.plot(
            x,
            y,
            linewidth=2.2,
            marker="o",
            markersize=3.8,
            label=rf"{DISPLAY_NAME[dataset]} ($\rho_s={rho:.2f}$)",
        )

    ax_a.set_yscale("log")
    ax_a.set_xlim(0, 100)
    ax_a.set_xlabel("Candidate margin percentile  (low margin  →  high margin)")
    ax_a.set_ylabel(r"Median residual mass $1-p_{i,y_{\rm adv}}$")
    ax_a.set_title("(a) Lower-margin candidates have larger residuals", loc="left")
    ax_a.legend(frameon=False, fontsize=8.6, loc="best")
    style_axis(ax_a)

    # Panel B: distribution over target x surrogate units.
    box_data = [np.asarray(correlations[d], dtype=np.float64) for d in DATASET_ORDER]
    positions = np.arange(1, len(DATASET_ORDER) + 1)
    ax_b.boxplot(
        box_data,
        positions=positions,
        widths=0.55,
        showfliers=False,
        patch_artist=False,
        medianprops={"linewidth": 2.0},
        whiskerprops={"linewidth": 1.2},
        capprops={"linewidth": 1.2},
        boxprops={"linewidth": 1.3},
    )
    rng = np.random.default_rng(seed)
    for pos, dataset, values in zip(positions, DATASET_ORDER, box_data):
        jitter = rng.normal(0.0, 0.045, size=len(values))
        ax_b.scatter(
            np.full(len(values), pos, dtype=np.float64) + jitter,
            values,
            s=13,
            alpha=0.38,
            edgecolors="none",
            zorder=3,
        )

    ax_b.axhline(0.0, linewidth=0.9, linestyle="--", alpha=0.45)
    ax_b.set_xticks(positions)
    ax_b.set_xticklabels([
        f"{DISPLAY_NAME[d]}\n(n={len(correlations[d])})" for d in DATASET_ORDER
    ])
    ax_b.set_ylim(-1.02, 1.02)
    ax_b.set_ylabel(r"Spearman $\rho_s(-M_i,\, r_i^\top r_t)$")
    ax_b.set_title("(b) Margin preserves residual-interaction ranking", loc="left")
    ax_b.text(
        0.02,
        0.03,
        "Each point = one target × surrogate",
        transform=ax_b.transAxes,
        fontsize=8.8,
        alpha=0.75,
    )
    style_axis(ax_b)

    output_dir.mkdir(parents=True, exist_ok=True)
    stem = output_dir / "margin_residual_main"
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".png"), dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def write_correlations(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = (
        "dataset",
        "class_pair",
        "target_id",
        "surrogate_id",
        "n_candidates",
        "spearman_neg_margin_residual_dot",
        "margin_source",
        "cache_shard",
    )
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main(cli: argparse.Namespace) -> None:
    root = cli.input.resolve()
    output_dir = cli.output_dir.resolve()
    panel_a_paths, panel_b_paths = discover_shards(root)
    print(f"Cache root: {root}", flush=True)
    print(f"Panel A shards: {len(panel_a_paths)}", flush=True)
    print(f"Panel B shards: {len(panel_b_paths)}", flush=True)

    panel_a, panel_a_index = load_panel_a(panel_a_paths)
    correlations, rows = load_panel_b_correlations(
        root, panel_b_paths, panel_a_index
    )

    print("\nSummary", flush=True)
    for dataset in DATASET_ORDER:
        m = panel_a[dataset]["M"]
        u = panel_a[dataset]["u"]
        valid = np.isfinite(m) & np.isfinite(u) & (u > 0)
        margin_rho = spearman(-m[valid], u[valid])
        vals = np.asarray(correlations[dataset], dtype=np.float64)
        print(
            f"  {DISPLAY_NAME[dataset]}: "
            f"rho(-M,u)={margin_rho:.4f}; "
            f"rho(-M,r_i^T r_t) median={np.median(vals):.4f}, "
            f"mean={np.mean(vals):.4f}, n={len(vals)}",
            flush=True,
        )

    make_main_figure(
        panel_a,
        correlations,
        output_dir,
        bins=cli.bins,
        dpi=cli.dpi,
        seed=cli.seed,
    )
    csv_path = output_dir / "margin_residual_main_spearman.csv"
    write_correlations(rows, csv_path)

    print("\nWROTE", flush=True)
    print(output_dir / "margin_residual_main.pdf", flush=True)
    print(output_dir / "margin_residual_main.png", flush=True)
    print(csv_path, flush=True)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Make the simple main-paper margin/residual plot from cached NPZ shards."
    )
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Root containing panel_a_surrogate_*.npz and panel_b_surrogate_*.npz",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bins", type=int, default=20)
    parser.add_argument("--dpi", type=int, default=350)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)
    if args.bins < 8 or args.bins > 50:
        parser.error("--bins must be between 8 and 50")
    if args.dpi <= 0:
        parser.error("--dpi must be positive")
    return args


if __name__ == "__main__":
    main(parse_args())


    
