#!/usr/bin/env python3
"""Plot raw negative logit margin against probability-residual magnitude.

For each eligible candidate, averaged across the saved surrogate models:

    x = -M_i
    y = ||r_i||_2 = ||p_theta(x_i) - e_adv||_2.

Each candidate is plotted once per dataset/model/class-pair setting. Candidates
are not duplicated across targets because both quantities are target-independent.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import pearsonr, spearmanr


COLORS = (
    "#0072B2", "#D55E00", "#009E73", "#CC79A7",
    "#E69F00", "#56B4E9", "#6F4E7C", "#000000",
)


@dataclass
class Setting:
    dataset: str
    model: str
    class_pair: str
    candidate_idx: np.ndarray
    negative_margin: np.ndarray
    residual_norm: np.ndarray
    surrogate_count: int

    @property
    def label(self) -> str:
        dataset = {"CIFAR10": "CIFAR-10", "CIFAR100": "CIFAR-100",
                   "TinyImageNet": "Tiny ImageNet"}.get(self.dataset,
                                                          self.dataset)
        return f"{dataset} · {self.model} · {self.class_pair}"


def die(message: str) -> "NoReturn":
    raise SystemExit("ERROR: " + message)


def atomic_csv(path: Path, fieldnames: Sequence[str], rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def correlation(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    keep = np.isfinite(x) & np.isfinite(y)
    x, y = x[keep], y[keep]
    if len(x) < 3 or np.ptp(x) == 0 or np.ptp(y) == 0:
        return float("nan"), float("nan")
    return (float(spearmanr(x, y).statistic),
            float(pearsonr(x, y).statistic))


def load_settings(root: Path) -> tuple[list[Setting], list[str]]:
    settings: list[Setting] = []
    skipped: list[str] = []
    for manifest_path in sorted(root.glob("*/*/*/manifest.json")):
        combo = manifest_path.parent
        try:
            manifest = json.loads(manifest_path.read_text())
            targets = [int(value) for value in manifest["target_indices"]]
            surrogate_ids = [int(value) for value in manifest["surrogate_ids"]]
        except (OSError, KeyError, TypeError, ValueError,
                json.JSONDecodeError) as exc:
            skipped.append(f"{combo}: unreadable manifest ({exc})")
            continue
        if not targets or not surrogate_ids:
            skipped.append(f"{combo}: no targets or surrogates in manifest")
            continue

        # Candidate logits/residuals do not depend on the target. Read one
        # target directory only, preventing tenfold duplication and I/O.
        target_dir = combo / f"target_{targets[0]}"
        shard_paths = [target_dir / f"surrogate_{sid:03d}.npz"
                       for sid in surrogate_ids]
        missing = [path for path in shard_paths if not path.is_file()]
        if missing:
            skipped.append(f"{combo}: missing {len(missing)} surrogate shards")
            continue

        candidate_idx = None
        margin_sum = None
        residual_norm_sum = None
        for path in shard_paths:
            with np.load(path, allow_pickle=False) as data:
                required = ("candidate_idx", "M_logit_margin",
                            "candidate_residual")
                absent = [name for name in required if name not in data]
                if absent:
                    die(f"{path} is missing {', '.join(absent)}")
                current_idx = np.asarray(data["candidate_idx"], dtype=np.int64)
                margin = np.asarray(data["M_logit_margin"], dtype=np.float64)
                residual = np.asarray(data["candidate_residual"],
                                      dtype=np.float64)
            if residual.ndim != 2 or len(residual) != len(margin):
                die(f"invalid candidate residual shape in {path}: "
                    f"{residual.shape}")
            if candidate_idx is None:
                candidate_idx = current_idx
                margin_sum = np.zeros_like(margin)
                residual_norm_sum = np.zeros_like(margin)
            elif not np.array_equal(candidate_idx, current_idx):
                die(f"candidate order differs in {path}")
            margin_sum += margin
            residual_norm_sum += np.linalg.norm(residual, axis=1)

        assert candidate_idx is not None
        assert margin_sum is not None and residual_norm_sum is not None
        count = len(shard_paths)
        settings.append(Setting(
            dataset=str(manifest["dataset"]),
            model=str(manifest["model"]),
            class_pair=str(manifest["class_pair"]),
            candidate_idx=candidate_idx,
            negative_margin=-(margin_sum / count),
            residual_norm=residual_norm_sum / count,
            surrogate_count=count,
        ))
    if not settings:
        die("no complete settings with per-surrogate residuals were found")
    return settings, skipped


def configure_style() -> None:
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "axes.titlesize": 13,
        "axes.labelsize": 12,
        "legend.fontsize": 8,
        "pdf.fonttype": 42,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })


def save_figure(fig: plt.Figure, output: Path, formats: Sequence[str],
                dpi: int) -> None:
    for extension in formats:
        path = output / f"negative_margin_vs_residual_norm.{extension}"
        kwargs = {"bbox_inches": "tight"}
        if extension == "png":
            kwargs["dpi"] = dpi
        fig.savefig(path, **kwargs)
        print(f"wrote {path}", flush=True)
    plt.close(fig)


def plot(settings: Sequence[Setting], output: Path,
         formats: Sequence[str], dpi: int) -> None:
    configure_style()
    pooled_x = np.concatenate([setting.negative_margin for setting in settings])
    pooled_y = np.concatenate([setting.residual_norm for setting in settings])
    pooled_spearman, pooled_pearson = correlation(pooled_x, pooled_y)

    fig, ax = plt.subplots(figsize=(8.4, 6.3), constrained_layout=True)
    for index, setting in enumerate(settings):
        ax.scatter(
            setting.negative_margin,
            setting.residual_norm,
            s=5,
            alpha=.16,
            color=COLORS[index % len(COLORS)],
            edgecolors="none",
            rasterized=True,
            label=setting.label,
        )
    ax.set_xlabel(r"Negative logit margin $-M_i$")
    ax.set_ylabel(r"Probability residual magnitude "
                  r"$\|r_i\|_2=\|p_\theta(x_i)-e_{\mathrm{adv}}\|_2$")
    ax.set_title(r"Relationship between $-M_i$ and $\|r_i\|_2$"
                 "\nEvery eligible candidate is one raw ensemble-mean dot")
    ax.text(
        .03, .97,
        f"points = {len(pooled_x):,}\n"
        f"pooled Spearman = {pooled_spearman:.4f}\n"
        f"pooled Pearson = {pooled_pearson:.4f}",
        transform=ax.transAxes,
        ha="left",
        va="top",
        bbox={"facecolor": "white", "alpha": .9,
              "edgecolor": "#cccccc", "boxstyle": "round,pad=.35"},
    )
    ax.grid(color="#e2e2e2", linewidth=.7)
    ax.legend(frameon=False, loc="center left", bbox_to_anchor=(1.01, .5))
    save_figure(fig, output, formats, dpi)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path,
                        default=Path("./gradient_alignment_outputs"))
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--formats", default="pdf,png")
    parser.add_argument("--dpi", type=int, default=300)
    args = parser.parse_args(argv)
    args.input_root = args.input_root.expanduser().resolve()
    args.output_dir = ((args.input_root / "figures" / "margin_residual")
                       if args.output_dir is None
                       else args.output_dir.expanduser().resolve())
    args.formats = [value.strip().lower() for value in args.formats.split(",")
                    if value.strip()]
    if not args.input_root.is_dir():
        parser.error(f"input root does not exist: {args.input_root}")
    if not args.formats:
        parser.error("--formats cannot be empty")
    return args


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    settings, skipped = load_settings(args.input_root)
    if skipped:
        print("skipping incomplete settings:", flush=True)
        for item in skipped:
            print(f"  {item}", flush=True)
    print(f"loaded {len(settings)} settings", flush=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    for setting in settings:
        spearman, pearson = correlation(setting.negative_margin,
                                         setting.residual_norm)
        rows.append({
            "dataset": setting.dataset,
            "model": setting.model,
            "class_pair": setting.class_pair,
            "candidate_count": len(setting.candidate_idx),
            "surrogate_count": setting.surrogate_count,
            "spearman_neg_margin_vs_residual_norm": spearman,
            "pearson_neg_margin_vs_residual_norm": pearson,
        })
    atomic_csv(args.output_dir / "negative_margin_vs_residual_norm.csv",
               list(rows[0]), rows)
    plot(settings, args.output_dir, args.formats, args.dpi)
    print(f"analysis complete: {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
