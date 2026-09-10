import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.transforms import Bbox


# Settings
RESULTS_ROOT = Path("scatter_result")
OUTPUT_ROOT_PDF = Path("figures/pdf/scatter")
PLOT_DATA_ROOT = Path("figures/data/scatter")

EXPERIMENT_FILTER = ()
DATASET_FILTER = ("CIFAR10",)
MODEL_FILTER = ("ConvNetBN", "ResNet20BN", "VGG13BN")
TARGET_FILTER = ()

# Use {} to plot every target selected by TARGET_FILTER.
RANK_TARGET_BY_DATASET_MODEL = {
    ("CIFAR10", "ConvNetBN"): 5705,
}

KEEP_LOWEST = False
USE_PLOT_DATA_CACHE = True
SAVE_PLOT_DATA_CACHE = True
REBUILD_PLOT_DATA_CACHE = False
PLOT_DATA_CACHE_VERSION = 2

SAVE_PDF = True
SHOW_PLOTS = True
CLOSE_FIGURES_AFTER_SHOW = False

PANEL_SIZE = (3.375, 3.0)
FIGURE_DPI = 150
SAVE_DPI = 500
POINT_SIZE = 2.5
POINT_ALPHA = 0.65
TEXT_COLOR = "#111111"

RANK_X_LABEL = r"$R + M$"
RANK_Y_LABEL = r"$A$"
RANK_X_LABEL_FONT_SIZE = 20
RANK_Y_LABEL_FONT_SIZE = 30
RANK_Y_LABEL_MARGIN_FONT_SIZE = 36
RANK_Y_LABEL_X = -0.24
RANK_Y_LABEL_Y = 0.5
RANK_Y_LABEL_LEFT_PADDING_POINTS = 16.0
RANK_SHOW_Y_LABEL = {"ConvNetBN": True}
RANK_Y_LABEL_ROTATION = {"ConvNetBN": 90}

MODEL_COLORS = {
    "ConvNetBN": "#0072B2",
    "ResNet20BN": "#FF2C00",
    "VGG13BN": "#009E73",
}


def configure_matplotlib():
    mpl.rcParams.update({
        "figure.facecolor": "white",
        "figure.edgecolor": "white",
        "axes.facecolor": "white",
        "font.family": "serif",
        "font.serif": [
            "Times New Roman",
            "Times",
            "Nimbus Roman No9 L",
            "DejaVu Serif",
        ],
        "font.size": 14,
        "mathtext.fontset": "dejavuserif",
        "text.color": TEXT_COLOR,
        "axes.labelcolor": TEXT_COLOR,
        "xtick.color": TEXT_COLOR,
        "ytick.color": TEXT_COLOR,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


def discover_experiments():
    if not RESULTS_ROOT.is_dir():
        raise FileNotFoundError(f"{RESULTS_ROOT} does not exist")

    found = []
    for directory in sorted(RESULTS_ROOT.glob("SCATTER_*")):
        if not directory.is_dir():
            continue

        parts = directory.name[len("SCATTER_"):].split("_", 2)
        if len(parts) < 2:
            continue

        dataset, model = parts[:2]

        if EXPERIMENT_FILTER and not any(
            token in directory.name for token in EXPERIMENT_FILTER
        ):
            continue
        if DATASET_FILTER and dataset not in DATASET_FILTER:
            continue
        if MODEL_FILTER and model not in MODEL_FILTER:
            continue

        if all(
            (directory / name).is_file()
            for name in ("components.npz", "summary.json")
        ):
            found.append(directory)

    if not found:
        raise FileNotFoundError(
            "No matching experiments with components.npz and summary.json "
            f"under {RESULTS_ROOT}"
        )

    return found


class Experiment:
    def __init__(self, directory):
        self.directory = directory
        self.name = directory.name[len("SCATTER_"):]
        self.dataset = self.name.split("_", 1)[0]

        with (directory / "summary.json").open() as handle:
            summary = json.load(handle)

        self.model = summary.get("model", self.name.split("_", 2)[1])

        with np.load(directory / "components.npz", allow_pickle=False) as data:
            self.subset_pos = data["subset_pool_pos"]
            summary_targets = [int(t) for t in summary["targets"]]
            component_targets = [
                int(key[len("rm_target"):])
                for key in data.files
                if key.startswith("rm_target")
                and key[len("rm_target"):].isdigit()
            ]
            saved_targets = list(dict.fromkeys(
                summary_targets + component_targets
            ))
            self.target_aliases = {t: t for t in saved_targets}

            def register_aliases(name, values):
                if "target" not in name.lower():
                    return

                try:
                    aliases = np.asarray(values)
                except (TypeError, ValueError):
                    return

                if aliases.ndim != 1 or len(aliases) != len(summary_targets):
                    return

                for saved_target, alias in zip(summary_targets, aliases):
                    try:
                        self.target_aliases[int(alias)] = saved_target
                    except (TypeError, ValueError, OverflowError):
                        continue

            for name, values in summary.items():
                register_aliases(name, values)

            component_prefixes = (
                "rm_target",
                "a_target",
                "d_raw_target",
                "m_raw_target",
                "a_raw_target",
            )
            for name in data.files:
                if (
                    "target" in name.lower()
                    and not name.startswith(component_prefixes)
                ):
                    register_aliases(name, data[name])

            if TARGET_FILTER:
                missing = [t for t in TARGET_FILTER if t not in saved_targets]
                if missing:
                    raise KeyError(f"{self.name}: missing targets {missing}")

            self.targets = list(TARGET_FILTER) if TARGET_FILTER else saved_targets
            self.rm = {
                t: np.asarray(data[f"rm_target{t}"], dtype=np.float64)
                for t in self.targets
            }
            self.a = {
                t: data[f"a_target{t}"]
                for t in self.targets
            }


def percentile_rank(values):
    """Full-pool ranks in [0, 1]; ties are broken by position."""
    order = np.argsort(values, kind="stable")
    ranks = np.empty(len(values), dtype=np.float64)
    ranks[order] = np.arange(len(values))
    return ranks / max(len(values) - 1, 1)


def pearson(x, y):
    if len(x) < 2 or x.std() == 0 or y.std() == 0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def rank_plot_data(experiment, target):
    cache_path = PLOT_DATA_ROOT / experiment.name / f"rank_target{target}.npz"

    sources = {}
    for filename in ("components.npz", "summary.json"):
        stat = (experiment.directory / filename).stat()
        sources[filename] = [stat.st_size, stat.st_mtime_ns]

    metadata = json.dumps({
        "version": PLOT_DATA_CACHE_VERSION,
        "keep_lowest": KEEP_LOWEST,
        "target": int(target),
        "sources": sources,
    }, sort_keys=True)

    if (
        USE_PLOT_DATA_CACHE
        and not REBUILD_PLOT_DATA_CACHE
        and cache_path.is_file()
    ):
        try:
            with np.load(cache_path, allow_pickle=False) as cached:
                if str(cached["metadata_json"].item()) == metadata:
                    return (
                        cached["displayed_x"],
                        cached["y"],
                        float(cached["pearson"].item()),
                    )
        except (OSError, KeyError, ValueError):
            pass

    score = experiment.rm[target]
    subset = experiment.subset_pos
    best_is_zero = percentile_rank(score if KEEP_LOWEST else -score)

    x = 1.0 - best_is_zero[subset]
    y = percentile_rank(experiment.a[target])[subset]
    correlation = pearson(x, y)

    if SAVE_PLOT_DATA_CACHE:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            cache_path,
            metadata_json=np.asarray(metadata),
            displayed_x=x,
            y=y,
            pearson=np.asarray(correlation),
        )

    return x, y, correlation


def style_axes(ax):
    for side, spine in ax.spines.items():
        spine.set_visible(side in ("left", "bottom"))
        spine.set_color(TEXT_COLOR)
        spine.set_linewidth(1.6)

    ax.tick_params(
        axis="both",
        which="major",
        direction="out",
        length=8.0,
        width=1.5,
        pad=5.0,
        labelsize=22,
        color=TEXT_COLOR,
        labelcolor=TEXT_COLOR,
        top=False,
        right=False,
    )
    ax.grid(False)
    ax.set_axisbelow(True)


def save_rank_figure(fig, stem, show_y_label):
    if not SAVE_PDF:
        return

    # Freeze the layout before measuring the final crop.
    fig.canvas.draw()
    fig.set_layout_engine(None)
    fig.canvas.draw()
    bbox = fig.get_tightbbox(fig.canvas.get_renderer())

    if show_y_label:
        bbox = Bbox.from_extents(
            bbox.x0 - RANK_Y_LABEL_LEFT_PADDING_POINTS / 72.0,
            bbox.y0,
            bbox.x1,
            bbox.y1,
        )

    OUTPUT_ROOT_PDF.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        OUTPUT_ROOT_PDF / f"{stem}.pdf",
        dpi=SAVE_DPI,
        bbox_inches=bbox,
        pad_inches=0.0,
        facecolor="white",
        edgecolor="white",
        transparent=False,
    )


def plot_rank_grid(experiment):
    if RANK_TARGET_BY_DATASET_MODEL:
        requested = RANK_TARGET_BY_DATASET_MODEL.get(
            (experiment.dataset, experiment.model)
        )
        if requested is None:
            return {}

        target = experiment.target_aliases.get(int(requested))
        if target is None or target not in experiment.targets:
            return {}

        targets = [(requested, target)]
    else:
        targets = [(target, target) for target in experiment.targets]

    color = MODEL_COLORS[experiment.model]
    show_y_label = RANK_SHOW_Y_LABEL.get(experiment.model, True)
    figures = {}

    for output_target, target in targets:
        x, y, correlation = rank_plot_data(experiment, target)

        fig, ax = plt.subplots(
            figsize=PANEL_SIZE,
            dpi=FIGURE_DPI,
            constrained_layout=True,
        )
        ax.scatter(
            x,
            y,
            s=POINT_SIZE,
            alpha=POINT_ALPHA,
            color=color,
            edgecolors="none",
            rasterized=True,
            zorder=3,
        )
        ax.set_xlim(-0.015, 1.015)
        ax.set_ylim(-0.015, 1.015)
        ax.set_xticks((0.0, 0.5, 1.0))
        ax.set_yticks((0.0, 0.5, 1.0))
        style_axes(ax)

        ax.set_xlabel(RANK_X_LABEL, fontsize=RANK_X_LABEL_FONT_SIZE)

        if show_y_label:
            rotation = RANK_Y_LABEL_ROTATION.get(experiment.model, 90)
            white_y_title = ax.text(
                RANK_Y_LABEL_X,
                RANK_Y_LABEL_Y,
                RANK_Y_LABEL,
                transform=ax.transAxes,
                fontsize=RANK_Y_LABEL_MARGIN_FONT_SIZE,
                rotation=rotation,
                rotation_mode="anchor",
                color="white",
                ha="center",
                va="center",
                clip_on=False,
                zorder=3,
            )
            white_y_title.set_in_layout(True)

            real_y_title = ax.text(
                RANK_Y_LABEL_X,
                RANK_Y_LABEL_Y,
                RANK_Y_LABEL,
                transform=ax.transAxes,
                fontsize=RANK_Y_LABEL_FONT_SIZE,
                rotation=rotation,
                rotation_mode="anchor",
                color="black",
                ha="center",
                va="center",
                clip_on=False,
                zorder=4,
            )
            real_y_title.set_in_layout(True)
        else:
            ax.set_ylabel("")

        save_rank_figure(
            fig,
            f"rank_target{output_target}_pearson_{correlation:+.3f}",
            show_y_label,
        )
        figures[output_target] = fig

    return figures


def main():
    configure_matplotlib()
    all_figures = {}

    for directory in discover_experiments():
        experiment = Experiment(directory)
        all_figures[experiment.name] = {
            f"rank_target{target}": fig
            for target, fig in plot_rank_grid(experiment).items()
        }

    if SHOW_PLOTS:
        plt.show()

    if CLOSE_FIGURES_AFTER_SHOW:
        for figures in all_figures.values():
            for fig in figures.values():
                plt.close(fig)

    return all_figures


if __name__ == "__main__":
    all_figures = main()
