"""Jupyter-ready re-plotting of every experiment under scatter_result/.

Reads ONLY what final_update_scatter.py --scatter_mode saved (components.npz
and summary.json in each SCATTER_<run name>/ directory) and redraws the
figures from those arrays. Nothing is recomputed on a GPU, so every knob
below can be changed and the cell re-run in seconds.

Per experiment (one SCATTER_* directory) it can export:
  scatter_grid        raw x = rm, y = A for the saved random subset, one
                      panel per target
  rank_grid           the same subset as full-pool percentile ranks
  conditional_a       median A (with IQR band) in equal-count bins of x
  topk_overlap_linear |top-k by (R,M)  n  top-k by (A,R,M)| / k  against the
                      fraction of the pool selected, linear x axis
  topk_overlap_log    the same curve with a logarithmic x axis

The two rankings compared by the overlap curve are defined by cheap_score()
and full_score() in USER CONTROLS, so the combination rule, the weight beta,
and the keep-lowest / keep-highest convention are all editable here.
"""

from pathlib import Path
import json
import math

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FormatStrFormatter, NullLocator, PercentFormatter


# =============================================================================
# USER CONTROLS
# =============================================================================

RUN_SCATTER_GRID = True
RUN_RANK_GRID = True
RUN_CONDITIONAL = True
RUN_TOPK_LINEAR = True
RUN_TOPK_LOG = True

# Where the SCATTER_* directories are, and which of them to plot. The filter
# is a tuple of substrings; an experiment is plotted when its directory name
# contains ANY of them. () plots everything.
RESULTS_ROOT = Path("scatter_result")
EXPERIMENT_FILTER = ()
# EXPERIMENT_FILTER = ("ConvNetBN_fc_ours_dog-bird",)

# Which targets to draw. () = every target saved in summary.json; otherwise a
# tuple of target indices, in the order they should appear.
TARGET_FILTER = ()

# Standard budgets marked on the overlap curve, as fractions of the training
# set (each becomes N_p = round(budget * N_total) on the x axis).
BUDGET_MARKS = (0.001, 0.002, 0.005, 0.01, 0.02, 0.04)
SHOW_BUDGET_MARKS = True
SHOW_BUDGET_MARK_VALUES = True
# The run's own N_p is always available as a mark as well.
SHOW_RUN_NP_MARK = True

# Overlap curve x range, as a fraction of the poison-class pool. The curve is
# evaluated for every k from 1 to round(X_MAX_FRACTION * pool).
X_MAX_FRACTION = 0.8
# Log version: the smallest fraction shown (1 candidate = 1 / pool). None
# means start at the first candidate.
LOG_X_MIN_FRACTION = None

# Conditional-A plot: number of equal-count bins of x.
CONDITIONAL_BINS = 20

# Scatter grid y axis: "linear" or "symlog". symlog keeps the bulk near 0
# readable while still showing the long positive tail of A.
SCATTER_Y_SCALE = "symlog"
SCATTER_SYMLOG_LINTHRESH = 0.5

# Panel grid shape for the per-target figures.
GRID_COLUMNS = 4


# -----------------------------------------------------------------------------
# The two rankings. Both are evaluated over the FULL poison-class pool of one
# target, with the arrays final_update_scatter.py saved:
#   rm     = mean over surrogates of  std(d) + lam * std(M)      (cost)
#   a      = mean over surrogates of  std(A)                     (gain)
#   d_raw, m_raw, a_raw = the same three, NOT standardized
# KEEP_LOWEST = True  -> both functions return a COST and the k smallest win.
# KEEP_LOWEST = False -> both return a GAIN and the k largest win.
# -----------------------------------------------------------------------------

BETA = 1.0
KEEP_LOWEST = True


def cheap_score(rm, a, d_raw, m_raw, a_raw):
    """(R, M) ranking. Default: the saved cost rm = std(d) + lam*std(M)."""
    return rm


def full_score(rm, a, d_raw, m_raw, a_raw):
    """(A, R, M) ranking. Default: rm - beta * A, the real selector."""
    return rm - BETA * a


# =============================================================================
# LABELS
# =============================================================================

TITLE_TEMPLATE = "{model} / {attack} / {class_pair}"
SHOW_TITLES = True
SHOW_PANEL_TITLES = True
PANEL_TITLE_TEMPLATE = "target {target}"

SCATTER_X_LABEL = r"$\mathrm{std}(d_i) + \lambda\,\mathrm{std}(M_i)$"
SCATTER_Y_LABEL = r"$\mathrm{std}(A_i)$"

RANK_X_LABEL = r"rank of $\mathrm{std}(d_i) + \lambda\,\mathrm{std}(M_i)$  (0 = best)"
RANK_Y_LABEL = r"rank of $A_i$  (1 = highest)"

CONDITIONAL_X_LABEL = "quantile of the (R, M) score  (0 = best)"
CONDITIONAL_Y_LABEL = r"median $\mathrm{std}(A_i)$ in bin"

TOPK_X_LABEL = "fraction of the poison-class pool selected"
TOPK_Y_LABEL = r"$|S_{(A,R,M)} \cap S_{(R,M)}|\;/\;k$"

LEGEND_MEAN_LABEL = "mean over {n} targets"
LEGEND_TARGET_LABEL = "individual targets"
LEGEND_CHANCE_LABEL = "random selection"
LEGEND_IQR_LABEL = "inter-quartile band"

SHOW_STATS_BOX = True          # n / Pearson / Spearman box in scatter panels
STATS_BOX_FONT_SIZE = 12


# =============================================================================
# FIGURE STYLE
# =============================================================================

PANEL_SIZE = (4.6, 4.0)        # per panel, grids multiply this
SINGLE_FIG_SIZE = (9.0, 6.0)   # conditional and overlap figures
FIGURE_DPI = 150

FONT_FAMILY = "serif"
SERIF_FONTS = (
    "Times New Roman",
    "Times",
    "Nimbus Roman No9 L",
    "DejaVu Serif",
)
MATH_FONTSET = "dejavuserif"

BASE_FONT_SIZE = 16
AXIS_LABEL_FONT_SIZE = 22
TICK_FONT_SIZE = 18
TITLE_FONT_SIZE = 22
PANEL_TITLE_FONT_SIZE = 18
LEGEND_FONT_SIZE = 16
ANNOTATION_FONT_SIZE = 12
TEXT_COLOR = "#111111"

SPINE_LINE_WIDTH = 1.4
MAJOR_TICK_LENGTH = 7.0
MAJOR_TICK_WIDTH = 1.6
TICK_LABEL_PAD = 4.0

SHOW_GRID = True
GRID_COLOR = "#D5D8DC"
GRID_LINE_WIDTH = 0.6
GRID_ALPHA = 1.0

# Marks
POINT_COLOR = "#2a78d6"
POINT_SIZE = 9
POINT_ALPHA = 0.45

TARGET_LINE_COLOR = "#2a78d6"
TARGET_LINE_WIDTH = 1.0
TARGET_LINE_ALPHA = 0.30
MEAN_LINE_COLOR = "#004481"
MEAN_LINE_WIDTH = 2.6
CHANCE_LINE_COLOR = "#9A9A9A"
CHANCE_LINE_WIDTH = 1.2
CHANCE_LINE_STYLE = ":"
IQR_FILL_COLOR = "#2a78d6"
IQR_FILL_ALPHA = 0.12

BUDGET_MARK_COLOR = "#B0B0B0"
BUDGET_MARK_WIDTH = 0.9
BUDGET_MARK_STYLE = "--"
BUDGET_MARK_TEXT_COLOR = "#555555"
BUDGET_MARK_TEXT_Y = 0.02       # axes fraction
RUN_NP_MARK_COLOR = "#E69F00"

ZERO_LINE_COLOR = "#B0B0B0"
ZERO_LINE_WIDTH = 0.8

# Overlap y axis
TOPK_Y_LIMITS = (0.0, 1.02)
TOPK_Y_TICKS = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)
TOPK_Y_AS_PERCENT = False


# =============================================================================
# LEGEND AND EXPORT
# =============================================================================

SHOW_LEGEND = True
LEGEND_LOCATION = "lower right"
LEGEND_BBOX = None             # e.g. (0.98, 0.05); None = matplotlib default
LEGEND_FRAME_VISIBLE = True
LEGEND_BACKGROUND_COLOR = "white"
LEGEND_BACKGROUND_ALPHA = 0.95
LEGEND_BORDER_COLOR = "#777777"
LEGEND_BORDER_LINE_WIDTH = 1.0
LEGEND_NUMBER_OF_COLUMNS = 1

SAVE_PDF = True
SAVE_PNG = False
SHOW_PLOTS = True
CLOSE_FIGURES_AFTER_SHOW = False

OUTPUT_ROOT_PDF = Path("figures/pdf/scatter")
OUTPUT_ROOT_PNG = Path("figures/png/scatter")
SAVE_DPI = 150


# =============================================================================
# LOADING
# =============================================================================

def _configure_matplotlib():
    mpl.rcParams.update(
        {
            "figure.facecolor": "white",
            "figure.edgecolor": "white",
            "axes.facecolor": "white",
            "font.family": FONT_FAMILY,
            "font.serif": list(SERIF_FONTS),
            "font.size": BASE_FONT_SIZE,
            "text.color": TEXT_COLOR,
            "mathtext.fontset": MATH_FONTSET,
            "axes.labelcolor": TEXT_COLOR,
            "xtick.color": TEXT_COLOR,
            "ytick.color": TEXT_COLOR,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.dpi": SAVE_DPI,
            "savefig.facecolor": "white",
            "savefig.edgecolor": "white",
            "savefig.transparent": False,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.02,
        }
    )


def _discover_experiments():
    if not RESULTS_ROOT.is_dir():
        raise FileNotFoundError(f"{RESULTS_ROOT} does not exist")

    found = []
    for directory in sorted(RESULTS_ROOT.iterdir()):
        if not directory.is_dir() or not directory.name.startswith("SCATTER_"):
            continue
        if not (directory / "components.npz").is_file():
            print(f"skip {directory.name}: no components.npz")
            continue
        if not (directory / "summary.json").is_file():
            print(f"skip {directory.name}: no summary.json")
            continue
        if EXPERIMENT_FILTER and not any(
            token in directory.name for token in EXPERIMENT_FILTER
        ):
            continue
        found.append(directory)

    if not found:
        raise FileNotFoundError(
            f"no SCATTER_* experiment with components.npz under {RESULTS_ROOT}"
        )
    return found


class Experiment:
    """One SCATTER_* directory, loaded into plain numpy arrays."""

    def __init__(self, directory):
        self.directory = directory
        self.name = directory.name[len("SCATTER_"):]

        with open(directory / "summary.json") as handle:
            self.summary = json.load(handle)

        data = np.load(directory / "components.npz")
        self.pool_train_idx = data["pool_train_idx"]
        self.subset_pos = data["subset_pool_pos"]
        self.pool_size = int(len(self.pool_train_idx))

        saved_targets = [int(t) for t in self.summary["targets"]]
        if TARGET_FILTER:
            missing = [t for t in TARGET_FILTER if t not in saved_targets]
            if missing:
                raise KeyError(
                    f"{self.name}: TARGET_FILTER has targets not in this run: "
                    f"{missing}"
                )
            self.targets = list(TARGET_FILTER)
        else:
            self.targets = saved_targets

        self.rm = {t: data[f"rm_target{t}"] for t in self.targets}
        self.a = {t: data[f"a_target{t}"] for t in self.targets}
        self.d_raw = {t: data[f"d_raw_target{t}"] for t in self.targets}
        self.m_raw = {t: data[f"m_raw_target{t}"] for t in self.targets}
        self.a_raw = {t: data[f"a_raw_target{t}"] for t in self.targets}

        self.model = self.summary.get("model", "?")
        self.attack = self.summary.get("attack", "?")
        self.class_pair = self.summary.get("class_pair", "?")
        self.budget = float(self.summary.get("budget") or 0.0)
        self.num_poisons = int(self.summary.get("num_poisons") or 0)

        # N_total is not stored; recover it from the run's own budget so the
        # standard budget marks land on the right N_p.
        if self.budget > 0 and self.num_poisons > 0:
            self.n_total = int(round(self.num_poisons / self.budget))
        else:
            self.n_total = None

    # --- derived quantities -------------------------------------------------

    def cheap(self, target):
        return np.asarray(
            cheap_score(self.rm[target], self.a[target], self.d_raw[target],
                        self.m_raw[target], self.a_raw[target]),
            dtype=np.float64,
        )

    def full(self, target):
        return np.asarray(
            full_score(self.rm[target], self.a[target], self.d_raw[target],
                       self.m_raw[target], self.a_raw[target]),
            dtype=np.float64,
        )

    def title(self):
        return TITLE_TEMPLATE.format(
            model=self.model, attack=self.attack, class_pair=self.class_pair,
            budget=self.budget, num_poisons=self.num_poisons, name=self.name,
        )

    def budget_marks(self, k_max):
        """[(label, k, color)] for the dashed verticals of the overlap curve."""
        marks = []
        if SHOW_BUDGET_MARKS and self.n_total:
            for budget in BUDGET_MARKS:
                k = int(round(budget * self.n_total))
                if 1 <= k <= k_max:
                    marks.append((f"{budget:g}", k, BUDGET_MARK_COLOR))
        if SHOW_RUN_NP_MARK and 1 <= self.num_poisons <= k_max:
            marks.append((r"$N_p$", self.num_poisons, RUN_NP_MARK_COLOR))
        return marks


# =============================================================================
# STATISTICS
# =============================================================================

def _pearson(x, y):
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if len(x) < 2 or x.std() == 0 or y.std() == 0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def _rank(values):
    """0..1 percentile rank, 0 = smallest. Ties broken by position."""
    order = np.argsort(values, kind="stable")
    ranks = np.empty(len(values), dtype=np.float64)
    ranks[order] = np.arange(len(values))
    return ranks / max(len(values) - 1, 1)


def _spearman(x, y):
    try:
        from scipy.stats import spearmanr
        return float(spearmanr(x, y).correlation)
    except Exception:
        return _pearson(_rank(x), _rank(y))


def _best_first(score):
    """Indices ordered best first under the KEEP_LOWEST convention."""
    if KEEP_LOWEST:
        return np.argsort(score, kind="stable")
    return np.argsort(-score, kind="stable")


def _best_is_zero_rank(score):
    """Percentile rank with 0 = best under the KEEP_LOWEST convention."""
    return _rank(score) if KEEP_LOWEST else _rank(-score)


def _overlap_curve(cheap, full, ks):
    """|top-k by cheap  n  top-k by full| / k for every k in ks."""
    order_cheap = _best_first(cheap)
    order_full = _best_first(full)
    out = np.empty(len(ks), dtype=np.float64)
    for index, k in enumerate(ks):
        shared = np.intersect1d(order_cheap[:k], order_full[:k],
                                assume_unique=True)
        out[index] = len(shared) / float(k)
    return out


def _binned_quantiles(x_rank, y, n_bins):
    """Median and quartiles of y inside equal-count bins of x_rank (0..1)."""
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    index = np.clip(np.searchsorted(edges, x_rank, side="right") - 1,
                    0, n_bins - 1)
    median = np.full(n_bins, np.nan)
    low = np.full(n_bins, np.nan)
    high = np.full(n_bins, np.nan)
    for b in range(n_bins):
        mask = index == b
        if mask.any():
            low[b], median[b], high[b] = np.percentile(y[mask], [25, 50, 75])
    centers = (np.arange(n_bins) + 0.5) / n_bins
    return centers, median, low, high


# =============================================================================
# SHARED DRAWING
# =============================================================================

def _style_axes(ax):
    for side in ("left", "bottom"):
        ax.spines[side].set_visible(True)
        ax.spines[side].set_color(TEXT_COLOR)
        ax.spines[side].set_linewidth(SPINE_LINE_WIDTH)
    for side in ("right", "top"):
        ax.spines[side].set_visible(False)

    ax.tick_params(
        axis="both", which="major", direction="out",
        length=MAJOR_TICK_LENGTH, width=MAJOR_TICK_WIDTH, pad=TICK_LABEL_PAD,
        labelsize=TICK_FONT_SIZE, color=TEXT_COLOR, labelcolor=TEXT_COLOR,
        top=False, right=False,
    )
    if SHOW_GRID:
        ax.grid(visible=True, which="major", color=GRID_COLOR,
                linewidth=GRID_LINE_WIDTH, alpha=GRID_ALPHA, zorder=0)
    else:
        ax.grid(visible=False)
    ax.set_axisbelow(True)


def _add_legend(ax):
    if not SHOW_LEGEND:
        return None
    kwargs = {
        "loc": LEGEND_LOCATION,
        "ncol": LEGEND_NUMBER_OF_COLUMNS,
        "frameon": LEGEND_FRAME_VISIBLE,
        "prop": {"size": LEGEND_FONT_SIZE},
    }
    if LEGEND_BBOX is not None:
        kwargs["bbox_to_anchor"] = LEGEND_BBOX
    legend = ax.legend(**kwargs)
    frame = legend.get_frame()
    frame.set_facecolor(LEGEND_BACKGROUND_COLOR)
    frame.set_alpha(LEGEND_BACKGROUND_ALPHA)
    frame.set_edgecolor(LEGEND_BORDER_COLOR)
    frame.set_linewidth(LEGEND_BORDER_LINE_WIDTH)
    return legend


def _stats_box(ax, x, y):
    if not SHOW_STATS_BOX:
        return
    ax.text(
        0.03, 0.97,
        f"n = {len(x)}\nPearson = {_pearson(x, y):.3f}\n"
        f"Spearman = {_spearman(x, y):.3f}",
        transform=ax.transAxes, ha="left", va="top",
        fontsize=STATS_BOX_FONT_SIZE,
        bbox={"facecolor": "white", "alpha": 0.9, "edgecolor": "#CCCCCC",
              "boxstyle": "round,pad=0.35"},
    )


def _grid(n_panels):
    columns = max(1, min(GRID_COLUMNS, n_panels))
    rows = int(math.ceil(n_panels / float(columns)))
    fig, axes = plt.subplots(
        rows, columns,
        figsize=(PANEL_SIZE[0] * columns, PANEL_SIZE[1] * rows),
        dpi=FIGURE_DPI, squeeze=False, constrained_layout=True,
    )
    for index in range(n_panels, rows * columns):
        axes[index // columns][index % columns].set_axis_off()
    return fig, axes, rows, columns


def _save(fig, experiment, stem):
    common = {
        "dpi": SAVE_DPI, "bbox_inches": "tight", "pad_inches": 0.02,
        "facecolor": "white", "edgecolor": "white", "transparent": False,
    }
    if SAVE_PDF:
        destination = OUTPUT_ROOT_PDF / experiment.name
        destination.mkdir(parents=True, exist_ok=True)
        fig.savefig(destination / f"{stem}.pdf", **common)
        print(f"wrote {destination / f'{stem}.pdf'}")
    if SAVE_PNG:
        destination = OUTPUT_ROOT_PNG / experiment.name
        destination.mkdir(parents=True, exist_ok=True)
        fig.savefig(destination / f"{stem}.png", **common)
        print(f"wrote {destination / f'{stem}.png'}")


# =============================================================================
# FIGURE 1: raw scatter grid (saved random subset)
# =============================================================================

def plot_scatter_grid(experiment):
    fig, axes, rows, columns = _grid(len(experiment.targets))
    subset = experiment.subset_pos

    for index, target in enumerate(experiment.targets):
        ax = axes[index // columns][index % columns]
        x = experiment.cheap(target)[subset]
        y = experiment.a[target][subset]

        ax.scatter(x, y, s=POINT_SIZE, alpha=POINT_ALPHA, color=POINT_COLOR,
                   edgecolors="none", rasterized=True, zorder=3)
        ax.axhline(0.0, color=ZERO_LINE_COLOR, linewidth=ZERO_LINE_WIDTH, zorder=1)
        ax.axvline(0.0, color=ZERO_LINE_COLOR, linewidth=ZERO_LINE_WIDTH, zorder=1)
        if SCATTER_Y_SCALE == "symlog":
            ax.set_yscale("symlog", linthresh=SCATTER_SYMLOG_LINTHRESH)
        _style_axes(ax)
        _stats_box(ax, x, y)
        if SHOW_PANEL_TITLES:
            ax.set_title(PANEL_TITLE_TEMPLATE.format(target=target),
                         fontsize=PANEL_TITLE_FONT_SIZE)
        if index % columns == 0:
            ax.set_ylabel(SCATTER_Y_LABEL, fontsize=AXIS_LABEL_FONT_SIZE)
        if index // columns == rows - 1 or index + columns >= len(experiment.targets):
            ax.set_xlabel(SCATTER_X_LABEL, fontsize=AXIS_LABEL_FONT_SIZE)

    if SHOW_TITLES:
        fig.suptitle(experiment.title(), fontsize=TITLE_FONT_SIZE)
    _save(fig, experiment, "scatter_grid")
    return fig


# =============================================================================
# FIGURE 2: rank-rank grid (saved random subset, full-pool ranks)
# =============================================================================

def plot_rank_grid(experiment):
    fig, axes, rows, columns = _grid(len(experiment.targets))
    subset = experiment.subset_pos
    np_fraction = experiment.num_poisons / float(experiment.pool_size)

    for index, target in enumerate(experiment.targets):
        ax = axes[index // columns][index % columns]
        x = _best_is_zero_rank(experiment.cheap(target))[subset]
        y = _rank(experiment.a[target])[subset]

        ax.scatter(x, y, s=POINT_SIZE, alpha=POINT_ALPHA, color=POINT_COLOR,
                   edgecolors="none", rasterized=True, zorder=3)
        if SHOW_RUN_NP_MARK:
            ax.axvline(np_fraction, color=RUN_NP_MARK_COLOR,
                       linewidth=BUDGET_MARK_WIDTH, linestyle=BUDGET_MARK_STYLE)
        ax.set_xlim(0.0, 1.0)
        ax.set_ylim(0.0, 1.0)
        _style_axes(ax)
        _stats_box(ax, x, y)
        if SHOW_PANEL_TITLES:
            ax.set_title(PANEL_TITLE_TEMPLATE.format(target=target),
                         fontsize=PANEL_TITLE_FONT_SIZE)
        if index % columns == 0:
            ax.set_ylabel(RANK_Y_LABEL, fontsize=AXIS_LABEL_FONT_SIZE)
        if index // columns == rows - 1 or index + columns >= len(experiment.targets):
            ax.set_xlabel(RANK_X_LABEL, fontsize=AXIS_LABEL_FONT_SIZE)

    if SHOW_TITLES:
        fig.suptitle(experiment.title(), fontsize=TITLE_FONT_SIZE)
    _save(fig, experiment, "rank_grid")
    return fig


# =============================================================================
# FIGURE 3: A conditional on the cheap score (full pool)
# =============================================================================

def plot_conditional(experiment):
    fig, ax = plt.subplots(figsize=SINGLE_FIG_SIZE, dpi=FIGURE_DPI,
                           constrained_layout=True)

    medians, lows, highs = [], [], []
    centers = None
    for target in experiment.targets:
        x_rank = _best_is_zero_rank(experiment.cheap(target))
        centers, median, low, high = _binned_quantiles(
            x_rank, experiment.a[target], CONDITIONAL_BINS)
        medians.append(median)
        lows.append(low)
        highs.append(high)
        ax.plot(centers, median, color=TARGET_LINE_COLOR,
                alpha=TARGET_LINE_ALPHA, linewidth=TARGET_LINE_WIDTH,
                label=LEGEND_TARGET_LABEL if target == experiment.targets[0] else None)

    ax.fill_between(centers, np.nanmean(np.stack(lows), axis=0),
                    np.nanmean(np.stack(highs), axis=0),
                    color=IQR_FILL_COLOR, alpha=IQR_FILL_ALPHA, linewidth=0,
                    label=LEGEND_IQR_LABEL)
    ax.plot(centers, np.nanmean(np.stack(medians), axis=0),
            color=MEAN_LINE_COLOR, linewidth=MEAN_LINE_WIDTH,
            label=LEGEND_MEAN_LABEL.format(n=len(experiment.targets)))
    ax.axhline(0.0, color=ZERO_LINE_COLOR, linewidth=ZERO_LINE_WIDTH, zorder=1)
    if SHOW_RUN_NP_MARK:
        ax.axvline(experiment.num_poisons / float(experiment.pool_size),
                   color=RUN_NP_MARK_COLOR, linewidth=BUDGET_MARK_WIDTH,
                   linestyle=BUDGET_MARK_STYLE)

    ax.set_xlim(0.0, 1.0)
    _style_axes(ax)
    ax.set_xlabel(CONDITIONAL_X_LABEL, fontsize=AXIS_LABEL_FONT_SIZE)
    ax.set_ylabel(CONDITIONAL_Y_LABEL, fontsize=AXIS_LABEL_FONT_SIZE)
    if SHOW_TITLES:
        ax.set_title(experiment.title(), fontsize=TITLE_FONT_SIZE)
    _add_legend(ax)
    _save(fig, experiment, "conditional_a")
    return fig


# =============================================================================
# FIGURE 4 / 5: overlap of the (R,M) and (A,R,M) selections
# =============================================================================

def _overlap_curves(experiment):
    k_max = int(round(X_MAX_FRACTION * experiment.pool_size))
    k_max = max(1, min(k_max, experiment.pool_size))
    ks = np.arange(1, k_max + 1)
    curves = [
        _overlap_curve(experiment.cheap(target), experiment.full(target), ks)
        for target in experiment.targets
    ]
    return ks, curves


def plot_topk(experiment, log_x):
    ks, curves = _overlap_curves(experiment)
    fraction = ks / float(experiment.pool_size)
    mean = np.mean(np.stack(curves), axis=0)

    fig, ax = plt.subplots(figsize=SINGLE_FIG_SIZE, dpi=FIGURE_DPI,
                           constrained_layout=True)

    for index, curve in enumerate(curves):
        ax.plot(fraction, curve, color=TARGET_LINE_COLOR,
                alpha=TARGET_LINE_ALPHA, linewidth=TARGET_LINE_WIDTH,
                label=LEGEND_TARGET_LABEL if index == 0 else None)
    ax.plot(fraction, mean, color=MEAN_LINE_COLOR, linewidth=MEAN_LINE_WIDTH,
            label=LEGEND_MEAN_LABEL.format(n=len(experiment.targets)))
    ax.plot(fraction, fraction, color=CHANCE_LINE_COLOR,
            linewidth=CHANCE_LINE_WIDTH, linestyle=CHANCE_LINE_STYLE,
            label=LEGEND_CHANCE_LABEL)

    for label, k, color in experiment.budget_marks(int(ks[-1])):
        x = k / float(experiment.pool_size)
        ax.axvline(x, color=color, linewidth=BUDGET_MARK_WIDTH,
                   linestyle=BUDGET_MARK_STYLE, zorder=1)
        if SHOW_BUDGET_MARK_VALUES:
            text = f"{label}\n{mean[k - 1]:.2f}"
            ax.text(x, BUDGET_MARK_TEXT_Y, text, transform=ax.get_xaxis_transform(),
                    ha="center", va="bottom", fontsize=ANNOTATION_FONT_SIZE,
                    color=BUDGET_MARK_TEXT_COLOR)

    if log_x:
        ax.set_xscale("log")
        x_min = LOG_X_MIN_FRACTION or fraction[0]
        ax.set_xlim(x_min, fraction[-1])
    else:
        ax.set_xlim(0.0, fraction[-1])

    ax.set_ylim(*TOPK_Y_LIMITS)
    ax.set_yticks(TOPK_Y_TICKS)
    if TOPK_Y_AS_PERCENT:
        ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=0))
    else:
        ax.yaxis.set_major_formatter(FormatStrFormatter("%.1f"))
    ax.yaxis.set_minor_locator(NullLocator())

    _style_axes(ax)
    ax.set_xlabel(TOPK_X_LABEL, fontsize=AXIS_LABEL_FONT_SIZE)
    ax.set_ylabel(TOPK_Y_LABEL, fontsize=AXIS_LABEL_FONT_SIZE)
    if SHOW_TITLES:
        ax.set_title(experiment.title(), fontsize=TITLE_FONT_SIZE)
    _add_legend(ax)
    _save(fig, experiment, "topk_overlap_log" if log_x else "topk_overlap_linear")
    return fig


# =============================================================================
# JUPYTER EXECUTION
# =============================================================================

_configure_matplotlib()

all_figures = {}
for _directory in _discover_experiments():
    _experiment = Experiment(_directory)
    print(f"=== {_experiment.name}: pool {_experiment.pool_size}, "
          f"N_p {_experiment.num_poisons}, targets {_experiment.targets}")
    _figs = {}
    if RUN_SCATTER_GRID:
        _figs["scatter_grid"] = plot_scatter_grid(_experiment)
    if RUN_RANK_GRID:
        _figs["rank_grid"] = plot_rank_grid(_experiment)
    if RUN_CONDITIONAL:
        _figs["conditional_a"] = plot_conditional(_experiment)
    if RUN_TOPK_LINEAR:
        _figs["topk_overlap_linear"] = plot_topk(_experiment, log_x=False)
    if RUN_TOPK_LOG:
        _figs["topk_overlap_log"] = plot_topk(_experiment, log_x=True)
    all_figures[_experiment.name] = _figs

if SHOW_PLOTS:
    plt.show()

if CLOSE_FIGURES_AFTER_SHOW:
    for _figs in all_figures.values():
        for _fig in _figs.values():
            plt.close(_fig)
