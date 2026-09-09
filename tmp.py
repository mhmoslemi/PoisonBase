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

Across all models selected by MODEL_FILTER it exports one combined
topk_overlap_linear figure and/or one combined topk_overlap_log figure. Each
model is represented by a thick mean curve and faint individual-target curves.

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
from matplotlib.ticker import (
    FormatStrFormatter,
    LogFormatterMathtext,
    NullLocator,
    PercentFormatter,
)


# =============================================================================
# USER CONTROLS
# =============================================================================

RUN_SCATTER_GRID = False
RUN_CONDITIONAL = False
RUN_TOPK_LOG_Y = False


RUN_RANK_GRID = True
RUN_TOPK_LINEAR = False
RUN_TOPK_LOG = False

RUN_TOPK_STACKED = False   # top: log-x overlap; bottom: linear-x overlap
RUN_TOPK_SIDE_BY_SIDE = True  # left: linear-x overlap; right: log-x overlap

# Rank figures to export, keyed by (dataset, model). An empty mapping restores
# the default behavior of plotting every target selected by TARGET_FILTER.
RANK_TARGET_BY_DATASET_MODEL = {
    ("CIFAR10", "ConvNetBN"): 5705,
    ("CIFAR10", "ResNet20BN"): 6697,
    ("CIFAR10", "VGG13BN"): 2127,
}

# Rank-plot y-label controls. False keeps the label but makes it white so all
# three saved figures retain the same layout and spacing.
RANK_SHOW_Y_LABEL = {
    "ConvNetBN": True,
    "ResNet20BN": False,
    "VGG13BN": False,
}
RANK_Y_LABEL_ROTATION = {
    "ConvNetBN": 90,
    "ResNet20BN": 90,
    "VGG13BN": 90,
}
RANK_HIDDEN_Y_LABEL_COLOR = "white"

# Rank y-label position in axes coordinates.
# More-negative X moves it left; smaller Y moves it down.
RANK_Y_LABEL_FONT_SIZE = 20
RANK_X_LABEL_FONT_SIZE = 20
RANK_Y_LABEL_X =-0.31
RANK_Y_LABEL_Y = 0.43

# Where the SCATTER_* directories are, and which of them to plot. The filter
# is a tuple of substrings; an experiment is plotted when its directory name
# contains ANY of them. () plots everything.
RESULTS_ROOT = Path("scatter_result")
EXPERIMENT_FILTER = ()
# EXPERIMENT_FILTER = ("ConvNetBN_fc_ours_dog-bird",)

# Which datasets to plot, e.g. ("CIFAR100", "SVHN"). A directory name is
# SCATTER_<dataset>_<model>_..., so this matches the token right after
# SCATTER_. () plots every dataset found under RESULTS_ROOT.
DATASET_FILTER = ("CIFAR10",)
# DATASET_FILTER = ("CIFAR100",)

# Which models to plot, e.g. ("ConvNetBN", "ResNet20BN"). A directory name is
# SCATTER_<dataset>_<model>_..., so this matches the second token. () plots
# every model found under RESULTS_ROOT.
MODEL_FILTER = ("ConvNetBN", "ResNet20BN", "VGG13BN")
# MODEL_FILTER = ("ResNet20BN", "VGG13BN")

# Which targets to draw. () = every target saved in summary.json; otherwise a
# tuple of target indices, in the order they should appear.
TARGET_FILTER = ()

# Numeric plot-data cache. The first run writes reusable NPZ files; subsequent
# runs load them so visual-only tuning does not recompute ranks, overlaps, or
# the random baseline. Set REBUILD_PLOT_DATA_CACHE=True after changing score
# definitions, or increment PLOT_DATA_CACHE_VERSION for manual invalidation.
USE_PLOT_DATA_CACHE = True
SAVE_PLOT_DATA_CACHE = True
REBUILD_PLOT_DATA_CACHE = False
PLOT_DATA_CACHE_VERSION = 1
PLOT_DATA_ROOT = Path("figures/data/scatter")

# =============================================================================
# OVERLAP PLOT CONTROLS — tune the complete overlap plot here
# =============================================================================

# --- Figure canvas and margins ----------------------------------------------
TOPK_FIG_SIZE = (5.0, 3.35)            # width, height in inches
TOPK_FIGURE_DPI = 150                  # notebook/display resolution
TOPK_FIGURE_FACE_COLOR = "white"
TOPK_AXES_FACE_COLOR = "white"
TOPK_SUBPLOT_LEFT = 0.10               # fraction of figure width
TOPK_SUBPLOT_RIGHT = 0.985
TOPK_SUBPLOT_BOTTOM = 0.27             # fraction of figure height
TOPK_SUBPLOT_TOP = 0.97

# Third overlap export: subplot(211) log-x above subplot(212) linear-x.
TOPK_STACKED_FIG_SIZE = (5.0, 6.7)
TOPK_STACKED_SUBPLOT_LEFT = 0.10
TOPK_STACKED_SUBPLOT_RIGHT = 0.985
TOPK_STACKED_SUBPLOT_BOTTOM = 0.12
TOPK_STACKED_SUBPLOT_TOP = 0.985
TOPK_STACKED_VERTICAL_SPACE = 0.22      # hspace: distance between the panels
TOPK_STACKED_HEIGHT_RATIOS = (1.0, 1.0)  # top height, bottom height
TOPK_STACKED_TOP_SHOW_X_LABEL = False  # top keeps ticks, hides only x label
TOPK_STACKED_BOTTOM_SHOW_X_LABEL = True
TOPK_STACKED_TOP_SHOW_LEGEND = False
TOPK_STACKED_BOTTOM_SHOW_LEGEND = True
TOPK_STACKED_OUTPUT_STEM = "topk_overlap_log_and_linear"

# Fourth overlap export: subplot(121) linear-x beside subplot(122) log-x.
# These are the exact axes widths and gap in inches. The complete figure width
# and Matplotlib's internal width ratio/wspace are calculated automatically.
TOPK_SIDE_BY_SIDE_LINEAR_WIDTH_INCHES = 3.3
TOPK_SIDE_BY_SIDE_LOG_WIDTH_INCHES = 2
TOPK_SIDE_BY_SIDE_GAP_INCHES = 0.6
TOPK_SIDE_BY_SIDE_FIG_HEIGHT_INCHES = 2.5
# Blank canvas appended above the completed panels so the top y tick is kept.
# This is side-by-side-only and does not move or resize either axes.
TOPK_SIDE_BY_SIDE_TOP_PADDING_POINTS = 8.0 # d right panel. This setting is
# side-by-side-only and does not resize or move either axes.
TOPK_SIDE_BY_SIDE_RIGHT_PADDING_POINTS = 5.0  # 72 points = 1 inch
# Leave enough canvas for the 20 pt y ticks and "Selection Overlap" label.
# The width calculation below preserves both requested axes widths.
TOPK_SIDE_BY_SIDE_SUBPLOT_LEFT = 0.12
TOPK_SIDE_BY_SIDE_SUBPLOT_RIGHT = 0.985
TOPK_SIDE_BY_SIDE_SUBPLOT_BOTTOM = 0.27
TOPK_SIDE_BY_SIDE_SUBPLOT_TOP = 0.97
TOPK_SIDE_BY_SIDE_LEFT_SHOW_X_LABEL = True
TOPK_SIDE_BY_SIDE_RIGHT_SHOW_X_LABEL = True
TOPK_SIDE_BY_SIDE_LEFT_SHOW_Y_LABEL = True    # 121/linear
TOPK_SIDE_BY_SIDE_RIGHT_SHOW_Y_LABEL = False  # 122/log: omitted entirely
TOPK_SIDE_BY_SIDE_LEFT_SHOW_LEGEND = True
TOPK_SIDE_BY_SIDE_RIGHT_SHOW_LEGEND = False
TOPK_SIDE_BY_SIDE_LINEAR_X_MAX_RHO = 0.041    # displayed maximum is 40
TOPK_SIDE_BY_SIDE_LINEAR_X_TICK_RHO_VALUES = (
    0.001, 0.005, 0.01, 0.02, 0.04,
)
TOPK_SIDE_BY_SIDE_OUTPUT_STEM = "topk_overlap_log_and_linear_side_by_side"

# --- X axis: rho displayed in units of 10^-3 -------------------------------
# Curves start as fractions of the poison-class pool. Dividing that fraction
# by 10 converts it to dataset-level rho for CIFAR-10; dividing rho by 1e-3
# then makes rho=0.001 appear as the tick label 1.
TOPK_X_LABEL = r"$\rho\; (\times 10^{-3})$"
TOPK_POOL_FRACTION_TO_RHO_DIVISOR = 10.0
TOPK_RHO_UNIT = 1e-3
TOPK_X_MIN_RHO = 0.0002                 # linear left limit; None starts at 0
TOPK_LOG_X_MIN_RHO = TOPK_X_MIN_RHO
TOPK_X_MAX_RHO = 0.04                   # None uses all available data
TOPK_X_TICK_RHO_VALUES = (0.001, 0.005, 0.01, 0.02, 0.04)
TOPK_X_TICK_LABEL_FORMAT = "g"          # gives 1, 5, 10, 20, 40
# Log plot extent and visible ticks are independent. Increasing MAX_RHO draws
# farther right, while the displayed tick labels remain fixed at 1, 5, and 15.
TOPK_LOG_X_MAX_RHO = 0.017
TOPK_LOG_X_TICK_RHO_VALUES = (
    0.001,
    0.005,
    0.015,
)
# White canvas added OUTSIDE the log plot's right edge, measured in points.
# It does not alter x limits, ticks, curves, grid, or the physical axes width.
# 72 points = 1 inch. Use 0.0 for no extra white canvas.
TOPK_LOG_X_RIGHT_PADDING = 0.0
TOPK_X_LOG_BASE = 10
TOPK_X_TICK_ROTATION = 0
TOPK_X_TICK_HORIZONTAL_ALIGNMENT = "center"
TOPK_X_TICK_VERTICAL_ALIGNMENT = "top"

# --- Y axis -----------------------------------------------------------------
TOPK_Y_LABEL = "Selection Overlap"
TOPK_Y_LIMITS = (0.0, 1.02)             # linear-y limits
TOPK_Y_TICKS = (0.0, 0.5, 1.0)
TOPK_Y_TICK_FORMAT = "%.1f"
TOPK_Y_AS_PERCENT = False
TOPK_Y_PERCENT_DECIMALS = 0
TOPK_LOG_Y_LIMITS = (1e-3, 1.05)
TOPK_LOG_Y_TICKS = (1e-3, 1e-2, 1e-1, 1.0)
TOPK_Y_LOG_BASE = 10
TOPK_SHOW_MINOR_TICKS = False

# --- Axis-label and tick typography -----------------------------------------
TOPK_X_LABEL_FONT_SIZE = 20
TOPK_Y_LABEL_FONT_SIZE = 20
TOPK_X_LABEL_FONT_WEIGHT = "normal"
TOPK_Y_LABEL_FONT_WEIGHT = "normal"
TOPK_X_LABEL_FONT_STYLE = "normal"
TOPK_Y_LABEL_FONT_STYLE = "normal"
TOPK_X_LABEL_COLOR = "#111111"
TOPK_Y_LABEL_COLOR = "#111111"
TOPK_X_LABEL_PAD = 4.0                  # points from x tick labels
TOPK_Y_LABEL_PAD = 4.0                  # points from y tick labels
TOPK_X_TICK_FONT_SIZE = 20
TOPK_Y_TICK_FONT_SIZE = 20
TOPK_X_TICK_FONT_WEIGHT = "normal"
TOPK_Y_TICK_FONT_WEIGHT = "normal"
TOPK_X_TICK_FONT_STYLE = "normal"
TOPK_Y_TICK_FONT_STYLE = "normal"
TOPK_X_TICK_COLOR = "#111111"
TOPK_Y_TICK_COLOR = "#111111"
TOPK_TICK_DIRECTION = "out"
TOPK_X_MAJOR_TICK_LENGTH = 3.5
TOPK_Y_MAJOR_TICK_LENGTH = 3.5
TOPK_X_MAJOR_TICK_WIDTH = 0.8
TOPK_Y_MAJOR_TICK_WIDTH = 0.8
TOPK_X_TICK_LABEL_PAD = 2.5
TOPK_Y_TICK_LABEL_PAD = 2.5
TOPK_Y_TICK_ROTATION = 0
TOPK_Y_TICK_HORIZONTAL_ALIGNMENT = "right"
TOPK_Y_TICK_VERTICAL_ALIGNMENT = "center"
TOPK_SHOW_BOTTOM_TICKS = True
TOPK_SHOW_TOP_TICKS = False
TOPK_SHOW_LEFT_TICKS = True
TOPK_SHOW_RIGHT_TICKS = False
TOPK_SHOW_BOTTOM_TICK_LABELS = True
TOPK_SHOW_TOP_TICK_LABELS = False
TOPK_SHOW_LEFT_TICK_LABELS = True
TOPK_SHOW_RIGHT_TICK_LABELS = False

# --- Spines and grid ---------------------------------------------------------
TOPK_SPINE_VISIBILITY = {
    "left": True,
    "right": False,
    "bottom": True,
    "top": False,
}
TOPK_SPINE_COLOR = "#111111"
TOPK_SPINE_LINE_WIDTH = 1.2
TOPK_SHOW_GRID = True
TOPK_GRID_AXIS = "both"                # "x", "y", or "both"
TOPK_GRID_WHICH = "major"              # "major", "minor", or "both"
TOPK_GRID_COLOR = "#D9D9D9"
TOPK_GRID_LINE_STYLE = "-"
TOPK_GRID_LINE_WIDTH = 0.45
TOPK_GRID_ALPHA = 0.75
TOPK_AXIS_BELOW_GRID = True

# --- Model mean curves and individual-target curves -------------------------
TOPK_MODEL_COLORS = {
    "ConvNetBN": "#4C78A8",
    "ResNet20BN": "#59A14F",
    "VGG13BN": "#E15759",
}
TOPK_MEAN_LINE_STYLE = "-"
TOPK_MEAN_LINE_WIDTH = 1.0
TOPK_MEAN_LINE_ALPHA = 1.0
TOPK_MEAN_LINE_ZORDER = 4
TOPK_TARGET_LINE_STYLE = ":"
TOPK_TARGET_LINE_WIDTH = 0.5
TOPK_TARGET_LINE_ALPHA = 0.2
TOPK_TARGET_LINE_ZORDER = 2
TOPK_LINE_ANTIALIASED = True

# --- Empirical random baseline ----------------------------------------------
RANDOM_BASELINE_REPEATS = 100
RANDOM_BASELINE_SEED = 2026
SHOW_RANDOM_TRIALS = False
SHOW_RANDOM_STD_BAND = False
TOPK_RANDOM_TRIAL_COLOR = "#8C8C8C"
TOPK_RANDOM_TRIAL_LINE_STYLE = ":"
TOPK_RANDOM_TRIAL_LINE_WIDTH = 0.7
TOPK_RANDOM_TRIAL_ALPHA = 0.22
TOPK_RANDOM_TRIAL_ZORDER = 1
TOPK_RANDOM_MEAN_COLOR = "#333333"
TOPK_RANDOM_MEAN_LINE_STYLE = ":"
TOPK_RANDOM_MEAN_LINE_WIDTH = 1.5  # dedicated random-curve width control
TOPK_RANDOM_MEAN_ALPHA = 1.0
TOPK_RANDOM_MEAN_ZORDER = 3
TOPK_RANDOM_STD_COLOR = "#A6A6A6"
TOPK_RANDOM_STD_ALPHA = 0.28
TOPK_RANDOM_STD_LINE_WIDTH = 0.0
TOPK_RANDOM_STD_ZORDER = 1
TOPK_RANDOM_BAND_UPPER_CLIP = 1.0

# --- Optional vertical poison-budget marks ----------------------------------
BUDGET_MARKS = (0.001,)                 # true rho fractions
SHOW_BUDGET_MARKS = False
SHOW_BUDGET_MARK_VALUES = True
SHOW_RUN_NP_MARK = True
TOPK_BUDGET_MARK_COLOR = "#B0B0B0"
TOPK_BUDGET_MARK_LINE_WIDTH = 0.7
TOPK_BUDGET_MARK_LINE_STYLE = "--"
TOPK_BUDGET_MARK_ZORDER = 1
TOPK_BUDGET_TEXT_COLOR = "#555555"
TOPK_BUDGET_TEXT_Y = 0.02               # axes-height fraction
TOPK_BUDGET_TEXT_FONT_SIZE = 11
TOPK_BUDGET_TEXT_HORIZONTAL_ALIGNMENT = "center"
TOPK_BUDGET_TEXT_VERTICAL_ALIGNMENT = "bottom"
TOPK_BUDGET_TEXT_FORMAT = "g"           # uses displayed 10^-3 units
TOPK_RUN_NP_MARK_COLOR = "#E69F00"

# --- Legend -----------------------------------------------------------------
TOPK_SHOW_LEGEND = True
TOPK_LEGEND_MODEL_LABEL = "{model}"
TOPK_LEGEND_LOCATION = "lower right"
TOPK_LEGEND_BBOX = (0.82, 0.1)
TOPK_LEGEND_NUMBER_OF_COLUMNS = 2
TOPK_LEGEND_FONT_SIZE = 15
TOPK_LEGEND_FONT_WEIGHT = "normal"
TOPK_LEGEND_FONT_STYLE = "normal"
TOPK_LEGEND_TEXT_COLOR = "#111111"
TOPK_LEGEND_FRAME_VISIBLE = False
TOPK_LEGEND_BACKGROUND_COLOR = "white"
TOPK_LEGEND_BACKGROUND_ALPHA = 0.0
TOPK_LEGEND_BORDER_COLOR = "#777777"
TOPK_LEGEND_BORDER_LINE_WIDTH = 0.0
TOPK_LEGEND_BORDER_PAD = 0.4
TOPK_LEGEND_BORDER_AXES_PAD = 0.5
TOPK_LEGEND_LABEL_SPACING = 0.5
TOPK_LEGEND_COLUMN_SPACING = 0.5
TOPK_LEGEND_HANDLE_LENGTH = 1
TOPK_LEGEND_HANDLE_LINE_WIDTH = 1.5
TOPK_LEGEND_HANDLE_TEXT_PAD = 0.8
TOPK_LEGEND_MARKER_SCALE = 1.0

# --- Export -----------------------------------------------------------------
TOPK_SAVE_PDF = True
TOPK_OUTPUT_ROOT_PDF = Path("figures/pdf/scatter")
TOPK_SAVE_DPI = 500
TOPK_SAVE_BBOX_INCHES = "tight"         # None disables tight cropping
TOPK_SAVE_PAD_INCHES = 0.02
TOPK_SAVE_FACE_COLOR = "white"
TOPK_SAVE_EDGE_COLOR = "white"
TOPK_SAVE_TRANSPARENT = False

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
# target, with the arrays final_update_scatter.py's scatter_components() saved:
#   rm     = mean over surrogates of  std(d) - std(M),  d = 1 / ||f(x)-f(x_t)||^2
#   a      = mean over surrogates of  std(A)
# d_raw, m_raw, a_raw = the same three, NOT standardized. With d defined as a
# reciprocal squared distance, both rm and A are GAINS: larger is closer /
# more confident / more helpful, and the real selector keeps the LARGEST
# rm + beta * A (equivalently the smallest -rm - beta * A). This matches
# final_update_scatter.py's own topk_overlap curve exactly -- see its
# scatter_main(), which ranks by `-rm_np` (cheap) and `-rm_np - a_np` (full)
# and keeps the lowest of each; verified bit-for-bit against the
# topk_overlap_vs_full_score array it also saves into components.npz.
# KEEP_LOWEST = True  -> both functions return a COST and the k smallest win.
# KEEP_LOWEST = False -> both return a GAIN and the k largest win.
# -----------------------------------------------------------------------------

BETA = 1.0
KEEP_LOWEST = False


def cheap_score(rm, a, d_raw, m_raw, a_raw):
    """(R, M) ranking. Default: the saved gain rm = std(d) - std(M)."""
    return rm


def full_score(rm, a, d_raw, m_raw, a_raw):
    """(A, R, M) ranking. Default: rm + beta * A, the real selector."""
    return rm + BETA * a


# =============================================================================
# LABELS
# =============================================================================

SHOW_PANEL_TITLES = True

SCATTER_X_LABEL = r"$R + M$"
SCATTER_Y_LABEL = r"$A$"

RANK_X_LABEL = r"$R + M$"
RANK_Y_LABEL = r"$A$"

CONDITIONAL_X_LABEL = "quantile of the (R, M) score  (0 = best)"
CONDITIONAL_Y_LABEL = r"median $A_i$ in bin"

MODEL_LABELS = {
    "ConvNetBN": "ConvNet",
    "ResNet20BN": "ResNet20",
    "VGG13BN": "VGG13",
}
LEGEND_TARGET_LABEL = "Target runs"
LEGEND_RANDOM_STD_LABEL = r"Random $\pm$ 1 s.d."
LEGEND_IQR_LABEL = "inter-quartile band"

SHOW_STATS_BOX = True          # n / Pearson / Spearman box in scatter panels
STATS_BOX_FONT_SIZE = 7


# =============================================================================
# FIGURE STYLE
# =============================================================================

# All four exports use the same source size so they align cleanly when placed
# at equal widths in a one-row ICLR figure.
PANEL_SIZE = (3.375, 3.0)
FIGURE_DPI = 150

FONT_FAMILY = "serif"
SERIF_FONTS = (
    "Times New Roman",
    "Times",
    "Nimbus Roman No9 L",
    "DejaVu Serif",
)
MATH_FONTSET = "dejavuserif"

BASE_FONT_SIZE = 14
AXIS_LABEL_FONT_SIZE = 18
TICK_FONT_SIZE = 22
TITLE_FONT_SIZE = 18
PANEL_TITLE_FONT_SIZE = 16
LEGEND_FONT_SIZE = 11.5
ANNOTATION_FONT_SIZE = 11
TEXT_COLOR = "#111111"

SPINE_LINE_WIDTH = 1.6
SHOW_LEFT_SPINE = True
SHOW_RIGHT_SPINE = False
SHOW_BOTTOM_SPINE = True
SHOW_TOP_SPINE = False
MAJOR_TICK_LENGTH = 8.0
MAJOR_TICK_WIDTH = 1.5
TICK_LABEL_PAD = 5.0

SHOW_GRID = False
GRID_COLOR = "#D5D8DC"
GRID_LINE_WIDTH = 0.0
GRID_ALPHA = 0.0

# Marks
POINT_SIZE = 2.5
POINT_ALPHA = 0.65

MODEL_COLORS = {
    "ConvNetBN": "#0072B2",
    "ResNet20BN": "#FF2C00",
    "VGG13BN": "#009E73",
}
IQR_FILL_COLOR = "#2a78d6"
IQR_FILL_ALPHA = 0.12

ZERO_LINE_COLOR = "#B0B0B0"
ZERO_LINE_WIDTH = 0.8


# =============================================================================
# LEGEND AND EXPORT
# =============================================================================

SHOW_LEGEND = True
LEGEND_LOCATION = "lower left"
LEGEND_BBOX = None             # e.g. (0.98, 0.05); None = matplotlib default
LEGEND_FRAME_VISIBLE = True
LEGEND_BACKGROUND_COLOR = "white"
LEGEND_BACKGROUND_ALPHA = 0.95
LEGEND_BORDER_COLOR = "#777777"
LEGEND_BORDER_LINE_WIDTH = 1.0
LEGEND_NUMBER_OF_COLUMNS = 2

SAVE_PDF = True
SHOW_PLOTS = True
CLOSE_FIGURES_AFTER_SHOW = False

OUTPUT_ROOT_PDF = Path("figures/pdf/scatter")
OUTPUT_ROOT_PNG = Path("figures/png/scatter")
# SAVE_DPI = 300 /


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
            "savefig.dpi": TOPK_SAVE_DPI,
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
        if DATASET_FILTER:
            dataset = directory.name[len("SCATTER_"):].split("_", 1)[0]
            if dataset not in DATASET_FILTER:
                continue
        if MODEL_FILTER:
            model = directory.name[len("SCATTER_"):].split("_", 2)[1]
            if model not in MODEL_FILTER:
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

        summary_targets = [int(t) for t in self.summary["targets"]]
        component_targets = []
        for key in data.files:
            if not key.startswith("rm_target"):
                continue
            suffix = key[len("rm_target"):]
            if suffix.isdigit():
                component_targets.append(int(suffix))
        saved_targets = list(dict.fromkeys(summary_targets + component_targets))

        # Some result formats key component arrays by a compact target number
        # while storing the original CIFAR target ID in a parallel metadata
        # array. Accept either representation when selecting a rank target.
        self.target_alias_to_saved = {target: target for target in saved_targets}

        def register_target_aliases(name, values):
            if "target" not in name.lower():
                return
            try:
                aliases = np.asarray(values)
            except Exception:
                return
            if aliases.ndim != 1 or len(aliases) != len(summary_targets):
                return
            for saved_target, alias in zip(summary_targets, aliases):
                try:
                    self.target_alias_to_saved[int(alias)] = saved_target
                except (TypeError, ValueError):
                    continue

        for name, values in self.summary.items():
            register_target_aliases(name, values)
        for name in data.files:
            register_target_aliases(name, data[name])
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
        self.dataset = self.name.split("_", 1)[0]
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

    def resolve_target(self, target):
        """Resolve an original dataset target ID to its saved component key."""
        return self.target_alias_to_saved.get(int(target))



    def budget_marks(self, k_max):
        """[(label, k, color)] for the dashed verticals of the overlap curve."""
        marks = []
        if SHOW_BUDGET_MARKS and self.n_total:
            for budget in BUDGET_MARKS:
                k = int(round(budget * self.n_total))
                if 1 <= k <= k_max:
                    marks.append((f"{budget:g}", k, TOPK_BUDGET_MARK_COLOR))
        # if SHOW_RUN_NP_MARK and 1 <= self.num_poisons <= k_max:
        #     marks.append(("", 2000, TOPK_RUN_NP_MARK_COLOR))
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


def _ordered_overlap_curve(order_left, order_right, ks):
    """Prefix-set overlap for two candidate orderings, computed in one sweep."""
    max_k = int(ks[-1])
    in_left = np.zeros(len(order_left), dtype=bool)
    in_right = np.zeros(len(order_right), dtype=bool)
    overlap = np.empty(max_k, dtype=np.float64)
    shared = 0

    for index in range(max_k):
        left = order_left[index]
        right = order_right[index]
        if in_right[left]:
            shared += 1
        in_left[left] = True
        if in_left[right] and not in_right[right]:
            shared += 1
        in_right[right] = True
        overlap[index] = shared / float(index + 1)

    return overlap[np.asarray(ks) - 1]


def _overlap_curve(cheap, full, ks):
    """|top-k by cheap intersect top-k by full| / k for every k in ks."""
    return _ordered_overlap_curve(_best_first(cheap), _best_first(full), ks)


# =============================================================================
# SHARED DRAWING
# =============================================================================

def _style_axes(ax):
    spine_visibility = {
        "left": SHOW_LEFT_SPINE,
        "right": SHOW_RIGHT_SPINE,
        "bottom": SHOW_BOTTOM_SPINE,
        "top": SHOW_TOP_SPINE,
    }
    for side, visible in spine_visibility.items():
        ax.spines[side].set_visible(visible)
        ax.spines[side].set_color(TEXT_COLOR)
        ax.spines[side].set_linewidth(SPINE_LINE_WIDTH)

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


def _style_topk_axes(ax):
    """Apply only the controls in OVERLAP PLOT CONTROLS."""
    ax.set_facecolor(TOPK_AXES_FACE_COLOR)
    for side, visible in TOPK_SPINE_VISIBILITY.items():
        ax.spines[side].set_visible(visible)
        ax.spines[side].set_color(TOPK_SPINE_COLOR)
        ax.spines[side].set_linewidth(TOPK_SPINE_LINE_WIDTH)

    ax.tick_params(
        axis="x", which="major", direction=TOPK_TICK_DIRECTION,
        length=TOPK_X_MAJOR_TICK_LENGTH,
        width=TOPK_X_MAJOR_TICK_WIDTH,
        pad=TOPK_X_TICK_LABEL_PAD,
        labelsize=TOPK_X_TICK_FONT_SIZE,
        color=TOPK_X_TICK_COLOR,
        labelcolor=TOPK_X_TICK_COLOR,
        bottom=TOPK_SHOW_BOTTOM_TICKS,
        top=TOPK_SHOW_TOP_TICKS,
        labelbottom=TOPK_SHOW_BOTTOM_TICK_LABELS,
        labeltop=TOPK_SHOW_TOP_TICK_LABELS,
    )
    ax.tick_params(
        axis="y", which="major", direction=TOPK_TICK_DIRECTION,
        length=TOPK_Y_MAJOR_TICK_LENGTH,
        width=TOPK_Y_MAJOR_TICK_WIDTH,
        pad=TOPK_Y_TICK_LABEL_PAD,
        labelsize=TOPK_Y_TICK_FONT_SIZE,
        color=TOPK_Y_TICK_COLOR,
        labelcolor=TOPK_Y_TICK_COLOR,
        left=TOPK_SHOW_LEFT_TICKS,
        right=TOPK_SHOW_RIGHT_TICKS,
        labelleft=TOPK_SHOW_LEFT_TICK_LABELS,
        labelright=TOPK_SHOW_RIGHT_TICK_LABELS,
    )
    for label in ax.get_xticklabels():
        label.set_fontweight(TOPK_X_TICK_FONT_WEIGHT)
        label.set_fontstyle(TOPK_X_TICK_FONT_STYLE)
    for label in ax.get_yticklabels():
        label.set_fontweight(TOPK_Y_TICK_FONT_WEIGHT)
        label.set_fontstyle(TOPK_Y_TICK_FONT_STYLE)
        label.set_rotation(TOPK_Y_TICK_ROTATION)
        label.set_horizontalalignment(TOPK_Y_TICK_HORIZONTAL_ALIGNMENT)
        label.set_verticalalignment(TOPK_Y_TICK_VERTICAL_ALIGNMENT)

    if not TOPK_SHOW_MINOR_TICKS:
        ax.xaxis.set_minor_locator(NullLocator())
        ax.yaxis.set_minor_locator(NullLocator())

    if TOPK_SHOW_GRID:
        ax.grid(
            visible=True,
            axis=TOPK_GRID_AXIS,
            which=TOPK_GRID_WHICH,
            color=TOPK_GRID_COLOR,
            linestyle=TOPK_GRID_LINE_STYLE,
            linewidth=TOPK_GRID_LINE_WIDTH,
            alpha=TOPK_GRID_ALPHA,
            zorder=0,
        )
    else:
        ax.grid(False)
    ax.set_axisbelow(TOPK_AXIS_BELOW_GRID)


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


def _add_topk_legend(ax, visible=None):
    """Overlap-plot legend, controlled entirely by the TOPK settings."""
    if visible is None:
        visible = TOPK_SHOW_LEGEND
    if not visible:
        return None
    kwargs = {
        "loc": TOPK_LEGEND_LOCATION,
        "ncol": TOPK_LEGEND_NUMBER_OF_COLUMNS,
        "frameon": TOPK_LEGEND_FRAME_VISIBLE,
        "prop": {
            "size": TOPK_LEGEND_FONT_SIZE,
            "weight": TOPK_LEGEND_FONT_WEIGHT,
            "style": TOPK_LEGEND_FONT_STYLE,
        },
        "borderpad": TOPK_LEGEND_BORDER_PAD,
        "borderaxespad": TOPK_LEGEND_BORDER_AXES_PAD,
        "labelspacing": TOPK_LEGEND_LABEL_SPACING,
        "columnspacing": TOPK_LEGEND_COLUMN_SPACING,
        "handlelength": TOPK_LEGEND_HANDLE_LENGTH,
        "handletextpad": TOPK_LEGEND_HANDLE_TEXT_PAD,
        "markerscale": TOPK_LEGEND_MARKER_SCALE,
    }
    if TOPK_LEGEND_BBOX is not None:
        kwargs["bbox_to_anchor"] = TOPK_LEGEND_BBOX
    legend = ax.legend(**kwargs)
    frame = legend.get_frame()
    frame.set_facecolor(TOPK_LEGEND_BACKGROUND_COLOR)
    frame.set_alpha(TOPK_LEGEND_BACKGROUND_ALPHA)
    frame.set_edgecolor(TOPK_LEGEND_BORDER_COLOR)
    frame.set_linewidth(TOPK_LEGEND_BORDER_LINE_WIDTH)
    for line_handle in legend.get_lines():
        line_handle.set_linewidth(TOPK_LEGEND_HANDLE_LINE_WIDTH)
    for text_item in legend.get_texts():
        text_item.set_color(TOPK_LEGEND_TEXT_COLOR)
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
        "dpi": 500, "bbox_inches": None, "pad_inches": 0.0,
        "facecolor": "white", "edgecolor": "white", "transparent": False,
    }
    if SAVE_PDF:
        destination = '/Users/mohammad/Downloads/PoisonBase/figures/pdf/scatter'
        # destination.mkdir(parents=True, exist_ok=True)
        fig.savefig( '/Users/mohammad/Downloads/PoisonBase/figures/pdf/scatter' +f"{stem}.pdf", **common)
        # print(f"wrote {destination / f'{stem}.pdf'}")


def _right_padded_canvas(base_width, subplot_left, subplot_right, padding_points):
    """Append white canvas on the right without moving or resizing the axes."""
    if padding_points < 0:
        raise ValueError("right canvas padding cannot be negative")
    padded_width = base_width + padding_points / 72.0
    scale = base_width / padded_width
    return padded_width, subplot_left * scale, subplot_right * scale


def _top_padded_canvas(base_height, subplot_bottom, subplot_top, padding_points):
    """Append white canvas on top without moving or resizing the axes."""
    if padding_points < 0:
        raise ValueError("top canvas padding cannot be negative")
    padded_height = base_height + padding_points / 72.0
    scale = base_height / padded_height
    return padded_height, subplot_bottom * scale, subplot_top * scale


def _save_combined(fig, stem, preserve_canvas=False):
    """Save a figure that combines every model selected for this invocation."""
    common = {
        "dpi": TOPK_SAVE_DPI,
        # An explicit figure bbox is required here: bbox_inches=None falls back
        # to rcParams["savefig.bbox"] ("tight") and crops intentional padding.
        "bbox_inches": fig.bbox_inches if preserve_canvas else TOPK_SAVE_BBOX_INCHES,
        "pad_inches": TOPK_SAVE_PAD_INCHES,
        "facecolor": TOPK_SAVE_FACE_COLOR,
        "edgecolor": TOPK_SAVE_EDGE_COLOR,
        "transparent": TOPK_SAVE_TRANSPARENT,
    }
    if TOPK_SAVE_PDF:
        TOPK_OUTPUT_ROOT_PDF.mkdir(parents=True, exist_ok=True)
        destination = TOPK_OUTPUT_ROOT_PDF / f"{stem}.pdf"
        fig.savefig(destination, **common)
        print(f"wrote {destination}")


def _experiment_source_state(experiment):
    """Stable cache inputs for one experiment's saved numeric artifacts."""
    sources = []
    for filename in ("components.npz", "summary.json"):
        path = experiment.directory / filename
        stat = path.stat()
        sources.append({
            "name": filename,
            "size": stat.st_size,
            "mtime_ns": stat.st_mtime_ns,
        })
    return {
        "name": experiment.name,
        "targets": list(experiment.targets),
        "pool_size": experiment.pool_size,
        "sources": sources,
    }


def _cache_metadata(kind, experiments, **extra):
    metadata = {
        "version": PLOT_DATA_CACHE_VERSION,
        "kind": kind,
        "beta": BETA,
        "keep_lowest": KEEP_LOWEST,
        "experiments": [
            _experiment_source_state(experiment) for experiment in experiments
        ],
    }
    metadata.update(extra)
    return metadata


def _metadata_json(metadata):
    return json.dumps(metadata, sort_keys=True, separators=(",", ":"))


def _load_plot_data(path, metadata):
    if (not USE_PLOT_DATA_CACHE or REBUILD_PLOT_DATA_CACHE
            or not path.is_file()):
        return None
    try:
        with np.load(path, allow_pickle=False) as cached:
            if str(cached["metadata_json"].item()) != _metadata_json(metadata):
                return None
            arrays = {
                name: cached[name].copy()
                for name in cached.files
                if name != "metadata_json"
            }
    except (OSError, KeyError, ValueError):
        return None
    # print(f"loaded plot data {path}")
    return arrays


def _save_plot_data(path, metadata, **arrays):
    if not SAVE_PLOT_DATA_CACHE:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        metadata_json=np.asarray(_metadata_json(metadata)),
        **arrays,
    )
    print(f"saved plot data {path}")


def _rank_plot_data(experiment, target):
    """Load or compute the exact x/y arrays used by one rank figure."""
    cache_path = (
        PLOT_DATA_ROOT / experiment.name / f"rank_target{target}.npz"
    )
    metadata = _cache_metadata(
        "rank", [experiment], saved_target=int(target)
    )
    cached = _load_plot_data(cache_path, metadata)
    if cached is not None:
        return (
            cached["displayed_x"], cached["y"],
            float(cached["pearson"].item()),
        )

    subset = experiment.subset_pos
    x = _best_is_zero_rank(experiment.cheap(target))[subset]
    y = _rank(experiment.a[target])[subset]
    displayed_x = 1.0 - x
    pearson = _pearson(displayed_x, y)
    _save_plot_data(
        cache_path, metadata, displayed_x=displayed_x, y=y,
        pearson=np.asarray(pearson),
    )
    return displayed_x, y, pearson



# =============================================================================
# FIGURE 2: rank-rank plots (saved random subset, full-pool ranks)
#           one separate figure per target, not a shared grid
# =============================================================================

def plot_rank_grid(experiment):
    if RANK_TARGET_BY_DATASET_MODEL:
        requested_target = RANK_TARGET_BY_DATASET_MODEL.get(
            (experiment.dataset, experiment.model)
        )
        if requested_target is None:
            return {}
        target = experiment.resolve_target(requested_target)
        if target is None or target not in experiment.targets:
            # Several runs can share the same dataset/model. Only the run that
            # actually saved this requested target should produce a rank plot.
            return {}
        rank_targets = ((requested_target, target),)
    else:
        rank_targets = tuple((target, target) for target in experiment.targets)

    model_color = _model_color(experiment.model)

    figs = {}
    for output_target, target in rank_targets:
        fig, ax = plt.subplots(figsize=PANEL_SIZE, dpi=FIGURE_DPI,
                               constrained_layout=True)
        displayed_x, y, pearson = _rank_plot_data(experiment, target)

        ax.scatter(displayed_x, y, s=POINT_SIZE, alpha=POINT_ALPHA,
                   color=model_color,
                   edgecolors="none", rasterized=True, zorder=3)
        # Keep boundary markers fully visible while retaining rank ticks at
        # their meaningful 0--1 values.
        ax.set_xlim(-0.015, 1.015)
        ax.set_ylim(-0.015, 1.015)
        ax.set_xticks((0.0, 0.5, 1.0))
        ax.set_yticks((0.0, 0.5, 1.0))
        _style_axes(ax)
        ax.set_ylabel(
            RANK_Y_LABEL,
            fontsize=RANK_Y_LABEL_FONT_SIZE,
            rotation=RANK_Y_LABEL_ROTATION[experiment.model],
            color=(
                TEXT_COLOR
                if RANK_SHOW_Y_LABEL[experiment.model]
                else RANK_HIDDEN_Y_LABEL_COLOR
            ),
        )
        ax.yaxis.set_label_coords(RANK_Y_LABEL_X, RANK_Y_LABEL_Y)
        ax.set_xlabel(RANK_X_LABEL, fontsize=RANK_X_LABEL_FONT_SIZE)

        _save(
            fig, experiment,
            f"rank_target{output_target}_pearson_{pearson:+.3f}",
        )
        figs[output_target] = fig
    return figs




# =============================================================================
# FIGURE 4 / 5: overlap of the (R,M) and (A,R,M) selections
# =============================================================================

def _overlap_curves(experiment):
    cache_path = (
        PLOT_DATA_ROOT / experiment.name / "topk_overlap_curves.npz"
    )
    metadata = _cache_metadata("topk_overlap", [experiment])
    cached = _load_plot_data(cache_path, metadata)
    if cached is not None:
        return cached["ks"], [curve for curve in cached["curves"]]

    k_max = int(round( experiment.pool_size))
    k_max = max(1, min(k_max, experiment.pool_size))
    ks = np.arange(1, k_max + 1)
    curves = [
        _overlap_curve(experiment.cheap(target), experiment.full(target), ks)
        for target in experiment.targets
    ]
    _save_plot_data(cache_path, metadata, ks=ks, curves=np.stack(curves))
    return ks, curves


def _random_cache_metadata(experiments):
    return _cache_metadata(
        "topk_random", experiments,
        repeats=RANDOM_BASELINE_REPEATS,
        seed=RANDOM_BASELINE_SEED,
    )


def _load_random_plot_data(experiments):
    cache_path = PLOT_DATA_ROOT / "topk_random_trials.npz"
    metadata = _random_cache_metadata(experiments)
    cached = _load_plot_data(cache_path, metadata)
    if cached is None:
        return None
    return [
        (cached["fraction"], curve)
        for curve in cached["trial_means"]
    ]


def _save_random_plot_data(experiments, trial_means):
    cache_path = PLOT_DATA_ROOT / "topk_random_trials.npz"
    metadata = _random_cache_metadata(experiments)
    fraction = trial_means[0][0]
    curves = np.stack([curve for _, curve in trial_means])
    _save_plot_data(
        cache_path, metadata, fraction=fraction, trial_means=curves,
    )


def _ordered_model_groups(experiments):
    """Return (model, experiments) groups in MODEL_FILTER order."""
    grouped = {}
    for experiment in experiments:
        grouped.setdefault(experiment.model, []).append(experiment)

    requested = ((MODEL_FILTER,) if isinstance(MODEL_FILTER, str)
                 else tuple(MODEL_FILTER))
    ordered_models = [model for model in requested if model in grouped]
    ordered_models.extend(model for model in grouped if model not in ordered_models)
    if len(ordered_models) > len(MODEL_COLORS):
        raise ValueError(
            f"selected {len(ordered_models)} models but only "
            f"{len(MODEL_COLORS)} publication colors are configured"
        )
    return [(model, grouped[model]) for model in ordered_models]


def _model_color(model):
    """Use the same model-to-color assignment in rank and TOPK figures."""
    if model not in MODEL_COLORS:
        raise KeyError(f"no publication color configured for model {model!r}")
    return MODEL_COLORS[model]


def _topk_model_color(model):
    if model not in TOPK_MODEL_COLORS:
        raise KeyError(f"no TOPK color configured for model {model!r}")
    return TOPK_MODEL_COLORS[model]


def _stack_on_common_x(curves):
    """Stack (x, y) curves, interpolating only when x grids differ."""
    reference_x = curves[0][0]
    if all(len(x) == len(reference_x) and np.array_equal(x, reference_x)
           for x, _ in curves[1:]):
        return reference_x, np.stack([y for _, y in curves])

    lower = max(x[0] for x, _ in curves)
    upper = min(x[-1] for x, _ in curves)
    reference_x = max(curves, key=lambda item: len(item[0]))[0]
    common_x = reference_x[(reference_x >= lower) & (reference_x <= upper)]
    interpolated = [np.interp(common_x, x, y) for x, y in curves]
    return common_x, np.stack(interpolated)


def _mean_on_common_x(curves):
    """Average (x, y) curves on a shared x grid."""
    common_x, stacked = _stack_on_common_x(curves)
    return common_x, np.mean(stacked, axis=0)


def _rho_to_display(rho):
    """Convert a dataset-level rho value to displayed units of 10^-3."""
    return np.asarray(rho, dtype=np.float64) / TOPK_RHO_UNIT


def _pool_fraction_to_rho_display(pool_fraction):
    """Convert poison-class-pool fractions to displayed dataset-level rho."""
    rho = (
        np.asarray(pool_fraction, dtype=np.float64)
        / TOPK_POOL_FRACTION_TO_RHO_DIVISOR
    )
    return _rho_to_display(rho)


def plot_topk(
    experiments,
    log_x=False,
    log_y=False,
    *,
    ax=None,
    show_x_label=True,
    show_y_label=True,
    show_legend=None,
    x_max_rho=TOPK_X_MAX_RHO,
    x_tick_rho_values=TOPK_X_TICK_RHO_VALUES,
    x_right_padding=0.0,
    save_figure=True,
):
    if log_x and log_y:
        raise ValueError("TOPK supports one logarithmic axis at a time")

    if ax is None:
        figure_width = TOPK_FIG_SIZE[0]
        subplot_left = TOPK_SUBPLOT_LEFT
        subplot_right = TOPK_SUBPLOT_RIGHT
        if log_x and x_right_padding:
            figure_width, subplot_left, subplot_right = _right_padded_canvas(
                figure_width,
                subplot_left,
                subplot_right,
                x_right_padding,
            )
        fig, ax = plt.subplots(
            figsize=(figure_width, TOPK_FIG_SIZE[1]),
            dpi=TOPK_FIGURE_DPI,
            facecolor=TOPK_FIGURE_FACE_COLOR,
        )
        fig.subplots_adjust(
            left=subplot_left,
            right=subplot_right,
            bottom=TOPK_SUBPLOT_BOTTOM,
            top=TOPK_SUBPLOT_TOP,
        )
    else:
        fig = ax.figure

    data_x_max = (
        float(_rho_to_display(x_max_rho))
        if x_max_rho is not None
        else None
    )

    def clipped_to_data_max(x_values, *y_values):
        """Stop drawn data at the configured x maximum."""
        x_array = np.asarray(x_values)
        if data_x_max is None:
            return (x_array, *(np.asarray(y) for y in y_values))
        tolerance = np.finfo(float).eps * max(1.0, abs(data_x_max))
        keep = x_array <= data_x_max + tolerance
        return (x_array[keep], *(np.asarray(y)[keep] for y in y_values))

    random_trial_fraction_means = _load_random_plot_data(experiments)
    if random_trial_fraction_means is None:
        rng = np.random.default_rng(RANDOM_BASELINE_SEED)
        random_trials = [[] for _ in range(RANDOM_BASELINE_REPEATS)]
    else:
        rng = None
        random_trials = None
    minimum_x = float("inf")
    maximum_x = 0.0
    for model_index, (model, model_experiments) in enumerate(
            _ordered_model_groups(experiments)):
        color = _topk_model_color(model)
        model_curves = []

        for experiment in model_experiments:
            ks, curves = _overlap_curves(experiment)
            fraction = ks / float(experiment.pool_size)
            x_axis = _pool_fraction_to_rho_display(fraction)
            minimum_x = min(minimum_x, float(x_axis[0]))
            maximum_x = max(maximum_x, float(x_axis[-1]))
            for target, curve in zip(experiment.targets, curves):
                model_curves.append((x_axis, curve))
                line_x, line_y = clipped_to_data_max(x_axis, curve)
                ax.plot(
                    line_x,
                    line_y,
                    color=color,
                    linestyle=TOPK_TARGET_LINE_STYLE,
                    alpha=TOPK_TARGET_LINE_ALPHA,
                    linewidth=TOPK_TARGET_LINE_WIDTH,
                    antialiased=TOPK_LINE_ANTIALIASED,
                    zorder=TOPK_TARGET_LINE_ZORDER,
                )

                if random_trials is not None:
                    full_order = _best_first(experiment.full(target))
                    for trial in random_trials:
                        random_order = rng.permutation(experiment.pool_size)
                        random_curve = _ordered_overlap_curve(
                            full_order, random_order, ks
                        )
                        # Cache the true pool fraction; display scaling is
                        # deliberately applied only while drawing.
                        trial.append((fraction, random_curve))

        mean_x, mean = _mean_on_common_x(model_curves)
        visible_mean_x, visible_mean = clipped_to_data_max(mean_x, mean)
        ax.plot(
            visible_mean_x,
            visible_mean,
            color=color,
            linestyle=TOPK_MEAN_LINE_STYLE,
            linewidth=TOPK_MEAN_LINE_WIDTH,
            alpha=TOPK_MEAN_LINE_ALPHA,
            antialiased=TOPK_LINE_ANTIALIASED,
            label=TOPK_LEGEND_MODEL_LABEL.format(
                model=MODEL_LABELS.get(model, model), n=len(model_curves)
            ),
            zorder=TOPK_MEAN_LINE_ZORDER,
        )

    if random_trial_fraction_means is None:
        random_trial_fraction_means = []
        for trial in random_trials:
            trial_fraction, trial_mean = _mean_on_common_x(trial)
            random_trial_fraction_means.append((trial_fraction, trial_mean))
        _save_random_plot_data(experiments, random_trial_fraction_means)

    random_trial_means = [
        (_pool_fraction_to_rho_display(fraction), curve)
        for fraction, curve in random_trial_fraction_means
    ]
    for trial_x, trial_mean in random_trial_means:
        if SHOW_RANDOM_TRIALS:
            visible_trial_x, visible_trial_mean = clipped_to_data_max(
                trial_x, trial_mean
            )
            ax.plot(
                visible_trial_x,
                visible_trial_mean,
                color=TOPK_RANDOM_TRIAL_COLOR,
                linestyle=TOPK_RANDOM_TRIAL_LINE_STYLE,
                linewidth=TOPK_RANDOM_TRIAL_LINE_WIDTH,
                alpha=TOPK_RANDOM_TRIAL_ALPHA,
                antialiased=TOPK_LINE_ANTIALIASED,
                zorder=TOPK_RANDOM_TRIAL_ZORDER,
            )

    random_mean_x, random_stack = _stack_on_common_x(random_trial_means)
    random_mean = np.mean(random_stack, axis=0)
    random_std = np.std(
        random_stack, axis=0, ddof=1 if RANDOM_BASELINE_REPEATS > 1 else 0
    )
    if SHOW_RANDOM_STD_BAND:
        random_band_floor = TOPK_LOG_Y_LIMITS[0] if log_y else 0.0
        band_lower = np.clip(
            random_mean - random_std,
            random_band_floor,
            TOPK_RANDOM_BAND_UPPER_CLIP,
        )
        band_upper = np.clip(
            random_mean + random_std,
            0.0,
            TOPK_RANDOM_BAND_UPPER_CLIP,
        )
        visible_band_x, visible_band_lower, visible_band_upper = (
            clipped_to_data_max(random_mean_x, band_lower, band_upper)
        )
        ax.fill_between(
            visible_band_x,
            visible_band_lower,
            visible_band_upper,
            color=TOPK_RANDOM_STD_COLOR,
            alpha=TOPK_RANDOM_STD_ALPHA,
            linewidth=TOPK_RANDOM_STD_LINE_WIDTH,
            zorder=TOPK_RANDOM_STD_ZORDER,
        )
    visible_random_x, visible_random_mean = clipped_to_data_max(
        random_mean_x, random_mean
    )
    ax.plot(
        visible_random_x,
        visible_random_mean,
        color=TOPK_RANDOM_MEAN_COLOR,
        linestyle=TOPK_RANDOM_MEAN_LINE_STYLE,
        linewidth=TOPK_RANDOM_MEAN_LINE_WIDTH,
        alpha=TOPK_RANDOM_MEAN_ALPHA,
        antialiased=TOPK_LINE_ANTIALIASED,
        zorder=TOPK_RANDOM_MEAN_ZORDER,
    )

    seen_marks = set()
    for experiment in experiments:
        for label, k, color in experiment.budget_marks(experiment.pool_size):
            x = _pool_fraction_to_rho_display(
                k / float(experiment.pool_size)
            )
            mark_key = (label, round(x, 12))
            if mark_key in seen_marks:
                continue
            seen_marks.add(mark_key)
            ax.axvline(
                x,
                color=color,
                linewidth=TOPK_BUDGET_MARK_LINE_WIDTH,
                linestyle=TOPK_BUDGET_MARK_LINE_STYLE,
                zorder=TOPK_BUDGET_MARK_ZORDER,
            )
            if SHOW_BUDGET_MARK_VALUES:
                ax.text(
                    x,
                    TOPK_BUDGET_TEXT_Y,
                    format(float(x), TOPK_BUDGET_TEXT_FORMAT),
                    transform=ax.get_xaxis_transform(),
                    ha=TOPK_BUDGET_TEXT_HORIZONTAL_ALIGNMENT,
                    va=TOPK_BUDGET_TEXT_VERTICAL_ALIGNMENT,
                    fontsize=TOPK_BUDGET_TEXT_FONT_SIZE,
                    color=TOPK_BUDGET_TEXT_COLOR,
                )

    x_tick_positions = _rho_to_display(x_tick_rho_values)
    x_tick_labels = tuple(
        format(float(value), TOPK_X_TICK_LABEL_FORMAT)
        for value in x_tick_positions
    )
    if log_x:
        ax.set_xscale("log", base=TOPK_X_LOG_BASE)
        x_min = (
            _rho_to_display(TOPK_LOG_X_MIN_RHO)
            if TOPK_LOG_X_MIN_RHO is not None
            else minimum_x
        )
        x_max = (
            data_x_max if data_x_max is not None else maximum_x
        )
        ax.set_xticks(x_tick_positions)
        ax.set_xticklabels(
            x_tick_labels,
            rotation=TOPK_X_TICK_ROTATION,
            ha=TOPK_X_TICK_HORIZONTAL_ALIGNMENT,
            va=TOPK_X_TICK_VERTICAL_ALIGNMENT,
        )
        # Limits are set after ticks so Matplotlib cannot expand them again.
        ax.set_xlim(x_min, x_max)
    else:
        x_min = (
            _rho_to_display(TOPK_X_MIN_RHO)
            if TOPK_X_MIN_RHO is not None
            else 0.0
        )
        x_max = (
            data_x_max if data_x_max is not None else maximum_x
        )
        ax.set_xticks(x_tick_positions)
        ax.set_xticklabels(
            x_tick_labels,
            rotation=TOPK_X_TICK_ROTATION,
            ha=TOPK_X_TICK_HORIZONTAL_ALIGNMENT,
            va=TOPK_X_TICK_VERTICAL_ALIGNMENT,
        )
        ax.set_xlim(x_min, x_max)

    if log_y:
        ax.set_yscale("log", base=TOPK_Y_LOG_BASE)
        ax.set_ylim(*TOPK_LOG_Y_LIMITS)
        ax.set_yticks(TOPK_LOG_Y_TICKS)
        ax.yaxis.set_major_formatter(LogFormatterMathtext())
    else:
        ax.set_ylim(*TOPK_Y_LIMITS)
        ax.set_yticks(TOPK_Y_TICKS)
        if TOPK_Y_AS_PERCENT:
            ax.yaxis.set_major_formatter(
                PercentFormatter(xmax=1.0, decimals=TOPK_Y_PERCENT_DECIMALS)
            )
        else:
            ax.yaxis.set_major_formatter(
                FormatStrFormatter(TOPK_Y_TICK_FORMAT)
            )

    _style_topk_axes(ax)
    if show_x_label:
        ax.set_xlabel(
            TOPK_X_LABEL,
            fontsize=TOPK_X_LABEL_FONT_SIZE,
            fontweight=TOPK_X_LABEL_FONT_WEIGHT,
            fontstyle=TOPK_X_LABEL_FONT_STYLE,
            color=TOPK_X_LABEL_COLOR,
            labelpad=TOPK_X_LABEL_PAD,
        )
    else:
        # Keep the x ticks and tick labels; omit only the axis label.
        ax.set_xlabel("")
    if show_y_label:
        ax.set_ylabel(
            TOPK_Y_LABEL,
            fontsize=TOPK_Y_LABEL_FONT_SIZE,
            fontweight=TOPK_Y_LABEL_FONT_WEIGHT,
            fontstyle=TOPK_Y_LABEL_FONT_STYLE,
            color=TOPK_Y_LABEL_COLOR,
            labelpad=TOPK_Y_LABEL_PAD,
        )
    else:
        # Do not hide text by painting it white: in a side-by-side layout the
        # later axes can draw that text over the earlier axes and erase curves.
        ax.set_ylabel("")
    _add_topk_legend(ax, visible=show_legend)
    if log_x:
        stem = "topk_overlap_log"
    elif log_y:
        stem = "topk_overlap_log_y"
    else:
        stem = "topk_overlap_linear"
    if save_figure:
        _save_combined(
            fig,
            stem,
            preserve_canvas=bool(log_x and x_right_padding),
        )
    return fig


def plot_topk_stacked(experiments):
    """Create subplot(211) log-x above subplot(212) linear-x."""
    figure_width, subplot_left, subplot_right = _right_padded_canvas(
        TOPK_STACKED_FIG_SIZE[0],
        TOPK_STACKED_SUBPLOT_LEFT,
        TOPK_STACKED_SUBPLOT_RIGHT,
        TOPK_LOG_X_RIGHT_PADDING,
    )
    fig, (log_ax, linear_ax) = plt.subplots(
        2,
        1,
        figsize=(figure_width, TOPK_STACKED_FIG_SIZE[1]),
        dpi=TOPK_FIGURE_DPI,
        facecolor=TOPK_FIGURE_FACE_COLOR,
        gridspec_kw={"height_ratios": TOPK_STACKED_HEIGHT_RATIOS},
    )
    fig.subplots_adjust(
        left=subplot_left,
        right=subplot_right,
        bottom=TOPK_STACKED_SUBPLOT_BOTTOM,
        top=TOPK_STACKED_SUBPLOT_TOP,
        hspace=TOPK_STACKED_VERTICAL_SPACE,
    )

    plot_topk(
        experiments,
        log_x=True,
        ax=log_ax,
        show_x_label=TOPK_STACKED_TOP_SHOW_X_LABEL,
        show_legend=TOPK_STACKED_TOP_SHOW_LEGEND,
        x_max_rho=TOPK_LOG_X_MAX_RHO,
        x_tick_rho_values=TOPK_LOG_X_TICK_RHO_VALUES,
        x_right_padding=TOPK_LOG_X_RIGHT_PADDING,
        save_figure=False,
    )
    plot_topk(
        experiments,
        log_x=False,
        ax=linear_ax,
        show_x_label=TOPK_STACKED_BOTTOM_SHOW_X_LABEL,
        show_legend=TOPK_STACKED_BOTTOM_SHOW_LEGEND,
        save_figure=False,
    )

    _save_combined(
        fig,
        TOPK_STACKED_OUTPUT_STEM,
        preserve_canvas=bool(TOPK_LOG_X_RIGHT_PADDING),
    )
    return fig


def plot_topk_side_by_side(experiments):
    """Create subplot(121) linear-x beside subplot(122) log-x."""
    axes_width = (
        TOPK_SIDE_BY_SIDE_LINEAR_WIDTH_INCHES
        + TOPK_SIDE_BY_SIDE_LOG_WIDTH_INCHES
    )
    usable_figure_fraction = (
        TOPK_SIDE_BY_SIDE_SUBPLOT_RIGHT
        - TOPK_SIDE_BY_SIDE_SUBPLOT_LEFT
    )
    base_figure_width = (
        axes_width + TOPK_SIDE_BY_SIDE_GAP_INCHES
    ) / usable_figure_fraction
    figure_width, subplot_left, subplot_right = _right_padded_canvas(
        base_figure_width,
        TOPK_SIDE_BY_SIDE_SUBPLOT_LEFT,
        TOPK_SIDE_BY_SIDE_SUBPLOT_RIGHT,
        TOPK_SIDE_BY_SIDE_RIGHT_PADDING_POINTS,
    )
    figure_height, subplot_bottom, subplot_top = _top_padded_canvas(
        TOPK_SIDE_BY_SIDE_FIG_HEIGHT_INCHES,
        TOPK_SIDE_BY_SIDE_SUBPLOT_BOTTOM,
        TOPK_SIDE_BY_SIDE_SUBPLOT_TOP,
        TOPK_SIDE_BY_SIDE_TOP_PADDING_POINTS,
    )
    internal_wspace = (
        2.0 * TOPK_SIDE_BY_SIDE_GAP_INCHES / axes_width
    )

    fig, (linear_ax, log_ax) = plt.subplots(
        1,
        2,
        figsize=(figure_width, figure_height),
        dpi=TOPK_FIGURE_DPI,
        facecolor=TOPK_FIGURE_FACE_COLOR,
        gridspec_kw={
            "width_ratios": (
                TOPK_SIDE_BY_SIDE_LINEAR_WIDTH_INCHES,
                TOPK_SIDE_BY_SIDE_LOG_WIDTH_INCHES,
            ),
        },
    )
    fig.subplots_adjust(
        left=subplot_left,
        right=subplot_right,
        bottom=subplot_bottom,
        top=subplot_top,
        wspace=internal_wspace,
    )

    plot_topk(
        experiments,
        log_x=False,
        ax=linear_ax,
        show_x_label=TOPK_SIDE_BY_SIDE_LEFT_SHOW_X_LABEL,
        show_y_label=TOPK_SIDE_BY_SIDE_LEFT_SHOW_Y_LABEL,
        show_legend=TOPK_SIDE_BY_SIDE_LEFT_SHOW_LEGEND,
        x_max_rho=TOPK_SIDE_BY_SIDE_LINEAR_X_MAX_RHO,
        x_tick_rho_values=TOPK_SIDE_BY_SIDE_LINEAR_X_TICK_RHO_VALUES,
        save_figure=False,
    )
    plot_topk(
        experiments,
        log_x=True,
        ax=log_ax,
        show_x_label=TOPK_SIDE_BY_SIDE_RIGHT_SHOW_X_LABEL,
        show_y_label=TOPK_SIDE_BY_SIDE_RIGHT_SHOW_Y_LABEL,
        show_legend=TOPK_SIDE_BY_SIDE_RIGHT_SHOW_LEGEND,
        x_max_rho=TOPK_LOG_X_MAX_RHO,
        x_tick_rho_values=TOPK_LOG_X_TICK_RHO_VALUES,
        x_right_padding=TOPK_LOG_X_RIGHT_PADDING,
        save_figure=False,
    )

    _save_combined(
        fig,
        TOPK_SIDE_BY_SIDE_OUTPUT_STEM,
        preserve_canvas=bool(
            TOPK_SIDE_BY_SIDE_RIGHT_PADDING_POINTS
            or TOPK_SIDE_BY_SIDE_TOP_PADDING_POINTS
        ),
    )
    return fig


# =============================================================================
# JUPYTER EXECUTION
# =============================================================================

_configure_matplotlib()

_experiments = [Experiment(directory) for directory in _discover_experiments()]
all_figures = {}
for _experiment in _experiments:
    print(f"=== {_experiment.name}: pool {_experiment.pool_size}, "
          f"N_p {_experiment.num_poisons}, targets {_experiment.targets}")
    _figs = {}
    if RUN_RANK_GRID:
        for _target, _fig in plot_rank_grid(_experiment).items():
            _figs[f"rank_target{_target}"] = _fig
    all_figures[_experiment.name] = _figs

_topk_figs = {}
if RUN_TOPK_LINEAR:
    _topk_figs["topk_overlap_linear"] = plot_topk(_experiments, log_x=False)
if RUN_TOPK_LOG:
    _topk_figs["topk_overlap_log"] = plot_topk(
        _experiments,
        log_x=True,
        x_max_rho=TOPK_LOG_X_MAX_RHO,
        x_tick_rho_values=TOPK_LOG_X_TICK_RHO_VALUES,
        x_right_padding=TOPK_LOG_X_RIGHT_PADDING,
    )
if RUN_TOPK_LOG_Y:
    _topk_figs["topk_overlap_log_y"] = plot_topk(_experiments, log_y=True)
if RUN_TOPK_STACKED:
    _topk_figs[TOPK_STACKED_OUTPUT_STEM] = plot_topk_stacked(_experiments)
if RUN_TOPK_SIDE_BY_SIDE:
    _topk_figs[TOPK_SIDE_BY_SIDE_OUTPUT_STEM] = (
        plot_topk_side_by_side(_experiments)
    )
if _topk_figs:
    all_figures["combined_models"] = _topk_figs

if SHOW_PLOTS:
    plt.show()

if CLOSE_FIGURES_AFTER_SHOW:
    for _figs in all_figures.values():
        for _fig in _figs.values():
            plt.close(_fig)
