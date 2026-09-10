from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.path import Path as MplPath
from matplotlib.patheffects import AbstractPathEffect
from matplotlib.transforms import Bbox


HERE = Path(__file__).resolve().parent
CACHE = Path(
    "figures/data/scatter/"
    "CIFAR10_ConvNetBN_fc_ours_dog-bird_b0.02_eps8_seed42_lam1_"
    "cosine_seldpp2_jacw1_ce5_tgt50/rank_target5705.npz"
)

mpl.rcParams.update(
    {
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "font.family": "serif",
        "font.serif": [
            "Times New Roman",
            "Times",
            "Nimbus Roman No9 L",
            "DejaVu Serif",
        ],
        "font.size": 14,
        "text.color": "#111111",
        "mathtext.fontset": "dejavuserif",
        "axes.labelcolor": "#111111",
        "xtick.color": "#111111",
        "ytick.color": "#111111",
        "pdf.fonttype": 42,
    }
)


class PointMathGlyphApex(AbstractPathEffect):
    def draw_path(self, renderer, gc, tpath, affine, rgbFace=None):
        vertices = tpath.vertices.copy()
        if len(vertices):
            top = vertices[:, 1].max()
            at_top = np.isclose(vertices[:, 1], top, rtol=0, atol=1e-9)
            if at_top.sum() >= 2:
                vertices[at_top, 0] = 0.5 * (
                    vertices[at_top, 0].min() + vertices[at_top, 0].max()
                )
        renderer.draw_path(gc, MplPath(vertices, tpath.codes), affine, rgbFace)


with np.load(CACHE, allow_pickle=False) as cached:
    x = cached["displayed_x"]
    y = cached["y"]

fig, ax = plt.subplots(figsize=(3.375, 3.0), dpi=150, constrained_layout=True)
ax.scatter(
    x,
    y,
    s=2.5,
    alpha=0.65,
    color="#0072B2",
    edgecolors="none",
    rasterized=True,
    zorder=3,
)
ax.set_xlim(-0.015, 1.015)
ax.set_ylim(-0.015, 1.015)
ax.set_xticks((0.0, 0.5, 1.0))
ax.set_yticks((0.0, 0.5, 1.0))
for side, visible in {"left": True, "right": False, "bottom": True, "top": False}.items():
    ax.spines[side].set_visible(visible)
    ax.spines[side].set_color("#111111")
    ax.spines[side].set_linewidth(1.6)
ax.tick_params(
    axis="both",
    which="major",
    direction="out",
    length=8,
    width=1.5,
    pad=5,
    labelsize=22,
    color="#111111",
    labelcolor="#111111",
    top=False,
    right=False,
)
ax.grid(visible=False)

ylabel = ax.set_ylabel(r"$A$", fontsize=23, rotation=90, color="#111111")
ylabel.set_clip_on(False)
ylabel.set_path_effects([PointMathGlyphApex()])
ax.yaxis.set_label_coords(-0.24, 0.5)
ax.set_xlabel(r"$R + M$", fontsize=20)

fig.canvas.draw()
fig.set_layout_engine(None)
fig.canvas.draw()
tight = fig.get_tightbbox(fig.canvas.get_renderer())
export = Bbox.from_extents(tight.x0 - 0.25, tight.y0, tight.x1, tight.y1)
fig.savefig(
    HERE / "actual-rank-pointed.pdf",
    dpi=500,
    bbox_inches=export,
    pad_inches=0,
    facecolor="white",
    edgecolor="white",
    transparent=False,
)
