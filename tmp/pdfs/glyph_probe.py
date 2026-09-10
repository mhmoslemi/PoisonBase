import matplotlib as mpl

mpl.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from matplotlib.patches import PathPatch
from matplotlib.textpath import TextPath
from matplotlib.transforms import Affine2D


mpl.rcParams.update({
    "font.family": "serif",
    "font.serif": [
        "Times New Roman",
        "Times",
        "Nimbus Roman No9 L",
        "DejaVu Serif",
    ],
    "mathtext.fontset": "dejavuserif",
    "pdf.fonttype": 42,
})

fig, axes = plt.subplots(1, 4, figsize=(8, 2), dpi=150)
for ax in axes:
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xticks([])
    ax.set_yticks([])

label = axes[0].set_ylabel(r"$A$", fontsize=23, rotation=90)
label.set_clip_on(False)
label.set_clip_box(None)
label.set_clip_path(None)
axes[0].set_title("ylabel")

axes[1].text(
    0.5, 0.5, r"$A$", fontsize=23, rotation=90,
    ha="center", va="center", clip_on=False,
)
axes[1].set_title("Text")

axes[2].text(
    0.5, 0.5, r"$A$", fontsize=23, rotation=0,
    ha="center", va="center", clip_on=False,
)
axes[2].set_title("unrotated")

path = TextPath(
    (0, 0), r"$A$", size=23,
    prop=FontProperties(math_fontfamily="dejavuserif"),
    usetex=False,
)
bounds = path.get_extents()
transform = (
    Affine2D()
    .translate(-bounds.x0 - bounds.width / 2, -bounds.y0 - bounds.height / 2)
    .rotate_deg(90)
    .scale(1 / 72)
    .translate(0.5, 0.5)
    + axes[3].transAxes
)
axes[3].add_patch(PathPatch(path, transform=transform, color="#111111"))
axes[3].set_title("TextPath")

fig.savefig("tmp/pdfs/glyph_probe.pdf", bbox_inches=None, pad_inches=0)
fig.savefig("tmp/pdfs/glyph_probe.png", dpi=600, bbox_inches=None, pad_inches=0)
