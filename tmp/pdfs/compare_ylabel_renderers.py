import os
os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from matplotlib.patches import PathPatch
from matplotlib.textpath import TextPath
from matplotlib.transforms import Affine2D

mpl.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "Nimbus Roman No9 L", "DejaVu Serif"],
    "mathtext.fontset": "dejavuserif",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

OUT = "tmp/pdfs"
SIZE = 23
COLOR = "#111111"


def prep():
    fig, ax = plt.subplots(figsize=(3.375, 3.0), dpi=150, constrained_layout=True)
    ax.set_xlim(-0.015, 1.015)
    ax.set_ylim(-0.015, 1.015)
    ax.set_xticks((0.0, 0.5, 1.0))
    ax.set_yticks((0.0, 0.5, 1.0))
    ax.tick_params(labelsize=22, length=8, width=1.5)
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    for spine in (ax.spines["left"], ax.spines["bottom"]):
        spine.set_linewidth(1.6)
    ax.set_xlabel(r"$R + M$", fontsize=20)
    return fig, ax


def save(fig, stem):
    fig.canvas.draw()
    fig.set_layout_engine(None)
    fig.savefig(f"{OUT}/{stem}.pdf", dpi=500, bbox_inches="tight", pad_inches=0.25,
                facecolor="white", edgecolor="white", transparent=False)
    fig.savefig(f"{OUT}/{stem}.png", dpi=600, bbox_inches="tight", pad_inches=0.25,
                facecolor="white", edgecolor="white", transparent=False)
    plt.close(fig)


# Exact current Text ylabel.
fig, ax = prep()
t = ax.set_ylabel(r"$A$", fontsize=SIZE, rotation=90, color=COLOR)
t.set_clip_on(False)
ax.yaxis.set_label_coords(-0.24, 0.5)
save(fig, "ylabel_standard")

# Exact current Text ylabel, explicitly rasterized.
fig, ax = prep()
t = ax.set_ylabel(r"$A$", fontsize=SIZE, rotation=90, color=COLOR)
t.set_clip_on(False)
t.set_rasterized(True)
ax.yaxis.set_label_coords(-0.24, 0.5)
save(fig, "ylabel_rasterized")

# A figure-level Text at the display-coordinate equivalent (still MathText).
fig, ax = prep()
fig.canvas.draw()
x_disp, y_disp = ax.transAxes.transform((-0.24, 0.5))
x_fig, y_fig = fig.transFigure.inverted().transform((x_disp, y_disp))
t = fig.text(x_fig, y_fig, r"$A$", fontsize=SIZE, rotation=90, color=COLOR,
             ha="center", va="center", clip_on=False)
save(fig, "ylabel_figure_text")

# Convert the identical MathText glyph to a vector path and place its visual
# center at the identical display coordinate before applying the 90-deg turn.
fig, ax = prep()
fig.canvas.draw()
path = TextPath((0, 0), r"$A$", size=SIZE,
                prop=FontProperties(math_fontfamily="dejavuserif"), usetex=False)
bbox = path.get_extents()
cx, cy = bbox.x0 + bbox.width / 2, bbox.y0 + bbox.height / 2
x_disp, y_disp = ax.transAxes.transform((-0.24, 0.5))
transform = (Affine2D().translate(-cx, -cy).rotate_deg(90).translate(x_disp, y_disp)
             + mpl.transforms.IdentityTransform())
patch = PathPatch(path, transform=transform, facecolor=COLOR, edgecolor="none",
                  clip_on=False)
fig.add_artist(patch)
save(fig, "ylabel_textpath")

# Same label parameters and string, but Computer Modern's math glyph has an
# almost-pointed cap (the current DejaVu glyph itself has a broad flat cap).
fig, ax = prep()
t = ax.set_ylabel(r"$A$", fontsize=SIZE, rotation=90, color=COLOR)
t.set_clip_on(False)
t.set_math_fontfamily("cm")
ax.yaxis.set_label_coords(-0.24, 0.5)
save(fig, "ylabel_cm_mathtext")

# Same string/size/rotation/coordinates through TeX's Computer Modern renderer.
fig, ax = prep()
t = ax.set_ylabel(r"$A$", fontsize=SIZE, rotation=90, color=COLOR, usetex=True)
t.set_clip_on(False)
ax.yaxis.set_label_coords(-0.24, 0.5)
save(fig, "ylabel_tex")

# Same point size/rotation/anchor, using the configured document serif face
# directly. This is Times New Roman Italic on this machine.
fig, ax = prep()
t = ax.set_ylabel("A", fontsize=SIZE, rotation=90, color=COLOR,
                  fontfamily="serif", fontstyle="italic")
t.set_clip_on(False)
ax.yaxis.set_label_coords(-0.24, 0.5)
save(fig, "ylabel_times_text")

# Inspect the same glyph unrotated, once as Text and once as its TextPath.
fig, axes = plt.subplots(1, 2, figsize=(5, 2), dpi=150)
for ax in axes:
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
axes[0].text(.5, .5, r"$A$", fontsize=100, ha="center", va="center")
p = TextPath((0, 0), r"$A$", size=100,
             prop=FontProperties(math_fontfamily="dejavuserif"), usetex=False)
b = p.get_extents()
tr = Affine2D().translate(-b.x0-b.width/2, -b.y0-b.height/2).scale(1/72).translate(3.75, 1)
fig.add_artist(PathPatch(p, transform=tr + fig.dpi_scale_trans,
                         facecolor=COLOR, edgecolor="none", clip_on=False))
fig.savefig(f"{OUT}/unrotated_text_vs_path.png", dpi=600, bbox_inches="tight", pad_inches=.2)
