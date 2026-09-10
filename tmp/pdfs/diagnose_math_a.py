from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.textpath import TextPath
from matplotlib.transforms import Affine2D
from PIL import Image


HERE = Path(__file__).resolve().parent

mpl.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": [
            "Times New Roman",
            "Times",
            "Nimbus Roman No9 L",
            "DejaVu Serif",
        ],
        "mathtext.fontset": "dejavuserif",
        "pdf.fonttype": 42,
        "text.color": "#111111",
    }
)


def ink_bbox(path):
    image = Image.open(path).convert("L")
    pixels = image.load()
    xs = []
    ys = []
    for y in range(image.height):
        for x in range(image.width):
            if pixels[x, y] < 245:
                xs.append(x)
                ys.append(y)
    return image.size, (min(xs), min(ys), max(xs), max(ys))


# A deliberately huge, fixed page.  The text is nowhere near any page edge,
# no axes exists, and both artist-level and patch-level clipping are disabled.
fig = plt.figure(figsize=(8, 8), dpi=300, facecolor="white")
label = fig.text(
    0.5,
    0.5,
    r"$A$",
    fontsize=23,
    rotation=90,
    ha="center",
    va="center",
    color="#111111",
    clip_on=False,
)
fig.canvas.draw()
renderer = fig.canvas.get_renderer()
print("artist_window_extent_pixels", tuple(label.get_window_extent(renderer).bounds))
print("artist_tightbbox_pixels", tuple(label.get_tightbbox(renderer).bounds))
print("artist_clip_on", label.get_clip_on())
print("artist_clip_box", label.get_clip_box())
print("artist_clip_path", label.get_clip_path())
fig.savefig(HERE / "standalone-fixed-page.pdf", bbox_inches=None, pad_inches=0)
fig.savefig(HERE / "standalone-fixed-page.png", bbox_inches=None, pad_inches=0, dpi=600)
plt.close(fig)


# Direct vector outline from MathText. This bypasses the Text artist and PDF
# text operators entirely, so no text bbox or clip path can trim its contour.
glyph = TextPath((0, 0), r"$A$", size=23, usetex=False)
bounds = glyph.get_extents()
print("textpath_bounds_points", tuple(bounds.bounds))
rotated = Affine2D().rotate_deg(90).transform_path(glyph)
print("rotated_textpath_bounds_points", tuple(rotated.get_extents().bounds))

fig = plt.figure(figsize=(8, 8), dpi=300, facecolor="white")
ax = fig.add_axes((0, 0, 1, 1), xlim=(-80, 80), ylim=(-80, 80), aspect="equal")
ax.set_axis_off()
patch = mpl.patches.PathPatch(
    rotated,
    facecolor="#111111",
    edgecolor="none",
    clip_on=False,
)
patch.set_transform(Affine2D().translate(0, 0) + ax.transData)
ax.add_patch(patch)
fig.savefig(HERE / "direct-outline-fixed-page.pdf", bbox_inches=None, pad_inches=0)
fig.savefig(HERE / "direct-outline-fixed-page.png", bbox_inches=None, pad_inches=0, dpi=600)
plt.close(fig)


for name in ("standalone-fixed-page.png", "direct-outline-fixed-page.png"):
    print(name, ink_bbox(HERE / name))
