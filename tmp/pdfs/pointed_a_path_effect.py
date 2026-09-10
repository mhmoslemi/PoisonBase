from pathlib import Path as FsPath

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.path import Path
from matplotlib.patheffects import AbstractPathEffect


HERE = FsPath(__file__).resolve().parent

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
    }
)


class PointMathGlyphApex(AbstractPathEffect):
    """Collapse a font's deliberately flat top cap to a mathematical point."""

    def draw_path(self, renderer, gc, tpath, affine, rgbFace=None):
        vertices = tpath.vertices.copy()
        if len(vertices):
            top = vertices[:, 1].max()
            mask = abs(vertices[:, 1] - top) < 1e-9
            if mask.sum() >= 2:
                vertices[mask, 0] = 0.5 * (
                    vertices[mask, 0].min() + vertices[mask, 0].max()
                )
        renderer.draw_path(gc, Path(vertices, tpath.codes), affine, rgbFace)


fig = plt.figure(figsize=(8, 8), dpi=300, facecolor="white")
original = fig.text(
    0.35,
    0.5,
    r"$A$",
    fontsize=23,
    rotation=90,
    ha="center",
    va="center",
    clip_on=False,
)
pointed = fig.text(
    0.65,
    0.5,
    r"$A$",
    fontsize=23,
    rotation=90,
    ha="center",
    va="center",
    clip_on=False,
)
pointed.set_path_effects([PointMathGlyphApex()])
fig.savefig(HERE / "original-vs-pointed.pdf", bbox_inches=None, pad_inches=0)
fig.savefig(HERE / "original-vs-pointed.png", bbox_inches=None, pad_inches=0, dpi=600)
