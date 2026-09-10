import matplotlib as mpl

mpl.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.patches import Polygon


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

extensions = (0.0, 0.75, 1.25, 1.75, 2.5)
fig, axes = plt.subplots(1, len(extensions), figsize=(8, 2), dpi=150)
artists = []
for ax, extension_points in zip(axes, extensions):
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_axis_off()
    artist = ax.text(
        0.5, 0.5, r"$A$",
        fontsize=23,
        rotation=90,
        ha="center",
        va="center",
        clip_on=False,
        color="#111111",
    )
    ax.set_title(f"cap {extension_points:.2f} pt", fontsize=9)
    artists.append((artist, extension_points))

fig.canvas.draw()
renderer = fig.canvas.get_renderer()
for artist, extension_points in artists:
    if extension_points == 0:
        continue
    bbox = artist.get_window_extent(renderer)
    pixels_per_point = fig.dpi / 72.0
    base_x = bbox.x0 + 0.15 * pixels_per_point
    center_y = (bbox.y0 + bbox.y1) / 2.0
    half_base = 0.57 * pixels_per_point
    point_x = bbox.x0 - extension_points * pixels_per_point
    display_vertices = [
        (base_x, center_y - half_base),
        (point_x, center_y),
        (base_x, center_y + half_base),
    ]
    figure_vertices = fig.transFigure.inverted().transform(display_vertices)
    fig.add_artist(Polygon(
        figure_vertices,
        closed=True,
        transform=fig.transFigure,
        facecolor="#111111",
        edgecolor="none",
        clip_on=False,
        zorder=10,
    ))

fig.savefig("tmp/pdfs/glyph_cap_probe.pdf", bbox_inches=None, pad_inches=0)
fig.savefig("tmp/pdfs/glyph_cap_probe.png", dpi=600, bbox_inches=None, pad_inches=0)
