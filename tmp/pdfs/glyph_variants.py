import matplotlib as mpl

mpl.use("Agg")

import matplotlib.patheffects as path_effects
import matplotlib.pyplot as plt


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

fig, axes = plt.subplots(1, 6, figsize=(9, 2), dpi=150)
variants = (
    ("DejaVu math", {"math_fontfamily": "dejavuserif"}),
    ("STIX math", {"math_fontfamily": "stix"}),
    ("CM math", {"math_fontfamily": "cm"}),
    ("Times italic", {"fontfamily": "Times New Roman", "fontstyle": "italic"}),
    ("TeX", {"usetex": True}),
    ("DejaVu stroke", {"math_fontfamily": "dejavuserif"}),
)

for ax, (title, kwargs) in zip(axes, variants):
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_axis_off()
    text = "A" if title == "Times italic" else r"$A$"
    artist = ax.text(
        0.5, 0.5, text,
        fontsize=23,
        rotation=90,
        ha="center",
        va="center",
        clip_on=False,
        color="#111111",
        **kwargs,
    )
    if title == "DejaVu stroke":
        artist.set_path_effects([
            path_effects.withStroke(linewidth=0.6, foreground="#111111")
        ])
    ax.set_title(title, fontsize=9)

fig.savefig("tmp/pdfs/glyph_variants.pdf", bbox_inches=None, pad_inches=0)
fig.savefig("tmp/pdfs/glyph_variants.png", dpi=600, bbox_inches=None, pad_inches=0)
