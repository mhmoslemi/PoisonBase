from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt


HERE = Path(__file__).resolve().parent
fontsets = ("dejavuserif", "stix", "cm", "stixsans", "dejavusans")

fig, axes = plt.subplots(len(fontsets), 1, figsize=(4, 10), dpi=300)
for ax, fontset in zip(axes, fontsets):
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_axis_off()
    ax.text(0.08, 0.5, fontset, fontsize=14, va="center")
    ax.text(
        0.72,
        0.5,
        r"$A$",
        fontsize=23,
        rotation=90,
        ha="center",
        va="center",
        clip_on=False,
        math_fontfamily=fontset,
    )
fig.savefig(HERE / "fontset-comparison.pdf", bbox_inches=None, pad_inches=0)
fig.savefig(HERE / "fontset-comparison.png", bbox_inches=None, pad_inches=0, dpi=600)
