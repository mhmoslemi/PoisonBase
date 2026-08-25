#!/usr/bin/env python
"""
visualization/plotstyle.py -- the paper's visual identity, with NO torch import.

Split out of common.py so the results-only figures (which read csv files and
nothing else) start in under a second on a login node. common.py re-exports
everything here, so there is exactly one definition of the palette, the
typography and the save path.
"""

import csv
import os

import numpy as np

import matplotlib
matplotlib.use('Agg')                      # never depend on an interactive backend
import matplotlib.pyplot as plt


# Okabe-Ito based, colorblind safe. Every script uses the SAME entry for the
# same object, so a reader learns the visual identity once.
COLORS = {
    'random':  '#8C8C8C',       # neutral grey  -- the uninformed baseline
    'greedy':  '#E69F00',       # orange        -- pointwise selection
    'dpp':     '#0072B2',       # blue          -- ours
    'target':  '#D55E00',       # vermillion    -- the target x_t
    'pool':    '#C9C9C9',       # light grey    -- the candidate cloud
    'poison':  '#009E73',       # green         -- optimized poisons
    'tclass':  '#56B4E9',       # sky blue      -- target-class cloud
    'aclass':  '#CC79A7',       # purple        -- adversarial-class cloud
    'rule':    '#444444',
}
MARKERS = {'random': 'o', 'greedy': 's', 'dpp': 'D'}
LABELS = {'random': 'Random', 'greedy': 'Greedy', 'dpp': 'DPP (ours)'}
METHOD_ORDER = ['random', 'greedy', 'dpp']

# per-attack accents, used only where a point cloud must be split by attack
ATTACK_COLORS = {'fc': '#009E73', 'gradmatch': '#0072B2', 'sapa': '#CC79A7'}
ATTACK_LABELS = {'fc': 'FC', 'gradmatch': 'GM', 'sapa': 'SAPA'}
MODEL_MARKERS = {'ConvNetBN': 'o', 'ResNet20BN': '^', 'VGG13BN': 's'}
MODEL_LABELS = {'ConvNetBN': 'ConvNet', 'ResNet20BN': 'ResNet-20', 'VGG13BN': 'VGG-13'}

# perceptually uniform, colorblind friendly, no rainbow
CMAP_SIM = 'cividis'
CMAP_SEQ = 'viridis'

# ICLR geometry, in inches. The figure is AUTHORED at the width it is included
# at, so \includegraphics[width=\textwidth] never rescales it: rescaling is what
# silently shrinks every font in the figure (a 6.9in figure dropped into a 5.5in
# text block prints an 8pt label at 6.4pt).
WIDTH_FULL = 5.5        # ICLR's text block
WIDTH_HALF = 2.7
WIDTH_SINGLE = 2.7


def set_paper_style():
    """rcParams shared by every figure. Text stays vector text in the PDF."""
    plt.rcParams.update({
        'figure.dpi': 150,
        'savefig.dpi': 400,
        'pdf.fonttype': 42,          # TrueType: selectable / searchable vector text
        'ps.fonttype': 42,
        'svg.fonttype': 'none',
        'pdf.compression': 6,
        'text.usetex': False,
        'font.family': 'sans-serif',
        'font.sans-serif': ['DejaVu Sans', 'Helvetica', 'Arial', 'sans-serif'],
        'mathtext.fontset': 'dejavusans',
        'font.size': FS['label'],
        'axes.labelsize': FS['label'],
        'axes.titlesize': FS['title'],
        'xtick.labelsize': FS['tick'],
        'ytick.labelsize': FS['tick'],
        'legend.fontsize': FS['legend'],
        'legend.frameon': False,
        'legend.handlelength': 1.4,
        'legend.handletextpad': 0.5,
        'legend.columnspacing': 1.0,
        'legend.borderaxespad': 0.2,
        'axes.linewidth': 0.6,
        'axes.spines.top': False,
        'axes.spines.right': False,
        'axes.labelpad': 2.0,
        'xtick.major.width': 0.6,
        'ytick.major.width': 0.6,
        'xtick.major.size': 2.5,
        'ytick.major.size': 2.5,
        'xtick.direction': 'out',
        'ytick.direction': 'out',
        'grid.linewidth': 0.4,
        'grid.alpha': 0.35,
        'grid.color': '#B4B4B4',
        'lines.linewidth': 1.2,
        'lines.markersize': 3.0,
        'figure.facecolor': 'white',
        'axes.facecolor': 'white',
        'savefig.facecolor': 'white',
        'savefig.transparent': False,
    })


# One type scale for every figure. Nothing in a panel may be smaller than
# FS['note']: at ICLR's 10pt body text a 6pt annotation is unreadable in print.
# Sizes are FINAL printed sizes because the figure is never rescaled. Against
# ICLR's 10pt body text this is the usual range for figure type.
FS = {'label': 8.5, 'tick': 7.5, 'legend': 7.5, 'note': 7.5, 'panel': 10.0,
      'title': 8.5, 'small': 7.0}


def stats_box(ax, lines, loc='lower right', fontsize=None, color='black', pad=0.030):
    """A corner block of numbers, same look in every panel.

    Always drawn on an opaque white patch and always out of the layout, so it can
    sit over a point cloud without either colliding with it or deforming the axes.
    """
    fs = fontsize or FS['note']
    va, ha = ('bottom', 'left')
    x, y = pad, pad
    if 'upper' in loc:
        va, y = 'top', 1.0 - pad
    if 'right' in loc:
        ha, x = 'right', 1.0 - pad
    if 'center' in loc:
        ha, x = 'center', 0.5
    txt = lines if isinstance(lines, str) else '\n'.join(lines)
    a = ax.annotate(txt, xy=(x, y), xycoords='axes fraction', ha=ha, va=va,
                    fontsize=fs, color=color, linespacing=1.35,
                    bbox=dict(facecolor='white', alpha=0.85, edgecolor='none',
                              boxstyle='round,pad=0.25'))
    a.set_in_layout(False)
    return a


def bottom_legend(fig, handles, ncol=None, fontsize=None):
    """One legend for the whole figure, under the panels.

    Panels that each carry their own legend spend a third of their area on
    repeated keys; a single row below the figure is what a paper does.
    """
    fs = fontsize or FS['legend']
    n = ncol or len(handles)
    try:
        return fig.legend(handles=handles, loc='outside lower center', ncol=n,
                          fontsize=fs, frameon=False, handletextpad=0.4,
                          columnspacing=1.1, borderaxespad=0.1)
    except ValueError:                 # matplotlib without 'outside' locations
        return fig.legend(handles=handles, loc='lower center', ncol=n, fontsize=fs,
                          frameon=False, bbox_to_anchor=(0.5, -0.02))


# Marker geometry, one definition for the whole paper. Dense layers are small
# and semi-transparent; the layers a reader must count are larger and opaque.
SIZES = {'cloud': 1.6, 'set': 7.0, 'point': 11.0, 'big': 16.0, 'star': 95.0}
ALPHA = {'cloud': 0.16, 'set': 0.75, 'point': 0.85}


def kde_contours(ax, x, y, color, levels=(0.5, 0.9), grid=110, fill=True,
                 lw=0.6, alpha_fill=0.16, zorder=1, label=None, linestyle='-'):
    """Draw a point cloud as highest-density contours instead of dots.

    A few hundred overlapping dots print as a smear: the eye cannot tell 50
    points from 400, and at ICLR size the ink dominates whatever is drawn on top
    of it. Contours at fixed probability mass show the same distribution with a
    twentieth of the ink, and they stay readable when two clouds overlap.

    levels are FRACTIONS OF THE MASS (0.5 = the region containing half the
    points), computed from the density evaluated at the sample itself, so a
    contour means the same thing in every panel.
    """
    from scipy.stats import gaussian_kde
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if len(x) < 12:                       # too few points for a density estimate
        ax.scatter(x, y, s=SIZES['cloud'] * 2, color=color, alpha=0.5,
                   linewidths=0, zorder=zorder, label=label)
        return
    try:
        kde = gaussian_kde(np.vstack([x, y]))
    except Exception:                     # degenerate cloud (all points equal)
        ax.scatter(x, y, s=SIZES['cloud'] * 2, color=color, alpha=0.5,
                   linewidths=0, zorder=zorder, label=label)
        return
    pad_x = 0.12 * (x.max() - x.min() + 1e-9)
    pad_y = 0.12 * (y.max() - y.min() + 1e-9)
    gx = np.linspace(x.min() - pad_x, x.max() + pad_x, grid)
    gy = np.linspace(y.min() - pad_y, y.max() + pad_y, grid)
    GX, GY = np.meshgrid(gx, gy)
    Z = kde(np.vstack([GX.ravel(), GY.ravel()])).reshape(GX.shape)

    # threshold that encloses the requested share of the sample
    ds = np.sort(kde(np.vstack([x, y])))[::-1]
    cuts = []
    for lv in sorted(levels):
        k = max(1, int(round(lv * len(ds))))
        cuts.append(ds[k - 1])
    cuts = sorted(set(float(c) for c in cuts))
    if fill:
        ax.contourf(GX, GY, Z, levels=cuts + [Z.max() * 1.01], colors=[color],
                    alpha=alpha_fill, zorder=zorder)
    ax.contour(GX, GY, Z, levels=cuts, colors=[color], linewidths=lw,
               linestyles=linestyle, alpha=0.9, zorder=zorder + 0.1)
    if label is not None:                 # proxy so the cloud can be in a legend
        ax.plot([], [], color=color, linewidth=lw * 1.6, linestyle=linestyle,
                label=label)


def density_hexbin(ax, x, y, cmap='Greys', gridsize=44, mincnt=1, zorder=1,
                   alpha=0.9):
    """A dense cloud as a binned density, rasterized.

    Use where the axes are linear: hexbin bins in data coordinates, so it cannot
    be combined with a symlog axis.
    """
    hb = ax.hexbin(x, y, gridsize=gridsize, cmap=cmap, mincnt=mincnt,
                   linewidths=0.0, zorder=zorder, alpha=alpha,
                   norm=matplotlib.colors.LogNorm())
    hb.set_rasterized(True)
    return hb


def binned_trend(ax, x, y, bins=12, color='black', lw=1.2, zorder=5,
                 band=True, label=None):
    """Median of y in equal-count bins of x, with an interquartile ribbon.

    The honest summary of a dense cloud: it survives overplotting and states the
    trend the correlation coefficient reports.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    order = np.argsort(x)
    parts = np.array_split(order, bins)
    bx = np.array([np.median(x[p]) for p in parts if len(p)])
    med = np.array([np.median(y[p]) for p in parts if len(p)])
    if band:
        lo = np.array([np.percentile(y[p], 25) for p in parts if len(p)])
        hi = np.array([np.percentile(y[p], 75) for p in parts if len(p)])
        ax.fill_between(bx, lo, hi, color=color, alpha=0.14, linewidth=0,
                        zorder=zorder - 0.1)
    ax.plot(bx, med, color=color, linewidth=lw, zorder=zorder, label=label,
            solid_capstyle='round')
    return bx, med


def panel_label(ax, text, dx=-0.16, dy=1.04):
    """(a) / (b) / (c) in the panel's top-left corner."""
    # Kept IN the layout on purpose: the label is small, so the margin it asks
    # for is small, and leaving it out makes constrained_layout clip it against
    # the figure edge. The long captions are the ones that must opt out -- note().
    return ax.text(dx, dy, text, transform=ax.transAxes, fontsize=FS['panel'],
                   fontweight='bold', va='bottom', ha='left')


def note(ax, text, xy, ha='left', va='bottom', fontsize=None, color=None, **kwargs):
    """An annotation in axes coordinates that does NOT drive the layout.

    constrained_layout sizes every axes from its tight bbox, and a tight bbox
    includes annotations whose in_layout flag is set. A caption centred above a
    panel is therefore read as "this axes needs to be much wider than its cell",
    and the engine answers by shrinking the axes to a sliver. Every decorative
    label in these figures goes through here.
    """
    a = ax.annotate(text, xy=xy, xycoords='axes fraction', ha=ha, va=va,
                    fontsize=(fontsize or FS['note']),
                    color=(color or COLORS['rule']), **kwargs)
    a.set_in_layout(False)
    return a


def light_grid(ax, axis='y'):
    ax.grid(True, axis=axis, linestyle='-', linewidth=0.4, alpha=0.3, zorder=0)
    ax.set_axisbelow(True)


def identity_line(ax, lo, hi, color=None, lw=0.8):
    """y = x, the reference every paired scatter is read against."""
    ax.plot([lo, hi], [lo, hi], color=(color or COLORS['rule']), linewidth=lw,
            linestyle='--', alpha=0.7, zorder=1)


def save_fig(fig, out_dir, stem, dpi=400):
    """Vector PDF + >=300 dpi PNG, same stem. Returns the two paths."""
    os.makedirs(out_dir, exist_ok=True)
    pdf = os.path.join(out_dir, stem + '.pdf')
    png = os.path.join(out_dir, stem + '.png')
    fig.savefig(pdf)                       # fonttype 42 keeps text as text
    fig.savefig(png, dpi=max(300, dpi))
    print('  wrote %s' % pdf)
    print('  wrote %s' % png)
    return pdf, png


def save_csv(path, fieldnames, rows):
    """Every plotted point must be auditable later, so each figure writes one."""
    os.makedirs(os.path.dirname(os.path.abspath(path)) or '.', exist_ok=True)
    with open(path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, '') for k in fieldnames})
    print('  wrote %s (%d rows)' % (path, len(rows)))
