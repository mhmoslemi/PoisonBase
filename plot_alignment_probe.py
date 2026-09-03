#!/usr/bin/env python
"""Figures for the alignment_probe.py audit.

Reads whatever alignment_probe.py has written under --in_dir and produces the
panels that answer the reviewer note on Section 3.2. Each figure also writes a
sibling .csv holding exactly the numbers plotted, so every claim in the rebuttal
can be quoted from a table rather than read off a chart.

    python plot_alignment_probe.py                       # every figure
    python plot_alignment_probe.py --figures 3 7         # just those
    python plot_alignment_probe.py --in_dir alignment_probe_result \
                                   --out_dir alignment_probe_figs --format pdf png

Figures
  1  identity        brute-force <g_i,g_t> vs head_W+head_b+A, float64.
                     The claim that A_i is exactly the backbone half of Eq. (3).
  2  decomposition   what share of |<g_i,g_t>| the classifier-head term carries,
                     i.e. how much A_i leaves on the table, per combination.
  3  rank_vs_exact   Spearman(each Algorithm-1 ingredient, exact <g_i,g_t>).
                     The central "how loosely connected" panel.
  4  ntk_claim       <h_i,h_t> vs tr(J_i J_t^T). The paper asserts these are
                     strongly correlated; this measures it.
  5  margin_proxy    M_i against ||r_i|| and against <r_i,r_t>. The reviewer's
                     point that M_i is an indirect proxy that ignores r_t.
  6  scale           raw magnitudes of R, M and A on one log axis, i.e. the
                     "not on an evidently common scale" objection.
  7  topm_overlap    top-m agreement between each score and a ranking by the
                     exact Eq. (3) quantity, against a random-selection floor.
  8  selected_quality mean exact-alignment percentile of the bases each score
                     actually selects.

Figures 7 and 8 carry three rankings, which are NOT three methods. Two are GRAFT
and GRAFT+ as final_update.py runs them; the third is the same method scored by
the formula Algorithm 1 prints, which uses R_i = <h_i,h_t> where the code uses
1 - cos(h_i,h_t). See the SCORES comment for why those are not the same ranking.

Nothing here is specific to one dataset: every combination directory found under
--in_dir is picked up, and figures that need a single combination take the first
one unless --combo names another.
"""

import argparse
import csv
import glob
import json
import os
import re

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm

try:
    from scipy.stats import spearmanr
except Exception:
    spearmanr = None


# --------------------------------------------------------------------------- #
# palette
#
# Slots 1-3 of the reference categorical theme, used in fixed order and never
# cycled. These three are the documented all-pairs-validated subset (CVD dE 9.2 /
# normal-vision dE 24.0 on the light surface), which is what scatter and facetted
# forms require. Slot 4 is only ever used where marks sit adjacent (boxes, grouped
# bars), which validates on the adjacent pairlist.
#
# Aqua and yellow fall below 3:1 contrast on the light surface, so the relief rule
# applies throughout: every categorical figure carries direct value labels and a
# companion CSV.
# --------------------------------------------------------------------------- #

SERIES = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100']   # blue, orange, aqua, yellow
SURFACE = '#fcfcfb'
INK = '#0b0b0b'
INK_2 = '#52514e'
INK_MUTED = '#8a8880'
GRID = '#e4e3df'

# sequential: one hue, light -> dark. Used for density and magnitude only.
SEQ = LinearSegmentedColormap.from_list(
    'seq_blue', ['#eef4fc', '#c3d9f4', '#8fb8e8', '#5896dd', '#2a78d6', '#1a4f8f'])

# diverging: two hues either side of a neutral gray midpoint. Never a hue at the
# midpoint, never a rainbow. Used for correlations, whose job is polarity.
DIV = LinearSegmentedColormap.from_list(
    'div_orange_blue',
    ['#a83c14', '#eb6834', '#f6b394', '#ecebe7', '#9dc0ea', '#2a78d6', '#1a4f8f'])


def style():
    plt.rcParams.update({
        'figure.facecolor': SURFACE, 'axes.facecolor': SURFACE,
        'savefig.facecolor': SURFACE,
        'font.family': 'DejaVu Sans', 'font.size': 8,
        'axes.titlesize': 9, 'axes.labelsize': 8,
        'axes.titleweight': 'semibold', 'axes.titlelocation': 'left',
        'axes.titlepad': 8,
        'axes.edgecolor': GRID, 'axes.linewidth': 0.8,
        'axes.labelcolor': INK_2, 'text.color': INK,
        'xtick.color': INK_2, 'ytick.color': INK_2,
        'xtick.labelsize': 7, 'ytick.labelsize': 7,
        'xtick.major.size': 0, 'ytick.major.size': 0,
        'legend.frameon': False, 'legend.fontsize': 7,
        'grid.color': GRID, 'grid.linewidth': 0.6,
        'lines.linewidth': 2.0, 'lines.solid_capstyle': 'round',
        'figure.dpi': 160,
    })


def clean(ax, grid='y'):
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)
    for s in ('left', 'bottom'):
        ax.spines[s].set_color(GRID)
    if grid:
        ax.set_axisbelow(True)
        ax.grid(True, axis=grid, alpha=0.9)
        ax.grid(False, axis='x' if grid == 'y' else 'y')


# --------------------------------------------------------------------------- #
# loading
# --------------------------------------------------------------------------- #

QUANTITIES = ['A', 'jker', 'ntk_tr', 'hTh', 'd_cos', 'margin', 'r_norm', 'r_dot',
              'head_W', 'g_full']

# how each Algorithm-1 ingredient is meant to point, so a Spearman against the
# exact objective is signed the way the score uses it
ORIENT = {'A': +1, 'jker': +1, 'ntk_tr': +1, 'hTh': +1, 'd_cos': -1,
          'margin': -1, 'r_norm': +1, 'r_dot': +1, 'head_W': +1, 'g_full': +1}

LABEL = {
    'A': r'$A_i=\langle\nabla_\phi\ell_i,\nabla_\phi\mathcal{L}_t\rangle$',
    'jker': r'$u_t^{\top}J_iJ_t^{\top}u_t$',
    'ntk_tr': r'$\mathrm{tr}(J_iJ_t^{\top})$',
    'hTh': r'$R_i=\langle h_i,h_t\rangle$',
    'd_cos': r'$1-\cos(h_i,h_t)$  (as coded)',
    'margin': r'$M_i$  (logit margin)',
    'r_norm': r'$\|r_i\|$',
    'r_dot': r'$\langle r_i,r_t\rangle$',
    'head_W': r'$(r_i\!\cdot\!r_t)(h_i\!\cdot\!h_t)$',
    'g_full': r'exact $\langle g_i,g_t\rangle$',
}


def parse_combo(name):
    parts = name.split('_', 2)
    return parts if len(parts) == 3 else [name, '', '']


def load(in_dir):
    """[{name, dataset, model, pair, targets: {tidx: npz-dict}}] for every combo."""
    combos = []
    for d in sorted(glob.glob(os.path.join(in_dir, '*'))):
        if not os.path.isdir(d):
            continue
        files = sorted(glob.glob(os.path.join(d, 'target_*.npz')),
                       key=lambda p: int(re.search(r'target_(\d+)', p).group(1)))
        if not files:
            continue
        name = os.path.basename(d)
        meta = {}
        mp = os.path.join(d, 'meta.json')
        if os.path.isfile(mp):
            try:
                meta = json.load(open(mp))
            except Exception:
                meta = {}
        ds, model, pair = (meta.get('dataset'), meta.get('model'),
                           meta.get('class_pair'))
        if not (ds and model and pair):
            ds, model, pair = parse_combo(name)
        targets = {}
        for f in files:
            ti = int(re.search(r'target_(\d+)', f).group(1))
            z = np.load(f)
            targets[ti] = {k: z[k] for k in z.files}
        combos.append(dict(name=name, dataset=ds, model=model, pair=pair,
                           short='%s/%s' % (model, pair), targets=targets,
                           meta=meta, path=d))
    if not combos:
        raise SystemExit('no combination directories with target_*.npz under %s'
                         % in_dir)
    return combos


def sp(a, b):
    """Spearman, nan-safe, on the finite overlap."""
    a, b = np.asarray(a, float).ravel(), np.asarray(b, float).ravel()
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 8 or np.ptp(a[ok]) == 0 or np.ptp(b[ok]) == 0:
        return np.nan
    if spearmanr is not None:
        return float(spearmanr(a[ok], b[ok]).statistic)
    ra = np.argsort(np.argsort(a[ok])).astype(float)
    rb = np.argsort(np.argsort(b[ok])).astype(float)
    return float(np.corrcoef(ra, rb)[0, 1])


def write_table(path, header, rows):
    with open(path, 'w', newline='') as fh:
        w = csv.writer(fh)
        w.writerow(header)
        w.writerows(rows)


def save(fig, out_dir, stem, formats):
    os.makedirs(out_dir, exist_ok=True)
    for ext in formats:
        fig.savefig(os.path.join(out_dir, '%s.%s' % (stem, ext)),
                    bbox_inches='tight', pad_inches=0.02)
    plt.close(fig)
    print('  wrote %s.{%s}' % (os.path.join(out_dir, stem), ','.join(formats)))


# --------------------------------------------------------------------------- #
# 1. identity
# --------------------------------------------------------------------------- #

def fig_identity(combos, out_dir, formats, **_):
    bru, cla, tags = [], [], []
    for c in combos:
        for ti, t in c['targets'].items():
            if 'verify_gfull_brute_f64' not in t:
                continue
            b = t['verify_gfull_brute_f64'].ravel()
            q = t['verify_gfull_claim_f64'].ravel()
            bru.append(b)
            cla.append(q)
            tags += [(c['short'], ti)] * len(b)
    if not bru:
        print('  [1] skipped: no verify_* arrays (the run used --verify 0)')
        return
    b, q = np.concatenate(bru), np.concatenate(cla)
    resid = np.abs(b - q)
    scale = float(np.sqrt(np.mean(b ** 2)))

    fig, axes = plt.subplots(1, 2, figsize=(5.6, 2.5))

    ax = axes[0]
    lim_lo = max(min(np.abs(b).min(), np.abs(q).min()), 1e-18)
    lim_hi = max(np.abs(b).max(), np.abs(q).max()) * 2
    ax.plot([lim_lo, lim_hi], [lim_lo, lim_hi], color=INK_MUTED, lw=1.0,
            ls=(0, (4, 3)), zorder=1)
    # one series -> no legend box; the title names it
    ax.scatter(np.abs(b), np.abs(q), s=14, facecolor=SERIES[0], alpha=0.55,
               linewidths=0.6, edgecolors=SURFACE, zorder=2)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlim(lim_lo, lim_hi)
    ax.set_ylim(lim_lo, lim_hi)
    ax.set_xlabel(r'brute force  $|\langle g_i,g_t\rangle|$')
    ax.set_ylabel(r'$|\,$head$_W$ + head$_b$ + $A_i|$')
    ax.set_title('Eq. (3) reproduces the exact alignment')
    clean(ax, grid='both')
    ax.text(0.04, 0.94, 'identity line', transform=ax.transAxes, color=INK_MUTED,
            fontsize=7, va='top')

    ax = axes[1]
    pos = resid[resid > 0]
    if len(pos):
        bins = np.logspace(np.log10(pos.min()), np.log10(max(pos.max(), pos.min() * 10)), 24)
        ax.hist(pos, bins=bins, color=SERIES[0], edgecolor=SURFACE, linewidth=0.6)
        ax.set_xscale('log')
    ax.set_xlabel('absolute residual (float64)')
    ax.set_ylabel('checked candidates')
    ax.set_title('Residual is at machine precision')
    clean(ax, grid='y')
    ax.text(0.96, 0.94,
            'n = %d\nmax %.1e\nrel. to RMS %.1e' % (len(b), resid.max(),
                                                    resid.max() / max(scale, 1e-30)),
            transform=ax.transAxes, ha='right', va='top', fontsize=7, color=INK_2)

    fig.tight_layout()
    save(fig, out_dir, 'fig1_identity', formats)
    write_table(os.path.join(out_dir, 'fig1_identity.csv'),
                ['combo', 'target', 'brute_f64', 'claim_f64', 'abs_residual'],
                [[t[0], t[1], '%.17g' % x, '%.17g' % y, '%.3e' % abs(x - y)]
                 for t, x, y in zip(tags, b, q)])


# --------------------------------------------------------------------------- #
# 2. decomposition shares
# --------------------------------------------------------------------------- #

def fig_decomposition(combos, out_dir, formats, **_):
    rows = []
    for c in combos:
        h, a = [], []
        for t in c['targets'].values():
            h.append(np.abs(t['head_W'] + t['head_b']).ravel())
            a.append(np.abs(t['A']).ravel())
        h, a = np.concatenate(h), np.concatenate(a)
        tot = h + a
        ok = tot > 0
        share_head = float(np.median(h[ok] / tot[ok]))
        rows.append((c['short'], share_head, 1.0 - share_head, int(ok.sum())))
    rows.sort(key=lambda r: r[1])

    fig, ax = plt.subplots(figsize=(5.6, 0.34 * len(rows) + 1.5))
    y = np.arange(len(rows))
    head = np.array([r[1] for r in rows])
    back = np.array([r[2] for r in rows])
    # 2px surface gap between adjacent stacked segments
    ax.barh(y, head, height=0.6, color=SERIES[0], label='classifier head  '
            r'$(r_i\!\cdot\!r_t)(h_i\!\cdot\!h_t + 1)$', zorder=2)
    ax.barh(y, back, height=0.6, left=head, color=SERIES[1],
            label=r'backbone  $A_i$', zorder=2,
            edgecolor=SURFACE, linewidth=2.0)
    for i, (h_, b_) in enumerate(zip(head, back)):
        ax.text(h_ / 2, i, '%.0f%%' % (100 * h_), ha='center', va='center',
                fontsize=7, color='white')
        ax.text(h_ + b_ / 2, i, '%.0f%%' % (100 * b_), ha='center', va='center',
                fontsize=7, color='white')
    ax.set_yticks(y)
    ax.set_yticklabels([r[0] for r in rows])
    ax.set_xlim(0, 1)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_xticklabels(['0', '25%', '50%', '75%', '100%'])
    ax.set_xlabel(r'median share of $|\langle g_i,g_t\rangle|$ magnitude')
    ax.set_title(r'$A_i$ keeps the backbone term and drops the classifier term')
    clean(ax, grid='x')
    ax.legend(loc='lower center', bbox_to_anchor=(0.5, 1.06), ncol=2)
    fig.tight_layout()
    save(fig, out_dir, 'fig2_decomposition', formats)
    write_table(os.path.join(out_dir, 'fig2_decomposition.csv'),
                ['combo', 'median_share_head', 'median_share_backbone', 'n'],
                [[r[0], '%.4f' % r[1], '%.4f' % r[2], r[3]] for r in rows])


# --------------------------------------------------------------------------- #
# 3. rank agreement with the exact objective
# --------------------------------------------------------------------------- #

def fig_rank_vs_exact(combos, out_dir, formats, **_):
    quants = ['A', 'jker', 'ntk_tr', 'hTh', 'd_cos', 'margin', 'r_norm', 'r_dot',
              'head_W']
    M = np.full((len(quants), len(combos)), np.nan)
    for j, c in enumerate(combos):
        for i, q in enumerate(quants):
            vals = []
            for t in c['targets'].values():
                if q not in t:
                    continue
                g, x = t['g_full'], t[q]
                for k in range(g.shape[0]):          # per surrogate, then average
                    r = sp(ORIENT[q] * x[k], g[k])
                    if np.isfinite(r):
                        vals.append(r)
            if vals:
                M[i, j] = float(np.mean(vals))

    fig, ax = plt.subplots(figsize=(1.05 * len(combos) + 3.2, 0.42 * len(quants) + 1.6))
    im = ax.imshow(M, cmap=DIV, norm=TwoSlopeNorm(vmin=-1, vcenter=0, vmax=1),
                   aspect='auto')
    ax.set_xticks(range(len(combos)))
    ax.set_xticklabels([c['short'] for c in combos], rotation=35, ha='right')
    ax.set_yticks(range(len(quants)))
    ax.set_yticklabels([LABEL[q] for q in quants])
    for i in range(len(quants)):
        for j in range(len(combos)):
            if np.isfinite(M[i, j]):
                ax.text(j, i, '%.2f' % M[i, j], ha='center', va='center', fontsize=6.5,
                        color='white' if abs(M[i, j]) > 0.55 else INK)
    ax.set_xticks(np.arange(-.5, len(combos), 1), minor=True)
    ax.set_yticks(np.arange(-.5, len(quants), 1), minor=True)
    ax.grid(which='minor', color=SURFACE, linewidth=2.0)
    ax.tick_params(which='minor', length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_title(r'Rank agreement with the exact $\langle g_i,g_t\rangle$'
                 '\n' r'Spearman $\rho$, signed as the score uses each term, '
                 'averaged over surrogates and targets')
    cb = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02,
                      ticks=[-1, -0.5, 0, 0.5, 1])
    cb.outline.set_visible(False)
    cb.ax.tick_params(length=0, labelsize=7)
    fig.tight_layout()
    save(fig, out_dir, 'fig3_rank_vs_exact', formats)
    write_table(os.path.join(out_dir, 'fig3_rank_vs_exact.csv'),
                ['quantity'] + [c['short'] for c in combos],
                [[q] + ['%.4f' % M[i, j] for j in range(len(combos))]
                 for i, q in enumerate(quants)])


# --------------------------------------------------------------------------- #
# 4. the NTK claim
# --------------------------------------------------------------------------- #

def _hexpanel(ax, x, y, xlabel, ylabel, title):
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    hb = ax.hexbin(x, y, gridsize=42, cmap=SEQ, mincnt=1, linewidths=0)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    clean(ax, grid='both')
    r = sp(x, y)
    ax.text(0.04, 0.95, r'Spearman $\rho$ = %.2f' % r, transform=ax.transAxes,
            va='top', fontsize=7.5, color=INK,
            bbox=dict(boxstyle='round,pad=0.25', fc=SURFACE, ec=GRID, lw=0.6))
    return hb, r


def fig_ntk_claim(combos, out_dir, formats, **_):
    usable = [c for c in combos
              if any(np.isfinite(t.get('ntk_tr', np.array([np.nan]))).any()
                     for t in c['targets'].values())]
    if not usable:
        print('  [4] skipped: no finite ntk_tr (the run used --ntk_probes 0)')
        return
    n = min(3, len(usable))
    fig, axes = plt.subplots(1, n, figsize=(2.15 * n + 0.9, 2.5), squeeze=False)
    rows = []
    for ax, c in zip(axes[0], usable[:n]):
        x = np.concatenate([t['hTh'].ravel() for t in c['targets'].values()])
        y = np.concatenate([t['ntk_tr'].ravel() for t in c['targets'].values()])
        hb, r = _hexpanel(ax, x, y, r'$\langle h_i,h_t\rangle$',
                          r'$\mathrm{tr}(J_iJ_t^{\top})$', c['short'])
        rows.append([c['short'], '%.4f' % r, int(np.isfinite(x + y).sum())])
    cb = fig.colorbar(hb, ax=axes[0].tolist(), fraction=0.02, pad=0.02)
    cb.set_label('candidates', fontsize=7)
    cb.outline.set_visible(False)
    cb.ax.tick_params(length=0, labelsize=7)
    fig.suptitle('The paper asserts these two are strongly correlated', x=0.02,
                 ha='left', fontsize=9, fontweight='semibold')
    save(fig, out_dir, 'fig4_ntk_claim', formats)
    write_table(os.path.join(out_dir, 'fig4_ntk_claim.csv'),
                ['combo', 'spearman_hTh_vs_tr_JiJt', 'n'], rows)


# --------------------------------------------------------------------------- #
# 5. M_i as a residual proxy
# --------------------------------------------------------------------------- #

def fig_margin_proxy(combos, out_dir, formats, combo=None, **_):
    c = pick(combos, combo)
    m = np.concatenate([t['margin'].ravel() for t in c['targets'].values()])
    rn = np.concatenate([t['r_norm'].ravel() for t in c['targets'].values()])
    rd = np.concatenate([t['r_dot'].ravel() for t in c['targets'].values()])
    fig, axes = plt.subplots(1, 2, figsize=(5.6, 2.5))
    hb1, r1 = _hexpanel(axes[0], m, rn, r'$M_i$  (logit margin)', r'$\|r_i\|$',
                        r'$M_i$ vs the residual magnitude it proxies')
    hb2, r2 = _hexpanel(axes[1], m, rd, r'$M_i$  (logit margin)',
                        r'$\langle r_i,r_t\rangle$',
                        r'$M_i$ vs the residual coupling it ignores')
    axes[1].set_yscale('symlog', linthresh=1e-6)
    cb = fig.colorbar(hb2, ax=axes.tolist(), fraction=0.02, pad=0.02)
    cb.set_label('candidates', fontsize=7)
    cb.outline.set_visible(False)
    cb.ax.tick_params(length=0, labelsize=7)
    fig.suptitle('%s' % c['short'], x=0.02, ha='left', fontsize=9,
                 fontweight='semibold')
    save(fig, out_dir, 'fig5_margin_proxy', formats)
    write_table(os.path.join(out_dir, 'fig5_margin_proxy.csv'),
                ['combo', 'spearman_margin_vs_r_norm', 'spearman_margin_vs_r_dot', 'n'],
                [[c['short'], '%.4f' % r1, '%.4f' % r2, len(m)]])


# --------------------------------------------------------------------------- #
# 6. scale mismatch
# --------------------------------------------------------------------------- #

def fig_scale(combos, out_dir, formats, combo=None, **_):
    c = pick(combos, combo)
    quants = ['hTh', 'margin', 'A', 'g_full']
    data, rows = [], []
    for q in quants:
        v = np.abs(np.concatenate([t[q].ravel() for t in c['targets'].values()]))
        v = v[np.isfinite(v) & (v > 0)]
        data.append(v)
        rows.append([q, '%.4g' % np.percentile(v, 5), '%.4g' % np.median(v),
                     '%.4g' % np.percentile(v, 95), len(v)])

    fig, ax = plt.subplots(figsize=(5.0, 2.6))
    bp = ax.boxplot(data, vert=True, widths=0.5, patch_artist=True, showfliers=False,
                    medianprops=dict(color=SURFACE, lw=1.6),
                    whiskerprops=dict(color=INK_MUTED, lw=1.0),
                    capprops=dict(color=INK_MUTED, lw=1.0))
    for patch, col in zip(bp['boxes'], SERIES):
        patch.set_facecolor(col)
        patch.set_edgecolor(SURFACE)      # 2px surface gap between adjacent fills
        patch.set_linewidth(2.0)
    ax.set_yscale('log')
    ax.set_xticks(range(1, len(quants) + 1))
    ax.set_xticklabels([LABEL[q].replace('  (logit margin)', '') for q in quants],
                       fontsize=7.5)
    ax.set_ylabel('absolute raw value  (log)')
    ax.set_title('The score adds three terms that live orders of magnitude apart\n'
                 '%s, before standardization' % c['short'])
    clean(ax, grid='y')
    for i, v in enumerate(data, start=1):
        ax.text(i, np.median(v), '  %.2g' % np.median(v), fontsize=6.5,
                color=INK_2, va='center', ha='left')
    fig.tight_layout()
    save(fig, out_dir, 'fig6_scale', formats)
    write_table(os.path.join(out_dir, 'fig6_scale.csv'),
                ['quantity', 'p5_abs', 'median_abs', 'p95_abs', 'n'], rows)


# --------------------------------------------------------------------------- #
# 7. top-m overlap with the exact ranking
# --------------------------------------------------------------------------- #

# These are not three methods. The first two are GRAFT and GRAFT+ exactly as
# final_update.py runs them. The third is the SAME method scored by the formula
# Algorithm 1 prints, which differs in one term:
#
#   printed   line 6:  R_i = <h_i, h_t>          -> s_i = R~ - M~ + beta*A~, take largest
#   shipped         :  d_i = 1 - cos(h_i, h_t)   -> cost = d~ + lam*M~ - beta*A~, take smallest
#
# -d is an affine function of cos, so standardize(-d) == standardize(cos), whereas
# standardize(<h_i,h_t>) also carries ||h_i||. The two rank identically only if the
# candidate feature norms are constant. Plotting both says whether the mismatch
# between the printed algorithm and the shipped one changes anything.
SCORES = [
    ('cost_prod_b0', 'GRAFT  (as implemented, beta=0)', True),
    ('cost_prod_b1', 'GRAFT+  (as implemented, beta=1)', True),
    ('s_paper_b1', r'GRAFT+ with $R_i=\langle h_i,h_t\rangle$  (Alg. 1 as printed)',
     False),
]


def _overlap_curve(score, ref, lower_better, ms):
    order_s = np.argsort(score if lower_better else -score)
    order_r = np.argsort(-ref)
    out = []
    for m in ms:
        out.append(len(set(order_s[:m].tolist()) & set(order_r[:m].tolist())) / m)
    return np.asarray(out)


def fig_topm_overlap(combos, out_dir, formats, **_):
    n = min(3, len(combos))
    fig, axes = plt.subplots(1, n, figsize=(2.3 * n + 0.6, 2.7), squeeze=False,
                             sharey=True)
    rows = []
    for ax, c in zip(axes[0], combos[:n]):
        N = len(next(iter(c['targets'].values()))['cand_idx'])
        ms = np.unique(np.round(np.logspace(np.log10(5), np.log10(N / 2), 22))
                       .astype(int))
        for si, (key, label, lower) in enumerate(SCORES):
            curves = [_overlap_curve(t[key], t['s_eq3'], lower, ms)
                      for t in c['targets'].values() if key in t]
            if not curves:
                continue
            y = np.mean(curves, axis=0)
            ax.plot(ms, y, color=SERIES[si], label=label, zorder=3)
            rows += [[c['short'], label, int(m), '%.4f' % v] for m, v in zip(ms, y)]
        ax.plot(ms, ms / N, color=INK_MUTED, lw=1.2, ls=(0, (4, 3)), zorder=2,
                label='random selection')
        rows += [[c['short'], 'random selection', int(m), '%.4f' % (m / N)]
                 for m in ms]
        ax.set_xscale('log')
        ax.set_ylim(0, 1)
        ax.set_xlabel('poison-set size  m')
        ax.set_title(c['short'])
        clean(ax, grid='both')
    axes[0][0].set_ylabel(r'overlap with top-$m$ by exact $\langle g_i,g_t\rangle$')
    axes[0][0].legend(loc='upper left', bbox_to_anchor=(0, -0.28), ncol=2)
    fig.suptitle('How much of the stated objective does the score actually pick?',
                 x=0.02, ha='left', fontsize=9, fontweight='semibold')
    fig.tight_layout()
    save(fig, out_dir, 'fig7_topm_overlap', formats)
    write_table(os.path.join(out_dir, 'fig7_topm_overlap.csv'),
                ['combo', 'ranking', 'm', 'overlap'], rows)


# --------------------------------------------------------------------------- #
# 8. quality of the selected set
# --------------------------------------------------------------------------- #

def fig_selected_quality(combos, out_dir, formats, top_m=250, **_):
    labels = [s[1] for s in SCORES] + ['ranking by exact ' r'$\langle g_i,g_t\rangle$']
    vals = np.full((len(labels), len(combos)), np.nan)
    for j, c in enumerate(combos):
        acc = [[] for _ in labels]
        for t in c['targets'].values():
            ref = t['s_eq3']
            pct = np.argsort(np.argsort(ref)) / (len(ref) - 1.0)   # 0..1
            m = min(top_m, len(ref) // 2)
            for si, (key, _lab, lower) in enumerate(SCORES):
                if key not in t:
                    continue
                order = np.argsort(t[key] if lower else -t[key])[:m]
                acc[si].append(float(np.mean(pct[order])))
            acc[-1].append(float(np.mean(pct[np.argsort(-ref)[:m]])))
        for i in range(len(labels)):
            if acc[i]:
                vals[i, j] = float(np.mean(acc[i]))

    x = np.arange(len(combos))
    w = 0.8 / len(labels)
    fig, ax = plt.subplots(figsize=(1.15 * len(combos) + 2.6, 2.9))
    colors = SERIES[:len(SCORES)] + [INK_MUTED]
    for i, lab in enumerate(labels):
        pos = x - 0.4 + w * (i + 0.5)
        ax.bar(pos, vals[i], width=w, color=colors[i], label=lab, zorder=3,
               edgecolor=SURFACE, linewidth=2.0)   # 2px gap between adjacent bars
        for xi, v in zip(pos, vals[i]):
            if np.isfinite(v):
                ax.text(xi, v + 0.015, '%.2f' % v, ha='center', va='bottom',
                        fontsize=6, color=INK_2, rotation=90)
    ax.axhline(0.5, color=INK_MUTED, lw=1.0, ls=(0, (4, 3)), zorder=2)
    ax.text(len(combos) - 0.45, 0.515, 'random selection', fontsize=6.5,
            color=INK_MUTED, ha='right')
    ax.set_xticks(x)
    ax.set_xticklabels([c['short'] for c in combos], rotation=25, ha='right')
    ax.set_ylim(0, 1.12)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_ylabel('mean alignment percentile of the\nselected bases  (m = %d)' % top_m)
    ax.set_title('Does the score select the bases the objective wants?')
    clean(ax, grid='y')
    ax.legend(loc='lower center', bbox_to_anchor=(0.5, 1.08), ncol=2)
    fig.tight_layout()
    save(fig, out_dir, 'fig8_selected_quality', formats)
    write_table(os.path.join(out_dir, 'fig8_selected_quality.csv'),
                ['ranking'] + [c['short'] for c in combos],
                [[labels[i]] + ['%.4f' % vals[i, j] for j in range(len(combos))]
                 for i in range(len(labels))])


# --------------------------------------------------------------------------- #

def pick(combos, name):
    if not name:
        return combos[0]
    for c in combos:
        if name in (c['name'], c['short']):
            return c
    raise SystemExit('no combination %r; have: %s'
                     % (name, ', '.join(c['name'] for c in combos)))


FIGURES = {
    1: fig_identity, 2: fig_decomposition, 3: fig_rank_vs_exact, 4: fig_ntk_claim,
    5: fig_margin_proxy, 6: fig_scale, 7: fig_topm_overlap, 8: fig_selected_quality,
}


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--in_dir', default='alignment_probe_result')
    p.add_argument('--out_dir', default='alignment_probe_figs')
    p.add_argument('--figures', nargs='*', type=int, default=sorted(FIGURES),
                   help='which panels to draw (default: all)')
    p.add_argument('--format', nargs='+', default=['pdf', 'png'])
    p.add_argument('--combo', default=None,
                   help='combination for the single-combination panels (5, 6); '
                        'default is the first one found')
    p.add_argument('--top_m', type=int, default=250,
                   help='poison-set size for figure 8')
    args = p.parse_args()

    style()
    combos = load(args.in_dir)
    print('loaded %d combination(s): %s'
          % (len(combos), ', '.join('%s [%d targets]' % (c['short'], len(c['targets']))
                                    for c in combos)))
    os.makedirs(args.out_dir, exist_ok=True)
    for n in args.figures:
        if n not in FIGURES:
            print('  [%s] no such figure' % n)
            continue
        print('[%d] %s' % (n, FIGURES[n].__name__))
        try:
            FIGURES[n](combos, args.out_dir, args.format, combo=args.combo,
                       top_m=args.top_m)
        except Exception as exc:                       # one bad panel is not fatal
            print('  [%d] FAILED: %s: %s' % (n, type(exc).__name__, exc))


if __name__ == '__main__':
    main()
