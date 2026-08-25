#!/usr/bin/env python
"""
fig_results_paired_scatter.py -- "selection helps target by target, not on
average only".

Every point is one ATTACKED TARGET: its attack success rate under Random base
selection on the x-axis, and under our DPP selector on the y-axis, computed over
the SAME victim seeds. A point above the diagonal is a target that got easier to
poison purely by choosing which clean images to perturb -- the perturbation
budget, the attack objective, the crafting hyper-parameters and the victim
training are identical between the two arms.

Reads nothing but <out_dir>/*/results.csv through results_index.py: no gpu, no
dataset, no model. A full run takes a couple of seconds.

Panels
------
(a) one point per (configuration, target), all models / attacks / pairs / budgets
    that have both arms. Colour = attack, marker = architecture. Points are
    jittered because per-target ASR is a multiple of 1/n_victims, so identical
    values would otherwise stack invisibly.
(b) one point per configuration -- the unit the paper's main table reports --
    with the mean over its targets. Colour = poison budget.
(c) the paired per-target difference by budget, which is where the effect is
    largest: at 1e-3 the clean-base choice decides almost everything, and the
    gap narrows as the budget grows.

Statistics: targets inside one configuration are not independent, so panel (a)
reports a sign count and a Wilcoxon signed-rank test over targets, while panel
(b) shows the configuration-level view. Neither is presented as an i.i.d. sample.
"""

import argparse
import collections
import os

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.lines import Line2D
from scipy.stats import wilcoxon

import plotstyle as P
import results_index as R


STEM = 'fig_results_paired_scatter'
HERE = os.path.dirname(os.path.abspath(__file__))


def plot(args):
    rows = R.load(args.index_csv, args.results_dir, refresh=args.refresh)
    pts = R.paired_targets(rows, args.selection_a, args.selection_b,
                           args.min_victims)
    if not pts:
        raise SystemExit('no paired (configuration, target) points for %s vs %s'
                         % (args.selection_a, args.selection_b))
    x = np.array([p['asr_a'] for p in pts])
    y = np.array([p['asr_b'] for p in pts])
    gain = y - x
    n_cfg = len({(p['model'], p['attack'], p['pair'], p['budget'], p['tags'])
                 for p in pts})
    wins, ties = int((gain > 0).sum()), int((gain == 0).sum())
    losses = len(gain) - wins - ties
    try:
        stat = wilcoxon(y, x, zero_method='wilcox', alternative='greater')
        ptxt = ('$p<10^{-16}$' if stat.pvalue < 1e-16 else '$p=%.1e$' % stat.pvalue)
    except ValueError:
        ptxt = ''
    print('  %d paired targets in %d configurations: %s better on %d, tied %d, '
          'worse %d; mean gain %+.1f points'
          % (len(pts), n_cfg, args.selection_b, wins, ties, losses, gain.mean()))

    P.set_paper_style()
    rng = np.random.default_rng(args.viz_seed)
    # Authored at ICLR's text width, so nothing is rescaled at include time.
    # Two square scatters on top, the budget strip across the bottom.
    fig = plt.figure(figsize=(P.WIDTH_FULL, 4.30), constrained_layout=True)
    outer = fig.add_gridspec(2, 1, height_ratios=[1.55, 1.0], hspace=0.16)
    top = outer[0].subgridspec(1, 2, wspace=0.24)
    ax_a = fig.add_subplot(top[0, 0])
    ax_b = fig.add_subplot(top[0, 1])
    ax_c = fig.add_subplot(outer[1])

    # ---- (a) every target -------------------------------------------------
    # jitter is cosmetic only: per-target ASR is a multiple of 1/n_victims, so
    # identical values would stack into one dot. The csv keeps the exact values.
    j = args.jitter
    for att in ('fc', 'gradmatch', 'sapa'):
        for mod in ('ConvNetBN', 'ResNet20BN', 'VGG13BN'):
            sel = [i for i, p in enumerate(pts)
                   if p['attack'] == att and p['model'] == mod]
            if not sel:
                continue
            ax_a.scatter(x[sel] + rng.uniform(-j, j, len(sel)),
                         y[sel] + rng.uniform(-j, j, len(sel)),
                         s=5.0, marker=P.MODEL_MARKERS[mod],
                         facecolor=P.ATTACK_COLORS[att], edgecolor='none',
                         alpha=0.40, zorder=2)
    P.identity_line(ax_a, -4, 104)
    ax_a.set_xlim(-4, 104)
    ax_a.set_ylim(-4, 104)
    ax_a.set_xticks([0, 50, 100])
    ax_a.set_yticks([0, 50, 100])
    ax_a.set_xlabel('Random ASR (%)')
    ax_a.set_ylabel('DPP ASR (%)')
    ax_a.set_title('per attacked target', fontsize=P.FS['title'], pad=3)
    P.stats_box(ax_a, ['DPP better  %d' % wins,
                       'tied           %d' % ties,
                       'worse          %d' % losses,
                       ptxt], loc='lower right')
    P.light_grid(ax_a, axis='both')
    P.panel_label(ax_a, '(a)', dx=-0.26, dy=1.05)

    # ---- (b) every configuration ------------------------------------------
    cfg = collections.defaultdict(list)
    for p in pts:
        cfg[(p['model'], p['attack'], p['pair'], p['budget'], p['tags'])].append(p)
    cx = np.array([np.mean([q['asr_a'] for q in v]) for v in cfg.values()])
    cy = np.array([np.mean([q['asr_b'] for q in v]) for v in cfg.values()])
    cb = np.array([k[3] for k in cfg])
    cs = np.array([len(v) for v in cfg.values()])
    sc = ax_b.scatter(cx, cy, c=cb, s=9 + 1.5 * cs, cmap=P.CMAP_SEQ,
                      norm=LogNorm(vmin=cb.min(), vmax=cb.max()),
                      edgecolor='white', linewidth=0.3, alpha=0.9, zorder=2)
    P.identity_line(ax_b, -4, 104)
    ax_b.set_xlim(-4, 104)
    ax_b.set_ylim(-4, 104)
    ax_b.set_xticks([0, 50, 100])
    ax_b.set_yticks([0, 50, 100])
    ax_b.set_xlabel('Random ASR (%)')
    ax_b.set_ylabel('DPP ASR (%)')
    ax_b.set_title('per configuration', fontsize=P.FS['title'], pad=3)
    cbar = fig.colorbar(sc, ax=ax_b, shrink=0.88, aspect=14, pad=0.02)
    cbar.ax.set_aspect('auto')          # a fixed aspect here collapses the layout
    cbar.set_label('poison budget', fontsize=P.FS['note'])
    cbar.ax.tick_params(labelsize=P.FS['tick'])
    cbar.outline.set_linewidth(0.4)
    P.stats_box(ax_b, '%d of %d above\nthe diagonal'
                % (int((cy > cx).sum()), len(cx)), loc='lower right')
    P.light_grid(ax_b, axis='both')
    P.panel_label(ax_b, '(b)', dx=-0.26, dy=1.05)

    # ---- (c) where the gain lives -----------------------------------------
    budgets = sorted({p['budget'] for p in pts})
    for i, b in enumerate(budgets):
        g = gain[np.array([p['budget'] == b for p in pts])]
        ax_c.scatter(i + rng.uniform(-0.26, 0.26, len(g)), g, s=3.6,
                     facecolor=P.COLORS['dpp'], edgecolor='none', alpha=0.22,
                     zorder=2)
        mu = float(np.mean(g))
        se = float(np.std(g, ddof=1) / np.sqrt(len(g))) if len(g) > 1 else 0.0
        ax_c.errorbar([i], [mu], yerr=[se], fmt='D', markersize=4.6,
                      color=P.COLORS['dpp'], ecolor=P.COLORS['dpp'],
                      elinewidth=1.1, capsize=2.2, zorder=4)
        t = ax_c.text(i + 0.30, mu, '%+.0f' % mu, fontsize=P.FS['note'],
                      ha='left', va='center', color=P.COLORS['dpp'],
                      fontweight='bold', zorder=5)
        t.set_in_layout(False)
    ax_c.axhline(0.0, color=P.COLORS['rule'], linewidth=0.8, linestyle='--')
    ax_c.set_xticks(range(len(budgets)))
    ax_c.set_xticklabels([('%g%%' % (100 * b)) for b in budgets])
    ax_c.set_xlabel('poison budget (fraction of the training set)')
    ax_c.set_ylabel('ASR gain (points)')
    ax_c.set_xlim(-0.55, len(budgets) - 0.10)
    ax_c.set_ylim(-108, 112)
    ax_c.set_yticks([-100, -50, 0, 50, 100])
    P.stats_box(ax_c, '%d targets in %d configurations;  diamond = mean $\\pm$ s.e.m.'
                % (len(pts), n_cfg), loc='lower left')
    P.light_grid(ax_c)
    P.panel_label(ax_c, '(c)', dx=-0.108, dy=1.03)

    # one legend for the figure, under the panels
    handles = [Line2D([0], [0], marker='o', linestyle='none', markersize=3.8,
                      color=P.ATTACK_COLORS[a], label=P.ATTACK_LABELS[a])
               for a in ('fc', 'gradmatch', 'sapa')]
    handles += [Line2D([0], [0], marker=P.MODEL_MARKERS[m], linestyle='none',
                       markersize=3.8, color='#8C8C8C', label=P.MODEL_LABELS[m])
                for m in ('ConvNetBN', 'ResNet20BN', 'VGG13BN')]
    P.bottom_legend(fig, handles, ncol=6)

    P.save_fig(fig, args.out, STEM)
    plt.close(fig)

    P.save_csv(os.path.join(args.out, STEM + '.csv'),
               ['model', 'attack', 'pair', 'budget', 'eps', 'seed', 'tags',
                'target_idx', 'n_victims', 'sel_a', 'asr_a', 'sel_b', 'asr_b',
                'gain'], pts)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    R.add_index_args(p, os.path.join(HERE, 'figs'))
    p.add_argument('--out', default=os.path.join(HERE, 'figs'))
    p.add_argument('--viz_seed', type=int, default=0)
    p.add_argument('--jitter', type=float, default=1.6,
                   help='visual jitter in ASR points; per-target ASR is discrete '
                        '(a multiple of 1/n_victims) so identical values stack')
    p.add_argument('--dry_run', action='store_true')
    return p.parse_args()


def main():
    args = parse_args()
    print('=' * 74)
    print('Paired per-target scatter: %s vs %s (results only, no gpu)'
          % (args.selection_a, args.selection_b))
    print('=' * 74)
    if args.dry_run:
        print('dry run: nothing computed.')
        return
    plot(args)


if __name__ == '__main__':
    main()
