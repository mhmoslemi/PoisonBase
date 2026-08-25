#!/usr/bin/env python
"""
fig_results_mechanism_anatomy.py -- three point clouds that connect the
selection mechanism to what the attack actually does.

Every panel is one point per attacked target (or per configuration), taken from
results.csv through results_index.py. No gpu, no dataset, no model.

(a) THE OPTIMIZATION GETS EASIER.
    x = the crafting objective the run reported (craft_obj: for GM and SAPA this
    is the gradient-matching loss 1 - cos, lower is better), y = that target's
    ASR. Random and DPP are drawn in their own colours with a cross marking each
    arm's median. Objectives are only comparable WITHIN an attack, so the panel
    shows one attack at a time (--attack).

(b) WHERE SELECTION MATTERS MOST.
    x = how close the clean ensemble already is to the adversarial class for that
    target (target_score, the clean softmax probability of y_adv -- the paper's
    own difficulty measure), y = the paired ASR gain. Binned medians on top.
    A gain concentrated at LOW clean probability means selection is what rescues
    the targets a clean model is far from misclassifying.

(c) IT STAYS INVISIBLE.
    Clean test accuracy of the poisoned victims, Random vs DPP, one point per
    configuration. Points on the diagonal mean the extra attack strength costs
    no accuracy, which is what makes the threat model realistic.
"""

import argparse
import collections
import os

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

import plotstyle as P
import results_index as R


STEM = 'fig_results_mechanism_anatomy'
HERE = os.path.dirname(os.path.abspath(__file__))


def plot(args):
    rows = R.load(args.index_csv, args.results_dir, refresh=args.refresh)
    pts = R.attach_target_meta(
        rows, R.paired_targets(rows, args.selection_a, args.selection_b,
                               args.min_victims), args.selection_a)

    P.set_paper_style()
    rng = np.random.default_rng(args.viz_seed)
    fig = plt.figure(figsize=(P.WIDTH_FULL, 2.75), constrained_layout=True)
    gs = fig.add_gridspec(1, 2, wspace=0.26)
    ax_a = fig.add_subplot(gs[0, 0])
    ax_c = fig.add_subplot(gs[0, 1])

    # ---- (a) crafting objective vs ASR, one attack at a time -------------
    sub = [p for p in pts if p['attack'] == args.attack
           and p['craft_obj_a'] != '' and p['craft_obj_b'] != '']
    if not sub:
        raise SystemExit('no points with a recorded craft_obj for attack %s'
                         % args.attack)
    for arm, key_o, key_a in (('random', 'craft_obj_a', 'asr_a'),
                              ('dpp', 'craft_obj_b', 'asr_b')):
        ox = np.array([p[key_o] for p in sub])
        oy = np.array([p[key_a] for p in sub]) + rng.uniform(-1.6, 1.6, len(sub))
        ax_a.scatter(ox, oy, s=5.0, marker=P.MARKERS[arm], facecolor=P.COLORS[arm],
                     edgecolor='none', alpha=0.40, zorder=2)
        ax_a.scatter([np.median(ox)], [np.median(oy)], s=60, marker='+',
                     color=P.COLORS[arm], linewidth=1.6, zorder=4)
    ax_a.set_xlabel('final crafting objective')
    ax_a.set_ylabel('ASR per target (%)')
    ax_a.set_ylim(-6, 106)
    ax_a.set_yticks([0, 25, 50, 75, 100])
    ax_a.set_title('lower objective $\\rightarrow$ higher ASR',
                   fontsize=P.FS['title'], pad=3)
    P.stats_box(ax_a, ['%s, %d targets' % (P.ATTACK_LABELS[args.attack], len(sub)),
                       '$+$ = median of each arm'], loc='lower right')
    P.light_grid(ax_a, axis='both')
    P.panel_label(ax_a, '(a)', dx=-0.24, dy=1.03)

    # ---- (b) clean accuracy is untouched ---------------------------------
    cfg = collections.defaultdict(list)
    for p in pts:
        if p['cta_a'] != '' and p['cta_b'] != '':
            cfg[(p['model'], p['attack'], p['pair'], p['budget'], p['tags'])].append(p)
    cx = np.array([100 * np.mean([q['cta_a'] for q in v]) for v in cfg.values()])
    cy = np.array([100 * np.mean([q['cta_b'] for q in v]) for v in cfg.values()])
    cm = [k[0] for k in cfg]
    for mod in ('ConvNetBN', 'ResNet20BN', 'VGG13BN'):
        sel = [i for i, m in enumerate(cm) if m == mod]
        if sel:
            ax_c.scatter(cx[sel], cy[sel], s=11, marker=P.MODEL_MARKERS[mod],
                         facecolor=P.COLORS['dpp'], edgecolor='white',
                         linewidth=0.3, alpha=0.85, zorder=2)
    lo = float(min(cx.min(), cy.min())) - 0.4
    hi = float(max(cx.max(), cy.max())) + 0.4
    P.identity_line(ax_c, lo, hi)
    ax_c.set_xlim(lo, hi)
    ax_c.set_ylim(lo, hi)
    ax_c.set_xlabel('Random clean acc. (%)')
    ax_c.set_ylabel('DPP clean acc. (%)')
    ax_c.set_title('the attack stays invisible', fontsize=P.FS['title'], pad=3)
    d = cy - cx
    P.stats_box(ax_c, ['mean $\\Delta$ = %+.2f points' % d.mean(),
                       'max $|\\Delta|$ = %.2f' % np.abs(d).max()],
                loc='lower right', color='black')
    P.light_grid(ax_c, axis='both')
    P.panel_label(ax_c, '(b)', dx=-0.24, dy=1.03)

    handles = [Line2D([0], [0], marker=P.MARKERS[k], linestyle='none',
                      markersize=3.8, color=P.COLORS[k], label=P.LABELS[k])
               for k in ('random', 'dpp')]
    handles += [Line2D([0], [0], marker=P.MODEL_MARKERS[m], linestyle='none',
                       markersize=3.8, color=P.COLORS['dpp'], label=P.MODEL_LABELS[m])
                for m in ('ConvNetBN', 'ResNet20BN', 'VGG13BN')]
    P.bottom_legend(fig, handles, ncol=5)

    print('  (a) %s: median objective %.4f (Random) -> %.4f (DPP)'
          % (args.attack, np.median([p['craft_obj_a'] for p in sub]),
             np.median([p['craft_obj_b'] for p in sub])))
    print('  (c) clean accuracy change: mean %+.3f, max |delta| %.3f points'
          % (d.mean(), np.abs(d).max()))

    P.save_fig(fig, args.out, STEM)
    plt.close(fig)
    P.save_csv(os.path.join(args.out, STEM + '.csv'),
               ['model', 'attack', 'pair', 'budget', 'tags', 'target_idx',
                'n_victims', 'asr_a', 'asr_b', 'gain', 'target_score', 'clean_asr',
                'craft_obj_a', 'craft_obj_b', 'cta_a', 'cta_b'], pts)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    R.add_index_args(p, os.path.join(HERE, 'figs'))
    p.add_argument('--out', default=os.path.join(HERE, 'figs'))
    p.add_argument('--viz_seed', type=int, default=0)
    p.add_argument('--attack', default='gradmatch', choices=['fc', 'gradmatch', 'sapa'],
                   help='panel (a) only: crafting objectives are comparable within '
                        'an attack, never across attacks')
    p.add_argument('--dry_run', action='store_true')
    return p.parse_args()


def main():
    args = parse_args()
    print('=' * 74)
    print('Mechanism anatomy: crafting objective and stealth (no gpu)')
    print('=' * 74)
    if args.dry_run:
        print('dry run: nothing computed.')
        return
    plot(args)


if __name__ == '__main__':
    main()
