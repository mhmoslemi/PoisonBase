#!/usr/bin/env python
"""
fig_results_budget_efficiency.py -- "how much poison does the baseline need to
catch up?"

Same trials as the paired scatter, asked the other way round. For a configuration
(architecture, attack, class pair) and a budget b, DPP reaches some ASR. The
question is what budget Random would need to reach that same ASR, read off
Random's own budget-response curve for the same configuration by linear
interpolation in log-budget. The ratio

    multiplier(b) = b_random_equivalent / b

is how many times more poisoned images the uninformed baseline has to place in
the training set to buy what base selection buys for free. It is undefined --
and reported as a lower bound -- when Random never reaches that ASR at any
budget the sweep ran, which is itself the strongest case.

Reads nothing but results.csv through results_index.py: no gpu, no dataset.

Panels
------
(a) budget-response curves. Faint lines are individual configurations, bold lines
    the median across them, for Random and for DPP. The horizontal gap between
    the two bold curves is what panel (b) quantifies.
(b) one point per (configuration, budget): the multiplier, on a log axis.
    Triangles at the top edge are the censored cases where Random never catches
    up within the swept range.
(c) the same multipliers grouped by attack, so a reader can see the effect is not
    carried by one objective.
"""

import argparse
import collections
import os

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

import plotstyle as P
import results_index as R


STEM = 'fig_results_budget_efficiency'
HERE = os.path.dirname(os.path.abspath(__file__))


def curves(pts):
    """{group: {budget: (asr_random, asr_dpp, n_targets)}} for the paired points."""
    acc = collections.defaultdict(lambda: collections.defaultdict(list))
    for p in pts:
        acc[(p['model'], p['attack'], p['pair'], p['tags'])][p['budget']].append(p)
    out = {}
    for g, per_b in acc.items():
        out[g] = {b: (float(np.mean([q['asr_a'] for q in v])),
                      float(np.mean([q['asr_b'] for q in v])), len(v))
                  for b, v in per_b.items()}
    return out


def equivalent_budget(budgets, asr_random, target_asr):
    """Budget at which Random first reaches target_asr, interpolated in log-budget.

    Random's curve is monotone in expectation but not in every sample, so the
    running maximum is used: the question is "what budget does Random need",
    which is answered by the first crossing of the best-so-far curve.
    Returns (budget, censored) with censored=True when it never reaches it.
    """
    lb = np.log10(np.asarray(budgets, dtype=float))
    best = np.maximum.accumulate(np.asarray(asr_random, dtype=float))
    if target_asr <= best[0]:
        return float(10 ** lb[0]), False
    for i in range(1, len(best)):
        if best[i] >= target_asr:
            lo, hi = best[i - 1], best[i]
            t = 0.0 if hi == lo else (target_asr - lo) / (hi - lo)
            return float(10 ** (lb[i - 1] + t * (lb[i] - lb[i - 1]))), False
    return float(10 ** lb[-1]), True                      # censored: never catches up


def plot(args):
    rows = R.load(args.index_csv, args.results_dir, refresh=args.refresh)
    pts = R.paired_targets(rows, args.selection_a, args.selection_b, args.min_victims)
    cur = curves(pts)
    cur = {g: v for g, v in cur.items() if len(v) >= args.min_budgets}
    if not cur:
        raise SystemExit('no configuration has >= %d budgets with both arms'
                         % args.min_budgets)
    print('  %d configurations with at least %d budgets' % (len(cur), args.min_budgets))

    mult = []
    for g, per_b in cur.items():
        bs = sorted(per_b)
        rnd = [per_b[b][0] for b in bs]
        for b in bs:
            tgt = per_b[b][1]
            if tgt <= 0:
                continue                       # nothing to catch up to
            beq, cens = equivalent_budget(bs, rnd, tgt)
            mult.append(dict(model=g[0], attack=g[1], pair=g[2], tags=g[3],
                             budget=b, asr_dpp=tgt, asr_random=per_b[b][0],
                             budget_equiv=beq, multiplier=beq / b, censored=int(cens)))
    m_all = np.array([r['multiplier'] for r in mult])
    m_unc = np.array([r['multiplier'] for r in mult if not r['censored']])
    print('  %d (configuration, budget) points; median multiplier %.1fx '
          '(uncensored only: %.1fx, n=%d)'
          % (len(mult), np.median(m_all), np.median(m_unc) if len(m_unc) else float('nan'),
             len(m_unc)))

    P.set_paper_style()
    rng = np.random.default_rng(args.viz_seed)
    fig = plt.figure(figsize=(P.WIDTH_FULL, 2.75), constrained_layout=True)
    gs = fig.add_gridspec(1, 2, wspace=0.26)
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])

    # ---- (a) budget-response curves --------------------------------------
    all_b = sorted({b for v in cur.values() for b in v})
    for g, per_b in cur.items():
        bs = sorted(per_b)
        ax_a.plot(bs, [per_b[b][0] for b in bs], color=P.COLORS['random'],
                  linewidth=0.5, alpha=0.30, zorder=1)
        ax_a.plot(bs, [per_b[b][1] for b in bs], color=P.COLORS['dpp'],
                  linewidth=0.5, alpha=0.30, zorder=1)
    for key, arm in ((0, 'random'), (1, 'dpp')):
        med = [np.median([per_b[b][key] for per_b in cur.values() if b in per_b])
               for b in all_b]
        ax_a.plot(all_b, med, color=P.COLORS[arm], linewidth=1.8,
                  marker=P.MARKERS[arm], markersize=4.0, markeredgecolor='white',
                  markeredgewidth=0.4, label=P.LABELS[arm], zorder=3)
    ax_a.set_xscale('log')
    ax_a.set_xlabel('poison budget')
    ax_a.set_ylabel('ASR (%)')
    ax_a.set_ylim(-4, 104)
    ax_a.set_yticks([0, 25, 50, 75, 100])
    ax_a.legend(loc='upper left', fontsize=P.FS['legend'])
    P.stats_box(ax_a, 'thin lines: %d configurations' % len(cur), loc='lower right')
    P.light_grid(ax_a, axis='both')
    P.panel_label(ax_a, '(a)', dx=-0.24, dy=1.03)

    # ---- (b) the multiplier ----------------------------------------------
    top = max(20.0, float(np.max(m_all)) * 1.5)
    for r in mult:
        jx = r['budget'] * (10 ** rng.uniform(-0.05, 0.05))
        if r['censored']:
            ax_b.scatter([jx], [top * 0.70], s=13, marker='^', facecolor='none',
                         edgecolor=P.ATTACK_COLORS[r['attack']], linewidth=0.7,
                         zorder=3)
        else:
            ax_b.scatter([jx], [r['multiplier']], s=11,
                         marker=P.MODEL_MARKERS[r['model']],
                         facecolor=P.ATTACK_COLORS[r['attack']], edgecolor='white',
                         linewidth=0.3, alpha=0.9, zorder=2)
    for b in all_b:
        v = [r['multiplier'] for r in mult if r['budget'] == b and not r['censored']]
        if v:
            ax_b.plot([b * 0.75, b * 1.33], [np.median(v)] * 2, color='black',
                      linewidth=1.3, zorder=4)
    ax_b.axhline(1.0, color=P.COLORS['rule'], linewidth=0.8, linestyle='--')
    ax_b.set_xscale('log')
    ax_b.set_yscale('log')
    ax_b.set_ylim(0.5, top)
    ax_b.set_xlabel('poison budget')
    ax_b.set_ylabel('budget Random needs ($\\times$ DPP)')
    per_att = []
    for a in ('fc', 'gradmatch', 'sapa'):
        v = [r['multiplier'] for r in mult if r['attack'] == a and not r['censored']]
        if v:
            per_att.append('%s %.1f$\\times$' % (P.ATTACK_LABELS[a], np.median(v)))
    P.stats_box(ax_b, ['median %.1f$\\times$' % np.median(m_unc)]
                + [per_att[i] + ('   ' + per_att[i + 1] if i + 1 < len(per_att) else '')
                   for i in range(0, len(per_att), 2)],
                loc='lower left', color='black')
    P.light_grid(ax_b, axis='both')
    P.panel_label(ax_b, '(b)', dx=-0.24, dy=1.03)

    handles = [Line2D([0], [0], marker='o', linestyle='none', markersize=3.8,
                      color=P.ATTACK_COLORS[a], label=P.ATTACK_LABELS[a])
               for a in ('fc', 'gradmatch', 'sapa')]
    handles += [Line2D([0], [0], marker=P.MODEL_MARKERS[m], linestyle='none',
                       markersize=3.8, color='#8C8C8C', label=P.MODEL_LABELS[m])
                for m in ('ConvNetBN', 'ResNet20BN', 'VGG13BN')]
    handles += [Line2D([0], [0], marker='^', linestyle='none', markersize=4.4,
                       markerfacecolor='none', markeredgecolor='#444444',
                       label='never catches up')]
    P.bottom_legend(fig, handles, ncol=7)

    P.save_fig(fig, args.out, STEM)
    plt.close(fig)
    P.save_csv(os.path.join(args.out, STEM + '.csv'),
               ['model', 'attack', 'pair', 'tags', 'budget', 'asr_random', 'asr_dpp',
                'budget_equiv', 'multiplier', 'censored'], mult)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    R.add_index_args(p, os.path.join(HERE, 'figs'))
    p.add_argument('--out', default=os.path.join(HERE, 'figs'))
    p.add_argument('--viz_seed', type=int, default=0)
    p.add_argument('--min_budgets', type=int, default=4,
                   help='a configuration needs this many budgets with both arms '
                        'before a budget-response curve is drawn through it')
    p.add_argument('--dry_run', action='store_true')
    return p.parse_args()


def main():
    args = parse_args()
    print('=' * 74)
    print('Budget efficiency: how much more poison Random needs (results only, no gpu)')
    print('=' * 74)
    if args.dry_run:
        print('dry run: nothing computed.')
        return
    plot(args)


if __name__ == '__main__':
    main()
