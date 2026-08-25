#!/usr/bin/env python
"""
fig_intro_base_heterogeneity.py -- Figure 1, "not all clean bases are equally
useful".

Question
--------
Among CLEAN training examples of the SAME adversarial class y_adv, do different
candidate bases carry substantially different target-specific training signal,
BEFORE any poison perturbation is optimized -- and does the selector find the
useful ones?

No attack success rate anywhere: no victim is trained, no poison is crafted.

What is measured
----------------
For one fixed target x_t, on a HELD-OUT model theta that is NOT one of the K
selector surrogates:

    g_t = grad_theta CE(f_theta(x_t), y_adv)          <- y_adv, NOT the true class
    g_i = grad_theta CE(f_theta(x_i), y_adv)          <- candidate i, its own class
    U_i = <g_t, g_i>                                  <- full parameter vector

U_i > 0 means a small gradient-descent step on x_i decreases the target's loss
toward y_adv to first order (the learning rate eta is a common positive constant
and is omitted). Gradients are taken with the network in eval() -- the mode
final_update._target_grads uses -- so BatchNorm uses its running statistics and
measuring never updates them.

U_i is extremely heavy tailed: about half the candidates sit within |U| < 0.01
while the top decile carries most of the total favourable signal. Both panels
therefore use a symmetric-log axis, and the headline statistic is the
CONCENTRATION of utility, not the fraction of positive signs.

The three selections
--------------------
Random / Greedy / DPP are the repository's own selectors, all drawing the same m
bases from the same y_adv pool for the same target, so the figure shows what each
one actually picks up in held-out utility. Random uses the run's per-target
generator, seed * 100003 + target_idx, exactly as prepare_poisons does.

Selector scores come from the ORIGINAL K = 20 surrogates only. Repository
convention: LOWER s_i is better (Greedy takes the m smallest).

Held-out control
----------------
--heldout_checkpoint is REQUIRED and is checked against the K selector surrogates
by path AND by parameter fingerprint. cache/surrogates/<...>/net_20.pt and up are
natural choices: same training recipe, never used by a K = 20 selector.
"""

import argparse
import os

import numpy as np
import torch
import torch.nn as nn

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from scipy.stats import spearmanr

import common as C
from common import FU


STEM = 'fig_intro_base_heterogeneity'
SELECTIONS = ['random', 'greedy', 'dpp']


# --------------------------------------------------------------------------- #
# extraction
# --------------------------------------------------------------------------- #

def flat_param_grad(net, x_norm, y, crit):
    """grad_theta CE(f_theta(x), y) over the FULL parameter vector, flattened.

    net stays in eval(): BatchNorm running statistics are used, not updated, so
    measuring one candidate cannot change what the next candidate sees.
    """
    params = [p for p in net.parameters()]
    loss = crit(net(x_norm.unsqueeze(0)), y)
    g = torch.autograd.grad(loss, params)
    return FU.flat_grad([gi.detach() for gi in g])


def compute(args):
    device = C.pick_device(args)
    ctx = C.build_ctx(args, device)
    y_adv, target_class = C.classes_of(args, ctx)
    names = ctx['class_names']

    tidx = int(args.target_index)
    if int(ctx['test_labs'][tidx]) != target_class:
        raise SystemExit('test image %d has class %s, but --class_pair says the '
                         'target class is %s'
                         % (tidx, names[int(ctx['test_labs'][tidx])],
                            names[target_class]))
    x_t = ctx['test_imgs'][tidx]

    # ---- the K selector surrogates, and a model none of them is ---------------
    sel_nets, sel_paths = C.load_selector_surrogates(args, ctx)
    heldout, fp = C.load_heldout_model(args, ctx, args.heldout_checkpoint,
                                       sel_nets, sel_paths)
    print('  held-out model %s (fingerprint %s) is disjoint from the %d selector '
          'surrogates' % (args.heldout_checkpoint, fp[:12], len(sel_nets)))

    # ---- selector score on the WHOLE y_adv pool, K = 20 nets only -------------
    st = C.selector_state(sel_nets, ctx, x_t, y_adv, args.lambda_margin,
                          args.base_dist)
    cls_idx = st['cls_idx']
    n_pool = len(cls_idx)

    # ---- what each selector would actually take -------------------------------
    m = C.num_poisons(args, ctx)
    sel = {'random': C.select_random(ctx, y_adv, m, args.seed, tidx),
           'greedy': C.select_greedy(sel_nets, ctx, x_t, y_adv, m, args),
           'dpp': C.select_dpp(sel_nets, ctx, x_t, y_adv, m, args)}
    sel_pos = {k: C.positions_in_pool(cls_idx, v).cpu().numpy()
               for k, v in sel.items()}
    print('  m = %d bases per selection out of a pool of %d' % (m, n_pool))

    # ---- which candidates get a utility -------------------------------------
    # The correlation sample is drawn independently of the score. The selected
    # bases are ADDED so their utility can be shown, and are flagged separately:
    # the Spearman statistic is computed on the score-independent sample only,
    # never on the sample plus three score-dependent sets.
    rng = C.viz_seed(args.viz_seed)
    if args.num_candidates and args.num_candidates < n_pool:
        sample = np.sort(rng.choice(n_pool, size=int(args.num_candidates),
                                    replace=False))
    else:
        sample = np.arange(n_pool)
    take = np.unique(np.concatenate([sample] + list(sel_pos.values())))
    in_sample = np.isin(take, sample)
    print('  utilities for %d candidates (%d in the score-independent sample)'
          % (len(take), int(in_sample.sum())))

    # ---- held-out utilities ---------------------------------------------------
    heldout.eval()
    FU.set_requires_grad([heldout], True)
    crit = nn.CrossEntropyLoss().to(device)
    y_adv_t = torch.full((1,), y_adv, dtype=torch.long, device=device)

    # the TARGET gradient is taken at the ADVERSARIAL label y_adv, never at its
    # true class: U_i must measure motion toward y_adv
    g_t = flat_param_grad(heldout, x_t, y_adv_t, crit)

    U = np.empty(len(take), dtype=np.float64)
    for j, p in enumerate(take.tolist()):
        x_i = ctx['train_imgs'][int(cls_idx[int(p)])]
        g_i = flat_param_grad(heldout, x_i, y_adv_t, crit)   # candidates ARE y_adv
        U[j] = float(torch.dot(g_t, g_i))
        if (j + 1) % 500 == 0:
            print('    utility %d / %d' % (j + 1, len(take)))

    pos_t = torch.tensor(take, dtype=torch.long, device=device)
    data = dict(
        target_index=np.array([tidx]),
        candidate_index=cls_idx[pos_t].cpu().numpy(),
        pool_position=take,
        in_sample=in_sample,
        selector_score=st['score'][pos_t].cpu().numpy(),
        boundary_score=st['boundary'][pos_t].cpu().numpy(),
        target_relevance=st['relevance'][pos_t].cpu().numpy(),
        distance_term=st['distance'][pos_t].cpu().numpy(),
        heldout_gradient_utility=U,
        pool_size=np.array([n_pool]), m=np.array([m]),
        sel_K=np.array([len(sel_nets)]),
        y_adv=np.array([y_adv]), target_class=np.array([target_class]),
        heldout_fingerprint=np.array([fp]),
        heldout_checkpoint=np.array([os.path.abspath(args.heldout_checkpoint)]),
    )
    for k in SELECTIONS:
        data['in_' + k] = np.isin(take, sel_pos[k])
        data['selorder_' + k] = sel[k].cpu().numpy()   # selection order preserved

    # ---- cache the images the figure can possibly draw -----------------------
    # 32x32 uint8 thumbnails are tiny (~3 KB each) and make --plot_only fully
    # self-contained: re-plotting never loads CIFAR-10 and never needs a gpu.
    order_s = np.argsort(data['selector_score'])
    show = np.unique(np.concatenate([
        order_s[:args.cache_images], order_s[-args.cache_images:],
        np.where(data['in_random'])[0][:args.cache_images],
        np.where(data['in_greedy'])[0][:args.cache_images],
        np.where(data['in_dpp'])[0][:args.cache_images]]))
    data['img_rows'] = show                       # rows of the arrays above
    data['img_cand'] = np.stack([
        C.to_uint8(ctx['train_imgs'][int(data['candidate_index'][j])], ctx)
        for j in show])
    data['img_target'] = C.to_uint8(x_t, ctx)
    data['class_names'] = np.array([str(c) for c in names])
    print('  cached %d candidate thumbnails for offline re-plotting' % len(show))

    os.makedirs(args.out, exist_ok=True)
    npz = os.path.join(args.out, STEM + '.npz')
    np.savez(npz, **data)
    print('  wrote %s' % npz)

    # ---- a few statistics worth quoting in the text --------------------------
    smp = U[in_sample]
    srt = np.sort(smp)[::-1]
    tot_pos = srt[srt > 0].sum()
    dec = max(1, int(round(0.10 * len(srt))))
    print('  utility concentration: top 10%% of candidates carry %.1f%% of the '
          'total favourable signal' % (100 * srt[:dec].sum() / tot_pos))
    for k in SELECTIONS:
        u = U[data['in_' + k]]
        print('  %-6s selected bases: median U = %+.4g, mean U = %+.4g, '
              'share of total favourable signal = %.1f%%'
              % (k, np.median(u), u.mean(), 100 * u[u > 0].sum() / tot_pos))
    print('  spearman(s, U)  = %+.3f' % spearmanr(data['selector_score'][in_sample],
                                                  smp).statistic)
    print('  spearman(B, U)  = %+.3f' % spearmanr(data['boundary_score'][in_sample],
                                                  smp).statistic)
    print('  spearman(R, U)  = %+.3f' % spearmanr(data['target_relevance'][in_sample],
                                                  smp).statistic)

    rows = []
    order = np.argsort(data['selector_score'])
    rank = np.empty(len(order), dtype=int)
    rank[order] = np.arange(len(order))            # 0 = the selector's first pick
    for j in range(len(U)):
        rows.append(dict(candidate_index=int(data['candidate_index'][j]),
                         selector_score=float(data['selector_score'][j]),
                         selector_rank=int(rank[j]),
                         boundary_score=float(data['boundary_score'][j]),
                         target_relevance=float(data['target_relevance'][j]),
                         distance_term=float(data['distance_term'][j]),
                         heldout_gradient_utility=float(U[j]),
                         in_correlation_sample=int(in_sample[j]),
                         in_random=int(data['in_random'][j]),
                         in_greedy=int(data['in_greedy'][j]),
                         in_dpp=int(data['in_dpp'][j]),
                         target_index=tidx))
    C.save_csv(os.path.join(args.out, STEM + '.csv'),
               ['candidate_index', 'selector_score', 'selector_rank',
                'boundary_score', 'target_relevance', 'distance_term',
                'heldout_gradient_utility', 'in_correlation_sample',
                'in_random', 'in_greedy', 'in_dpp', 'target_index'], rows)
    return npz


# --------------------------------------------------------------------------- #
# plotting
# --------------------------------------------------------------------------- #

def symlog_bins(U, linthresh, n_pos=30, n_neg=12):
    """Bin edges that match a symlog axis: log-spaced on both tails, one linear
    bin across the near-zero core where ~half the candidates live."""
    hi = max(float(U.max()), linthresh * 1.001)
    lo = min(float(U.min()), -linthresh * 1.001)
    pos = np.logspace(np.log10(linthresh), np.log10(hi), n_pos)
    neg = -np.logspace(np.log10(linthresh), np.log10(abs(lo)), n_neg)[::-1]
    return np.unique(np.concatenate([neg, [-linthresh, 0.0, linthresh], pos]))


def plot(args):
    npz = os.path.join(args.out, STEM + '.npz')
    if not os.path.exists(npz):
        raise SystemExit('%s missing -- run with --compute first.' % npz)
    d = np.load(npz, allow_pickle=True)
    s = d['selector_score'].astype(float)
    U = d['heldout_gradient_utility'].astype(float)
    smp = d['in_sample'].astype(bool)
    tidx = int(d['target_index'][0])
    m = int(d['m'][0])
    lt = float(args.linthresh)

    # everything panel (c) needs was cached at compute time: --plot_only runs on
    # a login node with no gpu and no dataset
    if 'img_cand' not in d.files:
        raise SystemExit('%s predates the image cache -- rerun with --compute once '
                         '(it is the .npz that changed, not the figure).' % npz)
    names = [str(c) for c in d['class_names']]
    img_rows = {int(v): i for i, v in enumerate(d['img_rows'])}
    y_adv = int(d['y_adv'][0])
    target_class = int(d['target_class'][0])

    C.set_paper_style()
    # Authored at ICLR's text width: no rescaling at include time. Axis labels
    # are deliberately short -- at 2.4in per panel a full sentence on the axis
    # runs off the page.
    fig = plt.figure(figsize=(C.WIDTH_FULL, 4.35), constrained_layout=True)
    outer = fig.add_gridspec(2, 1, height_ratios=[1.70, 1.0], hspace=0.20)
    top = outer[0].subgridspec(1, 2, wspace=0.30)
    ax_a = fig.add_subplot(top[0, 0])
    ax_b = fig.add_subplot(top[0, 1])

    ulab = 'held-out utility $U_i$'

    # ---- (a) how heterogeneous the same-class candidates are ----------------
    bins = symlog_bins(U[smp], lt)
    ax_a.hist(U[smp][U[smp] <= 0], bins=bins, color=C.COLORS['pool'],
              edgecolor='white', linewidth=0.25)
    ax_a.hist(U[smp][U[smp] > 0], bins=bins, color=C.COLORS['dpp'], alpha=0.85,
              edgecolor='white', linewidth=0.25)
    ax_a.set_xscale('symlog', linthresh=lt)
    ax_a.axvline(0.0, color=C.COLORS['rule'], linewidth=0.8, linestyle='--')
    lo, hi = float(U[smp].min()), float(U[smp].max())
    ax_a.set_xticks([t for t in (-1.0, -lt, 0.0, lt, 1.0, 1e2)
                     if lo - abs(lo) * 0.1 <= t <= hi + abs(hi) * 0.1])
    ax_a.xaxis.set_minor_locator(mticker.NullLocator())

    ymax = ax_a.get_ylim()[1]
    med = {}
    for j, k in enumerate(SELECTIONS):
        med[k] = float(np.median(U[d['in_' + k].astype(bool)]))
        ax_a.plot([med[k], med[k]], [0, ymax * (0.995 - 0.05 * j)],
                  color=C.COLORS[k], linewidth=1.1, alpha=0.9, zorder=4,
                  linestyle=('--' if k == 'greedy' else '-'))
        ax_a.plot([med[k]], [ymax * (0.995 - 0.05 * j)], marker='v', markersize=4.2,
                  color=C.COLORS[k], clip_on=False, zorder=5)
    srt = np.sort(U[smp])[::-1]
    dec = max(1, int(round(0.10 * len(srt))))
    share = 100 * srt[:dec].sum() / srt[srt > 0].sum()
    # the medians of the three selections are the triangles on the top axis and
    # are quantified in panel (b); repeating them here only crowds the histogram
    # placed over the tail, where the histogram is flat: the near-zero spike
    # occupies the whole upper-left of this panel
    C.note(ax_a, 'top 10%% of bases carry\n%.0f%% of all favourable signal' % share,
           (0.97, 0.60), ha='right', va='top', color='black',
           bbox=dict(facecolor='white', alpha=0.9, edgecolor='none',
                     boxstyle='round,pad=0.25'))
    ax_a.set_xlabel(ulab + '   (symlog)')
    ax_a.set_ylabel('number of clean bases')
    C.light_grid(ax_a)
    C.panel_label(ax_a, '(a)', dx=-0.26, dy=1.03)

    # ---- (b) does the selector score predict the held-out utility? -----------
    # rho is computed on the score-independent sample only -- adding the three
    # selected sets would put score-chosen points into the correlation sample
    rho, pval = spearmanr(s[smp], U[smp])
    # The pool is 5000 points and each selection is 250 more: drawn as opaque
    # markers they merge into a single block. The pool therefore goes down as a
    # faint rasterized layer with a binned median/IQR summary on top, and each
    # selection is summarised by the region holding half of it.
    ax_b.scatter(s[smp], U[smp], s=C.SIZES['cloud'], c=C.COLORS['pool'],
                 alpha=0.28, linewidths=0, zorder=1, rasterized=True)
    C.binned_trend(ax_b, s[smp], U[smp], bins=14, color='#555555', lw=1.2,
                   zorder=6)
    # No density contours in this panel: the y-axis is symlog, and a kernel
    # density fitted in data coordinates would be drawn through a nonlinear
    # transform, producing shapes that mean nothing. Small semi-transparent
    # markers plus the binned median carry the same information honestly.
    styles = {'random': dict(facecolor=C.COLORS['random'], edgecolor='none',
                             alpha=0.45, s=3.6, zorder=2),
              'greedy': dict(facecolor='none', edgecolor=C.COLORS['greedy'],
                             linewidth=0.30, alpha=0.70, s=4.6, zorder=3),
              'dpp': dict(facecolor=C.COLORS['dpp'], edgecolor='none',
                          alpha=0.65, s=3.6, zorder=4)}
    for k in ('random', 'greedy', 'dpp'):
        sel = d['in_' + k].astype(bool)
        ax_b.scatter(s[sel], U[sel], marker=C.MARKERS[k], **styles[k])
    ax_b.set_yscale('symlog', linthresh=lt)
    ax_b.axhline(0.0, color=C.COLORS['rule'], linewidth=0.6, linestyle=':')
    ptxt = ('$p<10^{-16}$' if pval < 1e-16 else '$p=%.1e$' % pval)
    rho_b = spearmanr(d['boundary_score'][smp], U[smp]).statistic
    rho_r = spearmanr(d['target_relevance'][smp], U[smp]).statistic
    C.stats_box(ax_b, ['Spearman $\\rho=%.3f$' % rho, '%s  ($n=%d$)'
                       % (ptxt, int(smp.sum())), '',
                       'terms:  $\\rho(B_i)=%.3f$' % rho_b,
                       '            $\\rho(R_i)=%.3f$' % rho_r],
                loc='upper right', color='black')
    ax_b.set_xlabel('selector score $s_i$')
    ax_b.set_ylabel(ulab + '  (symlog)')
    C.note(ax_b, 'lower $s_i$ = preferred', (0.5, -0.30), ha='center', va='top',
           fontsize=C.FS['small'])
    C.light_grid(ax_b, axis='both')
    C.panel_label(ax_b, '(b)', dx=-0.26, dy=1.03)

    # ---- (c) the actual clean bases -----------------------------------------
    # Deterministic, not hand-picked: two bases a Random draw took, the two the
    # selector ranks first, and the two it ranks last. All are CLEAN.
    order = np.argsort(s)
    if len(order) < 6:
        raise SystemExit('only %d candidates -- panel (c) needs at least 6' % len(order))
    rnd = np.where(d['in_random'].astype(bool))[0][:2]
    groups = [('Random draw', list(rnd), C.COLORS['random']),
              ('selector-preferred', [order[0], order[1]], C.COLORS['dpp']),
              ('selector-rejected', [order[-2], order[-1]], C.COLORS['rule'])]

    bottom = outer[1].subgridspec(
        1, 10, width_ratios=[1.15, 0.34, 1, 1, 0.22, 1, 1, 0.22, 1, 1], wspace=0.10)
    ax_t = fig.add_subplot(bottom[0, 0])
    C.draw_rgb(ax_t, d['img_target'], edge=C.COLORS['target'], lw=1.4)
    ax_t.set_xlabel('target $x_t$\n%s #%d' % (names[target_class], tidx),
                    fontsize=C.FS['note'], color=C.COLORS['target'], linespacing=1.3)
    C.panel_label(ax_t, '(c)', dx=-0.42, dy=1.04)

    for (gname, members, gcol), cols in zip(groups, [(2, 3), (5, 6), (8, 9)]):
        for k, (j, col) in enumerate(zip(members, cols)):
            ax = fig.add_subplot(bottom[0, col])
            if int(j) not in img_rows:
                raise SystemExit('candidate %d was not cached; rerun --compute '
                                 'with a larger --cache_images' % int(j))
            C.draw_rgb(ax, d['img_cand'][img_rows[int(j)]], edge='#DDDDDD')
            utxt = ('$U\\approx0$' if abs(U[j]) < 1e-6 else '$U$=%+.2g' % U[j])
            ax.set_xlabel('$s$=%.2f\n%s' % (s[j], utxt), fontsize=C.FS['small'],
                          color=(C.COLORS['dpp'] if U[j] > 0 else C.COLORS['rule']),
                          linespacing=1.3, labelpad=1.5)
            if k == 0:                       # one title, centred over the pair
                ax.set_title(gname, fontsize=C.FS['note'], pad=2.5, color=gcol)
                ax.title.set_position((1.06, 1.0))

    handles = [
        Patch(facecolor=C.COLORS['pool'], label='$U_i\\leq0$'),
        Patch(facecolor=C.COLORS['dpp'], label='$U_i>0$'),
        Line2D([0], [0], color='#555555', linewidth=1.4, label='median'),
    ] + [Line2D([0], [0], marker=C.MARKERS[k], linestyle='none', markersize=3.6,
                markerfacecolor=(C.COLORS[k] if k != 'greedy' else 'none'),
                markeredgecolor=C.COLORS[k], color=C.COLORS[k],
                label=C.LABELS[k]) for k in SELECTIONS]
    C.bottom_legend(fig, handles, ncol=6)

    C.save_fig(fig, args.out, STEM)
    plt.close(fig)


# --------------------------------------------------------------------------- #

def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    C.add_repo_args(p)
    C.add_mode_args(p)
    p.add_argument('--target_index', type=int, required=True,
                   help="test-set index of x_t. Use one of the paper's pinned "
                        'targets, e.g. 3741 6857 8252 1126 3522 '
                        '(target_sets/appx_broad_ConvNetBN_dog-bird.json). Never '
                        'pick a target by attack success.')
    p.add_argument('--heldout_checkpoint', required=True,
                   help='REQUIRED. A model that is NOT one of the K selector '
                        'surrogates, e.g. cache/surrogates/ConvNetBN_60ep_lr0.1_'
                        'bs128_seed42/net_20.pt (the pool holds 50). Checked by '
                        'path and by parameter fingerprint.')
    p.add_argument('--budget', type=float, default=C.PAPER['budget'],
                   help='poison budget, i.e. how many bases each selector takes; '
                        'm = round(budget * 50000) = 250 at the default 5e-3')
    p.add_argument('--num_candidates', type=int, default=0,
                   help='size of the score-independent sample the correlation is '
                        'computed on; 0 = the whole y_adv pool (5000 for '
                        'CIFAR-10, ~4 min of gradients). The selected bases are '
                        'always measured on top of it.')
    p.add_argument('--cache_images', type=int, default=16,
                   help='how many thumbnails per group (best / worst by score, '
                        'and the head of each selection) to store in the .npz so '
                        '--plot_only never needs the dataset or a gpu')
    p.add_argument('--linthresh', type=float, default=0.01,
                   help='symlog linear region. U is heavy tailed -- about half '
                        'the pool sits inside |U| < 0.01 -- so a linear axis '
                        'shows one bar and an empty tail.')
    return p.parse_args()


def main():
    args = parse_args()
    do_compute, do_plot = C.resolve_mode(args)
    C.summarize('Figure 1 -- clean-base heterogeneity (no ASR, no victim training)', [
        ('dataset / model', '%s / %s' % (args.dataset, args.model)),
        ('class pair', '%s (%s)' % (args.class_pair, args.pair_order)),
        ('target index', args.target_index),
        ('selector', 'K=%d, lambda=%g, alpha=%g, base_dist=%s'
         % (args.sel_K, args.lambda_margin, args.sel_alpha, args.base_dist)),
        ('selections shown', ' '.join(C.LABELS[k] for k in SELECTIONS)),
        ('budget', '%g  (m bases per selection)' % args.budget),
        ('held-out model', args.heldout_checkpoint),
        ('candidates', args.num_candidates or 'whole y_adv pool'),
        ('utility', 'U_i = <grad_theta CE(x_t, y_adv), grad_theta CE(x_i, y_adv)>'),
        ('mode', 'compute=%s plot=%s' % (do_compute, do_plot)),
        ('outputs', os.path.join(args.out, STEM + '.{pdf,png,csv,npz}')),
    ])
    if args.dry_run:
        print('dry run: nothing computed.')
        return
    if do_compute:
        compute(args)
    if do_plot:
        plot(args)


if __name__ == '__main__':
    main()
