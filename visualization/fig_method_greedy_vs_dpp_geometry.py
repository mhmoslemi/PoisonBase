#!/usr/bin/env python
"""
fig_method_greedy_vs_dpp_geometry.py -- Figure 2, the geometry of the SELECTED
SET: Random vs Greedy vs DPP.

No attack is run and no victim is trained: this figure is about selection only,
and reports no attack success rate.

Why all three
-------------
Greedy and DPP differ only in how a set is assembled from the SAME pointwise
score, and at a large budget they can overlap almost completely -- the script
prints and stores |S_greedy AND S_dpp| so that overlap is a reported number
rather than something a reader has to guess from a heatmap. Random is the
baseline that makes the axes interpretable: it fixes what "low redundancy with no
quality" looks like, so the panel shows the actual trade-off,
    Random   = low quality, low redundancy
    Greedy   = high quality, high redundancy
    DPP      = high quality, redundancy pulled back toward Random.

Representation
--------------
For surrogate k let hhat_i^k = h_k(x_i) / ||h_k(x_i)||. The figure uses

    phi_i = (1 / sqrt(K)) [ hhat_i^1 ; ... ; hhat_i^K ]     so that
    phi_i . phi_j = (1/K) sum_k cos(h_k(x_i), h_k(x_j)) = C_ij,

exactly the kernel final_update's DPP uses -- phi comes from
final_update._ours_score_and_feats, it is not rebuilt here.

Panels
------
(a) PCA of phi over the whole candidate pool. ONE PCA is fitted on the common
    candidate representation and all three selections plus the target are
    transformed with that same basis. Illustrative only, and the explained
    variance is printed in the axis labels so a reader can see how little of a
    ~40k-dimensional geometry two components hold.
(b) the exact C_S of the three selections for one display target, shared colour
    limits, ordered by descending pointwise quality.
(c) target-wise paired metrics, one thin line per target:
        Q(S)   = mean selector score          (repository convention: LOWER = better)
        Red(S) = 2 / (m(m-1)) sum_{i<j} C_ij  (lower = less redundant)
        r_eff  = exp(-sum_j p_j log p_j), p_j = sigma_j / sum_l sigma_l over the
                 singular values of the selected representation matrix

det(C_S) is deliberately NOT plotted: the DPP objective maximises it, so it is
not independent evidence.
"""

import argparse
import os

import numpy as np
import torch

import matplotlib.pyplot as plt

import common as C


STEM = 'fig_method_greedy_vs_dpp_geometry'
METHODS = ['random', 'greedy', 'dpp']


# --------------------------------------------------------------------------- #
# extraction
# --------------------------------------------------------------------------- #

def compute(args):
    device = C.pick_device(args)
    ctx = C.build_ctx(args, device)
    y_adv, target_class = C.classes_of(args, ctx)
    targets = C.resolve_targets(args, ctx, target_class)
    m = C.num_poisons(args, ctx)
    sel_nets, _ = C.load_selector_surrogates(args, ctx)
    print('  m = %d poisons (budget %g of %d training images)'
          % (m, args.budget, ctx['train_imgs'].shape[0]))

    display = int(args.display_target) if args.display_target is not None else targets[0]
    if display not in targets:
        raise SystemExit('--display_target %d is not in the target list %s'
                         % (display, targets))

    per_target, store = [], {}
    for t in targets:
        if str(device).startswith('cuda'):
            torch.cuda.empty_cache()        # the per-target feature bank is ~0.8 GB
        x_t = ctx['test_imgs'][t]
        st = C.selector_state(sel_nets, ctx, x_t, y_adv, args.lambda_margin,
                              args.base_dist)
        cls_idx, score, feats = st['cls_idx'], st['score'], st['feats']

        # the repository's own selectors -- same target, same pool, same K
        sel = {'random': C.select_random(ctx, y_adv, m, args.seed, t),
               'greedy': C.select_greedy(sel_nets, ctx, x_t, y_adv, m, args),
               'dpp': C.select_dpp(sel_nets, ctx, x_t, y_adv, m, args)}
        sets = {k: set(int(i) for i in v.tolist()) for k, v in sel.items()}
        ov = len(sets['greedy'] & sets['dpp'])
        print('    target %d: |Greedy AND DPP| = %d/%d bases (%.1f%%)'
              % (t, ov, m, 100.0 * ov / m))

        for name, idx in sel.items():
            pos = C.positions_in_pool(cls_idx, idx)
            phi_S = feats[pos]
            Cs = C.similarity_matrix(phi_S)
            per_target.append(dict(
                target_index=int(t), selection=name, m=int(m),
                mean_score=float(score[pos].mean()),
                mean_relevance=float(st['relevance'][pos].mean()),
                mean_boundary=float(st['boundary'][pos].mean()),
                mean_offdiag_similarity=C.mean_offdiag(Cs),
                effective_rank=C.effective_rank(phi_S),
                overlap_with_greedy=len(sets[name] & sets['greedy']),
                overlap_with_random=len(sets[name] & sets['random'])))
            print('      %-6s Q=%+.3f  Red=%.4f  r_eff=%.2f'
                  % (name, per_target[-1]['mean_score'],
                     per_target[-1]['mean_offdiag_similarity'],
                     per_target[-1]['effective_rank']))

        if t == display:
            # ---- panel (a): ONE PCA on the common candidate representation ----
            rng = C.viz_seed(args.viz_seed)
            n_pool = feats.shape[0]
            keep = np.arange(n_pool)
            if args.pca_max_candidates and args.pca_max_candidates < n_pool:
                sub = rng.choice(n_pool, size=int(args.pca_max_candidates),
                                 replace=False)
                keep = np.unique(np.concatenate(
                    [sub] + [C.positions_in_pool(cls_idx, i).cpu().numpy()
                             for i in sel.values()]))
            keep_t = torch.tensor(keep, dtype=torch.long, device=device)
            coords, evr, transform = C.pca_2d(feats[keep_t], seed=args.viz_seed)
            phi_t = C.target_ensemble_feature(sel_nets, x_t)
            t_coord = transform(phi_t.unsqueeze(0))[0]

            where = {int(v): i for i, v in enumerate(keep)}
            cnp = coords.cpu().numpy()
            store['pca_pool'] = cnp
            store['pca_evr'] = np.asarray(evr, dtype=float)
            store['pca_target'] = t_coord.cpu().numpy()
            for name, idx in sel.items():
                pos = C.positions_in_pool(cls_idx, idx)
                store['pca_%s' % name] = cnp[np.array([where[int(p)]
                                                       for p in pos.cpu().numpy()])]
                # ---- panel (b): the exact C_S, ordered by descending quality ---
                o = torch.argsort(score[pos])       # ascending score = best first
                phi_S = feats[pos][o]
                store['Cs_%s' % name] = C.similarity_matrix(phi_S).cpu().numpy()
                store['sel_%s' % name] = idx.cpu().numpy()
            store['overlap_greedy_dpp'] = np.array([ov])

    os.makedirs(args.out, exist_ok=True)
    npz = os.path.join(args.out, STEM + '.npz')
    np.savez(npz,
             targets=np.array(targets),
             display_target=np.array([display]),
             m=np.array([m]),
             metrics_target=np.array([r['target_index'] for r in per_target]),
             metrics_selection=np.array([r['selection'] for r in per_target]),
             metrics_quality=np.array([r['mean_score'] for r in per_target]),
             metrics_redundancy=np.array([r['mean_offdiag_similarity']
                                          for r in per_target]),
             metrics_effrank=np.array([r['effective_rank'] for r in per_target]),
             **store)
    print('  wrote %s' % npz)

    # the headline diagnostic, printed rather than left for a reader to infer
    ovs = [r['overlap_with_greedy'] for r in per_target if r['selection'] == 'dpp']
    print('  Greedy AND DPP overlap across %d targets: mean %.1f/%d bases (%.1f%%)'
          % (len(ovs), float(np.mean(ovs)), m, 100.0 * float(np.mean(ovs)) / m))

    C.save_csv(os.path.join(args.out, STEM + '.csv'),
               ['target_index', 'selection', 'm', 'mean_score', 'mean_relevance',
                'mean_boundary', 'mean_offdiag_similarity', 'effective_rank',
                'overlap_with_greedy', 'overlap_with_random'], per_target)
    return npz


# --------------------------------------------------------------------------- #
# plotting
# --------------------------------------------------------------------------- #

def _paired_panel(ax, vals, label, better, targets):
    """One paired-dot plot: a thin line per target, a stronger marker for the mean."""
    xs = list(range(len(METHODS)))
    for i in range(len(targets)):
        ax.plot(xs, [vals[k][i] for k in METHODS], '-', color='#C9C9C9',
                linewidth=0.6, zorder=1)
    for x, k in zip(xs, METHODS):
        ax.scatter([x] * len(targets), vals[k], s=10, marker=C.MARKERS[k],
                   facecolor=C.COLORS[k], edgecolor='white', linewidth=0.3,
                   alpha=0.85, zorder=2)
        mu = float(np.mean(vals[k]))
        se = (float(np.std(vals[k], ddof=1) / np.sqrt(len(vals[k])))
              if len(vals[k]) > 1 else 0.0)
        ax.errorbar([x + 0.22], [mu], yerr=[se], fmt=C.MARKERS[k], markersize=4.0,
                    color=C.COLORS[k], ecolor=C.COLORS[k], elinewidth=0.9,
                    capsize=1.8, zorder=3)
    ax.set_xlim(-0.55, len(METHODS) - 0.25)
    ax.set_xticks(xs)
    # rotated: at ~1.4in per panel three horizontal words run into each other
    ax.set_xticklabels(['Random', 'Greedy', 'DPP'], fontsize=C.FS['tick'],
                       rotation=30, ha='right', rotation_mode='anchor')
    ax.tick_params(axis='x', pad=1.0)
    ax.tick_params(axis='x', pad=1.5)
    ax.set_ylabel('%s\n(%s)' % (label, better), fontsize=C.FS['note'] + 0.4)
    C.light_grid(ax)


def plot(args):
    npz = os.path.join(args.out, STEM + '.npz')
    if not os.path.exists(npz):
        raise SystemExit('%s missing -- run with --compute first.' % npz)
    d = np.load(npz, allow_pickle=True)
    targets = [int(t) for t in d['targets']]
    display = int(d['display_target'][0])
    m = int(d['m'][0])

    sel_names = np.array([str(x) for x in d['metrics_selection']])
    tgt_col = np.array([int(x) for x in d['metrics_target']])

    def by(metric):
        out = {}
        for k in METHODS:
            out[k] = np.array([float(d[metric][(sel_names == k) & (tgt_col == t)][0])
                               for t in targets])
        return out

    quality, redundancy, effrank = by('metrics_quality'), by('metrics_redundancy'), \
        by('metrics_effrank')

    C.set_paper_style()
    # Layout note: no fixed-aspect axes anywhere. constrained_layout resolves a
    # box_aspect by shrinking EVERY axes in the figure until the aspect fits, so
    # one square heatmap collapses the whole page. The heatmaps therefore use
    # aspect='auto' and are given near-square boxes by the grid instead.
    fig = plt.figure(figsize=(C.WIDTH_FULL, 4.65), constrained_layout=True)
    outer = fig.add_gridspec(2, 1, height_ratios=[0.74, 1.05], hspace=0.44)
    top = outer[0].subgridspec(1, 2, width_ratios=[1.05, 2.85], wspace=0.24)
    ax_a = fig.add_subplot(top[0, 0])
    hm = top[0, 1].subgridspec(1, 4, width_ratios=[1, 1, 1, 0.055], wspace=0.09)
    axes_b = [fig.add_subplot(hm[0, i]) for i in range(3)]
    cax = fig.add_subplot(hm[0, 3])

    # ---- (a) common PCA of the ensemble representation -----------------------
    # Plain points, small and translucent so the three overlapping 250-point
    # selections and the 5000-point pool underneath all stay visible instead of
    # one layer painting over the next.
    pool = d['pca_pool']
    ax_a.scatter(pool[:, 0], pool[:, 1], s=1.2, facecolor=C.COLORS['pool'],
                 edgecolor='none', alpha=0.35, zorder=1, rasterized=True)
    for k in ('random', 'greedy', 'dpp'):
        pp = d['pca_%s' % k]
        ax_a.scatter(pp[:, 0], pp[:, 1], marker=C.MARKERS[k], s=2.6,
                     facecolor=C.COLORS[k], edgecolor='none', alpha=0.45,
                     zorder=3, label=C.LABELS[k], rasterized=True)
    tc = d['pca_target']
    ax_a.scatter([tc[0]], [tc[1]], s=C.SIZES['star'], marker='*',
                 facecolor=C.COLORS['target'], edgecolor='white', linewidth=0.6,
                 label='target $x_t$', zorder=8)
    evr = d['pca_evr']
    ax_a.set_xlabel('PC1 (%.1f%% var.)' % (100 * evr[0]))
    ax_a.set_ylabel('PC2 (%.1f%% var.)' % (100 * evr[1]))
    C.light_grid(ax_a, axis='both')
    C.panel_label(ax_a, '(a)', dx=-0.22, dy=1.02)

    # ---- (b) the exact C_S matrices, identical colour limits -----------------
    mats = {k: d['Cs_%s' % k] for k in METHODS}
    off = {k: mats[k][~np.eye(mats[k].shape[0], dtype=bool)] for k in METHODS}
    allo = np.concatenate([off[k] for k in METHODS])
    vmin, vmax = np.percentile(allo, 1.0), np.percentile(allo, 99.0)
    cmap = plt.get_cmap(C.CMAP_SIM).copy()
    cmap.set_bad('white')
    im = None
    for ax, k in zip(axes_b, METHODS):
        disp = mats[k].astype(float).copy()
        np.fill_diagonal(disp, np.nan)          # drawn white, never colour-scaled
        im = ax.imshow(disp, cmap=cmap, vmin=vmin, vmax=vmax, aspect='auto',
                       interpolation='nearest', rasterized=True)
        ax.set_title('%s\nmean $C_{ij}$=%.3f' % (C.LABELS[k], off[k].mean()),
                     fontsize=C.FS['note'], pad=2.5, color=C.COLORS[k], linespacing=1.25)
        ax.set_xticks([])
        ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_visible(True)
            sp.set_linewidth(0.5)
            sp.set_color('#999999')
    ovl = int(d['overlap_greedy_dpp'][0]) if 'overlap_greedy_dpp' in d.files else None
    cap = 'the $m=%d$ selected bases, ordered by pointwise quality' % m
    cap += '\ndisplay target #%d' % display
    if ovl is not None:
        cap += ',  $|S_{\\mathrm{Greedy}} \\cap S_{\\mathrm{DPP}}|=%d/%d$' % (ovl, m)
    axes_b[1].set_xlabel(cap, fontsize=C.FS['note'], color=C.COLORS['rule'], labelpad=2,
                         linespacing=1.35)
    cb = fig.colorbar(im, cax=cax)
    # a colorbar axes is created with a FIXED aspect (20 by default). Under
    # constrained_layout that is another aspect constraint, and the engine
    # satisfies it by shrinking every other axes on the page -- which is what
    # collapsed panel (c) to a sliver. The colorbar does not need it: the grid
    # already gives it a thin column.
    cb.ax.set_aspect('auto')
    cb.set_label('ensemble cosine similarity', fontsize=C.FS['note'])
    cb.ax.tick_params(labelsize=6.0)
    cb.outline.set_linewidth(0.4)
    C.panel_label(axes_b[0], '(b)', dx=-0.10, dy=1.24)

    # ---- (c) target-wise paired geometry -------------------------------------
    bottom = outer[1].subgridspec(1, 3, wspace=0.24)
    ax1, ax2, ax3 = [fig.add_subplot(bottom[0, i]) for i in range(3)]
    _paired_panel(ax1, quality, 'mean pointwise score $Q(S)$', 'lower = better',
                  targets)
    _paired_panel(ax2, redundancy, 'mean off-diagonal $C_{ij}$',
                  'lower = less redundant', targets)
    _paired_panel(ax3, effrank, 'effective rank $r_{\\mathrm{eff}}$',
                  'higher = more spread', targets)
    C.panel_label(ax1, '(c)', dx=-0.30, dy=1.03)
    C.note(ax2, '%d targets, $m=%d$;  thin line = one target,  '
           'large marker = mean $\\pm$ s.e.m.' % (len(targets), m),
           (0.5, 1.04), ha='center', va='bottom', fontsize=C.FS['note'])

    from matplotlib.lines import Line2D
    handles = [Line2D([0], [0], marker='.', linestyle='none', markersize=6,
                      color=C.COLORS['pool'], label='candidate pool')]
    handles += [Line2D([0], [0], marker=C.MARKERS[k], linestyle='none',
                       markersize=4.2, color=C.COLORS[k], label=C.LABELS[k])
                for k in METHODS]
    handles += [Line2D([0], [0], marker='*', linestyle='none', markersize=8,
                       color=C.COLORS['target'], label='target $x_t$')]
    C.bottom_legend(fig, handles, ncol=5)

    C.save_fig(fig, args.out, STEM)
    plt.close(fig)


# --------------------------------------------------------------------------- #

def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    C.add_repo_args(p)
    C.add_mode_args(p)
    p.add_argument('--budget', type=float, default=C.PAPER['budget'],
                   help='poison budget; m = round(budget * 50000). The paper\'s '
                        'reduced protocol uses 5e-3 -> m = 250.')
    p.add_argument('--target_idx_file', default=C.PAPER['main_target_file'],
                   help='pinned target list. Default = the 10 main bird->dog '
                        'targets of the gradient-matching sweep. The reduced '
                        'appendix protocol\'s 5 targets are in '
                        'target_sets/appx_broad_ConvNetBN_dog-bird.json.')
    p.add_argument('--target_indices', type=int, nargs='*', default=None,
                   help='explicit target list; overrides --target_idx_file')
    p.add_argument('--display_target', type=int, default=None,
                   help='which target panels (a) and (b) draw; default = the '
                        'first of the list. Never choose it by attack success.')
    p.add_argument('--pca_max_candidates', type=int, default=0,
                   help='subsample the pool before fitting the panel-(a) PCA '
                        '(0 = the whole pool). Selected bases are always kept.')
    return p.parse_args()


def main():
    args = parse_args()
    do_compute, do_plot = C.resolve_mode(args)
    C.summarize('Figure 2 -- Random vs Greedy vs DPP set geometry (no ASR)', [
        ('dataset / model', '%s / %s' % (args.dataset, args.model)),
        ('class pair', '%s (%s)' % (args.class_pair, args.pair_order)),
        ('targets', args.target_indices or args.target_idx_file),
        ('budget', '%g  (m = round(budget * N_train))' % args.budget),
        ('selector', 'K=%d, lambda=%g, alpha=%g, base_dist=%s'
         % (args.sel_K, args.lambda_margin, args.sel_alpha, args.base_dist)),
        ('selections', 'select_base_random / select_base_ours / '
                       'select_base_ours_div(dpp)'),
        ('metrics', 'Q(S), mean off-diagonal C_ij, effective rank, set overlap'),
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
