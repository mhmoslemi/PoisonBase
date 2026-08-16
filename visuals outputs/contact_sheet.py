#!/usr/bin/env python
"""
contact_sheet.py -- one sheet per target showing the WHOLE selected set.

    [ target ]   50 random bases side by side, 1pt gaps
                 50 dpp    bases side by side, 1pt gaps      (3pt between rows)

The target sits at the left, vertically centred across both rows. Every base of
both selections is shown, in the order the selection returned them, so nothing is
sampled or reordered -- for DPP that is the greedy selection order final_update
saved, for random the order select_base_random returned.

Images are the original clean CIFAR pixels, denormalized through the dataset's
mean/std. Nothing is crafted, optimized or re-run. Written straight from the
cached index lists (poison_cache/base_<tid>.json for DPP; the random set from its
own cache when the run kept one, otherwise reconstructed by
final_update.select_base_random under manual_seed(seed*100003 + tid), which
reproduces the saved list exactly).

Geometry is done in points at a fixed scale, so the 1pt / 3pt gaps are real
typographic points in the PDF, not approximations in pixels.

    python "visuals outputs/contact_sheet.py"
    python "visuals outputs/contact_sheet.py" --budget 0.002 --cell_pt 12
"""

import argparse
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))

from plot_random_vs_dpp_bases import (_run_name_args, cached_base, cached_target_ids,
                                      difficulty_label, load_pinned_targets,
                                      random_bases, target_set_path, to_display)

PT = 1.0 / 72.0                      # a typographic point, in inches


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--repo_root', default=os.path.dirname(_HERE))
    p.add_argument('--out_dir', default=None, help='default: <repo_root>/ours_result')
    p.add_argument('--sweep_config', default=None)
    p.add_argument('--target_sets_dir', default=None)
    p.add_argument('--dataset', default='CIFAR10')
    p.add_argument('--data_path', default='/home/mmoslem3/scratch/data')

    p.add_argument('--model', default='ConvNetBN',
                   choices=['ConvNetBN', 'VGG13BN', 'ResNet20BN'])
    p.add_argument('--attack', default='fc', choices=['fc', 'gradmatch', 'sapa'])
    p.add_argument('--class_pair', default='dog-bird',
                   choices=['dog-bird', 'frog-airplane'])
    p.add_argument('--pair_order', default='poison-target',
                   choices=['poison-target', 'target-poison'])
    p.add_argument('--budget', type=float, default=0.001)
    p.add_argument('--epsilon', type=float, default=0.0313725)
    p.add_argument('--seed', type=int, default=42)

    p.add_argument('--sel_alpha', type=float, default=2.0)
    p.add_argument('--lambda_margin', type=float, default=1.0)
    p.add_argument('--base_dist', default='cosine', choices=['l2', 'cosine'])
    p.add_argument('--sel_pool', type=float, default=3.0)
    p.add_argument('--sel_mu', type=float, default=1.0)
    p.add_argument('--fc_mode', default='sample', choices=['sample', 'bullseye'])
    p.add_argument('--sharp_mode', default='worst', choices=['worst', 'avg'])
    p.add_argument('--sharp_sigma', type=float, default=0.05)
    p.add_argument('--sharp_samples', type=int, default=20)
    p.add_argument('--craft_ensemble', type=int, default=5)
    p.add_argument('--target_select', type=int, default=None)

    p.add_argument('--target_id', type=int, default=None,
                   help='only this target (default: every cached one)')
    p.add_argument('--cell_pt', type=float, default=14.0,
                   help='size of one base image, in points')
    p.add_argument('--gap_pt', type=float, default=1.0, help='gap within a row')
    p.add_argument('--row_gap_pt', type=float, default=3.0, help='gap between rows')
    p.add_argument('--target_gap_pt', type=float, default=8.0,
                   help='gap between the target and the rows')
    p.add_argument('--target_scale', type=float, default=2.0,
                   help='target size as a multiple of --cell_pt')
    p.add_argument('--label', action='store_true',
                   help='add small Random / DPP row labels')
    p.add_argument('--save_dir', default=os.path.join(_HERE, 'contact_sheets'))
    p.add_argument('--dpi', type=int, default=600)
    a = p.parse_args()

    a.repo_root = os.path.abspath(a.repo_root)
    a.out_dir = a.out_dir or os.path.join(a.repo_root, 'ours_result')
    a.sweep_config = a.sweep_config or os.path.join(a.repo_root, 'sweep_config.json')
    a.target_sets_dir = a.target_sets_dir or os.path.join(a.repo_root, 'target_sets')

    if a.repo_root not in sys.path:
        sys.path.insert(0, a.repo_root)
    import final_update as FU
    from utils import get_dataset

    if a.target_select is None:
        a.target_select = difficulty_label(a.sweep_config, a.model, a.attack,
                                           a.class_pair)

    rand_dir = os.path.join(a.out_dir, FU.build_run_name(_run_name_args(a, 'random', False)))
    dpp_dir = os.path.join(a.out_dir, FU.build_run_name(_run_name_args(a, 'ours', True)))
    tpath = target_set_path(a.target_sets_dir, a.model, a.attack, a.class_pair)
    pinned = load_pinned_targets(tpath, a.class_pair)
    have_dpp = cached_target_ids(dpp_dir)
    targets = ([a.target_id] if a.target_id is not None
               else [t for t in (pinned or have_dpp) if t in have_dpp])

    print('random run : %s' % rand_dir)
    print('dpp run    : %s' % dpp_dir)
    print('targets    : %s' % targets)
    if not targets:
        raise SystemExit('no target has cached DPP bases under %s' % dpp_dir)

    channel, im_size, num_classes, class_names, mean, std, dst_train, dst_test, _ = \
        get_dataset(a.dataset, a.data_path)
    y_adv, target_class = FU.parse_pair(a.class_pair, class_names, a.pair_order)
    N_total = len(dst_train)
    N_p = int(round(a.budget * N_total))
    raw = getattr(dst_train, 'targets', None)
    labels = (torch.tensor([int(v) for v in raw], dtype=torch.long) if raw is not None
              else torch.tensor([int(dst_train[i][1]) for i in range(N_total)],
                                dtype=torch.long))
    print('pair       : poisons from %s, target is %s'
          % (class_names[y_adv], class_names[target_class]))
    print('budget     : %g -> N_p = %d bases per selection' % (a.budget, N_p))

    os.makedirs(a.save_dir, exist_ok=True)
    for tid in targets:
        dpp_idx = cached_base(dpp_dir, tid)
        if dpp_idx is None:
            print('  %d: no cached DPP bases, skipped' % tid)
            continue
        rand_idx, rand_src = random_bases(FU, rand_dir, tid, labels, y_adv, N_p, a.seed)
        n = min(len(rand_idx), len(dpp_idx))
        sheet(a, dst_train, dst_test, mean, std, tid, rand_idx[:n], dpp_idx[:n])
        print('  %d: %d + %d bases  (random %s)' % (tid, n, n, rand_src))


def sheet(a, dst_train, dst_test, mean, std, tid, rand_idx, dpp_idx):
    """One target: all its random bases over all its dpp bases, exact gaps."""
    n = len(rand_idx)
    cell, gap, rgap, tgap = (a.cell_pt * PT, a.gap_pt * PT,
                             a.row_gap_pt * PT, a.target_gap_pt * PT)
    tsize = a.cell_pt * a.target_scale * PT

    rows_w = n * cell + (n - 1) * gap
    rows_h = 2 * cell + rgap
    fig_w = tsize + tgap + rows_w
    fig_h = max(rows_h, tsize)
    fig = plt.figure(figsize=(fig_w, fig_h))

    def put(img, x, y, w, h):
        """x, y, w, h in inches from the bottom-left of the figure."""
        ax = fig.add_axes([x / fig_w, y / fig_h, w / fig_w, h / fig_h])
        ax.imshow(to_display(img, mean, std), interpolation='nearest')
        ax.set_xticks([])
        ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_visible(False)
        return ax

    # target: left, vertically centred on the two rows
    put(dst_test[tid][0], 0.0, (fig_h - tsize) / 2.0, tsize, tsize)

    x0 = tsize + tgap
    top_y = (fig_h - rows_h) / 2.0 + cell + rgap        # random row
    bot_y = (fig_h - rows_h) / 2.0                      # dpp row
    for k, (idxs, y) in enumerate(((rand_idx, top_y), (dpp_idx, bot_y))):
        for i, j in enumerate(idxs):
            put(dst_train[int(j)][0], x0 + i * (cell + gap), y, cell, cell)
        if a.label:
            fig.text((x0 - gap) / fig_w, (y + cell / 2.0) / fig_h,
                     ('Random', 'DPP')[k], ha='right', va='center', fontsize=5)

    stem = os.path.join(a.save_dir, 'sheet_%s_%s_%s_b%g_t%d'
                        % (a.model, a.attack, a.class_pair, a.budget, tid))
    for ext in ('pdf', 'png'):
        fig.savefig('%s.%s' % (stem, ext), dpi=a.dpi,
                    bbox_inches='tight', pad_inches=0.01)
    plt.close(fig)
    print('       -> %s.{pdf,png}' % stem)


if __name__ == '__main__':
    main()
