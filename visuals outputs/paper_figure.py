#!/usr/bin/env python
"""
paper_figure.py -- the ICLR-sized version of the base-selection figure.

    [ target ]  16 random bases, one row
                16 dpp    bases, one row

Authored at exactly --width inches (ICLR's text block is 5.5in), so the figure
is included with \\includegraphics[width=\\textwidth] and LaTeX never rescales
it -- rescaling is what blurs the images and pushes a figure past the margin.

Which 16 of the 50
------------------
A subset, chosen by a fixed rule and not by eye: each selection's OWN order,
sampled at an even stride across the whole set. For DPP that order is the greedy
log-det selection order final_update saved, so the stride spans it from the
highest-quality picks through the ones added for diversity; for random it is the
order select_base_random returned, which carries no ranking. The same stride is
applied to both rows, so neither is favoured. --head takes the first 16 instead.

Say so in the caption: these are 16 of the N_p selected bases, not the whole
poison set.

Images are the original clean CIFAR pixels denormalized through the dataset's
mean/std -- the bases before any perturbation. Nothing is crafted or re-run;
indices come from poison_cache/base_<tid>.json (DPP) and the random run's cache
or the exact reconstruction final_update.select_base_random gives.

    python "visuals outputs/paper_figure.py" --target_id 3725
    python "visuals outputs/paper_figure.py" --targets 3725 7663 7488 3875 --stack
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


def stride_pick(idxs, m, head=False):
    """m of idxs, spanning the set at an even stride (or the first m)."""
    n = len(idxs)
    if m >= n:
        return list(idxs)
    if head:
        return list(idxs[:m])
    return [idxs[round(i * (n - 1) / (m - 1))] for i in range(m)] if m > 1 else [idxs[0]]


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--repo_root', default=os.path.dirname(_HERE))
    p.add_argument('--out_dir', default=None)
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

    p.add_argument('--target_id', type=int, default=None, help='single target')
    p.add_argument('--targets', type=int, nargs='*', default=None,
                   help='several targets (one file each, or --stack for one file)')
    p.add_argument('--stack', action='store_true',
                   help='stack --targets into a single figure, one panel per target')
    p.add_argument('--num_display', '-m', type=int, default=16)
    p.add_argument('--head', action='store_true',
                   help='take the first m of each set instead of an even stride')

    p.add_argument('--width', type=float, default=5.5,
                   help='figure width in inches (ICLR text block = 5.5)')
    p.add_argument('--gap_pt', type=float, default=1.0)
    p.add_argument('--row_gap_pt', type=float, default=3.0)
    p.add_argument('--target_gap_pt', type=float, default=6.0)
    p.add_argument('--panel_gap_pt', type=float, default=10.0,
                   help='vertical gap between stacked panels')
    p.add_argument('--labels', action='store_true',
                   help='draw Random / DPP row labels inside the figure '
                        '(default: leave them to the LaTeX caption)')
    p.add_argument('--label_pt', type=float, default=6.0)
    p.add_argument('--save_dir', default=os.path.join(_HERE, 'paper'))
    p.add_argument('--name', default=None)
    p.add_argument('--dpi', type=int, default=600)
    a = p.parse_args()

    a.repo_root = os.path.abspath(a.repo_root)
    a.out_dir = a.out_dir or os.path.join(a.repo_root, 'ours_result')
    a.sweep_config = a.sweep_config or os.path.join(a.repo_root, 'sweep_config.json')
    a.target_sets_dir = a.target_sets_dir or os.path.join(a.repo_root, 'target_sets')

    if a.repo_root not in sys.path:
        sys.path.insert(0, a.repo_root)
    import _old_.final_update as FU
    from _old_.utils import get_dataset

    if a.target_select is None:
        a.target_select = difficulty_label(a.sweep_config, a.model, a.attack,
                                           a.class_pair)

    rand_dir = os.path.join(a.out_dir, FU.build_run_name(_run_name_args(a, 'random', False)))
    dpp_dir = os.path.join(a.out_dir, FU.build_run_name(_run_name_args(a, 'ours', True)))
    tpath = target_set_path(a.target_sets_dir, a.model, a.attack, a.class_pair)
    pinned = load_pinned_targets(tpath, a.class_pair)
    have_dpp = cached_target_ids(dpp_dir)
    if a.targets:
        targets = list(a.targets)
    elif a.target_id is not None:
        targets = [a.target_id]
    else:
        targets = [t for t in (pinned or have_dpp) if t in have_dpp]
    print('random run : %s' % rand_dir)
    print('dpp run    : %s' % dpp_dir)

    channel, im_size, num_classes, class_names, mean, std, dst_train, dst_test, _ = \
        get_dataset(a.dataset, a.data_path)
    y_adv, target_class = FU.parse_pair(a.class_pair, class_names, a.pair_order)
    N_total = len(dst_train)
    N_p = int(round(a.budget * N_total))
    raw = getattr(dst_train, 'targets', None)
    labels = (torch.tensor([int(v) for v in raw], dtype=torch.long) if raw is not None
              else torch.tensor([int(dst_train[i][1]) for i in range(N_total)],
                                dtype=torch.long))
    print('budget %g -> N_p = %d; showing %d of them per row (%s)'
          % (a.budget, N_p, a.num_display, 'first' if a.head else 'even stride'))

    panels = []
    for tid in targets:
        dpp_idx = cached_base(dpp_dir, tid)
        if dpp_idx is None:
            print('  %d: no cached DPP bases, skipped' % tid)
            continue
        rand_idx, src = random_bases(FU, rand_dir, tid, labels, y_adv, N_p, a.seed)
        r = stride_pick(rand_idx, a.num_display, a.head)
        d = stride_pick(dpp_idx, a.num_display, a.head)
        panels.append((tid, r, d))
        print('  %d: random %s' % (tid, r))
        print('     dpp    %s' % d)

    if not panels:
        raise SystemExit('nothing to draw')

    os.makedirs(a.save_dir, exist_ok=True)
    if a.stack and len(panels) > 1:
        render(a, dst_train, dst_test, mean, std, panels,
               a.name or ('panel_%s_%s_%s_b%g_m%d'
                          % (a.model, a.attack, a.class_pair, a.budget, a.num_display)))
    else:
        for tid, r, d in panels:
            render(a, dst_train, dst_test, mean, std, [(tid, r, d)],
                   a.name or ('fig_%s_%s_%s_b%g_t%d_m%d'
                              % (a.model, a.attack, a.class_pair, a.budget, tid,
                                 a.num_display)))


def render(a, dst_train, dst_test, mean, std, panels, name):
    """Author at exactly --width inches; every gap is a real point."""
    PT = 1.0 / 72.0
    m = len(panels[0][1])
    gap, rgap, tgap, pgap = (a.gap_pt * PT, a.row_gap_pt * PT,
                             a.target_gap_pt * PT, a.panel_gap_pt * PT)
    lab_w = (a.label_pt * 2.6 * PT) if a.labels else 0.0

    # solve the cell size from the fixed total width:
    #   width = target(2*cell + rgap) + tgap + m*cell + (m-1)*gap + lab_w
    cell = (a.width - tgap - (m - 1) * gap - lab_w - rgap) / (m + 2.0)
    tsize = 2 * cell + rgap                       # target spans both rows
    panel_h = tsize
    fig_h = len(panels) * panel_h + (len(panels) - 1) * pgap
    fig = plt.figure(figsize=(a.width, fig_h))

    def put(img, x, y, w, h):
        ax = fig.add_axes([x / a.width, y / fig_h, w / a.width, h / fig_h])
        ax.imshow(to_display(img, mean, std), interpolation='nearest')
        ax.set_xticks([])
        ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_visible(False)

    for k, (tid, r_idx, d_idx) in enumerate(panels):
        base_y = fig_h - (k + 1) * panel_h - k * pgap
        put(dst_test[tid][0], 0.0, base_y, tsize, tsize)
        x0 = tsize + tgap + lab_w
        for row, idxs in enumerate((r_idx, d_idx)):
            y = base_y + (cell + rgap if row == 0 else 0.0)
            for i, j in enumerate(idxs):
                put(dst_train[int(j)][0], x0 + i * (cell + gap), y, cell, cell)
            if a.labels:
                fig.text((x0 - 0.35 * PT * a.label_pt) / a.width,
                         (y + cell / 2.0) / fig_h, ('Random', 'DPP')[row],
                         ha='right', va='center', fontsize=a.label_pt)

    stem = os.path.join(a.save_dir, name)
    for ext in ('pdf', 'png'):
        fig.savefig('%s.%s' % (stem, ext), dpi=a.dpi, pad_inches=0.0)
    plt.close(fig)
    print('  -> %s.{pdf,png}   %.2f x %.2f in, cell %.1fpt'
          % (stem, a.width, fig_h, cell / PT))


if __name__ == '__main__':
    main()
