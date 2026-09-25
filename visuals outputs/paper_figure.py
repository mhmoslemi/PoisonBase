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
    python "visuals outputs/paper_figure.py" --target_id 3875 -m 25
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

from plot_random_vs_dpp_bases import (
    _run_name_args,
    cached_base,
    cached_target_ids,
    difficulty_label,
    load_pinned_targets,
    random_bases,
    target_set_path,
    to_display,
)


def stride_pick(idxs, m, head=False):
    """m of idxs, spanning the set at an even stride (or the first m)."""
    n = len(idxs)
    if m >= n:
        return list(idxs)
    if head:
        return list(idxs[:m])
    return [idxs[round(i * (n - 1) / (m - 1))] for i in range(m)] if m > 1 else [idxs[0]]


@torch.no_grad()
def displayed_similarities(a, FU, channel, num_classes, im_size, dst_train,
                           dst_test, panels):
    """Cosine similarity of each displayed base to its test target.

    Values are averaged over the saved ConvNetBN surrogate checkpoints. This
    function only loads existing checkpoints; it never trains a model.
    """
    import torch.nn.functional as F

    device = a.device
    if device == 'auto':
        device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    checkpoint_dir = os.path.join(
        a.cache_dir, 'surrogates',
        'ConvNetBN_%dep_lr%g_bs%d_seed%d'
        % (a.surrogate_epochs, a.surrogate_lr, a.surrogate_bs, a.seed),
    )
    checkpoints = [
        os.path.join(checkpoint_dir, 'net_%d.pt' % model_id)
        for model_id in range(a.num_surrogates)
    ]
    missing = [checkpoint for checkpoint in checkpoints
               if not os.path.exists(checkpoint)]
    if missing:
        raise SystemExit(
            'trained ConvNetBN checkpoint(s) not found:\n%s\n'
            'Point --cache_dir at the existing surrogate cache; this script '
            'will not train them.' % '\n'.join(missing)
        )

    sums = {}
    for model_id, checkpoint in enumerate(checkpoints):
        net = FU.build_network(
            'ConvNetBN', channel, num_classes, im_size, device,
            seed=a.seed + 1000 + model_id,
        )
        net.load_state_dict(torch.load(checkpoint, map_location=device))
        net.eval()
        embed = FU.embed_of(net)
        for tid, r_idx, d_idx in panels:
            target = dst_test[int(tid)][0].unsqueeze(0).to(device)
            target_feature = F.normalize(embed(target).flatten(1), dim=1)
            indices = list(dict.fromkeys(list(r_idx) + list(d_idx)))
            bases = torch.stack(
                [dst_train[int(index)][0] for index in indices]
            ).to(device)
            base_features = F.normalize(embed(bases).flatten(1), dim=1)
            cosine = (base_features @ target_feature.T).squeeze(1).cpu().tolist()
            for index, value in zip(indices, cosine):
                key = (int(tid), int(index))
                sums[key] = sums.get(key, 0.0) + float(value)
        del net

    values = {key: total / a.num_surrogates for key, total in sums.items()}
    print('similarity : mean ConvNetBN cosine across %d saved surrogates in %s'
          % (a.num_surrogates, checkpoint_dir))
    return values


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--repo_root', default=os.path.dirname(_HERE))
    p.add_argument('--out_dir', default=None)
    p.add_argument('--sweep_config', default=None)
    p.add_argument('--target_sets_dir', default=None)
    p.add_argument('--dataset', default='CIFAR10')
    p.add_argument('--data_path', default='/home/mmoslem3/scratch/data')
    p.add_argument('--device', default='auto',
                   help="'auto' | 'cpu' | 'cuda:0'; used for similarity labels")
    p.add_argument('--cache_dir', default=None,
                   help='existing surrogate cache used for similarity labels '
                        '(default: <repo_root>/cache)')

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
    p.add_argument('--surrogate_epochs', type=int, default=60)
    p.add_argument('--surrogate_lr', type=float, default=0.1)
    p.add_argument('--surrogate_bs', type=int, default=128)
    p.add_argument('--num_surrogates', type=int, default=20,
                   help='number of saved ConvNetBN checkpoints to average')

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
    p.add_argument('--similarity_pt', type=float, default=4.3,
                   help='font size of the cosine number below each base')
    p.add_argument('--similarity_height_pt', type=float, default=7.0,
                   help='vertical space reserved below each base')
    p.add_argument('--average_pt', type=float, default=5.2,
                   help='font size of the displayed-row average at the right')
    p.add_argument('--average_width_pt', type=float, default=36.0,
                   help='horizontal space reserved for the row average')
    p.add_argument('--save_dir', default=None,
                   help='output directory (default: <repo_root>/out2)')
    p.add_argument('--name', default=None)
    p.add_argument('--dpi', type=int, default=600)
    a = p.parse_args()

    a.repo_root = os.path.abspath(a.repo_root)
    a.out_dir = a.out_dir or os.path.join(a.repo_root, 'ours_result')
    a.cache_dir = a.cache_dir or os.path.join(a.repo_root, 'cache')
    a.save_dir = a.save_dir or os.path.join(a.repo_root, 'out2')
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

    similarities = displayed_similarities(
        a, FU, channel, num_classes, im_size, dst_train, dst_test, panels,
    )

    os.makedirs(a.save_dir, exist_ok=True)
    if a.stack and len(panels) > 1:
        render(a, dst_train, dst_test, mean, std, panels,
               a.name or ('panel_%s_%s_%s_b%g_m%d'
                          % (a.model, a.attack, a.class_pair, a.budget, a.num_display)),
               similarities)
    else:
        for tid, r, d in panels:
            render(a, dst_train, dst_test, mean, std, [(tid, r, d)],
                   a.name or ('fig_%s_%s_%s_b%g_t%d_m%d'
                              % (a.model, a.attack, a.class_pair, a.budget, tid,
                                 a.num_display)),
                   similarities)


def render(a, dst_train, dst_test, mean, std, panels, name, similarities):
    """Author at exactly --width inches; every gap is a real point."""
    PT = 1.0 / 72.0
    m = len(panels[0][1])
    gap, rgap, tgap, pgap = (a.gap_pt * PT, a.row_gap_pt * PT,
                             a.target_gap_pt * PT, a.panel_gap_pt * PT)
    lab_w = (a.label_pt * 2.6 * PT) if a.labels else 0.0
    avg_w = a.average_width_pt * PT
    sim_h = a.similarity_height_pt * PT

    # solve the cell size from the fixed total width:
    # The target is square and spans from the top of the upper image row to
    # the bottom of the lower image row.  The labels below the lower row are
    # outside the target's extent.
    cell = (a.width - tgap - (m - 1) * gap - lab_w - avg_w
            - rgap - sim_h) / (m + 2.0)
    panel_h = 2 * cell + rgap + 2 * sim_h
    tsize = panel_h - sim_h
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
        put(dst_test[tid][0], 0.0, base_y + sim_h, tsize, tsize)
        x0 = tsize + tgap + lab_w
        for row, idxs in enumerate((r_idx, d_idx)):
            y = base_y + sim_h + (cell + rgap + sim_h if row == 0 else 0.0)
            for i, j in enumerate(idxs):
                x = x0 + i * (cell + gap)
                put(dst_train[int(j)][0], x, y, cell, cell)
                fig.text(
                    (x + cell / 2.0) / a.width,
                    (y - sim_h / 2.0) / fig_h,
                    '%.2f' % similarities[(int(tid), int(j))],
                    ha='center', va='center', fontsize=a.similarity_pt,
                    color='#4D4D4D',
                )
            row_mean = sum(similarities[(int(tid), int(j))] for j in idxs) / len(idxs)
            average_left = x0 + m * cell + (m - 1) * gap
            separator_x = average_left + 3.0 * PT
            average_center = separator_x + (avg_w - 3.0 * PT) / 2.0
            fig.add_artist(plt.Line2D(
                [separator_x / a.width, separator_x / a.width],
                [(y + 0.10 * cell) / fig_h, (y + 0.90 * cell) / fig_h],
                transform=fig.transFigure, color='#B8B8B8', linewidth=0.45,
            ))
            fig.text(
                average_center / a.width,
                (y + 0.70 * cell) / fig_h,
                'Avg. similarity',
                ha='center', va='center', fontsize=3.8,
                color='#555555',
            )
            fig.text(
                average_center / a.width,
                (y + 0.30 * cell) / fig_h,
                '%.3f' % row_mean,
                ha='center', va='center', fontsize=a.average_pt,
                color='#111111', fontweight='semibold',
            )
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
