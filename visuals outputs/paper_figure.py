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
    python "visuals outputs/paper_figure.py" --target_id 3875 -m 25 \
        --similarity_scatter
    python "visuals outputs/paper_figure.py" --targets 3725 7663 7488 3875 --stack
"""

import argparse
import csv
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
    candidate_bank,
    difficulty_label,
    load_pinned_targets,
    load_surrogates,
    random_bases,
    target_set_path,
    target_terms,
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


def all_target_similarity(a, FU, channel, num_classes, im_size, dst_train,
                          dst_test, mean, std, labels, y_adv, N_p, pinned,
                          have_dpp, rand_dir, dpp_dir):
    """Mean target--base cosine for RAND and BASIS over every pinned target.

    Each value averages over the full selected set. ``selection`` uses the
    original ensemble; ``convnet`` uses one repository ConvNetBN surrogate,
    training and caching it with the existing protocol when it is absent.
    """
    target_ids = [target for target in (pinned or have_dpp)
                  if target in have_dpp]
    if not target_ids:
        raise SystemExit('no pinned target has cached BASIS/DPP bases in %s'
                         % dpp_dir)

    if a.similarity_space == 'convnet':
        return trained_convnet_similarity(
            a, FU, channel, num_classes, im_size, dst_train, dst_test,
            labels, y_adv, N_p, target_ids, rand_dir, dpp_dir,
        )

    device = a.device
    if device == 'auto':
        device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    nets, sdir = load_surrogates(
        FU, a, channel, num_classes, im_size, device,
    )
    print('surrogates : %d x %s from %s' % (len(nets), a.model, sdir))

    cls_idx = (labels == y_adv).nonzero(as_tuple=True)[0]
    candidates = torch.stack(
        [dst_train[int(index)][0] for index in cls_idx]
    ).to(device)
    pos_of = torch.full((len(dst_train),), -1, dtype=torch.long)
    pos_of[cls_idx] = torch.arange(len(cls_idx))
    raw, unit, margin = candidate_bank(
        FU, nets, candidates, y_adv, a.batch_size,
    )

    rows = []
    for target in target_ids:
        dpp_idx = cached_base(dpp_dir, target)
        rand_idx, _ = random_bases(
            FU, rand_dir, target, labels, y_adv, N_p, a.seed,
        )
        relevance, _ = target_terms(
            FU, nets, raw, unit, margin,
            dst_test[target][0].to(device),
            a.lambda_margin, a.base_dist,
        )
        rand_pos = pos_of[torch.tensor(rand_idx)]
        dpp_pos = pos_of[torch.tensor(dpp_idx)]
        if int(rand_pos.min()) < 0 or int(dpp_pos.min()) < 0:
            raise SystemExit(
                'target %d: a cached base is outside poison class %d; check '
                'the run directories and --class_pair / --pair_order'
                % (target, y_adv)
            )
        rand_cosine = float(relevance[:, rand_pos.to(device)].mean().cpu())
        dpp_cosine = float(relevance[:, dpp_pos.to(device)].mean().cpu())
        rows.append({
            'target_id': int(target),
            'rand_mean_cosine': rand_cosine,
            'basis_mean_cosine': dpp_cosine,
            'basis_minus_rand': dpp_cosine - rand_cosine,
            'num_bases': min(len(rand_idx), len(dpp_idx)),
            'num_surrogates': len(nets),
            'representation': 'selection surrogate ensemble',
        })

    print('all-target target--base cosine: %d targets, %d bases per set'
          % (len(rows), rows[0]['num_bases']))
    return rows


def trained_convnet_similarity(a, FU, channel, num_classes, im_size, dst_train,
                               dst_test, labels, y_adv, N_p, target_ids,
                               rand_dir, dpp_dir):
    """All-target cosine comparison using one repository ConvNetBN surrogate."""
    import torch.nn.functional as F

    device = a.device
    if device == 'auto':
        if torch.cuda.is_available():
            device = 'cuda:0'
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            device = 'mps'
        else:
            device = 'cpu'

    checkpoint_dir = os.path.join(
        a.cache_dir, 'surrogates',
        '%s_%dep_lr%g_bs%d_seed%d'
        % (a.model, a.surrogate_epochs, a.surrogate_lr,
           a.surrogate_bs, a.seed),
    )
    checkpoint = os.path.join(checkpoint_dir, 'net_0.pt')
    net = FU.build_network(
        a.model, channel, num_classes, im_size, device,
        seed=a.seed + 1000,
    )
    if os.path.exists(checkpoint):
        net.load_state_dict(torch.load(checkpoint, map_location=device))
        print('similarity ConvNet: loaded %s' % checkpoint)
    else:
        print('similarity ConvNet: training one %s with the repository protocol '
              'on %s' % (a.model, device), flush=True)
        train_images = torch.stack(
            [dst_train[index][0] for index in range(len(dst_train))]
        ).to(device)
        train_labels = labels.to(device)
        test_images = torch.stack(
            [dst_test[index][0] for index in range(len(dst_test))]
        ).to(device)
        test_labels = torch.tensor(
            [int(dst_test[index][1]) for index in range(len(dst_test))],
            dtype=torch.long, device=device,
        )
        with torch.enable_grad():
            FU.train_from_scratch(
                net, train_images, train_labels,
                a.surrogate_epochs, a.surrogate_lr, a.surrogate_bs,
                a.surrogate_decay, device,
                weight_decay=a.surrogate_wd, aug=False,
            )
        accuracy = FU.test_acc(net, test_images, test_labels)
        os.makedirs(checkpoint_dir, exist_ok=True)
        torch.save(net.state_dict(), checkpoint)
        print('similarity ConvNet: test accuracy %.4f; saved %s'
              % (accuracy, checkpoint), flush=True)
        del train_images, train_labels, test_images, test_labels
    net.eval()
    encoder = FU.embed_of(net)

    selections = {}
    train_ids = set()
    for target in target_ids:
        dpp_idx = cached_base(dpp_dir, target)
        rand_idx, _ = random_bases(
            FU, rand_dir, target, labels, y_adv, N_p, a.seed,
        )
        selections[target] = (rand_idx, dpp_idx)
        train_ids.update(int(index) for index in rand_idx)
        train_ids.update(int(index) for index in dpp_idx)

    @torch.no_grad()
    def encode(images):
        outputs = []
        for start in range(0, len(images), a.similarity_batch_size):
            batch = torch.stack(
                images[start:start + a.similarity_batch_size]
            ).to(device)
            outputs.append(F.normalize(
                encoder(batch).detach().flatten(1), dim=1,
            ).cpu())
        return torch.cat(outputs)

    ordered_train_ids = sorted(train_ids)
    train_features = encode(
        [dst_train[index][0] for index in ordered_train_ids]
    )
    train_position = {
        index: position for position, index in enumerate(ordered_train_ids)
    }
    target_features = encode(
        [dst_test[target][0] for target in target_ids]
    )

    rows = []
    for position, target in enumerate(target_ids):
        rand_idx, dpp_idx = selections[target]
        rand_pos = [train_position[int(index)] for index in rand_idx]
        dpp_pos = [train_position[int(index)] for index in dpp_idx]
        target_feature = target_features[position]
        rand_cosine = float(
            train_features[rand_pos].matmul(target_feature).mean()
        )
        dpp_cosine = float(
            train_features[dpp_pos].matmul(target_feature).mean()
        )
        rows.append({
            'target_id': int(target),
            'rand_mean_cosine': rand_cosine,
            'basis_mean_cosine': dpp_cosine,
            'basis_minus_rand': dpp_cosine - rand_cosine,
            'num_bases': min(len(rand_idx), len(dpp_idx)),
            'num_surrogates': 1,
            'representation': '%s surrogate seed %d'
                              % (a.model, a.seed + 1000),
        })

    print('all-target %s cosine: %d targets, %d bases per set'
          % (a.model, len(rows), rows[0]['num_bases']))
    return rows


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
                   help="'auto' | 'cpu' | 'cuda:0'; used only by "
                        '--similarity_scatter')
    p.add_argument('--cache_dir', default=None,
                   help='surrogate cache used only by --similarity_scatter '
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
    p.add_argument('--num_surrogates', type=int, default=20)
    p.add_argument('--surrogate_epochs', type=int, default=60)
    p.add_argument('--surrogate_lr', type=float, default=0.1)
    p.add_argument('--surrogate_bs', type=int, default=128)
    p.add_argument('--surrogate_decay', nargs='*', type=int, default=[35, 45])
    p.add_argument('--surrogate_wd', type=float, default=0.0)
    p.add_argument('--batch_size', type=int, default=512)

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
    p.add_argument('--similarity_scatter', action='store_true',
                   help='append an all-target paired scatter: x = RAND and '
                        'y = BASIS mean target--base cosine, measured over the '
                        'full selected sets and surrogate ensemble')
    p.add_argument('--similarity_space', choices=('selection', 'convnet'),
                   default='selection',
                   help="'selection' uses the original surrogate ensemble; "
                        "'convnet' uses one repository ConvNetBN surrogate")
    p.add_argument('--similarity_batch_size', type=int, default=64,
                   help='feature-extraction batch size for --similarity_space '
                        'convnet')
    p.add_argument('--scatter_height', type=float, default=3.0,
                   help='height in inches of the optional quantitative panel')
    p.add_argument('--scatter_gap_pt', type=float, default=10.0,
                   help='vertical gap before the optional quantitative panel')
    p.add_argument('--save_dir', default=os.path.join(_HERE, 'paper'))
    p.add_argument('--name', default=None)
    p.add_argument('--dpi', type=int, default=600)
    a = p.parse_args()

    a.repo_root = os.path.abspath(a.repo_root)
    a.out_dir = a.out_dir or os.path.join(a.repo_root, 'ours_result')
    a.cache_dir = a.cache_dir or os.path.join(a.repo_root, 'cache')
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

    similarity_rows = None
    if a.similarity_scatter:
        similarity_rows = all_target_similarity(
            a, FU, channel, num_classes, im_size, dst_train, dst_test, mean,
            std, labels, y_adv, N_p, pinned, have_dpp, rand_dir, dpp_dir,
        )

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
                          % (a.model, a.attack, a.class_pair, a.budget, a.num_display)),
               similarity_rows)
    else:
        for tid, r, d in panels:
            render(a, dst_train, dst_test, mean, std, [(tid, r, d)],
                   a.name or ('fig_%s_%s_%s_b%g_t%d_m%d'
                              % (a.model, a.attack, a.class_pair, a.budget, tid,
                                 a.num_display)),
                   similarity_rows)


def render(a, dst_train, dst_test, mean, std, panels, name,
           similarity_rows=None):
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
    qualitative_h = len(panels) * panel_h + (len(panels) - 1) * pgap
    scatter_gap = a.scatter_gap_pt * PT if similarity_rows else 0.0
    scatter_h = a.scatter_height if similarity_rows else 0.0
    fig_h = qualitative_h + scatter_gap + scatter_h
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
                         (y + cell / 2.0) / fig_h, ('RAND', 'BASIS')[row],
                         ha='right', va='center', fontsize=a.label_pt)

    stem = os.path.join(a.save_dir, name)
    if similarity_rows:
        xs = [row['rand_mean_cosine'] for row in similarity_rows]
        ys = [row['basis_mean_cosine'] for row in similarity_rows]
        low, high = min(xs + ys), max(xs + ys)
        span = high - low
        padding = max(0.01, 0.08 * span)
        limits = (low - padding, high + padding)

        axes_bottom = 0.48
        axes_top = 0.18
        axes_size = min(
            a.width - 1.4,
            scatter_h - axes_bottom - axes_top,
        )
        axes_left = (a.width - axes_size) / 2.0
        ax = fig.add_axes([
            axes_left / a.width,
            axes_bottom / fig_h,
            axes_size / a.width,
            axes_size / fig_h,
        ])
        ax.plot(limits, limits, color='#444444', linestyle='--',
                linewidth=0.9, zorder=1)

        shown_targets = {int(panel[0]) for panel in panels}
        ordinary = [row for row in similarity_rows
                    if row['target_id'] not in shown_targets]
        highlighted = [row for row in similarity_rows
                       if row['target_id'] in shown_targets]
        if ordinary:
            ax.scatter(
                [row['rand_mean_cosine'] for row in ordinary],
                [row['basis_mean_cosine'] for row in ordinary],
                s=28, color='#0072B2', edgecolors='white', linewidths=0.45,
                zorder=3,
            )
        if highlighted:
            ax.scatter(
                [row['rand_mean_cosine'] for row in highlighted],
                [row['basis_mean_cosine'] for row in highlighted],
                s=42, color='#D55E00', edgecolors='white', linewidths=0.55,
                zorder=4,
            )
            for row in highlighted:
                ax.annotate(
                    str(row['target_id']),
                    (row['rand_mean_cosine'], row['basis_mean_cosine']),
                    xytext=(4, 3), textcoords='offset points', fontsize=6.5,
                    color='#7A2E00',
                )

        mean_gain = sum(row['basis_minus_rand']
                        for row in similarity_rows) / len(similarity_rows)
        representation = similarity_rows[0]['representation']
        ax.text(
            0.02, 0.98,
            r'$n=%d$, mean $\Delta=%+.3f$' % (len(similarity_rows), mean_gain),
            transform=ax.transAxes, ha='left', va='top', fontsize=7,
        )
        ax.set_xlim(limits)
        ax.set_ylim(limits)
        ax.set_aspect('equal', adjustable='box')
        ax.set_xlabel('RAND mean target--base cosine', fontsize=8)
        ax.set_ylabel('BASIS mean target--base cosine', fontsize=8)
        ax.set_title(
            'All targets; full selected sets; %s' % representation,
            fontsize=8.5, pad=4,
        )
        ax.tick_params(axis='both', labelsize=7, length=2.5, width=0.7)
        ax.grid(True, color='#DDDDDD', linewidth=0.45, zorder=0)
        for spine in ax.spines.values():
            spine.set_linewidth(0.8)

        csv_path = stem + '_similarity.csv'
        fields = (
            'target_id', 'rand_mean_cosine', 'basis_mean_cosine',
            'basis_minus_rand', 'num_bases', 'num_surrogates',
            'representation',
        )
        with open(csv_path, 'w', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(similarity_rows)
        print('  -> %s' % csv_path)

    for ext in ('pdf', 'png'):
        fig.savefig('%s.%s' % (stem, ext), dpi=a.dpi, pad_inches=0.0)
    plt.close(fig)
    print('  -> %s.{pdf,png}   %.2f x %.2f in, cell %.1fpt'
          % (stem, a.width, fig_h, cell / PT))


if __name__ == '__main__':
    main()
