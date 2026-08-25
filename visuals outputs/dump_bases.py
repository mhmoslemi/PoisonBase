#!/usr/bin/env python
"""
dump_bases.py -- write every selected base of every target out as its own PNG,
so the qualitative example can be picked by eye.

For each target of one combo it makes

    <out>/<target_id>/target.png
    <out>/<target_id>/1-random.png ... <N_p>-random.png
    <out>/<target_id>/1-dpp.png    ... <N_p>-dpp.png

The numbering is the position in the selected set: for DPP that is the greedy
selection order final_update saved, for random it is the order select_base_random
returned. Images are the ORIGINAL clean CIFAR pixels at native 32x32 -- the base
before any perturbation, denormalized back through the dataset's mean/std. No
attack, crafting or training happens here.

Indices come from the same places plot_random_vs_dpp_bases.py reads:
DPP always from <run_dir>/poison_cache/base_<tid>.json, random from its own
cache when the run kept one and otherwise reconstructed with
final_update.select_base_random under manual_seed(seed*100003 + tid), which
reproduces the saved list exactly.

    python "visuals outputs/dump_bases.py"
    python "visuals outputs/dump_bases.py" --budget 0.002 --model VGG13BN
"""

import argparse
import json
import os
import sys
from argparse import Namespace

import numpy as np
import torch
from PIL import Image

_HERE = os.path.dirname(os.path.abspath(__file__))

# same helpers the figure script uses, kept local so this file stands alone
from plot_random_vs_dpp_bases import (_run_name_args, cached_base, cached_target_ids,
                                      difficulty_label, load_pinned_targets,
                                      random_bases, target_set_path, to_display)


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

    p.add_argument('--save_dir', default=os.path.join(_HERE, 'base_dump'))
    p.add_argument('--scale', type=int, default=1,
                   help='nearest-neighbour upscale factor (1 = native 32x32)')
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
    targets = [t for t in (pinned or have_dpp) if t in have_dpp]

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
    print('pair       : poisons from %s, target is %s' % (class_names[y_adv],
                                                          class_names[target_class]))
    print('budget     : %g -> N_p = %d bases per selection' % (a.budget, N_p))

    def save(img_norm, path):
        arr = (to_display(img_norm, mean, std) * 255.0).round().astype(np.uint8)
        im = Image.fromarray(arr)
        if a.scale > 1:
            im = im.resize((im.width * a.scale, im.height * a.scale), Image.NEAREST)
        im.save(path)

    os.makedirs(a.save_dir, exist_ok=True)
    manifest = {}
    for tid in targets:
        dpp_idx = cached_base(dpp_dir, tid)
        rand_idx, rand_src = random_bases(FU, rand_dir, tid, labels, y_adv, N_p, a.seed)
        d = os.path.join(a.save_dir, str(tid))
        os.makedirs(d, exist_ok=True)
        save(dst_test[tid][0], os.path.join(d, 'target.png'))
        for tag, idxs in (('random', rand_idx), ('dpp', dpp_idx)):
            for i, j in enumerate(idxs, start=1):
                save(dst_train[int(j)][0], os.path.join(d, '%d-%s.png' % (i, tag)))
        manifest[str(tid)] = {'target_index': tid,
                              'target_class': class_names[target_class],
                              'poison_class': class_names[y_adv],
                              'random_source': rand_src,
                              'random_train_indices': rand_idx,
                              'dpp_train_indices': dpp_idx}
        print('  %d: %d random + %d dpp bases -> %s  (random %s)'
              % (tid, len(rand_idx), len(dpp_idx), d, rand_src))

    mpath = os.path.join(a.save_dir, 'manifest.json')
    with open(mpath, 'w') as f:
        json.dump({'run_random': rand_dir, 'run_dpp': dpp_dir, 'budget': a.budget,
                   'N_p': N_p, 'targets': manifest}, f, indent=1)
    print('manifest   : %s  (maps every N-dpp.png / N-random.png to its train index)')
    print('             %s' % mpath)


if __name__ == '__main__':
    main()
