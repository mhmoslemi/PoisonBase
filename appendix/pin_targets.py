"""Freeze the target--adversarial-class instances an appendix table shares.

The appendix tables insist that every selection method (and, for the broad
CIFAR benchmark, every attack) is scored on the SAME instances. Target choice
depends only on the clean victim ensemble, never on the attack or the base
selection, so the set can be fixed once here and then handed to every run with
--target_idx_file.

    python appendix/pin_targets.py --model ResNet20BN --pair cat-deer \
        --num_targets 20 --out target_sets/appx_ResNet20BN_cat-deer.json

Writes the usual {"pairs": {pair: {"indices": [...]}}} format. Idempotent: an
existing file is left alone unless --force is passed.
"""
import argparse, json, os, sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
import final_update as FU


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--dataset', default='CIFAR10')
    p.add_argument('--data_path', default='/home/mmoslem3/scratch/data')
    p.add_argument('--cache_dir', default='./cache')
    p.add_argument('--model', required=True)
    p.add_argument('--pair', required=True)
    p.add_argument('--pair_order', default='poison-target')
    p.add_argument('--num_targets', type=int, default=20)
    p.add_argument('--target_select', default='random')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--num_victims', type=int, default=5)
    p.add_argument('--victim_epochs', type=int, default=50)
    p.add_argument('--victim_lr', type=float, default=0.1)
    p.add_argument('--victim_bs', type=int, default=125)
    p.add_argument('--victim_decay', nargs='*', type=int, default=[40])
    p.add_argument('--victim_wd', type=float, default=0.0)
    p.add_argument('--victim_aug', action='store_true', default=False)
    p.add_argument('--dsa_strategy', default='color_crop_cutout_flip_scale_rotate')
    p.add_argument('--require_correct_target', action='store_true', default=False)
    p.add_argument('--out', required=True)
    p.add_argument('--force', action='store_true')
    a = p.parse_args()

    if os.path.exists(a.out) and not a.force:
        print('%s already exists -- leaving it alone (--force to redo)' % a.out)
        return

    a.target_select = FU.target_select_arg(a.target_select)
    a.rank_on_victims = True
    a.target_idx_file = None
    a.class_pair = a.pair

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if device == 'cpu':
        sys.exit('pin_targets: no GPU visible -- run this inside an allocation')

    ctx = FU.build_context(a, device)
    y_adv, target_class = FU.parse_pair(a.class_pair, ctx['class_names'], a.pair_order)
    victims = FU.get_clean_victims(a, ctx['train_imgs'], ctx['train_labs'], ctx['test_imgs'],
                                   ctx['test_labs'], ctx['channel'], ctx['num_classes'],
                                   ctx['im_size'], device, ctx['dsa_param'])
    gen = torch.Generator(device='cpu').manual_seed(a.seed)
    targets, _ = FU.select_targets(a, victims, ctx['test_imgs'], ctx['test_labs'],
                                   y_adv, target_class, gen)
    idx = [int(t) for t in targets]
    os.makedirs(os.path.dirname(a.out) or '.', exist_ok=True)
    with open(a.out, 'w') as f:
        json.dump({'_generated_by': 'appendix/pin_targets.py -- shared across every '
                                    'selection method and attack of this table',
                   '_combo': '%s / %s / %s' % (a.dataset, a.model, a.pair),
                   '_num_targets': len(idx),
                   'pairs': {a.pair: {'indices': idx}}}, f, indent=1)
    print('%s <- %d targets %s' % (a.out, len(idx), idx))
    # tab:additional-target-indices wants these verbatim
    adv, tgt = a.pair.split('-')
    print('  latex row:  %s & %s $\\rightarrow$ %s & %s \\\\'
          % (a.dataset.replace('CIFAR10', 'CIFAR-10'), tgt, adv, ', '.join(str(i) for i in idx)))


if __name__ == '__main__':
    main()
