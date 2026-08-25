#!/usr/bin/env python
"""
fig_exp_representation_dynamics.py -- Figure 4 (optional), "victim representation
dynamics".

Question
--------
Do poisons initialized from DPP-selected bases steer the target through
representation space differently from poisons initialized from Random bases?

This script is deliberately independent of Figures 1-3 so it can be included or
dropped late. It is ILLUSTRATIVE: one target, one victim seed per selection. No
causal claim may be made from a single trajectory, and no ASR is plotted.

What it does
------------
1. Loads the ALREADY OPTIMIZED poisons of the two runs from their poison caches,
   exactly the way defense.py replays an attack run:
       <out_dir>/<final_update.build_run_name(...)>/poison_cache/{delta_<t>.pt,
                                                                  base_<t>.json}
   Nothing is re-crafted, so both selections are compared on the poisons the
   paper's own runs produced. A missing cache is a hard error.
2. Trains one victim per selection with final_update.train_from_scratch, under
   the run's victim protocol and the run's own victim seed
   (seed * 100000 + target_idx * 100 + victim_id, as final_update.run_victims
   does), with the poisons written into the training set the same way.
3. Snapshots penultimate representations at ~0%, 15%, 40% and 100% of training.
   The snapshot hook counts optimizer steps and reads the network in eval() -- it
   changes no parameter, no BatchNorm statistic and no random draw, so the
   trajectory is the one the unmodified training loop produces.

Projection (no t-SNE / UMAP)
----------------------------
At each checkpoint, from the CLEAN representations of that checkpoint:

    u    = (mu_adv - mu_target) / ||mu_adv - mu_target||
    c1   = <h - mu_target, u> / ||mu_adv - mu_target||    (0 = target-class
                                                           centroid, 1 = adversarial
                                                           centroid)
    c2   = first principal component of the clean residuals orthogonal to u,
           in the same units

Each panel has its own basis, built by this same rule from its own checkpoint --
representations of two different victims do not live in a common space. Compare
the LAYOUT across panels, never the absolute coordinates.
"""

import argparse
import math
import os

import numpy as np
import torch

import matplotlib.pyplot as plt

import common as C
from common import FU


STEM = 'fig_exp_representation_dynamics'
SELECTIONS = ['random', 'dpp']


# --------------------------------------------------------------------------- #
# locating and loading the saved poisons
# --------------------------------------------------------------------------- #

def run_name_namespace(args, base, sel_dpp):
    """The subset of final_update's args that build_run_name reads."""
    return argparse.Namespace(
        dataset=args.dataset, model=args.model, attack=args.attack, base=base,
        class_pair=args.class_pair, budget=args.budget, epsilon=args.epsilon,
        seed=args.seed, lambda_margin=args.lambda_margin, base_dist=args.base_dist,
        sel_filter=False, sel_pca=False, sel_mmr=False, sel_dpp=sel_dpp,
        sel_pool=3.0, sel_mu=1.0, sel_alpha=args.sel_alpha,
        sel_model=None, sel_criterion=None, sel_K=None, base_topr=None,
        craft_aug=args.craft_aug, dsa_strategy=args.dsa_strategy,
        fc_mode='sample', sharp_mode=args.sharp_mode, sharp_sigma=args.sharp_sigma,
        sharp_samples=args.sharp_samples, craft_ensemble=args.craft_ensemble,
        target_select=args.target_select)


def load_poisons(args, ctx, selection, tidx):
    """(run_dir, base_idx, x_adv01) for one selection, from its poison cache."""
    ns = run_name_namespace(args, base=('random' if selection == 'random' else 'ours'),
                            sel_dpp=(selection == 'dpp'))
    run_dir = os.path.join(args.out_dir, FU.build_run_name(ns))
    if not os.path.isdir(run_dir):
        raise SystemExit('attack run directory missing: %s\nRun the attack first '
                         '(see appendix/ap1-broad.sh); this figure never crafts.'
                         % run_dir)
    delta, base = FU.load_poison_cache(run_dir, tidx,
                                       FU.load_legacy_cache(run_dir, False), False)
    if delta is None or base is None:
        raise SystemExit('no saved poisons for target %d in %s' % (tidx, run_dir))
    base_idx = torch.tensor(base, dtype=torch.long, device=ctx['device'])
    base01 = ctx['denorm'](ctx['train_imgs'][base_idx]).clamp(0.0, 1.0).detach()
    x_adv01 = torch.clamp(base01 + delta.to(ctx['device']), 0.0, 1.0)
    return run_dir, base_idx, x_adv01


# --------------------------------------------------------------------------- #
# victim training with representation snapshots
# --------------------------------------------------------------------------- #

@torch.no_grad()
def embed_batched(net, imgs, bs=512):
    net.eval()
    emb = FU.embed_of(net)
    return torch.cat([emb(imgs[i:i + bs]).flatten(1) for i in range(0, len(imgs), bs)])


def train_with_snapshots(args, ctx, net, snap_epochs, capture):
    """final_update.train_from_scratch, with a snapshot hook on the optimizer.

    The hook only counts SGD steps and, at an epoch boundary, reads the network
    in eval(); it writes nothing back. Forward passes in eval() consume no RNG
    and do not update BatchNorm running statistics, so the training trajectory is
    bit-identical to the unmodified call.
    """
    n = ctx['train_imgs'].shape[0]
    steps_per_epoch = math.ceil(n / args.victim_bs)
    wanted = {int(e) * steps_per_epoch: int(e) for e in snap_epochs if e > 0}
    seen = {'n': 0}
    fired = set()
    orig_step = torch.optim.SGD.step

    def patched(self, *a, **kw):
        out = orig_step(self, *a, **kw)
        seen['n'] += 1
        ep = wanted.get(seen['n'])
        if ep is not None:
            was_training = net.training
            capture(ep)                      # embed_batched puts the net in eval()
            net.train(was_training)
            fired.add(ep)
        return out

    torch.optim.SGD.step = patched          # restored in the finally below
    try:
        FU.train_from_scratch(net, ctx['train_imgs'], ctx['train_labs'],
                              args.victim_epochs, args.victim_lr, args.victim_bs,
                              args.victim_decay, ctx['device'],
                              weight_decay=args.victim_wd, aug=args.victim_aug,
                              dsa_strategy=args.dsa_strategy,
                              dsa_param=ctx['dsa_param'])
    finally:
        torch.optim.SGD.step = orig_step
    missing = set(wanted.values()) - fired
    if missing:
        raise RuntimeError(
            'snapshots for epochs %s never fired (%d optimizer steps seen, %d '
            'expected per epoch) -- the victim schedule is not what this script '
            'assumed.' % (sorted(missing), seen['n'], steps_per_epoch))


# --------------------------------------------------------------------------- #
# the interpretable 2D basis
# --------------------------------------------------------------------------- #

def project(h_tclass, h_aclass, others, seed=0):
    """Coordinates in the (u, PC1 of the orthogonal clean residual) frame."""
    mu_t = h_tclass.mean(dim=0)
    mu_a = h_aclass.mean(dim=0)
    diff = mu_a - mu_t
    scale = float(diff.norm())
    if scale <= 0:
        raise RuntimeError('the two class centroids coincide')
    u = diff / scale

    clean = torch.cat([h_tclass, h_aclass])
    resid = clean - mu_t
    resid = resid - torch.outer(resid @ u, u)          # orthogonal complement of u
    resid = resid - resid.mean(dim=0, keepdim=True)
    # PC1 of the clean residuals, exact and deterministic
    _, _, V = torch.linalg.svd(resid.double(), full_matrices=False)
    v = V[0].float()

    def coords(H):
        d = H - mu_t
        return torch.stack([(d @ u) / scale, (d @ v) / scale], dim=1)

    out = {'target_class': coords(h_tclass), 'adv_class': coords(h_aclass)}
    for k, H in others.items():
        out[k] = coords(H)
    # deterministic sign: the target sits at non-negative c2
    if 'target' in out and float(out['target'][0, 1]) < 0:
        for k in out:
            out[k][:, 1] *= -1.0
    return out


# --------------------------------------------------------------------------- #
# extraction
# --------------------------------------------------------------------------- #

def compute(args):
    device = C.pick_device(args)
    ctx = C.build_ctx(args, device)
    y_adv, target_class = C.classes_of(args, ctx)
    tidx = int(args.target_index)
    if int(ctx['test_labs'][tidx]) != target_class:
        raise SystemExit('test image %d is not of the target class' % tidx)
    x_t = ctx['test_imgs'][tidx]

    epochs = args.victim_epochs
    snap_epochs = sorted({int(round(f * epochs)) for f in args.checkpoint_fractions})
    print('  victim schedule: %d epochs, decay at %s -> snapshots at epochs %s'
          % (epochs, args.victim_decay, snap_epochs))

    # fixed clean samples, identical for both selections and every checkpoint
    rng = C.viz_seed(args.viz_seed)
    labs = ctx['train_labs']

    def sample(cls, exclude):
        pool = (labs == cls).nonzero(as_tuple=True)[0].cpu().numpy()
        pool = np.array([i for i in pool if int(i) not in exclude])
        k = min(args.num_clean, len(pool))
        return torch.tensor(np.sort(rng.choice(pool, size=k, replace=False)),
                            dtype=torch.long, device=device)

    store = {}
    for selection in SELECTIONS:
        run_dir, base_idx, x_adv01 = load_poisons(args, ctx, selection, tidx)
        print('  %s: %d poisons from %s' % (selection, len(base_idx), run_dir))
        poisoned = set(int(i) for i in base_idx.tolist())
        # clean clouds exclude the poisoned rows, which are no longer clean
        idx_t = sample(target_class, poisoned)
        idx_a = sample(y_adv, poisoned)

        seed_v = args.seed * 100000 + tidx * 100 + args.victim_id
        net = FU.build_network(args.model, ctx['channel'], ctx['num_classes'],
                               ctx['im_size'], device, seed=seed_v)

        snaps = {}

        def capture(ep, _net=net, _it=idx_t, _ia=idx_a, _p=x_adv01, _s=snaps):
            _s[int(ep)] = dict(
                target_class=embed_batched(_net, ctx['train_imgs'][_it]).cpu(),
                adv_class=embed_batched(_net, ctx['train_imgs'][_ia]).cpu(),
                poison=embed_batched(_net, ctx['norm'](_p)).cpu(),
                target=embed_batched(_net, x_t.unsqueeze(0)).cpu())

        clean_rows = ctx['train_imgs'][base_idx].clone()
        ctx['train_imgs'][base_idx] = ctx['norm'](x_adv01)     # inject the poisons
        try:
            capture(0)                                         # 0% = initialization
            train_with_snapshots(args, ctx, net, snap_epochs, capture)
        finally:
            ctx['train_imgs'][base_idx] = clean_rows           # always restore

        for ep in snap_epochs:
            if ep not in snaps:
                raise RuntimeError('snapshot for epoch %d was never taken' % ep)
            s = snaps[ep]
            co = project(s['target_class'].to(device), s['adv_class'].to(device),
                         {'poison': s['poison'].to(device),
                          'target': s['target'].to(device)}, seed=args.viz_seed)
            for k, v in co.items():
                store['%s__%d__%s' % (selection, ep, k)] = v.cpu().numpy()
        del net
        if str(device).startswith('cuda'):
            torch.cuda.empty_cache()

    os.makedirs(args.out, exist_ok=True)
    npz = os.path.join(args.out, STEM + '.npz')
    np.savez(npz, selections=np.array(SELECTIONS),
             epochs=np.array(snap_epochs), victim_epochs=np.array([epochs]),
             target_index=np.array([tidx]), victim_id=np.array([args.victim_id]),
             attack=np.array([args.attack]), **store)
    print('  wrote %s' % npz)

    rows = []
    for selection in SELECTIONS:
        for ep in snap_epochs:
            for group in ('target_class', 'adv_class', 'poison', 'target'):
                arr = store['%s__%d__%s' % (selection, ep, group)]
                for i in range(arr.shape[0]):
                    rows.append(dict(selection=selection, epoch=int(ep),
                                     group=group, point=i,
                                     coord1=float(arr[i, 0]), coord2=float(arr[i, 1]),
                                     target_index=tidx, victim_id=args.victim_id))
    C.save_csv(os.path.join(args.out, STEM + '.csv'),
               ['selection', 'epoch', 'group', 'point', 'coord1', 'coord2',
                'target_index', 'victim_id'], rows)
    return npz


# --------------------------------------------------------------------------- #
# plotting
# --------------------------------------------------------------------------- #

def plot(args):
    npz = os.path.join(args.out, STEM + '.npz')
    if not os.path.exists(npz):
        raise SystemExit('%s missing -- run with --compute first.' % npz)
    d = np.load(npz, allow_pickle=True)
    eps = [int(e) for e in d['epochs']]
    total = int(d['victim_epochs'][0])
    tidx = int(d['target_index'][0])

    allc = np.vstack([d['%s__%d__%s' % (s, e, g)]
                      for s in SELECTIONS for e in eps
                      for g in ('target_class', 'adv_class', 'poison', 'target')])
    lim = [(np.percentile(allc[:, i], 0.5), np.percentile(allc[:, i], 99.5))
           for i in range(2)]
    pad = [0.08 * (hi - lo) for lo, hi in lim]

    C.set_paper_style()
    fig, axes = plt.subplots(len(SELECTIONS), len(eps),
                             figsize=(C.WIDTH_FULL, 3.45), constrained_layout=True,
                             sharex=True, sharey=True, squeeze=False)
    for r, selection in enumerate(SELECTIONS):
        for c, ep in enumerate(eps):
            ax = axes[r][c]
            g = lambda k: d['%s__%d__%s' % (selection, ep, k)]
            ax.scatter(g('target_class')[:, 0], g('target_class')[:, 1], s=3.2,
                       c=C.COLORS['tclass'], alpha=0.45, linewidths=0, zorder=1)
            ax.scatter(g('adv_class')[:, 0], g('adv_class')[:, 1], s=3.2,
                       c=C.COLORS['aclass'], alpha=0.45, linewidths=0, zorder=1)
            # small and translucent: the poisons converge into a tight cluster
            # by the later checkpoints, and large opaque markers there paint
            # over each other AND over the clean clouds underneath
            ax.scatter(g('poison')[:, 0], g('poison')[:, 1], s=2.8, marker='o',
                       facecolor=C.COLORS['poison'], edgecolor='none',
                       alpha=0.40, zorder=2)
            t = g('target')
            ax.scatter(t[:, 0], t[:, 1], s=90, marker='*',
                       facecolor=C.COLORS['target'], edgecolor='white',
                       linewidth=0.5, zorder=3)
            ax.set_xlim(lim[0][0] - pad[0], lim[0][1] + pad[0])
            ax.set_ylim(lim[1][0] - pad[1], lim[1][1] + pad[1])
            C.light_grid(ax, axis='both')
            if r == 0:
                ax.set_title('epoch %d  (%.0f%%)' % (ep, 100.0 * ep / total),
                             fontsize=C.FS['label'], pad=3)
            if c == 0:
                ax.set_ylabel('%s\n$c_2$ (clean residual PC1)' % C.LABELS[selection],
                              fontsize=C.FS['label'], color=C.COLORS[selection])
            if r == len(SELECTIONS) - 1:
                ax.set_xlabel('$c_1$ along $\\mu_{adv}-\\mu_{target}$', fontsize=C.FS['label'])

    from matplotlib.lines import Line2D
    handles = [
        Line2D([0], [0], marker='o', linestyle='none', markersize=3,
               color=C.COLORS['tclass'], label='target-class clean'),
        Line2D([0], [0], marker='o', linestyle='none', markersize=3,
               color=C.COLORS['aclass'], label='adversarial-class clean'),
        Line2D([0], [0], marker='o', linestyle='none', markersize=4,
               color=C.COLORS['poison'], label='optimized poisons'),
        Line2D([0], [0], marker='*', linestyle='none', markersize=7,
               color=C.COLORS['target'], label='target $x_t$'),
    ]
    C.bottom_legend(fig, handles, ncol=4)
    # caption goes on the last panel of the top row, not the figure edge: at the
    # bottom edge it sits on top of the legend and the x-axis label below it
    C.note(axes[0][-1], 'target #%d, victim %d, %s'
           % (tidx, int(d['victim_id'][0]), str(d['attack'][0])),
           (0.97, 0.03), ha='right', va='bottom', fontsize=C.FS['small'])

    C.save_fig(fig, args.out, STEM)
    plt.close(fig)


# --------------------------------------------------------------------------- #

def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    C.add_repo_args(p)
    C.add_mode_args(p)
    p.add_argument('--target_index', type=int, required=True,
                   help='one pinned target, e.g. 3741 (do not pick it by ASR)')
    p.add_argument('--victim_id', type=int, default=0,
                   help='victim index; the seed is seed*100000 + target*100 + id, '
                        'exactly as final_update.run_victims computes it')
    p.add_argument('--attack', default='gradmatch',
                   choices=['gradmatch', 'sapa', 'fc'],
                   help='which attack run\'s saved poisons to replay')
    p.add_argument('--budget', type=float, default=C.PAPER['budget'])
    p.add_argument('--epsilon', type=float, default=C.PAPER['epsilon'])
    p.add_argument('--craft_ensemble', type=int, default=C.PAPER['craft_ensemble'])
    p.add_argument('--craft_aug', action='store_true', default=True)
    p.add_argument('--no_craft_aug', dest='craft_aug', action='store_false')
    p.add_argument('--dsa_strategy', default=C.PAPER['dsa_strategy'])
    p.add_argument('--sharp_mode', default=C.PAPER['sharp_mode'],
                   choices=['worst', 'avg'])
    p.add_argument('--sharp_sigma', type=float, default=C.PAPER['sharp_sigma'])
    p.add_argument('--sharp_samples', type=int, default=20)
    p.add_argument('--target_select', default='random',
                   help="naming only: the attack run's --target_select, so the run "
                        "directory is found. The reduced appendix protocol used "
                        "'random' (no _tgt suffix); the main sweep used a "
                        'difficulty degree, e.g. 70.')

    g = p.add_argument_group('victim protocol (defaults = the paper)')
    g.add_argument('--victim_epochs', type=int, default=C.PAPER['victim_epochs'])
    g.add_argument('--victim_lr', type=float, default=C.PAPER['victim_lr'])
    g.add_argument('--victim_bs', type=int, default=C.PAPER['victim_bs'])
    g.add_argument('--victim_decay', nargs='*', type=int,
                   default=C.PAPER['victim_decay'])
    g.add_argument('--victim_wd', type=float, default=C.PAPER['victim_wd'])
    g.add_argument('--victim_aug', action='store_true', default=False)

    p.add_argument('--checkpoint_fractions', type=float, nargs='+',
                   default=[0.0, 0.15, 0.40, 1.0],
                   help='fractions of the victim schedule to snapshot; converted '
                        'to epochs with the ACTUAL --victim_epochs, never hard '
                        'coded')
    p.add_argument('--num_clean', type=int, default=300,
                   help='clean examples sampled per class for the clouds')
    return p.parse_args()


def main():
    args = parse_args()
    do_compute, do_plot = C.resolve_mode(args)
    epochs = args.victim_epochs
    C.summarize('Figure 4 -- victim representation dynamics (illustrative)', [
        ('dataset / model', '%s / %s' % (args.dataset, args.model)),
        ('class pair', '%s (%s)' % (args.class_pair, args.pair_order)),
        ('target / victim', '%d / %d' % (args.target_index, args.victim_id)),
        ('attack replayed', args.attack),
        ('selections', ' '.join(C.LABELS[s] for s in SELECTIONS)),
        ('poisons', 'loaded from <out_dir>/<run_name>/poison_cache (never crafted)'),
        ('victim schedule', '%d epochs, lr %g, bs %d, decay %s'
         % (epochs, args.victim_lr, args.victim_bs, args.victim_decay)),
        ('snapshots at epochs', sorted({int(round(f * epochs))
                                        for f in args.checkpoint_fractions})),
        ('victims to train', '%d (one per selection)' % len(SELECTIONS)),
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
