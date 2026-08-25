#!/usr/bin/env python
"""
fig_poison_dynamics.py -- how the poison set and the target move through the
victim's representation space during training, and which poisons actually pull.

Modelled on the EPIC overview picture (Yang et al., ICLR 2023): columns are
training checkpoints, the two clean class clouds are drawn faint, the target is a
star, and the poisons are split into EFFECTIVE and INEFFECTIVE. The difference
here is the comparison: the top row is a Random base set, the bottom row is the
DPP base set, for the SAME target and the SAME victim seed, so the two rows
differ only in which clean images were perturbed.

Effectiveness is not decorative. At each checkpoint, for every poison,

    a_i = cos( grad_theta CE(x_p_i, y_adv),  grad_theta CE(x_t, y_adv) )

over the FULL parameter vector of the victim at that moment. a_i > 0 means the
gradient step this poison contributes has a positive component along the
direction that drives the target toward y_adv -- exactly the "closer to the
target in gradient space" notion EPIC uses to decide which poisons matter, and
the same quantity family as Figure 1's held-out utility.

Projection (no t-SNE / UMAP)
----------------------------
At each checkpoint, from that checkpoint's CLEAN representations,

    u  = (mu_adv - mu_target) / ||mu_adv - mu_target||
    c1 = <h - mu_target, u> / ||mu_adv - mu_target||   (0 = target-class centroid,
                                                        1 = adversarial centroid)
    c2 = first principal component of the clean residual orthogonal to u

Each panel is built by the same rule from its own checkpoint, so compare the
LAYOUT across panels, never the absolute coordinates.

Cost
----
--compute trains one victim per arm under the paper's victim protocol (2 x 50
epochs, ~40 min on an L40S) and adds ~2 min of per-poison gradients. Everything
is cached in the .npz, so --plot_only is free and needs no gpu.
"""

import argparse
import math
import os

import numpy as np
import torch
import torch.nn as nn

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

import common as C
import plotstyle as P
from common import FU


STEM = 'fig_poison_dynamics'
SELECTIONS = ['random', 'dpp']

# Same identity as fig_exp_representation_dynamics.py and every other figure in
# the paper: tclass/aclass/target are the shared palette entries, not a local
# override. Effective poisons (the ones actually pulling the target) get the
# paper's 'poison' green; ineffective ones fade to the same neutral grey the
# paper already uses elsewhere for "not the interesting point" (fig 1's
# unfavourable bases), so the panel reads at a glance without a new hue.
COL = {'tclass': P.COLORS['tclass'], 'aclass': P.COLORS['aclass'],
       'eff': P.COLORS['poison'], 'ineff': P.COLORS['pool'],
       'target': P.COLORS['target']}


# --------------------------------------------------------------------------- #
# extraction
# --------------------------------------------------------------------------- #

def run_name_namespace(args, base, sel_dpp):
    """The subset of final_update's args that build_run_name reads."""
    return argparse.Namespace(
        dataset=args.dataset, model=args.model, attack=args.attack, base=base,
        class_pair=args.class_pair, budget=args.budget, epsilon=args.epsilon,
        seed=args.seed, lambda_margin=args.lambda_margin, base_dist=args.base_dist,
        sel_filter=False, sel_pca=False, sel_mmr=False, sel_dpp=sel_dpp,
        sel_pool=3.0, sel_mu=1.0, sel_alpha=args.sel_alpha, sel_model=None,
        sel_criterion=None, sel_K=None, base_topr=None, craft_aug=args.craft_aug,
        dsa_strategy=args.dsa_strategy, fc_mode='sample',
        sharp_mode=args.sharp_mode, sharp_sigma=args.sharp_sigma,
        sharp_samples=args.sharp_samples, craft_ensemble=args.craft_ensemble,
        target_select=args.target_select)


def load_poisons(args, ctx, selection, tidx):
    """(run_dir, base_idx, x_adv01) from the attack run's poison cache."""
    ns = run_name_namespace(args, 'random' if selection == 'random' else 'ours',
                            selection == 'dpp')
    run_dir = os.path.join(args.out_dir, FU.build_run_name(ns))
    if not os.path.isdir(run_dir):
        raise SystemExit('attack run directory missing: %s\nRun the attack first; '
                         'this figure never crafts.' % run_dir)
    delta, base = FU.load_poison_cache(run_dir, tidx,
                                       FU.load_legacy_cache(run_dir, False), False)
    if delta is None or base is None:
        raise SystemExit('no saved poisons for target %d in %s' % (tidx, run_dir))
    base_idx = torch.tensor(base, dtype=torch.long, device=ctx['device'])
    base01 = ctx['denorm'](ctx['train_imgs'][base_idx]).clamp(0.0, 1.0).detach()
    return run_dir, base_idx, torch.clamp(base01 + delta.to(ctx['device']), 0.0, 1.0)


@torch.no_grad()
def embed_batched(net, imgs, bs=512):
    net.eval()
    emb = FU.embed_of(net)
    return torch.cat([emb(imgs[i:i + bs]).flatten(1) for i in range(0, len(imgs), bs)])


def gradient_alignment(net, x_poison_norm, x_t_norm, y_adv, device):
    """cos(grad CE(x_p, y_adv), grad CE(x_t, y_adv)) per poison, full parameters.

    net is left in eval() so BatchNorm uses its running statistics and measuring
    never perturbs them, the same convention final_update._target_grads uses.
    """
    was_training = net.training
    net.eval()
    crit = nn.CrossEntropyLoss().to(device)
    y = torch.full((1,), y_adv, dtype=torch.long, device=device)
    params = [p for p in net.parameters()]

    loss_t = crit(net(x_t_norm.unsqueeze(0)), y)
    g_t = FU.flat_grad([g.detach() for g in torch.autograd.grad(loss_t, params)])

    out = np.empty(len(x_poison_norm), dtype=np.float64)
    for i in range(len(x_poison_norm)):
        loss_p = crit(net(x_poison_norm[i].unsqueeze(0)), y)
        g_p = FU.flat_grad([g.detach() for g in torch.autograd.grad(loss_p, params)])
        out[i] = float(FU.cosine(g_p, g_t))
    net.train(was_training)
    return out


def train_with_snapshots(args, ctx, net, snap_epochs, capture):
    """final_update.train_from_scratch with a snapshot hook on the optimizer.

    The hook only counts SGD steps and reads the network; it writes nothing back.
    Forward/backward passes in eval() consume no RNG and do not update BatchNorm
    statistics, so the training trajectory is the unmodified one.
    """
    n = ctx['train_imgs'].shape[0]
    steps_per_epoch = math.ceil(n / args.victim_bs)
    wanted = {int(e) * steps_per_epoch: int(e) for e in snap_epochs if e > 0}
    seen, fired = {'n': 0}, set()
    orig_step = torch.optim.SGD.step

    def patched(self, *a, **kw):
        out = orig_step(self, *a, **kw)
        seen['n'] += 1
        ep = wanted.get(seen['n'])
        if ep is not None:
            was = net.training
            capture(ep)
            net.train(was)
            fired.add(ep)
        return out

    torch.optim.SGD.step = patched
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
        raise RuntimeError('snapshots for epochs %s never fired (%d steps seen, '
                           '%d per epoch)' % (sorted(missing), seen['n'],
                                              steps_per_epoch))


def project(h_tclass, h_aclass, others):
    """Coordinates in the (u, PC1 of the orthogonal clean residual) frame."""
    mu_t, mu_a = h_tclass.mean(dim=0), h_aclass.mean(dim=0)
    diff = mu_a - mu_t
    scale = float(diff.norm())
    if scale <= 0:
        raise RuntimeError('the two class centroids coincide')
    u = diff / scale
    resid = torch.cat([h_tclass, h_aclass]) - mu_t
    resid = resid - torch.outer(resid @ u, u)
    resid = resid - resid.mean(dim=0, keepdim=True)
    _, _, V = torch.linalg.svd(resid.double(), full_matrices=False)
    v = V[0].float()

    def coords(H):
        d = H - mu_t
        return torch.stack([(d @ u) / scale, (d @ v) / scale], dim=1)

    out = {'target_class': coords(h_tclass), 'adv_class': coords(h_aclass)}
    for k, H in others.items():
        out[k] = coords(H)
    if 'target' in out and float(out['target'][0, 1]) < 0:   # deterministic sign
        for k in out:
            out[k][:, 1] *= -1.0
    return out


def compute(args):
    device = C.pick_device(args)
    ctx = C.build_ctx(args, device)
    y_adv, target_class = C.classes_of(args, ctx)
    tidx = int(args.target_index)
    if int(ctx['test_labs'][tidx]) != target_class:
        raise SystemExit('test image %d is not of the target class' % tidx)
    x_t = ctx['test_imgs'][tidx]

    epochs = args.victim_epochs
    snap = sorted({int(round(f * epochs)) for f in args.checkpoint_fractions})
    print('  victim: %d epochs, decay %s -> snapshots at %s'
          % (epochs, args.victim_decay, snap))

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
        print('  %s: %d poisons from %s' % (selection, len(base_idx),
                                            os.path.basename(run_dir)))
        poisoned = {int(i) for i in base_idx.tolist()}
        idx_t, idx_a = sample(target_class, poisoned), sample(y_adv, poisoned)
        x_p_norm = ctx['norm'](x_adv01)

        seed_v = args.seed * 100000 + tidx * 100 + args.victim_id
        net = FU.build_network(args.model, ctx['channel'], ctx['num_classes'],
                               ctx['im_size'], device, seed=seed_v)
        snaps = {}

        def capture(ep, _net=net, _it=idx_t, _ia=idx_a, _xp=x_p_norm, _s=snaps):
            _s[int(ep)] = dict(
                target_class=embed_batched(_net, ctx['train_imgs'][_it]).cpu(),
                adv_class=embed_batched(_net, ctx['train_imgs'][_ia]).cpu(),
                poison=embed_batched(_net, _xp).cpu(),
                target=embed_batched(_net, x_t.unsqueeze(0)).cpu(),
                align=gradient_alignment(_net, _xp, x_t, y_adv, device))

        clean_rows = ctx['train_imgs'][base_idx].clone()
        ctx['train_imgs'][base_idx] = x_p_norm             # inject the poisons
        try:
            capture(0)                                     # 0% = initialization
            train_with_snapshots(args, ctx, net, snap, capture)
        finally:
            ctx['train_imgs'][base_idx] = clean_rows       # always restore

        for ep in snap:
            s = snaps[ep]
            co = project(s['target_class'].to(device), s['adv_class'].to(device),
                         {'poison': s['poison'].to(device),
                          'target': s['target'].to(device)})
            for k, v in co.items():
                store['%s__%d__%s' % (selection, ep, k)] = v.cpu().numpy()
            store['%s__%d__align' % (selection, ep)] = s['align']
            print('    %-6s epoch %2d: %.0f%% of poisons gradient-aligned with the '
                  'target' % (selection, ep, 100 * float((s['align'] > 0).mean())))
        del net
        if str(device).startswith('cuda'):
            torch.cuda.empty_cache()

    os.makedirs(args.out, exist_ok=True)
    npz = os.path.join(args.out, STEM + '.npz')
    np.savez(npz, selections=np.array(SELECTIONS), epochs=np.array(snap),
             victim_epochs=np.array([epochs]), target_index=np.array([tidx]),
             victim_id=np.array([args.victim_id]), attack=np.array([args.attack]),
             budget=np.array([args.budget]), **store)
    print('  wrote %s' % npz)

    rows = []
    for selection in SELECTIONS:
        for ep in snap:
            al = store.get('%s__%d__align' % (selection, ep))
            for g in ('target_class', 'adv_class', 'poison', 'target'):
                arr = store['%s__%d__%s' % (selection, ep, g)]
                for i in range(arr.shape[0]):
                    rows.append(dict(selection=selection, epoch=int(ep), group=g,
                                     point=i, coord1=float(arr[i, 0]),
                                     coord2=float(arr[i, 1]),
                                     grad_alignment=(float(al[i]) if
                                                     (g == 'poison' and al is not None)
                                                     else ''),
                                     target_index=tidx, victim_id=args.victim_id))
    P.save_csv(os.path.join(args.out, STEM + '.csv'),
               ['selection', 'epoch', 'group', 'point', 'coord1', 'coord2',
                'grad_alignment', 'target_index', 'victim_id'], rows)
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
    has_align = ('%s__%d__align' % (SELECTIONS[0], eps[0])) in d.files

    allc = np.vstack([d['%s__%d__%s' % (s, e, g)] for s in SELECTIONS for e in eps
                      for g in ('target_class', 'adv_class', 'poison', 'target')])
    lim = [(np.percentile(allc[:, i], 0.5), np.percentile(allc[:, i], 99.5))
           for i in range(2)]
    pad = [0.10 * (hi - lo) for lo, hi in lim]

    P.set_paper_style()
    fig = plt.figure(figsize=(P.WIDTH_FULL, 3.55), constrained_layout=True)
    ncol = len(eps) + (1 if has_align else 0)
    widths = [1.0] * len(eps) + ([0.92] if has_align else [])
    gs = fig.add_gridspec(2, ncol, width_ratios=widths, wspace=0.08, hspace=0.10)
    frac_eff = {s: [] for s in SELECTIONS}

    for r, selection in enumerate(SELECTIONS):
        for c, ep in enumerate(eps):
            ax = fig.add_subplot(gs[r, c])
            g = lambda k: d['%s__%d__%s' % (selection, ep, k)]
            ax.scatter(g('target_class')[:, 0], g('target_class')[:, 1], s=3.2,
                       c=COL['tclass'], alpha=0.45, linewidths=0, zorder=1)
            ax.scatter(g('adv_class')[:, 0], g('adv_class')[:, 1], s=3.2,
                       c=COL['aclass'], alpha=0.45, linewidths=0, zorder=1)
            pz = g('poison')
            if has_align:
                a = d['%s__%d__align' % (selection, ep)]
                eff = a > args.align_threshold
                frac_eff[selection].append(float(eff.mean()))
                # small and translucent: 250 poisons converge into a tight
                # cluster by the later checkpoints, and large opaque markers
                # there paint over each other AND over the clean clouds
                ax.scatter(pz[~eff, 0], pz[~eff, 1], s=2.6, marker='o',
                           facecolor=COL['ineff'], edgecolor='none', alpha=0.40,
                           zorder=2)
                ax.scatter(pz[eff, 0], pz[eff, 1], s=3.4, marker='s',
                           facecolor=COL['eff'], edgecolor='none', alpha=0.45,
                           zorder=3)
                P.stats_box(ax, '%.0f%% effective' % (100 * eff.mean()),
                            loc='lower right', fontsize=P.FS['note'] - 0.4,
                            color='black')
            else:
                ax.scatter(pz[:, 0], pz[:, 1], s=2.8, marker='o',
                           facecolor=COL['eff'], edgecolor='none', alpha=0.40,
                           zorder=2)
            t = g('target')
            ax.scatter(t[:, 0], t[:, 1], s=110, marker='*', facecolor=COL['target'],
                       edgecolor='white', linewidth=0.6, zorder=4)
            ax.set_xlim(lim[0][0] - pad[0], lim[0][1] + pad[0])
            ax.set_ylim(lim[1][0] - pad[1], lim[1][1] + pad[1])
            ax.set_xticks([])
            ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_visible(True)
                sp.set_linewidth(0.5)
                sp.set_color('#AAAAAA')
            if r == 0:
                ax.set_title('epoch %d  (%.0f%%)' % (ep, 100.0 * ep / total),
                             fontsize=P.FS['note'], pad=3)
            if c == 0:
                ax.set_ylabel(P.LABELS[selection], fontsize=P.FS['label'],
                              color=P.COLORS[selection])
                if r == 0:
                    P.panel_label(ax, '(a)', dx=-0.30, dy=1.02)
                else:
                    P.panel_label(ax, '(b)', dx=-0.30, dy=1.02)

    if has_align:
        # the quantitative counterpart of the two rows: how much of the poison
        # budget is actually pulling the target, at each checkpoint
        ax_s = fig.add_subplot(gs[:, -1])
        for sname in SELECTIONS:
            ax_s.plot(eps, [100 * f for f in frac_eff[sname]],
                      color=P.COLORS[sname], marker=P.MARKERS[sname],
                      markersize=3.4, linewidth=1.3, markeredgecolor='white',
                      markeredgewidth=0.3, label=P.LABELS[sname])
        ax_s.set_xlabel('epoch')
        ax_s.set_ylabel('effective poisons (%)')
        ax_s.set_ylim(0, 104)
        ax_s.legend(loc='lower right', fontsize=P.FS['legend'])
        P.light_grid(ax_s, axis='both')
        P.panel_label(ax_s, '(c)', dx=-0.34, dy=1.02)

    handles = [
        Line2D([0], [0], marker='*', linestyle='none', markersize=8,
               color=COL['target'], label='target $x_t$'),
        Line2D([0], [0], marker='o', linestyle='none', markersize=3.4,
               color=COL['tclass'], label='target class (clean)'),
        Line2D([0], [0], marker='o', linestyle='none', markersize=3.4,
               color=COL['aclass'], label='poison class (clean)'),
        Line2D([0], [0], marker='s', linestyle='none', markersize=3.6,
               color=COL['eff'], label='effective poison'),
        Line2D([0], [0], marker='o', linestyle='none', markersize=3.4,
               color=COL['ineff'], label='ineffective poison'),
    ]
    if not has_align:
        handles = handles[:3] + [Line2D([0], [0], marker='o', linestyle='none',
                                        markersize=3.4, color=COL['eff'],
                                        label='poison')]
    P.bottom_legend(fig, handles, ncol=len(handles))

    if has_align:
        for s in SELECTIONS:
            print('  %-6s fraction of effective poisons over training: %s'
                  % (s, ', '.join('%.0f%%' % (100 * f) for f in frac_eff[s])))

    P.save_fig(fig, args.out, STEM)
    plt.close(fig)


# --------------------------------------------------------------------------- #

def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    C.add_repo_args(p)
    C.add_mode_args(p)
    p.add_argument('--target_index', type=int, required=True)
    p.add_argument('--victim_id', type=int, default=0)
    p.add_argument('--attack', default='gradmatch', choices=['gradmatch', 'sapa', 'fc'])
    p.add_argument('--budget', type=float, default=C.PAPER['budget'])
    p.add_argument('--epsilon', type=float, default=C.PAPER['epsilon'])
    p.add_argument('--craft_ensemble', type=int, default=C.PAPER['craft_ensemble'])
    p.add_argument('--craft_aug', action='store_true', default=True)
    p.add_argument('--no_craft_aug', dest='craft_aug', action='store_false')
    p.add_argument('--dsa_strategy', default=C.PAPER['dsa_strategy'])
    p.add_argument('--sharp_mode', default=C.PAPER['sharp_mode'], choices=['worst', 'avg'])
    p.add_argument('--sharp_sigma', type=float, default=C.PAPER['sharp_sigma'])
    p.add_argument('--sharp_samples', type=int, default=20)
    p.add_argument('--target_select', default='random',
                   help="naming only: the attack run's --target_select, so the run "
                        'directory is found')
    g = p.add_argument_group('victim protocol (defaults = the paper)')
    g.add_argument('--victim_epochs', type=int, default=C.PAPER['victim_epochs'])
    g.add_argument('--victim_lr', type=float, default=C.PAPER['victim_lr'])
    g.add_argument('--victim_bs', type=int, default=C.PAPER['victim_bs'])
    g.add_argument('--victim_decay', nargs='*', type=int, default=C.PAPER['victim_decay'])
    g.add_argument('--victim_wd', type=float, default=C.PAPER['victim_wd'])
    g.add_argument('--victim_aug', action='store_true', default=False)
    p.add_argument('--checkpoint_fractions', type=float, nargs='+',
                   default=[0.0, 0.15, 0.40, 1.0])
    p.add_argument('--num_clean', type=int, default=400)
    p.add_argument('--align_threshold', type=float, default=0.0,
                   help='a poison counts as effective when its gradient cosine '
                        'with the target gradient exceeds this')
    return p.parse_args()


def main():
    args = parse_args()
    do_compute, do_plot = C.resolve_mode(args)
    epochs = args.victim_epochs
    C.summarize('Poison dynamics in the victim representation space', [
        ('dataset / model', '%s / %s' % (args.dataset, args.model)),
        ('class pair', '%s (%s)' % (args.class_pair, args.pair_order)),
        ('target / victim', '%d / %d' % (args.target_index, args.victim_id)),
        ('attack replayed', '%s, budget %g' % (args.attack, args.budget)),
        ('rows', ' vs '.join(P.LABELS[s] for s in SELECTIONS)),
        ('snapshots at epochs', sorted({int(round(f * epochs))
                                        for f in args.checkpoint_fractions})),
        ('effective poison', 'cos(grad CE(x_p, y_adv), grad CE(x_t, y_adv)) > %g'
         % args.align_threshold),
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
