#!/usr/bin/env python
"""
fig_exp_optimization_trajectories.py -- Figure 3, "how base initialization
changes the poison optimization".

No attack success rate anywhere: this figure asks whether a better base
initialization makes the DOWNSTREAM poison optimization problem easier, and it
plots the attacks' own objectives, not ASR.

Instrumentation, not re-implementation
--------------------------------------
The attacks are final_update.craft_gradmatch (Witches' Brew) and the same
function with sharp_mode='worst' (SAPA, He et al. ICLR 2024) -- one code path,
called exactly the way final_update.prepare_poisons calls it. The per-iteration
objective is captured by temporarily wrapping the module-level helper
final_update.cosine, which the crafting objective goes through:

    GM / SAPA objective at iteration k :   J(k) = mean_nets [ 1 - cos(g_p, g_t) ]

The wrapper RETURNS THE ORIGINAL TENSOR and only records a detached copy, so no
value, no gradient, no random draw and no optimizer step changes. As a check, the
minimum recorded J is compared against the best objective craft_gradmatch itself
returns, and the number of recorded calls is compared against
restarts * craft_steps * len(craft_nets); either mismatch aborts.

GM and SAPA objectives are different objectives and are never compared to each
other; each run is normalized within itself,

    J_norm(k) = J(k) / J(0),        lower is better for both.

Controls
--------
* the same five pinned targets for every selection and both attacks;
* the base set is fixed before optimization (Random drawn once with the run's own
  per-target generator, seed * 100003 + target_idx, exactly as prepare_poisons
  does -- never resampled);
* identical attack hyper-parameters for all three selections;
* final_update.set_seed(seed * 100003 + target_idx) immediately before every
  craft, so the delta initialisation and the DiffAugment draws are matched across
  Random / Greedy / DPP.

Nothing is written into ours_result/: this script never touches a poison_cache.

Cost
----
5 targets x 3 selections x 2 attacks = 30 crafts at the paper's 8 restarts x 250
steps. Use --targets / --restarts to shrink it, but note in the caption if you do.
"""

import argparse
import os
import time

import numpy as np
import torch

import matplotlib.pyplot as plt

import common as C
from common import FU


STEM = 'fig_exp_optimization_trajectories'
SELECTIONS = ['random', 'greedy', 'dpp']


# --------------------------------------------------------------------------- #
# objective recorder
# --------------------------------------------------------------------------- #

class ObjectiveRecorder:
    """Records final_update.cosine's value without changing it.

    craft_gradmatch computes its objective as 1 - cosine(g_p, g_t), once per
    surrogate per iteration, in every code path (default, --fast_gradmatch and
    --craft_lowmem). Wrapping the module-level name therefore observes exactly
    the objective the implementation optimizes.
    """

    def __init__(self, num_nets):
        self.num_nets = int(num_nets)
        self._vals = []
        self._orig = None

    def __enter__(self):
        self._orig = FU.cosine

        def wrapped(a, b, eps=1e-8):
            out = self._orig(a, b, eps)
            self._vals.append(out.detach())      # detached: no graph is retained
            return out                           # the caller gets the real tensor

        FU.cosine = wrapped
        return self

    def __exit__(self, *exc):
        FU.cosine = self._orig
        return False

    def objective(self, expect_iters):
        """J(k) = mean over surrogates of (1 - cos), one value per iteration."""
        if not self._vals:
            raise RuntimeError('no objective values recorded -- did final_update '
                               'change how craft_gradmatch computes its objective?')
        v = torch.stack(self._vals).detach().float().cpu().numpy()
        n = self.num_nets
        if len(v) != expect_iters * n:
            raise RuntimeError(
                'recorded %d cosine calls, expected %d = %d iterations x %d '
                'surrogates. The instrumentation no longer matches '
                'final_update.craft_gradmatch -- fix it rather than plotting it.'
                % (len(v), expect_iters * n, expect_iters, n))
        return (1.0 - v.reshape(expect_iters, n)).mean(axis=1)


# --------------------------------------------------------------------------- #
# extraction
# --------------------------------------------------------------------------- #

def craft_one(args, ctx, craft_nets, base_idx, x_t, y_adv, attack, tseed):
    """One craft, instrumented. Identical call to final_update.prepare_poisons."""
    device = ctx['device']
    base01 = ctx['denorm'](ctx['train_imgs'][base_idx]).clamp(0.0, 1.0).detach()

    # matched optimizer seeds: delta init + DiffAugment draws identical across
    # the three selections of this target (prepare_poisons seeds the same way)
    FU.set_seed(tseed)

    total_iters = args.restarts * args.craft_steps
    rec = ObjectiveRecorder(len(craft_nets))
    t0 = time.time()
    with rec:
        _x_adv01, best_obj = FU.craft_gradmatch(
            craft_nets, base01, x_t, y_adv, ctx['norm'], args.epsilon,
            args.craft_alpha, args.craft_steps, args.restarts, device,
            dsa_strategy=(args.dsa_strategy if args.craft_aug else None),
            dsa_param=ctx['dsa_param'], fast=args.fast_gradmatch,
            schedule=args.craft_schedule, lowmem=args.craft_lowmem,
            chunk=args.craft_batch,
            sharp_mode=(args.sharp_mode if attack == 'sapa' else None),
            sharp_sigma=args.sharp_sigma, sharp_samples=args.sharp_samples)
    J = rec.objective(total_iters)

    # the recording must reproduce the value the implementation reports
    if not np.isfinite(best_obj) or abs(float(J.min()) - float(best_obj)) > 1e-4:
        raise RuntimeError('recorded min objective %.6f != craft_gradmatch\'s '
                           'best_obj %.6f' % (float(J.min()), float(best_obj)))
    print('      %-9s %d iters in %.0f s, J(0)=%.5f -> best %.5f'
          % (attack, total_iters, time.time() - t0, J[0], best_obj))
    return J


def compute(args):
    device = C.pick_device(args)
    ctx = C.build_ctx(args, device)
    y_adv, target_class = C.classes_of(args, ctx)
    targets = C.resolve_targets(args, ctx, target_class)
    m = C.num_poisons(args, ctx)

    sel_nets, _ = C.load_selector_surrogates(args, ctx)
    craft_nets = sel_nets[:args.craft_ensemble] if args.craft_ensemble else sel_nets
    print('  m=%d poisons, K=%d selector surrogates, %d crafting surrogates'
          % (m, len(sel_nets), len(craft_nets)))

    store, rows = {}, []
    for t in targets:
        x_t = ctx['test_imgs'][t]
        tseed = args.seed * 100003 + int(t)

        # ---- base sets, fixed once per target, before any optimization -------
        FU.set_seed(tseed)
        bases = {
            'random': C.select_random(ctx, y_adv, m, args.seed, t),
            'greedy': C.select_greedy(sel_nets, ctx, x_t, y_adv, m, args),
            'dpp': C.select_dpp(sel_nets, ctx, x_t, y_adv, m, args),
        }
        print('  target %d: base sets fixed (%s)'
              % (t, ', '.join('%s=%d' % (k, len(v)) for k, v in bases.items())))

        for attack in args.attacks:
            for sname in SELECTIONS:
                print('    target %d | %s | %s' % (t, attack, sname))
                J = craft_one(args, ctx, craft_nets, bases[sname], x_t, y_adv,
                              attack, tseed)
                key = '%s__%s__%d' % (attack, sname, t)
                store['J__' + key] = J
                store['base__' + key] = bases[sname].cpu().numpy()

                best = np.minimum.accumulate(J)
                keep = set(range(0, len(J), args.log_every)) | {0, len(J) - 1}
                for k in sorted(keep):
                    rows.append(dict(
                        target_index=int(t), attack=attack, selection=sname,
                        restart=int(k // args.craft_steps),
                        step=int(k % args.craft_steps), iteration=int(k),
                        objective=float(J[k]),
                        normalized_objective=float(J[k] / J[0]),
                        best_objective=float(best[k]),
                        normalized_best_objective=float(best[k] / J[0])))

    os.makedirs(args.out, exist_ok=True)
    npz = os.path.join(args.out, STEM + '.npz')
    np.savez(npz, targets=np.array(targets),
             attacks=np.array(args.attacks), selections=np.array(SELECTIONS),
             craft_steps=np.array([args.craft_steps]),
             restarts=np.array([args.restarts]), m=np.array([m]), **store)
    print('  wrote %s' % npz)

    C.save_csv(os.path.join(args.out, STEM + '.csv'),
               ['target_index', 'attack', 'selection', 'restart', 'step',
                'iteration', 'objective', 'normalized_objective',
                'best_objective', 'normalized_best_objective'], rows)
    return npz


# --------------------------------------------------------------------------- #
# plotting
# --------------------------------------------------------------------------- #

# short: at 2.2in per panel the full names collide with the panel labels
PRETTY_ATTACK = {'gradmatch': 'GM', 'sapa': 'SAPA'}


def plot(args):
    npz = os.path.join(args.out, STEM + '.npz')
    if not os.path.exists(npz):
        raise SystemExit('%s missing -- run with --compute first.' % npz)
    d = np.load(npz, allow_pickle=True)
    targets = [int(t) for t in d['targets']]
    attacks = [str(a) for a in d['attacks']]
    steps = int(d['craft_steps'][0])
    restarts = int(d['restarts'][0])

    def curves(attack, sname):
        out = []
        for t in targets:
            J = d['J__%s__%s__%d' % (attack, sname, t)]
            y = np.minimum.accumulate(J) if args.curve == 'best' else J
            out.append(y / J[0])          # J_norm(k) = J(k) / J(0); lower is better
        return np.vstack(out)

    C.set_paper_style()
    ncol = len(attacks)
    fig = plt.figure(figsize=(C.WIDTH_FULL, 2.55), constrained_layout=True)
    gs = fig.add_gridspec(1, 2 * ncol,
                          width_ratios=sum([[1.0, 0.19] for _ in attacks], []),
                          wspace=0.06)

    rng = np.random.default_rng(args.viz_seed)
    for ci, attack in enumerate(attacks):
        ax = fig.add_subplot(gs[0, 2 * ci])
        axe = fig.add_subplot(gs[0, 2 * ci + 1], sharey=ax)

        for r in range(1, restarts):      # restart boundaries, very light
            ax.axvline(r * steps, color='#DDDDDD', linewidth=0.5, zorder=0)

        for si, sname in enumerate(SELECTIONS):
            Y = curves(attack, sname)
            k = np.arange(Y.shape[1])
            sub = np.unique(np.concatenate([np.arange(0, len(k), args.log_every),
                                            [len(k) - 1]]))
            mu = Y.mean(axis=0)
            se = (Y.std(axis=0, ddof=1) / np.sqrt(Y.shape[0])) if Y.shape[0] > 1 \
                else np.zeros_like(mu)
            ax.plot(k[sub], mu[sub], color=C.COLORS[sname], linewidth=1.2,
                    label=C.LABELS[sname], zorder=3)
            # band = standard error ACROSS TARGETS (the statistical unit), never
            # across optimizer iterations
            ax.fill_between(k[sub], (mu - se)[sub], (mu + se)[sub],
                            color=C.COLORS[sname], alpha=0.16, linewidth=0, zorder=2)

            jit = (rng.random(Y.shape[0]) - 0.5) * 0.26
            axe.scatter(si + jit, Y[:, -1], s=8, marker=C.MARKERS[sname],
                        facecolor=C.COLORS[sname], edgecolor='white',
                        linewidth=0.3, alpha=0.9, zorder=3)
            axe.plot([si - 0.28, si + 0.28], [Y[:, -1].mean()] * 2,
                     color=C.COLORS[sname], linewidth=1.1, zorder=4)

        ax.set_xlabel('poison optimization iteration')
        if ci == 0:
            ax.set_ylabel('normalized objective $J(k)/J(0)$\n(lower = better $\\downarrow$)')
            ax.legend(loc='upper right', fontsize=C.FS['legend'])
        ax.set_title(PRETTY_ATTACK.get(attack, attack), fontsize=C.FS['title'], pad=3)
        ax.set_xlim(0, restarts * steps)
        C.light_grid(ax)
        C.panel_label(ax, '(%s)' % 'abcdef'[ci], dx=-0.26 if ci == 0 else -0.16, dy=1.06)

        axe.set_xlim(-0.6, len(SELECTIONS) - 0.4)
        axe.set_xticks(range(len(SELECTIONS)))
        axe.set_xticklabels(['R', 'G', 'D'], fontsize=C.FS['note'])
        axe.tick_params(axis='y', labelleft=False, left=False)
        axe.spines['left'].set_visible(False)
        axe.set_title('final', fontsize=C.FS['note'], pad=3, color=C.COLORS['rule'])
        C.light_grid(axe)

    C.stats_box(fig.axes[0], ['%d targets' % len(targets),
                              'band = s.e.m.'], loc='lower left')

    C.save_fig(fig, args.out, STEM)
    plt.close(fig)


# --------------------------------------------------------------------------- #

def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    C.add_repo_args(p)
    C.add_mode_args(p)
    p.add_argument('--budget', type=float, default=C.PAPER['budget'],
                   help="the appendix protocol's poison budget 5e-3 -> m = 250")
    p.add_argument('--target_idx_file', default=C.PAPER['appendix_target_file'],
                   help='the five pinned bird->dog targets of the reduced '
                        'appendix protocol')
    p.add_argument('--target_indices', type=int, nargs='*', default=None,
                   help='explicit target list; overrides --target_idx_file')
    p.add_argument('--attacks', nargs='+', default=['gradmatch', 'sapa'],
                   choices=['gradmatch', 'sapa'])

    g = p.add_argument_group('attack hyper-parameters (defaults = the paper; '
                             'identical for every selection)')
    g.add_argument('--epsilon', type=float, default=C.PAPER['epsilon'],
                   help='Linf radius, 8/255')
    g.add_argument('--craft_steps', type=int, default=C.PAPER['craft_steps'])
    g.add_argument('--craft_alpha', type=float, default=C.PAPER['craft_alpha'])
    g.add_argument('--restarts', type=int, default=C.PAPER['restarts'])
    g.add_argument('--craft_ensemble', type=int, default=C.PAPER['craft_ensemble'])
    g.add_argument('--craft_aug', action='store_true', default=True)
    g.add_argument('--no_craft_aug', dest='craft_aug', action='store_false')
    g.add_argument('--dsa_strategy', default=C.PAPER['dsa_strategy'])
    g.add_argument('--craft_schedule', action='store_true', default=False)
    g.add_argument('--fast_gradmatch', action='store_true', default=False)
    g.add_argument('--craft_lowmem', action='store_true', default=False)
    g.add_argument('--craft_batch', type=int, default=256)
    g.add_argument('--sharp_mode', default=C.PAPER['sharp_mode'],
                   choices=['worst', 'avg'])
    g.add_argument('--sharp_sigma', type=float, default=C.PAPER['sharp_sigma'])
    g.add_argument('--sharp_samples', type=int, default=20)

    p.add_argument('--log_every', type=int, default=10,
                   help='thinning of the saved / plotted trajectory. Every '
                        'iteration is recorded; this only controls how many are '
                        'written to the csv and drawn.')
    p.add_argument('--curve', default='best', choices=['best', 'raw'],
                   help="'best' plots the best-so-far objective, which is what "
                        'craft_gradmatch actually keeps; \'raw\' plots the '
                        'per-iteration value.')
    return p.parse_args()


def main():
    args = parse_args()
    do_compute, do_plot = C.resolve_mode(args)
    if args.restarts != C.PAPER['restarts'] or args.craft_steps != C.PAPER['craft_steps']:
        print('!! attack schedule differs from the paper (%d restarts x %d steps '
              'vs %d x %d) -- say so in the caption'
              % (args.restarts, args.craft_steps, C.PAPER['restarts'],
                 C.PAPER['craft_steps']))
    n_targets = (len(args.target_indices) if args.target_indices
                 else len(C.load_pinned_targets(args.target_idx_file, args.class_pair)))
    C.summarize('Figure 3 -- poison-optimization trajectories (no ASR)', [
        ('dataset / model', '%s / %s' % (args.dataset, args.model)),
        ('class pair', '%s (%s)' % (args.class_pair, args.pair_order)),
        ('targets', args.target_indices or args.target_idx_file),
        ('attacks', ' '.join(args.attacks)),
        ('selections', ' '.join(SELECTIONS)),
        ('budget / epsilon', '%g / %g (%.2f/255)'
         % (args.budget, args.epsilon, args.epsilon * 255)),
        ('schedule', '%d restarts x %d steps, signed-Adam lr %g'
         % (args.restarts, args.craft_steps, args.craft_alpha)),
        ('crafting ensemble', args.craft_ensemble or 'all'),
        ('crafts to run', '%d targets x %d selections x %d attacks'
         % (n_targets, len(SELECTIONS), len(args.attacks))),
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
