#!/usr/bin/env python
"""
plot_random_vs_dpp_bases.py -- qualitative figure: which CLEAN bases the two
selections pick for the SAME target, plus an automatic, ASR-free search for a
representative target to show.

The figure is one target on the left and two aligned rows of clean poison-class
images:

    Random       the uniform baseline  (--base random)
    DPP (ours)   greedy log-det MAP    (--base ours --sel_dpp --sel_alpha ALPHA)

Nothing is crafted, optimized or re-run: these are the clean bases BEFORE the
perturbation, so the figure is about selection only. Both rows come from the
same combo -- same model, attack, class pair, dataset split, budget (hence the
same N_p), seed and the same target -- so the only difference is how the bases
were chosen.

Where the indices come from
---------------------------
final_update.py caches the exact base set it used, per run directory:

    <out_dir>/<run_name>/poison_cache/base_<target_idx>.json   (train indices)

run_name comes from final_update.build_run_name, which is imported rather than
re-implemented, so the two directories opened here are the ones sel_dpp.sh wrote.
DPP indices are ALWAYS read from that cache (reproducing them needs the surrogate
ensemble). Random indices are read from the cache when the run kept one, and
otherwise reconstructed by calling final_update.select_base_random with the same
per-target generator prepare_poisons uses, manual_seed(seed*100003 + target_idx),
which reproduces the saved list exactly.

Choosing the target (--auto-target), without touching ASR
---------------------------------------------------------
The example must not be cherry-picked on attack outcome, so no victim result,
success flag or results.csv is read anywhere in this file. Every quantity below
is available at BASE-SELECTION time, computed with the surrogate ensemble in the
same representation space the selector itself uses (final_update's embed_of,
per-surrogate features L2-normalised so an ensemble cosine equals the mean of the
per-surrogate cosines -- exactly what _ours_score_and_feats builds):

    relevance  R_i   mean over surrogates of cos(f_i, f_t)        higher = closer to the target
    redundancy       mean pairwise cosine inside the selected set  lower  = less redundant
    boundary   B_i   mean over surrogates of the standardized
                     logit margin toward y_adv                     lower  = nearer the boundary
    score            the selector's own standardized
                     d(x) + lambda * M(x)                          lower  = what 'ours' prefers

For every target that has a cached DPP set, both selections are scored and the
targets are ranked by a deliberately simple criterion:

    relevance_gain = mean_R_dpp  - mean_R_random
    diversity_gain = mean_sim_random - mean_sim_dpp
    rank_score     = relevance_gain / std(relevance_gain)
                   + diversity_gain / std(diversity_gain)

(each gain is divided by its own spread across the evaluated targets, so the two
terms are comparable; nothing else is weighted or tuned). Two guards keep the
winner honest, and a target failing either is reported but not auto-picked:

    boundary    DPP must not buy its diversity with worse candidates:
                mean_B_dpp <= mean_B_random + --boundary_tol
    consistency the effect must be an ensemble property, not one network:
                on at least --consistency_frac of the surrogates, DPP must have
                BOTH higher relevance AND lower redundancy than random

Which bases are drawn
---------------------
Not the first 8. Both rows go through the SAME representative-subset rule, so
the two rows stay comparable: keep the --display_pool_frac most relevant of the
selected set, then greedily take farthest-point (minimum max-cosine) picks in the
ensemble space, so the drawn images span the spread of the set they come from
while staying target-relevant. The selected sets themselves are never modified --
the drawn images are a sample of N_p bases, and the caption says so.

Surrogates are only ever LOADED from --cache_dir; a missing checkpoint is an
error, never a training run. --fast skips them entirely (first m bases, no stats).

Examples
--------
    python "visuals outputs/plot_random_vs_dpp_bases.py" --auto-target
    python "visuals outputs/plot_random_vs_dpp_bases.py" --auto-target --top-targets 10
    python "visuals outputs/plot_random_vs_dpp_bases.py" --target-id 5118 --num-display 8
    python "visuals outputs/plot_random_vs_dpp_bases.py" \
        --model VGG13BN --attack gradmatch --class_pair frog-airplane --budget 0.04 \
        --auto-target
    python "visuals outputs/plot_random_vs_dpp_bases.py" --list_targets
"""

import argparse
import json
import os
import sys
from argparse import Namespace

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

_HERE = os.path.dirname(os.path.abspath(__file__))

PRETTY_MODEL = {'ConvNetBN': 'ConvNet', 'VGG13BN': 'VGG-13', 'ResNet20BN': 'ResNet-20'}
PRETTY_ATTACK = {'fc': 'feature collision', 'gradmatch': 'gradient matching',
                 'sapa': 'sharpness-aware poisoning'}
ROW_LABELS = ('Random', 'DPP (ours)')


# --------------------------------------------------------------------------- #
# repo lookups -- run names, target sets and difficulty labels, exactly as the
# experiment scripts define them
# --------------------------------------------------------------------------- #

def _run_name_args(a, base, sel_dpp):
    """The subset of final_update's args that build_run_name reads."""
    return Namespace(
        dataset=a.dataset, model=a.model, attack=a.attack, base=base,
        class_pair=a.class_pair, budget=a.budget, epsilon=a.epsilon, seed=a.seed,
        lambda_margin=a.lambda_margin, base_dist=a.base_dist,
        sel_filter=False, sel_pca=False, sel_mmr=False, sel_dpp=sel_dpp,
        sel_pool=a.sel_pool, sel_mu=a.sel_mu, sel_alpha=a.sel_alpha,
        fc_mode=a.fc_mode, sharp_mode=a.sharp_mode, sharp_sigma=a.sharp_sigma,
        sharp_samples=a.sharp_samples, craft_ensemble=a.craft_ensemble,
        target_select=a.target_select)


def difficulty_key(attack):
    """sapa reads the gradmatch entry -- same difficulty, same target set."""
    return 'gradmatch' if attack == 'sapa' else attack


def difficulty_label(cfg_path, model, attack, pair):
    with open(cfg_path) as f:
        cfg = json.load(f)
    key = difficulty_key(attack)
    try:
        return int(cfg['difficulty'][model][key][pair])
    except KeyError:
        raise SystemExit('%s has no difficulty for %s / %s / %s'
                         % (cfg_path, model, key, pair))


def target_set_path(root, model, attack, pair):
    """target_sets/<MODEL>_<ATTACK>_<PAIR>.json, sapa falling back to gradmatch."""
    p = os.path.join(root, '%s_%s_%s.json' % (model, attack, pair))
    if not os.path.exists(p):
        key = difficulty_key(attack)
        if key != attack:
            alt = os.path.join(root, '%s_%s_%s.json' % (model, key, pair))
            if os.path.exists(alt):
                return alt
    return p


def load_pinned_targets(path, pair):
    """The pinned test indices, read the way select_targets reads them."""
    if not os.path.exists(path):
        return []
    with open(path) as f:
        blob = json.load(f)
    want = blob['pairs'][pair]['indices'] if 'pairs' in blob else blob[pair]
    return [int(i) for i in want]


def cached_base(run_dir, tidx):
    """The base index list final_update saved for this target, or None."""
    p = os.path.join(run_dir, 'poison_cache', 'base_%d.json' % tidx)
    if not os.path.exists(p):
        return None
    with open(p) as f:
        return [int(i) for i in json.load(f)]


def cached_target_ids(run_dir):
    d = os.path.join(run_dir, 'poison_cache')
    if not os.path.isdir(d):
        return []
    ids = []
    for f in os.listdir(d):
        if f.startswith('base_') and f.endswith('.json'):
            ids.append(int(f[len('base_'):-len('.json')]))
    return sorted(ids)


def random_bases(FU, run_dir, tidx, labels, y_adv, N_p, seed):
    """(indices, provenance). Cached if the run kept one, else reconstructed."""
    idx = cached_base(run_dir, tidx)
    if idx is not None:
        return idx, 'poison_cache'
    # exactly prepare_poisons(): per-target generator, then select_base_random
    gen = torch.Generator(device='cpu').manual_seed(seed * 100003 + int(tidx))
    idx = FU.select_base_random(labels, y_adv, N_p, 'cpu', gen).tolist()
    return idx, 'reconstructed (seed %d * 100003 + %d)' % (seed, tidx)


# --------------------------------------------------------------------------- #
# selection-time diagnostics, in the selector's own representation space
# --------------------------------------------------------------------------- #

def load_surrogates(FU, a, channel, num_classes, im_size, device):
    """Load the surrogate ensemble the selector used. Never trains anything."""
    d = os.path.join(a.cache_dir, 'surrogates',
                     '%s_%dep_lr%g_bs%d_seed%d'
                     % (a.model, a.surrogate_epochs, a.surrogate_lr,
                        a.surrogate_bs, a.seed))
    nets, missing = [], []
    for i in range(a.num_surrogates):
        p = os.path.join(d, 'net_%d.pt' % i)
        if not os.path.exists(p):
            missing.append(os.path.basename(p))
            continue
        net = FU.build_network(a.model, channel, num_classes, im_size, device,
                               seed=a.seed + 1000 + i)
        net.load_state_dict(torch.load(p, map_location=device))
        net.eval()
        nets.append(net)
    if missing:
        raise SystemExit(
            'surrogate checkpoint(s) missing from %s: %s\n'
            'This script only ever loads them -- it will not train. Point '
            '--cache_dir / --num_surrogates at the pool the run used, or pass '
            '--fast to skip the diagnostics.' % (d, ', '.join(missing)))
    return nets, d


@torch.no_grad()
def candidate_bank(FU, nets, cand, y_adv, bs):
    """Per-surrogate candidate features and standardized margins.

    Target independent, so this runs once and every target then costs one
    embedding of x_t plus a matrix-vector product.
    """
    raw, unit, marg = [], [], []
    for net in nets:
        emb = FU.embed_of(net)
        fs, ms = [], []
        for i in range(0, len(cand), bs):
            b = cand[i:i + bs]
            fb = emb(b).detach().flatten(1)
            z = net(b)
            z_adv = z[:, y_adv].clone()
            z_o = z.clone()
            z_o[:, y_adv] = float('-inf')
            ms.append(z_adv - z_o.max(dim=1).values)
            fs.append(fb)
        f = torch.cat(fs)
        raw.append(f)
        unit.append(F.normalize(f, dim=1))
        marg.append(FU.standardize(torch.cat(ms)))      # B_i, the boundary term
    return raw, unit, torch.stack(marg)


@torch.no_grad()
def target_terms(FU, nets, raw, unit, marg_std, x_t, lam, base_dist):
    """(relevance per surrogate [K, N], standardized selection score [N]).

    score is assembled with the same formula select_base_ours uses:
    mean_k [ standardize(d_k) + lam * standardize(margin_k) ].
    """
    rel, score = [], None
    for k, net in enumerate(nets):
        f_t = FU.embed_of(net)(x_t.unsqueeze(0)).detach().flatten(1)[0]
        cos = unit[k] @ F.normalize(f_t, dim=0)
        d = (1.0 - cos) if base_dist == 'cosine' else ((raw[k] - f_t) ** 2).sum(dim=1)
        term = FU.standardize(d) + lam * marg_std[k]
        score = term if score is None else score + term
        rel.append(cos)
    return torch.stack(rel), FU.standardize(score / len(nets))


@torch.no_grad()
def set_stats(pos, rel, unit, marg_std, score):
    """Diagnostics of one selected set, per surrogate and ensemble averaged.

    The ensemble redundancy is the mean of the per-surrogate cosines, which is
    exactly the cosine in the concatenated normalized space the selector's DPP
    kernel lives in.
    """
    n = len(pos)
    sim_k = []
    for k in range(len(unit)):
        S = unit[k][pos]
        G = S @ S.T
        sim_k.append(((G.sum() - G.diagonal().sum()) / max(n * (n - 1), 1)).item())
    sim_k = torch.tensor(sim_k)
    rel_k = rel[:, pos].mean(dim=1).cpu()
    bnd_k = marg_std[:, pos].mean(dim=1).cpu()
    return {'n': n,
            'relevance': float(rel_k.mean()), 'relevance_k': rel_k,
            'redundancy': float(sim_k.mean()), 'redundancy_k': sim_k,
            'boundary': float(bnd_k.mean()), 'boundary_k': bnd_k,
            'score': float(score[pos].mean())}


def compare(rand_stats, dpp_stats, boundary_tol, consistency_frac):
    """The two gains, the two guards, and why a target was rejected."""
    rel_gain = dpp_stats['relevance'] - rand_stats['relevance']
    div_gain = rand_stats['redundancy'] - dpp_stats['redundancy']
    agree = ((dpp_stats['relevance_k'] > rand_stats['relevance_k']) &
             (dpp_stats['redundancy_k'] < rand_stats['redundancy_k']))
    consistency = float(agree.float().mean())
    boundary_cost = dpp_stats['boundary'] - rand_stats['boundary']
    why = []
    if boundary_cost > boundary_tol:
        why.append('boundary cost %+.3f > %.3f' % (boundary_cost, boundary_tol))
    if consistency < consistency_frac:
        why.append('consistent on %.0f%% < %.0f%% of surrogates'
                   % (100 * consistency, 100 * consistency_frac))
    return {'relevance_gain': rel_gain, 'diversity_gain': div_gain,
            'boundary_cost': boundary_cost, 'consistency': consistency,
            'ok': not why, 'why': '; '.join(why)}


def rank_targets(rows):
    """rank_score = relevance_gain/std + diversity_gain/std. Nothing else.

    Each gain is divided by its own spread across the evaluated targets so the
    two terms are on a comparable scale before being added; the sign and the
    relative size within a term are untouched.
    """
    def scale(vals):
        t = torch.tensor(vals, dtype=torch.float64)
        s = float(t.std()) if len(t) > 1 else 0.0
        return [v / s for v in vals] if s > 1e-12 else list(vals)

    rel = scale([r['relevance_gain'] for r in rows])
    div = scale([r['diversity_gain'] for r in rows])
    for r, a_, b_ in zip(rows, rel, div):
        r['rank_score'] = a_ + b_
    return sorted(rows, key=lambda r: (r['ok'], r['rank_score']), reverse=True)


# --------------------------------------------------------------------------- #
# which of the selected bases to draw
# --------------------------------------------------------------------------- #

@torch.no_grad()
def representative(pos, rel_ens, unit, m, keep_frac):
    """A representative subset of an ALREADY selected set (display only).

    Keep the most target-relevant keep_frac, then farthest-point in the ensemble
    space, so the drawn images span the spread of the set without dropping to its
    least relevant members. Applied identically to Random and DPP.
    """
    order = sorted(pos, key=lambda p: -float(rel_ens[p]))
    k = min(len(order), max(m, int(round(keep_frac * len(order)))))
    pool = order[:k]
    m = min(m, len(pool))
    P = torch.cat([u[pool] for u in unit], dim=1) / (len(unit) ** 0.5)
    G = P @ P.T
    sel = [0]                                    # most relevant of the pool
    maxsim = G[0].clone()
    while len(sel) < m:
        maxsim[torch.tensor(sel, device=G.device)] = float('inf')
        j = int(torch.argmin(maxsim).item())     # least similar to everything picked
        sel.append(j)
        maxsim = torch.maximum(maxsim, G[j])
    chosen = [pool[i] for i in sel]
    return sorted(chosen, key=lambda p: -float(rel_ens[p]))


# --------------------------------------------------------------------------- #
# images
# --------------------------------------------------------------------------- #

def to_display(img_norm, mean, std):
    """Normalized CHW tensor -> HWC array in [0, 1], i.e. the original image."""
    m = torch.tensor(mean).view(-1, 1, 1)
    s = torch.tensor(std).view(-1, 1, 1)
    x = (img_norm.detach().cpu() * s + m).clamp(0.0, 1.0)
    return x[0].numpy() if x.shape[0] == 1 else x.permute(1, 2, 0).numpy()


def draw(ax, img, mean, std):
    ax.imshow(to_display(img, mean, std), interpolation='nearest')
    ax.set_xticks([])
    ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)


# --------------------------------------------------------------------------- #

def parse_args():
    p = argparse.ArgumentParser(
        description='Random vs DPP base selection for one target (clean bases only).',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # what run to open -- defaults match sel_dpp.sh / ours.sh
    p.add_argument('--repo_root', default=os.path.dirname(_HERE),
                   help='where final_update.py, target_sets/ and sweep_config.json live')
    p.add_argument('--out_dir', default=None,
                   help='results root holding the run directories '
                        '(default: <repo_root>/ours_result)')
    p.add_argument('--cache_dir', default=None,
                   help='surrogate cache (default: <repo_root>/cache)')
    p.add_argument('--sweep_config', default=None,
                   help='default: <repo_root>/sweep_config.json')
    p.add_argument('--target_sets_dir', default=None,
                   help='default: <repo_root>/target_sets')
    p.add_argument('--dataset', default='CIFAR10')
    p.add_argument('--data_path', default='/home/mmoslem3/scratch/data')
    p.add_argument('--device', default='auto', help="'auto' | 'cpu' | 'cuda:0'")

    p.add_argument('--model', default='ResNet20BN',
                   choices=['ConvNetBN', 'VGG13BN', 'ResNet20BN'])
    p.add_argument('--attack', default='gradmatch', choices=['fc', 'gradmatch', 'sapa'])
    p.add_argument('--class_pair', default='dog-bird',
                   choices=['dog-bird', 'frog-airplane'],
                   help="'dog-bird' = poisons from dog (y_adv), target is a bird")
    p.add_argument('--pair_order', default='poison-target',
                   choices=['poison-target', 'target-poison'])
    p.add_argument('--budget', type=float, default=0.02,
                   help='poison budget as a fraction of the train set -> N_p')
    p.add_argument('--epsilon', type=float, default=0.0313725)
    p.add_argument('--seed', type=int, default=42)

    # selection knobs -- name the run directories and reproduce the score
    p.add_argument('--sel_alpha', type=float, default=2.0, help='the DPP run alpha')
    p.add_argument('--lambda_margin', type=float, default=1.0)
    p.add_argument('--base_dist', default='cosine', choices=['l2', 'cosine'])
    p.add_argument('--sel_pool', type=float, default=3.0)
    p.add_argument('--sel_mu', type=float, default=1.0)
    p.add_argument('--fc_mode', default='sample', choices=['sample', 'bullseye'])
    p.add_argument('--sharp_mode', default='worst', choices=['worst', 'avg'])
    p.add_argument('--sharp_sigma', type=float, default=0.05)
    p.add_argument('--sharp_samples', type=int, default=20)
    p.add_argument('--craft_ensemble', type=int, default=5)
    p.add_argument('--target_select', type=int, default=None,
                   help='difficulty degree in the run name (_tgt<N>); default: the '
                        "combo's label in sweep_config.json")

    # the surrogate pool the SELECTOR used (prepare_poisons hands it every
    # surrogate, not just the --craft_ensemble crafting subset)
    p.add_argument('--num_surrogates', type=int, default=20)
    p.add_argument('--surrogate_epochs', type=int, default=60)
    p.add_argument('--surrogate_lr', type=float, default=0.1)
    p.add_argument('--surrogate_bs', type=int, default=128)
    p.add_argument('--batch_size', type=int, default=512)

    # target choice
    p.add_argument('--auto-target', '--auto_target', dest='auto_target',
                   action='store_true',
                   help='rank every cached target by the selection-time criterion '
                        'and show the best one (never uses ASR)')
    p.add_argument('--top-targets', '--top_targets', dest='top_targets',
                   type=int, default=0,
                   help='print the top N targets of that ranking (implies '
                        '--auto-target); 0 prints only the winner')
    p.add_argument('--target-id', '--target_id', dest='target_id',
                   type=int, default=None,
                   help='explicit test-set index; overrides the automatic choice')
    p.add_argument('--target_idx', type=int, default=0,
                   help='position in the pinned target set, when neither '
                        '--auto-target nor --target-id is given')
    p.add_argument('--boundary_tol', type=float, default=0.25,
                   help='guard: how much worse DPP\'s mean boundary score may be '
                        'than random\'s (standardized margin units)')
    p.add_argument('--consistency_frac', type=float, default=0.6,
                   help='guard: fraction of surrogates on which DPP must be both '
                        'more relevant and less redundant')

    # display
    p.add_argument('--num-display', '--num_display', '--num_bases', '-m',
                   dest='num_display', type=int, default=8,
                   help='bases drawn per row (a sample of the selected set)')
    p.add_argument('--display_pool_frac', type=float, default=0.5,
                   help='fraction of each selected set, most relevant first, the '
                        'displayed subset is drawn from')
    p.add_argument('--fast', action='store_true',
                   help='no surrogates: draw the first bases of each set and print '
                        'no diagnostics')
    p.add_argument('--list_targets', action='store_true',
                   help='print the pinned targets and which have cached bases, exit')

    # output
    p.add_argument('--save_dir', default=_HERE)
    p.add_argument('--name', default=None, help='output basename (no extension)')
    p.add_argument('--dpi', type=int, default=400)
    p.add_argument('--cell', type=float, default=0.95, help='inches per base image')
    p.add_argument('--no_row_stats', dest='row_stats', action='store_false',
                   help='hide the small per-row summary')
    p.add_argument('--index_labels', action='store_true',
                   help='caption every drawn base with its train-set index')
    return p.parse_args()


def main():
    a = parse_args()
    a.repo_root = os.path.abspath(a.repo_root)
    a.out_dir = a.out_dir or os.path.join(a.repo_root, 'ours_result')
    a.cache_dir = a.cache_dir or os.path.join(a.repo_root, 'cache')
    a.sweep_config = a.sweep_config or os.path.join(a.repo_root, 'sweep_config.json')
    a.target_sets_dir = a.target_sets_dir or os.path.join(a.repo_root, 'target_sets')
    if a.top_targets:
        a.auto_target = True

    # the repo's own run naming / selection code, not a copy of it
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

    print('random run : %s' % rand_dir)
    print('dpp run    : %s  (alpha=%g)' % (dpp_dir, a.sel_alpha))
    print('targets    : %s (%d pinned, %d with cached DPP bases)'
          % (tpath, len(pinned), len(have_dpp)))

    if a.list_targets:
        print('\n  pos  target_id  dpp bases cached  random bases cached')
        for i, t in enumerate(pinned):
            print('  %3d  %9d  %-16s  %s'
                  % (i, t, 'yes' if t in have_dpp else 'no',
                     'yes' if cached_base(rand_dir, t) is not None else 'no'))
        extra = [t for t in have_dpp if t not in pinned]
        if extra:
            print('  dpp cache also holds unpinned targets: %s' % extra)
        return

    # ---- dataset ---------------------------------------------------------- #
    device = a.device
    if device == 'auto':
        device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    channel, im_size, num_classes, class_names, mean, std, dst_train, dst_test, _ = \
        get_dataset(a.dataset, a.data_path)
    y_adv, target_class = FU.parse_pair(a.class_pair, class_names, a.pair_order)
    N_total = len(dst_train)
    N_p = int(round(a.budget * N_total))
    raw_labels = getattr(dst_train, 'targets', None)
    labels = (torch.tensor([int(v) for v in raw_labels], dtype=torch.long)
              if raw_labels is not None
              else torch.tensor([int(dst_train[i][1]) for i in range(N_total)],
                                dtype=torch.long))

    print('pair       : %s -> poisons from %s (y_adv=%d), target is %s (class %d)'
          % (a.class_pair, class_names[y_adv], y_adv, class_names[target_class],
             target_class))
    print('budget     : %g of %d -> N_p = %d bases per selection' % (a.budget, N_total, N_p))

    # candidate order in the pinned set, used when nothing is auto-picked
    if a.target_id is not None:
        default_tid = int(a.target_id)
    elif pinned:
        if not 0 <= a.target_idx < len(pinned):
            raise SystemExit('--target_idx %d out of range (%d pinned targets)'
                             % (a.target_idx, len(pinned)))
        default_tid = pinned[a.target_idx]
    elif have_dpp:
        default_tid = have_dpp[min(a.target_idx, len(have_dpp) - 1)]
    else:
        raise SystemExit('no pinned target set at %s and no cached bases under %s'
                         % (tpath, dpp_dir))

    # ---- fast path: no surrogates, no diagnostics -------------------------- #
    if a.fast:
        tid = default_tid
        dpp_idx = cached_base(dpp_dir, tid)
        if dpp_idx is None:
            raise SystemExit('no cached DPP bases for target %d in %s (cached: %s)'
                             % (tid, dpp_dir, have_dpp or 'none'))
        rand_idx, rand_src = random_bases(FU, rand_dir, tid, labels, y_adv, N_p, a.seed)
        m = min(a.num_display, len(rand_idx), len(dpp_idx))
        shown = {ROW_LABELS[0]: rand_idx[:m], ROW_LABELS[1]: dpp_idx[:m]}
        print('target id  : %d (test set)' % tid)
        print('random     : %d bases from %s' % (len(rand_idx), rand_src))
        print('             shown: %s' % shown[ROW_LABELS[0]])
        print('dpp        : %d bases from poison_cache' % len(dpp_idx))
        print('             shown: %s' % shown[ROW_LABELS[1]])
        make_figure(a, dst_train, dst_test, mean, std, class_names, tid,
                    target_class, y_adv, N_p, shown, None)
        return

    # ---- selection-time diagnostics ---------------------------------------- #
    nets, sdir = load_surrogates(FU, a, channel, num_classes, im_size, device)
    print('surrogates : %d x %s from %s' % (len(nets), a.model, sdir))

    cls_idx = (labels == y_adv).nonzero(as_tuple=True)[0]
    cand = torch.stack([dst_train[int(i)][0] for i in cls_idx]).to(device)
    pos_of = torch.full((N_total,), -1, dtype=torch.long)
    pos_of[cls_idx] = torch.arange(len(cls_idx))
    raw, unit, marg_std = candidate_bank(FU, nets, cand, y_adv, a.batch_size)

    todo = [default_tid] if a.target_id is not None or not a.auto_target else \
        [t for t in (pinned or have_dpp) if t in have_dpp]
    if not todo:
        raise SystemExit('none of the pinned targets has cached DPP bases under %s'
                         % dpp_dir)

    rows = []
    for tid in todo:
        dpp_idx = cached_base(dpp_dir, tid)
        if dpp_idx is None:
            print('  target %d: no cached DPP bases, skipped' % tid)
            continue
        rand_idx, rand_src = random_bases(FU, rand_dir, tid, labels, y_adv, N_p, a.seed)
        rel, score = target_terms(FU, nets, raw, unit, marg_std,
                                  dst_test[tid][0].to(device), a.lambda_margin,
                                  a.base_dist)
        rel_ens = rel.mean(dim=0)
        p_rand = pos_of[torch.tensor(rand_idx)]
        p_dpp = pos_of[torch.tensor(dpp_idx)]
        if int(p_rand.min()) < 0 or int(p_dpp.min()) < 0:
            raise SystemExit(
                'target %d: some cached base is not in class %d (%s) -- the run '
                'directories and --class_pair / --pair_order disagree'
                % (tid, y_adv, class_names[y_adv]))
        p_rand, p_dpp = p_rand.to(device), p_dpp.to(device)
        st_rand = set_stats(p_rand, rel, unit, marg_std, score)
        st_dpp = set_stats(p_dpp, rel, unit, marg_std, score)
        row = {'tid': tid, 'rand': st_rand, 'dpp': st_dpp, 'rand_src': rand_src,
               'rand_idx': rand_idx, 'dpp_idx': dpp_idx,
               'p_rand': p_rand, 'p_dpp': p_dpp, 'rel_ens': rel_ens}
        row.update(compare(st_rand, st_dpp, a.boundary_tol, a.consistency_frac))
        rows.append(row)

    if not rows:
        raise SystemExit('no target with cached DPP bases to score under %s '
                         '(cached: %s)' % (dpp_dir, have_dpp or 'none'))
    rows = rank_targets(rows)

    if a.auto_target and a.target_id is None:
        n_show = a.top_targets if a.top_targets else len(rows)
        print('\nselection-time target ranking (no ASR, no victim result used)')
        print('  rank_score = relevance_gain/std + diversity_gain/std')
        print('  %-9s %8s %8s %8s %8s %6s  %s'
              % ('target', 'rel gain', 'div gain', 'bnd cost', 'score', 'cons', 'rank'))
        for r in rows[:n_show]:
            print('  %-9d %+8.4f %+8.4f %+8.4f %+8.3f %5.0f%%  %+6.3f%s'
                  % (r['tid'], r['relevance_gain'], r['diversity_gain'],
                     r['boundary_cost'], r['dpp']['score'] - r['rand']['score'],
                     100 * r['consistency'], r['rank_score'],
                     '' if r['ok'] else '   rejected: ' + r['why']))
        eligible = [r for r in rows if r['ok']]
        if not eligible:
            print('  !! no target passes both guards -- falling back to the best '
                  'ranked one; treat it as a weak example')
            eligible = rows
        best = eligible[0]
    else:
        hit = [r for r in rows if r['tid'] == default_tid]
        if not hit:
            raise SystemExit('target %d has no cached DPP bases in %s (cached: %s)'
                             % (default_tid, dpp_dir, have_dpp or 'none'))
        best = hit[0]

    tid = best['tid']
    m = min(a.num_display, len(best['rand_idx']), len(best['dpp_idx']))
    picks = {}
    for label, key in ((ROW_LABELS[0], 'p_rand'), (ROW_LABELS[1], 'p_dpp')):
        sub = representative(best[key].tolist(), best['rel_ens'], unit, m,
                             a.display_pool_frac)
        picks[label] = [int(cls_idx[p]) for p in sub]

    # ---- what was chosen, and why ------------------------------------------ #
    r, d = best['rand'], best['dpp']
    print('\nchosen target id : %d (test set, %s)' % (tid, class_names[target_class]))
    print('  selected sets  : %d bases each (random: %s)'
          % (d['n'], best['rand_src']))
    print('  %-22s %10s %10s %10s' % ('', 'Random', 'DPP', 'gain'))
    print('  %-22s %10.4f %10.4f %+10.4f'
          % ('mean target relevance', r['relevance'], d['relevance'],
             best['relevance_gain']))
    print('  %-22s %10.4f %10.4f %+10.4f'
          % ('mean redundancy', r['redundancy'], d['redundancy'],
             best['diversity_gain']))
    print('  %-22s %10.4f %10.4f %+10.4f'
          % ('mean boundary score', r['boundary'], d['boundary'],
             best['boundary_cost']))
    print('  %-22s %10.4f %10.4f %+10.4f'
          % ('mean selection score', r['score'], d['score'],
             d['score'] - r['score']))
    print('  relevance higher = closer to the target; redundancy lower = less '
          'repetitive;\n  boundary lower = nearer the poison-class boundary; '
          'selection score lower = preferred')
    print('  consistent on %.0f%% of the %d surrogates; rank score %+.3f%s'
          % (100 * best['consistency'], len(nets), best['rank_score'],
             '' if best['ok'] else '  (guard failed: %s)' % best['why']))
    print('  displayed Random bases : %s' % picks[ROW_LABELS[0]])
    print('  displayed DPP bases    : %s' % picks[ROW_LABELS[1]])

    stats = {ROW_LABELS[0]: r, ROW_LABELS[1]: d} if a.row_stats else None
    make_figure(a, dst_train, dst_test, mean, std, class_names, tid, target_class,
                y_adv, N_p, picks, stats)


# --------------------------------------------------------------------------- #

def make_figure(a, dst_train, dst_test, mean, std, class_names, tid, target_class,
                y_adv, N_p, picks, stats):
    """Target on the left, Random and DPP in aligned rows of equal length."""
    labels = list(picks)
    m = len(picks[labels[0]])
    cell = a.cell
    stat_w = 1.5 if stats else 0.15
    fig = plt.figure(figsize=(cell * (m + 2.9 + stat_w), cell * 2 + 0.85))
    gs = fig.add_gridspec(2, m + 3, width_ratios=[2.0, 0.95] + [1.0] * m + [stat_w],
                          wspace=0.10, hspace=0.16,
                          left=0.01, right=0.99, top=0.90, bottom=0.11)

    ax_t = fig.add_subplot(gs[:, 0])
    draw(ax_t, dst_test[tid][0], mean, std)
    for sp in ax_t.spines.values():
        sp.set_visible(True)
        sp.set_linewidth(1.1)
        sp.set_edgecolor('0.25')
    ax_t.set_title('Target image (%s)' % class_names[target_class],
                   fontsize=11, pad=7)

    for row, label in enumerate(labels):
        ax_lab = fig.add_subplot(gs[row, 1])
        ax_lab.axis('off')
        ax_lab.text(0.9, 0.5, label, transform=ax_lab.transAxes,
                    ha='right', va='center', fontsize=12)
        for col, j in enumerate(picks[label]):
            ax = fig.add_subplot(gs[row, col + 2])
            draw(ax, dst_train[j][0], mean, std)
            if a.index_labels:
                ax.set_xlabel(str(j), fontsize=6, labelpad=1)
        if stats:
            s = stats[label]
            ax_s = fig.add_subplot(gs[row, m + 2])
            ax_s.axis('off')
            ax_s.text(0.06, 0.5,
                      'Target similarity: %.3f\nRedundancy: %.3f'
                      % (s['relevance'], s['redundancy']),
                      transform=ax_s.transAxes, ha='left', va='center',
                      fontsize=8, color='0.25', linespacing=1.6)

    caption = ('%s · %s · %s · poisons from %s, target a %s · '
               'budget %g (%d bases) — %d representative bases shown from each '
               'selected set'
               % (a.dataset, PRETTY_MODEL.get(a.model, a.model),
                  PRETTY_ATTACK.get(a.attack, a.attack), class_names[y_adv],
                  class_names[target_class], a.budget, N_p, m))
    if stats:
        caption += '; target similarity and redundancy are over the full selected set'
    fig.text(0.5, 0.035, caption, ha='center', va='center', fontsize=8, color='0.35')

    name = a.name or ('random_vs_dpp_%s_%s_%s_b%g_t%d_m%d'
                      % (a.model, a.attack, a.class_pair, a.budget, tid, m))
    os.makedirs(a.save_dir, exist_ok=True)
    stem = os.path.join(a.save_dir, name)
    for ext in ('pdf', 'png'):
        fig.savefig('%s.%s' % (stem, ext), dpi=a.dpi, bbox_inches='tight')
        print('wrote      : %s.%s' % (stem, ext))
    plt.close(fig)


if __name__ == '__main__':
    main()
