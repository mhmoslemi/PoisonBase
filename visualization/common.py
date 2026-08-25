#!/usr/bin/env python
"""
visualization/common.py -- shared plotting / data-extraction helpers for the
paper figures.

Scope
-----
This module contains ONLY visualization and data-extraction helpers. Every piece
of science -- the networks, the dataset and its normalization, the selector
score, the greedy / DPP / random selections, the attacks and the victim training
loop -- is imported from the repository (`final_update.py`, `networks.py`,
`utils.py`) and called, never re-implemented here. If a figure needs a quantity
the repository does not expose (e.g. the two terms of the selector score
separately), it is recomputed here in a function that also CHECKS its
reconstruction against the repository's own value and fails loudly on a mismatch.

Repository objects used
-----------------------
    final_update.build_context            dataset on device + norm/denorm
    final_update.parse_pair               '<adv>-<target>' -> (y_adv, target_class)
    final_update.build_network            model factory (ConvNetBN, ...)
    final_update.surrogate_dir            where the cached surrogates live
    final_update.embed_of                 penultimate representation h_k(x)
    final_update.standardize              the selector's z-scoring
    final_update._ours_score_and_feats    the selector score s_i and the ensemble
                                          features phi_i (the DPP kernel space)
    final_update.select_base_ours         Greedy
    final_update.select_base_ours_div     DPP (mode='dpp')
    final_update.select_base_random       Random
    final_update.build_run_name           locates an attack run directory
    final_update.load_poison_cache        the saved deltas / base sets

Paper configuration
-------------------
PAPER below mirrors the appendix protocol used by appendix/ap1-broad.sh and
appendix/ap8-redundancy.sh (ConvNetBN, CIFAR-10, budget 5e-3, 8/255 Linf,
selector lambda=1, alpha=2, K=20 surrogates, cosine base distance, seed 42).
Nothing here invents a hyper-parameter: every default is taken from a script in
the repository, and anything that could not be resolved is a required CLI
argument.
"""

import argparse
import csv
import hashlib
import json
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

import matplotlib
matplotlib.use('Agg')                      # never depend on an interactive backend
import matplotlib.pyplot as plt

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import final_update as FU                                        # noqa: E402


# --------------------------------------------------------------------------- #
# the paper's configuration (see appendix/ap1-broad.sh, appendix/ap8-redundancy.sh,
# sel_dpp.sh -- all three agree on these values for ConvNetBN / CIFAR-10)
# --------------------------------------------------------------------------- #

def _default_data_path():
    """Where CIFAR-10 already lives. The repo's scripts pass
    /home/mmoslem3/scratch/data; ./data is the final_update.py default. Take the
    first one that is really there so nothing is downloaded by accident."""
    for p in (os.environ.get('DATA_PATH'),
              '/home/mmoslem3/scratch/data',
              os.path.join(REPO_ROOT, 'data')):
        if p and os.path.isdir(p):
            return p
    return os.path.join(REPO_ROOT, 'data')


PAPER = dict(
    dataset='CIFAR10',
    data_path=_default_data_path(),
    model='ConvNetBN',
    # appendix.tex writes the direction as bird -> dog; final_update.py takes
    # '<adversarial>-<target>' under --pair_order poison-target, i.e. 'dog-bird'
    # means poisons are drawn from dog and the target image is a bird.
    class_pair='dog-bird',
    pair_order='poison-target',
    seed=42,
    cache_dir=os.path.join(REPO_ROOT, 'cache'),
    out_dir=os.path.join(REPO_ROOT, 'ours_result'),
    budget=0.005,                 # the appendix's "eps = 5e-3" poison budget
    epsilon=0.0313725,            # 8/255 Linf perturbation radius
    craft_steps=250,
    craft_alpha=0.0039216,        # 1/255, signed-Adam lr
    restarts=8,
    craft_ensemble=5,
    num_surrogates=20,            # K, the selector ensemble
    surrogate_epochs=60,
    surrogate_lr=0.1,
    surrogate_bs=128,
    lambda_margin=1.0,            # lambda
    base_dist='cosine',
    sel_alpha=2.0,                # alpha of the DPP quality term
    sharp_mode='worst',
    sharp_sigma=0.05,             # SAPA (ICLR 2024) default
    dsa_strategy=FU.DSA_DEFAULT,
    victim_epochs=50,
    victim_lr=0.1,
    victim_bs=125,
    victim_decay=[40],
    victim_wd=0.0,
    # the 5 targets appendix/pin_targets.py froze for ConvNetBN / dog-bird, shared
    # by every selection method and attack of the reduced appendix protocol
    appendix_target_file=os.path.join(REPO_ROOT, 'target_sets',
                                      'appx_broad_ConvNetBN_dog-bird.json'),
    # the 10 targets the main gradient-matching sweep attacked for this combo
    main_target_file=os.path.join(REPO_ROOT, 'target_sets',
                                  'ConvNetBN_gradmatch_dog-bird.json'),
)


# --------------------------------------------------------------------------- #
# typography and palette -- defined once in plotstyle.py (no torch import there,
# so the results-only figures start instantly) and re-exported here
# --------------------------------------------------------------------------- #

from plotstyle import (FS, COLORS, MARKERS, LABELS, METHOD_ORDER, ATTACK_COLORS,  # noqa: E402,F401
                       ATTACK_LABELS, MODEL_MARKERS, MODEL_LABELS, CMAP_SIM,
                       CMAP_SEQ, WIDTH_SINGLE, WIDTH_HALF, WIDTH_FULL, set_paper_style,
                       panel_label, note, light_grid, identity_line, save_fig,
                       save_csv, stats_box, bottom_legend, SIZES, ALPHA,
                       kde_contours, density_hexbin, binned_trend)


# --------------------------------------------------------------------------- #
# reproducibility
# --------------------------------------------------------------------------- #

def viz_seed(seed):
    """Seed for VISUALIZATION-ONLY sampling (candidate subsets, jitter).

    Deliberately separate from the experiment seeds: nothing scientific may
    depend on it, and it never touches the experiment's own generators.
    """
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    return rng


# --------------------------------------------------------------------------- #
# argparse blocks shared by the scripts
# --------------------------------------------------------------------------- #

def add_repo_args(p):
    """The flags that identify the repository configuration to reproduce."""
    g = p.add_argument_group('repository / paper configuration')
    g.add_argument('--dataset', default=PAPER['dataset'])
    g.add_argument('--data_path', default=PAPER['data_path'])
    g.add_argument('--model', default=PAPER['model'], choices=FU.SUPPORTED_MODELS)
    g.add_argument('--class_pair', default=PAPER['class_pair'],
                   help="'<adversarial>-<target>'. The paper's bird -> dog is "
                        "'dog-bird' under --pair_order poison-target.")
    g.add_argument('--pair_order', default=PAPER['pair_order'],
                   choices=['poison-target', 'target-poison'])
    g.add_argument('--seed', type=int, default=PAPER['seed'])
    g.add_argument('--cache_dir', default=PAPER['cache_dir'])
    g.add_argument('--out_dir', default=PAPER['out_dir'],
                   help='where the attack runs live (read-only for the figures)')
    g.add_argument('--device', default=None,
                   help="'cuda:0' | 'cpu'; default = cuda:0 if available")

    g = p.add_argument_group('selector (defaults = the paper: lambda=1, alpha=2, K=20)')
    g.add_argument('--sel_K', type=int, default=PAPER['num_surrogates'],
                   help='K, the number of selector surrogates')
    g.add_argument('--lambda_margin', type=float, default=PAPER['lambda_margin'])
    g.add_argument('--base_dist', default=PAPER['base_dist'], choices=['l2', 'cosine'])
    g.add_argument('--sel_alpha', type=float, default=PAPER['sel_alpha'])
    g.add_argument('--surrogate_epochs', type=int, default=PAPER['surrogate_epochs'])
    g.add_argument('--surrogate_lr', type=float, default=PAPER['surrogate_lr'])
    g.add_argument('--surrogate_bs', type=int, default=PAPER['surrogate_bs'])

    g = p.add_argument_group('output')
    g.add_argument('--out', default=os.path.join(REPO_ROOT, 'visualization', 'figs'),
                   help='directory for the pdf / png / csv (created if missing)')
    g.add_argument('--viz_seed', type=int, default=0,
                   help='seed for visualization-only sampling; nothing scientific '
                        'depends on it')
    g.add_argument('--dry_run', action='store_true',
                   help='print what would be computed and exit without touching '
                        'the GPU')
    return p


def add_mode_args(p):
    """--compute / --plot-only around an .npz intermediate."""
    g = p.add_argument_group('mode')
    g.add_argument('--compute', action='store_true',
                   help='run the extraction and write the .npz/.csv intermediate')
    g.add_argument('--plot_only', action='store_true',
                   help='skip the extraction, read the intermediate, draw the figure')
    return p


def resolve_mode(args):
    """Default: do both. --compute alone stops before plotting."""
    if args.compute and args.plot_only:
        raise SystemExit('--compute and --plot_only are mutually exclusive')
    if not args.compute and not args.plot_only:
        return True, True
    return args.compute, (args.plot_only or False)


def summarize(title, lines):
    print('=' * 74)
    print(title)
    print('=' * 74)
    for k, v in lines:
        print('  %-28s %s' % (k + ':', v))
    print('-' * 74)


# --------------------------------------------------------------------------- #
# repository lookups: device, dataset context, classes, checkpoints
# --------------------------------------------------------------------------- #

def pick_device(args):
    if args.device:
        return args.device
    return 'cuda:0' if torch.cuda.is_available() else 'cpu'


def build_ctx(args, device):
    """final_update.build_context: dataset as GPU tensors + norm/denorm closures."""
    return FU.build_context(argparse.Namespace(dataset=args.dataset,
                                               data_path=args.data_path), device)


def classes_of(args, ctx):
    """(y_adv, target_class) exactly as final_update resolves --class_pair."""
    return FU.parse_pair(args.class_pair, ctx['class_names'], args.pair_order)


def require_file(path, what):
    if not os.path.exists(path):
        raise SystemExit('%s not found: %s\nThis script only ever LOADS; it will '
                         'not train or fabricate it.' % (what, path))
    return path


def surrogate_dir(args):
    """cache/surrogates/<tag><model>_<ep>ep_lr<lr>_bs<bs>_seed<seed>/"""
    return FU.surrogate_dir(argparse.Namespace(
        cache_dir=args.cache_dir, dataset=args.dataset, model=args.model,
        surrogate_epochs=args.surrogate_epochs, surrogate_lr=args.surrogate_lr,
        surrogate_bs=args.surrogate_bs, seed=args.seed))


def _load_net(args, ctx, path, seed):
    net = FU.build_network(args.model, ctx['channel'], ctx['num_classes'],
                           ctx['im_size'], ctx['device'], seed=seed)
    net.load_state_dict(torch.load(path, map_location=ctx['device']))
    net.eval()
    return net


def load_selector_surrogates(args, ctx, k=None):
    """The K selector surrogates, LOADED from cache. Never trains.

    Same files final_update.get_surrogates would load (net_0 ... net_{K-1} of
    surrogate_dir), so these are literally the nets the experiments selected with.
    """
    k = int(k if k is not None else args.sel_K)
    d = surrogate_dir(args)
    nets, paths = [], []
    for i in range(k):
        p = os.path.join(d, 'net_%d.pt' % i)
        require_file(p, 'selector surrogate %d' % i)
        nets.append(_load_net(args, ctx, p, seed=args.seed + 1000 + i))
        paths.append(p)
    return nets, paths


def param_fingerprint(net):
    """sha1 of the flat parameter vector -- used to PROVE a held-out model is not
    one of the selector surrogates, rather than trusting a file path."""
    h = hashlib.sha1()
    with torch.no_grad():
        for p in net.parameters():
            h.update(p.detach().float().cpu().numpy().tobytes())
    return h.hexdigest()


def load_heldout_model(args, ctx, path, selector_nets, selector_paths, seed=None):
    """A model that must NOT be one of the K selector surrogates.

    Enforced twice: by path and by parameter fingerprint. A silent overlap would
    invalidate the whole held-out control, so this raises instead of warning.
    """
    require_file(path, 'held-out checkpoint')
    real = os.path.realpath(path)
    for p in selector_paths:
        if os.path.realpath(p) == real:
            raise SystemExit(
                'the held-out checkpoint %s IS selector surrogate %s.\n'
                'Figure 1 measures utility on a model the selector never saw; '
                'point --heldout_checkpoint at a different net (e.g. net_%d.pt '
                'or a clean victim in cache/clean_victims/).'
                % (path, p, args.sel_K))
    net = _load_net(args, ctx, path, seed=seed)
    fp = param_fingerprint(net)
    for p, sn in zip(selector_paths, selector_nets):
        if param_fingerprint(sn) == fp:
            raise SystemExit(
                'the held-out checkpoint %s has the same weights as selector '
                'surrogate %s. It cannot be used as the held-out model.' % (path, p))
    return net, fp


def num_poisons(args, ctx):
    """m, exactly as final_update.main computes it from --budget."""
    n_total = ctx['train_imgs'].shape[0]
    return int(round(args.budget * n_total))


# --------------------------------------------------------------------------- #
# target lookup
# --------------------------------------------------------------------------- #

def load_pinned_targets(path, pair):
    """The pinned test indices, read the way final_update.select_targets reads
    a --target_idx_file."""
    require_file(path, 'target index file')
    with open(path) as f:
        blob = json.load(f)
    want = blob['pairs'][pair]['indices'] if 'pairs' in blob else blob[pair]
    idx = [int(i) for i in want]
    if not idx:
        raise SystemExit('%s lists no targets for pair %s' % (path, pair))
    return idx


def resolve_targets(args, ctx, target_class):
    """--target_indices wins over --target_idx_file. Every target is checked to
    really be an image of the target class."""
    if getattr(args, 'target_indices', None):
        idx = [int(i) for i in args.target_indices]
    else:
        idx = load_pinned_targets(args.target_idx_file, args.class_pair)
    labs = ctx['test_labs']
    bad = [i for i in idx if int(labs[i]) != target_class]
    if bad:
        raise SystemExit('test indices %s are not of the target class %d (%s) -- '
                         'wrong --class_pair / --pair_order?'
                         % (bad, target_class, ctx['class_names'][target_class]))
    return idx


# --------------------------------------------------------------------------- #
# the selector: score, its two terms, and the ensemble representation phi_i
# --------------------------------------------------------------------------- #

@torch.no_grad()
def selector_state(nets, ctx, x_t_norm, y_adv, lam, base_dist, bs=512):
    """Everything the selector knows about the candidate pool, for one target.

    Returns a dict with
        cls_idx    train indices of the whole y_adv candidate pool
        score      s_i, straight from final_update._ours_score_and_feats.
                   REPOSITORY CONVENTION: the selector takes the SMALLEST scores,
                   so LOWER s_i = better candidate. Greedy = the m smallest.
        feats      phi_i, the (N, K*d) block of per-surrogate L2-normalised
                   features divided by sqrt(K). By construction
                       phi_i . phi_j = (1/K) sum_k cos(h_k(x_i), h_k(x_j)) = C_ij,
                   i.e. the exact ensemble cosine kernel the DPP uses.
        boundary   B_i = (1/K) sum_k standardize(margin toward y_adv)_k
        distance   D_i = (1/K) sum_k standardize(d_k), d = 1 - cos for base_dist
                   cosine (this is the term the paper calls the target-relevance
                   term; it enters the score with a minus sign relative to R_i)
        relevance  R_i = (1/K) sum_k cos(h_k(x_i), h_k(x_t)), higher = closer to
                   the target. Reported, not used to re-derive anything.

    The score and the features come from the repository. B/D/R are recomputed
    here only because final_update does not return the terms separately, and the
    reconstruction  score == D + lam * B  is asserted below, so these columns can
    never silently drift from the selector the experiments ran.
    """
    device = ctx['device']
    cls_idx, score, feats = FU._ours_score_and_feats(
        nets, ctx['train_imgs'], ctx['train_labs'], x_t_norm, y_adv, lam,
        device, base_dist=base_dist, bs=bs)

    cand = ctx['train_imgs'][cls_idx]
    B = torch.zeros(len(cls_idx), device=device)
    D = torch.zeros(len(cls_idx), device=device)
    R = torch.zeros(len(cls_idx), device=device)
    for net in nets:
        net.eval()
        emb = FU.embed_of(net)
        f_t = emb(x_t_norm.unsqueeze(0))
        ds, ms, cs = [], [], []
        for i in range(0, len(cand), bs):
            b = cand[i:i + bs]
            fb = emb(b)
            cos = F.cosine_similarity(fb, f_t.expand(len(b), -1), dim=1)
            d = (1.0 - cos) if base_dist == 'cosine' else ((fb - f_t) ** 2).sum(dim=1)
            z = net(b)
            z_adv = z[:, y_adv].clone()
            z_o = z.clone()
            z_o[:, y_adv] = float('-inf')
            ms.append(z_adv - z_o.max(dim=1).values)
            ds.append(d)
            cs.append(cos)
        B += FU.standardize(torch.cat(ms))
        D += FU.standardize(torch.cat(ds))
        R += torch.cat(cs)
    k = float(len(nets))
    B, D, R = B / k, D / k, R / k

    err = (D + lam * B - score).abs().max().item()
    tol = 1e-3 * max(1.0, float(score.abs().max()))
    if not np.isfinite(err) or err > tol:
        raise RuntimeError(
            'the reconstructed selector score disagrees with '
            'final_update._ours_score_and_feats by %.3g (tol %.3g). The figure '
            'refuses to plot a score that is not the selector\'s own.' % (err, tol))

    return {'cls_idx': cls_idx, 'score': score, 'feats': feats,
            'boundary': B, 'distance': D, 'relevance': R}


@torch.no_grad()
def target_ensemble_feature(nets, x_t_norm):
    """phi_t, built exactly like phi_i in _ours_score_and_feats, so phi_t . phi_i
    is the ensemble cosine between the target and candidate i."""
    blocks = []
    for net in nets:
        net.eval()
        f = FU.embed_of(net)(x_t_norm.unsqueeze(0)).detach().flatten(1)
        blocks.append(F.normalize(f, dim=1))
    return torch.cat(blocks, dim=1)[0] / float(len(nets)) ** 0.5


def positions_in_pool(cls_idx, selected_idx):
    """Train indices -> positions inside the candidate pool (for indexing feats)."""
    lookup = {int(v): i for i, v in enumerate(cls_idx.tolist())}
    missing = [int(v) for v in selected_idx.tolist() if int(v) not in lookup]
    if missing:
        raise RuntimeError('selected indices %s are outside the y_adv pool' % missing[:5])
    return torch.tensor([lookup[int(v)] for v in selected_idx.tolist()],
                        dtype=torch.long, device=cls_idx.device)


# --------------------------------------------------------------------------- #
# the three selections, called on the repository implementations
# --------------------------------------------------------------------------- #

def select_greedy(nets, ctx, x_t_norm, y_adv, m, args):
    return FU.select_base_ours(nets, ctx['train_imgs'], ctx['train_labs'], x_t_norm,
                               y_adv, m, args.lambda_margin, ctx['device'],
                               base_dist=args.base_dist)


def select_dpp(nets, ctx, x_t_norm, y_adv, m, args):
    return FU.select_base_ours_div(nets, ctx['train_imgs'], ctx['train_labs'],
                                   x_t_norm, y_adv, m, args.lambda_margin,
                                   ctx['device'], base_dist=args.base_dist,
                                   mode='dpp', alpha=args.sel_alpha)


def select_random(ctx, y_adv, m, seed, tidx):
    """Random bases with the SAME per-target generator prepare_poisons uses,
    manual_seed(seed * 100003 + target_idx), so a run and a figure agree."""
    gen = torch.Generator(device='cpu').manual_seed(int(seed) * 100003 + int(tidx))
    return FU.select_base_random(ctx['train_labs'], y_adv, m, ctx['device'], gen)


# --------------------------------------------------------------------------- #
# set geometry in the ensemble representation space
# --------------------------------------------------------------------------- #

def similarity_matrix(phi_S):
    """C_S, the exact ensemble-cosine kernel of the selected set."""
    return phi_S @ phi_S.T


def mean_offdiag(C):
    """Red(S) = 2 / (m (m-1)) * sum_{i<j} C_ij."""
    m = C.shape[0]
    if m < 2:
        return float('nan')
    return float((C.sum() - C.diagonal().sum()) / (m * (m - 1)))


def effective_rank(phi_S):
    """r_eff = exp(-sum_j p_j log p_j), p_j = sigma_j / sum_l sigma_l.

    Singular values of the SELECTED representation matrix, i.e. the spectrum of
    C_S. Computed in float64: the tail of the spectrum is what the entropy is
    sensitive to.
    """
    s = torch.linalg.svdvals(phi_S.double())
    s = s.clamp_min(0)
    tot = s.sum()
    if float(tot) <= 0:
        return float('nan')
    p = s / tot
    p = p[p > 0]
    return float(torch.exp(-(p * p.log()).sum()))


def pca_2d(X, seed=0):
    """Exact 2-component PCA of X (n, D) via the Gram matrix.

    Deterministic (no randomized SVD), and cheap when D >> n, which is the case
    here: D = K * dim(h) is ~40k while n is a few thousand.

    Returns (coords, evr, transform) where
        coords     (n, 2) PC scores of the rows of X
        evr        explained variance ratio of PC1 / PC2
        transform  maps NEW rows (k, D) into the SAME fitted basis -- this is how
                   the target is placed on the plot, never by re-fitting.
    """
    torch.manual_seed(seed)                       # only for reproducible tie-breaks
    mu = X.mean(dim=0, keepdim=True)
    Xc = X - mu
    G = Xc @ Xc.T
    G = 0.5 * (G + G.T)
    evals, evecs = torch.linalg.eigh(G.double())
    order = torch.argsort(evals, descending=True)
    lam2 = evals[order[:2]].clamp_min(1e-12)
    U = evecs[:, order[:2]]
    coords = (U * lam2.sqrt()).float()
    total = evals.clamp_min(0).sum()
    evr = (lam2 / total).cpu().numpy()

    def transform(Y):
        # (Y - mu) V  with  V = Xc^T U / sqrt(lambda); written so the big matmul
        # stays in float32 and only the (k, n) x (n, 2) part is float64
        proj = ((Y - mu) @ Xc.T).double() @ U / lam2.sqrt()
        return proj.float()

    return coords, evr, transform


# --------------------------------------------------------------------------- #
# images
# --------------------------------------------------------------------------- #

def to_display(img_norm, ctx):
    """Normalized CHW tensor -> HWC array in [0, 1], i.e. the original pixels."""
    x = ctx['denorm'](img_norm.detach().unsqueeze(0)).clamp(0.0, 1.0)[0]
    return x.permute(1, 2, 0).cpu().numpy()


def to_uint8(img_norm, ctx):
    """Normalized CHW tensor -> (H, W, 3) uint8, small enough to cache in the
    .npz so a re-plot never has to load the dataset (or a gpu)."""
    return (to_display(img_norm, ctx) * 255.0 + 0.5).astype('uint8')


def draw_rgb(ax, arr, edge=None, lw=0.8):
    """Draw a cached uint8 image at its native resolution."""
    ax.imshow(arr, interpolation='nearest')
    ax.set_xticks([])
    ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(edge is not None)
        if edge is not None:
            sp.set_color(edge)
            sp.set_linewidth(lw)


def draw_image(ax, img_norm, ctx, edge=None, lw=0.8):
    """CIFAR image at its native 32x32, no interpolation artifacts."""
    ax.imshow(to_display(img_norm, ctx), interpolation='nearest')
    ax.set_xticks([])
    ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(edge is not None)
        if edge is not None:
            sp.set_color(edge)
            sp.set_linewidth(lw)


def method_handles(methods=METHOD_ORDER, line=True):
    """Legend handles with the shared visual identity."""
    from matplotlib.lines import Line2D
    out = []
    for k in methods:
        out.append(Line2D([0], [0], color=COLORS[k], marker=MARKERS[k],
                          linestyle='-' if line else 'none', markersize=3.2,
                          linewidth=1.2, label=LABELS[k]))
    return out
