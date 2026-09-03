#!/usr/bin/env python
"""Exact gradient-alignment audit for GRAFT's base-selection score.

Reviewer complaint this answers
-------------------------------
Eq. (3) decomposes the full parameter-gradient alignment

    <g_i, g_t> = r_i^T [ (h_i^T h_t) I + W J_i J_t^T W^T ] r_t ,
    J_x = d h_phi(x) / d phi ,

but Algorithm 1 ranks by  s_i = R~_i - M~_i + beta * A~_i  with a logit margin
M_i, a feature term R_i and a backbone-gradient term A_i. Nothing in the paper
shows how far that score is from the quantity it claims to approximate. This
script measures the gap directly: for every candidate base it computes the exact
<g_i, g_t>, every term of the Eq. (3) decomposition, and every ingredient of the
Algorithm 1 score, on the *same* cached surrogates the paper's runs used.

What is written per (dataset, architecture, class pair, target, surrogate)
--------------------------------------------------------------------------
exact alignment
    g_full      <g_i, g_t> over ALL parameters, exact.
    head_W      <grad_W l_i, grad_W L_t> = (r_i.r_t)(h_i.h_t)      [closed form]
    head_b      <grad_b l_i, grad_b L_t> = (r_i.r_t)               [closed form]
    A           <grad_phi l_i, grad_phi L_t> = u_i^T J_i J_t^T u_t [exact JVP]
                u_x = W^T r_x. This is Algorithm 1's A_i verbatim.
    The identity  g_full == head_W + head_b + A  is checked to machine
    precision on --verify candidates per (target, surrogate) by brute-force
    per-sample full-parameter gradients.

Eq. (3) ingredients
    hTh         <h_i, h_t>            -- the paper's R_i
    cos         cosine(h_i, h_t)
    d_cos       1 - cos               -- the feature term the CODE actually uses
    d_l2        ||h_i - h_t||^2
    r_dot       <r_i, r_t>            -- the residual coupling Eq. (3) has and
                                         Algorithm 1 drops
    r_norm      ||r_i||               -- what M_i is a proxy for
    margin      z_i[y_adv] - max_{c != y_adv} z_i[c]   -- the paper's M_i
    p_adv       softmax(z_i)[y_adv]

pure Jacobian kernel  (the "J_i^T J_t" quantities)
    jker        u_t^T J_i J_t^T u_t   -- same probe on both sides, so the
                                         candidate residual is removed and what
                                         is left is the Jacobian interaction
                                         alone. Exact, one JVP per batch.
    ntk_tr      Hutchinson estimate of tr(J_i J_t^T) over --ntk_probes
                Rademacher probes. This is the scalar directly comparable to
                h_i^T h_t, i.e. the quantity behind the paper's claim that the
                two are strongly correlated (Sec. 3.2, the NTK remark).
    ntk_tr_se   standard error of that estimate.

score reproduction
    z_dcos, z_margin, z_A   the standardized components, per surrogate.
    cost_prod_b0/b1         the production cost exactly as final_update.py
                            builds it: mean_k[ std(d_cos) + lam*std(margin)
                            - beta*std(A) ]. LOWER is selected.
    s_paper_b0/b1           the score as Algorithm 1 writes it, using the raw
                            R_i = <h_i, h_t>: mean_k[ std(hTh) - std(margin)
                            + beta*std(A) ]. HIGHER is selected.
    s_eq3_b0/b1             the same ranking driven by the exact g_full, i.e.
                            what Algorithm 1 would do if it ranked by the
                            quantity Eq. (3) actually defines.
    Comparing the top-m sets of these three is the quantitative answer to
    "how loosely is the method connected to its stated objective".

Everything runs on the cached surrogates in --cache_dir; nothing is retrained
unless a checkpoint is missing (then it is trained once and cached, exactly as
final_update.py would).

Output
------
    <out_dir>/<DATASET>_<MODEL>_<PAIR>/meta.json          run configuration
    <out_dir>/<DATASET>_<MODEL>_<PAIR>/target_<idx>.npz   arrays, see above
    <out_dir>/<DATASET>_<MODEL>_<PAIR>/summary.csv        per (target,surrogate)
                                                          correlations + identity
                                                          residuals + top-m overlap
    <out_dir>/manifest.jsonl                              one line per combo

Per-surrogate arrays are (K, N) float32, K = surrogates, N = candidates, with
`cand_idx` (N,) giving the training-set index of each candidate. With --csv a
long-format target_<idx>.csv.gz is written alongside for direct pandas use.
"""

import argparse
import csv
import glob
import json
import math
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import get_dataset, ParamDiffAug
from final_update import (build_network, parse_pair, standardize, embed_of,
                          get_surrogates)

try:
    from scipy.stats import spearmanr
except Exception:                                            # scipy is optional
    spearmanr = None


# --------------------------------------------------------------------------- #
# logging
# --------------------------------------------------------------------------- #

_T0 = time.time()


def log(msg):
    print('[%7.1fs] %s' % (time.time() - _T0, msg), flush=True)


# --------------------------------------------------------------------------- #
# dataset access that does NOT materialise the whole set on the GPU
#
# final_update.stack_dataset pushes train+test onto the device. TinyImageNet is
# 4.9 GB at float32, and here we only ever need one class pool plus a handful of
# targets, so pull labels cheaply and fetch only the images we index.
# --------------------------------------------------------------------------- #

def labels_of(dst):
    for attr in ('targets', 'labels'):
        if hasattr(dst, attr):
            v = getattr(dst, attr)
            if isinstance(v, torch.Tensor):
                return v.detach().cpu().long()
            return torch.as_tensor(np.asarray(v), dtype=torch.long)
    return torch.tensor([int(dst[i][1]) for i in range(len(dst))], dtype=torch.long)


def images_at(dst, idx, device, chunk=512):
    """Normalized images for the given indices, assembled chunkwise on the GPU."""
    idx = list(int(i) for i in idx)
    out = None
    for s in range(0, len(idx), chunk):
        block = torch.stack([dst[i][0] for i in idx[s:s + chunk]]).to(device)
        if out is None:
            out = torch.empty((len(idx),) + block.shape[1:], device=device,
                              dtype=block.dtype)
        out[s:s + len(block)] = block
    return out


# --------------------------------------------------------------------------- #
# target sets
# --------------------------------------------------------------------------- #

def resolve_target_file(args):
    if args.target_idx_file:
        return args.target_idx_file
    root = args.target_sets
    tries = [
        '%s_%s_%s.json' % (args.model, args.attack, args.class_pair),
        'xdata_%s_%s_%s.json' % (args.dataset, args.model, args.class_pair),
        'appx_tiny_%s_%s.json' % (args.model, args.class_pair),
        'appx_broad_%s_%s.json' % (args.model, args.class_pair),
        '%s_%s.json' % (args.model, args.class_pair),
    ]
    for name in tries:
        p = os.path.join(root, name)
        if os.path.isfile(p):
            return p
    for pattern in ('*_%s_%s.json' % (args.model, args.class_pair),
                    '*%s.json' % args.class_pair):
        hits = sorted(glob.glob(os.path.join(root, pattern)))
        if hits:
            return hits[0]
    return None


def load_targets(args):
    if args.target_idx:
        return [int(t) for t in args.target_idx], '--target_idx'
    path = resolve_target_file(args)
    if not path:
        raise SystemExit(
            'no pinned target set for %s / %s / %s under %s; pass --target_idx_file '
            'or --target_idx' % (args.dataset, args.model, args.class_pair,
                                 args.target_sets))
    with open(path) as f:
        blob = json.load(f)
    pairs = blob.get('pairs', {})
    entry = pairs.get(args.class_pair)
    if entry is None and len(pairs) == 1:
        entry = next(iter(pairs.values()))
    if entry is None:
        raise SystemExit('%s has no entry for pair %r (has %s)'
                         % (path, args.class_pair, list(pairs)))
    idx = [int(i) for i in entry['indices']]
    if args.num_targets:
        idx = idx[:args.num_targets]
    return idx, path


# --------------------------------------------------------------------------- #
# functional wrappers
#
# torch.func.functional_call always drives Module.__call__, so `embed` needs its
# own wrapper. Both wrappers hold the core under the SAME attribute name, which
# keeps one parameter-name namespace ("core.<...>") valid for either of them.
# --------------------------------------------------------------------------- #

class _LogitWrap(nn.Module):
    def __init__(self, core):
        super().__init__()
        self.core = core

    def forward(self, x):
        return self.core(x)


class _EmbedWrap(nn.Module):
    def __init__(self, core):
        super().__init__()
        self.core = core

    def forward(self, x):
        return self.core.embed(x)


def split_params(core, logit_wrap):
    """(backbone phi, frozen head, buffers) keyed by the wrapper's names."""
    head_ids = {id(p) for p in core.classifier.parameters()}
    backbone, fixed = {}, {}
    for name, p in logit_wrap.named_parameters():
        (fixed if id(p) in head_ids else backbone)[name] = p.detach()
    buffers = {name: b.detach() for name, b in logit_wrap.named_buffers()}
    if not backbone:
        raise ValueError('no parameters outside .classifier for %s' % type(core).__name__)
    return backbone, fixed, buffers


def backbone_grad(wrap, backbone, fixed, buffers, x, scalar_fn):
    """Reverse-mode d/dphi of scalar_fn(out).sum(). Used only on the target."""
    bb = {k: v.detach().clone().requires_grad_(True) for k, v in backbone.items()}
    params = dict(fixed)
    params.update(bb)
    with torch.enable_grad():
        out = torch.func.functional_call(wrap, (params, buffers), (x,))
        total = scalar_fn(out).sum()
        grads = torch.autograd.grad(total, tuple(bb.values()), allow_unused=True)
    return {k: (torch.zeros_like(bb[k]) if g is None else g.detach())
            for k, g in zip(bb, grads)}


def per_sample_dirderiv(wrap, backbone, fixed, buffers, x, tangent, scalar_fn):
    """Per-sample directional derivative of scalar_fn along `tangent` in phi.

    Returns (B,). Forward-mode first; if an operator has no forward-AD rule the
    dummy-weight double-backward identity gives the identical vector, so a run
    never silently degrades to an approximation. Mirrors the production path in
    final_update._backbone_gradient_interactions.
    """
    def f(bb):
        params = dict(fixed)
        params.update(bb)
        return scalar_fn(torch.func.functional_call(wrap, (params, buffers), (x,)))

    with torch.enable_grad():
        try:
            _, tang = torch.func.jvp(f, (backbone,), (tangent,))
            return tang.detach(), 'jvp'
        except Exception:
            bb = {k: v.detach().clone().requires_grad_(True)
                  for k, v in backbone.items()}
            params = dict(fixed)
            params.update(bb)
            w = torch.zeros(len(x), device=x.device, dtype=x.dtype,
                            requires_grad=True)
            out = torch.func.functional_call(wrap, (params, buffers), (x,))
            s = scalar_fn(out)
            mixed = torch.autograd.grad(torch.dot(w, s), tuple(bb.values()),
                                        create_graph=True, allow_unused=True)
            direct = sum((g * tangent[k]).sum()
                         for g, k in zip(mixed, bb) if g is not None)
            return torch.autograd.grad(direct, w)[0].detach(), 'double-backward'


# --------------------------------------------------------------------------- #
# exact full-parameter alignment, for the identity check
# --------------------------------------------------------------------------- #

def full_param_grad(core, x, y):
    params = [p for p in core.parameters()]
    for p in params:
        p.requires_grad_(True)
    with torch.enable_grad():
        loss = F.cross_entropy(core(x), y)
        grads = torch.autograd.grad(loss, params, allow_unused=True)
    return torch.cat([(torch.zeros_like(p) if g is None else g).reshape(-1)
                      for g, p in zip(grads, params)]).detach()


# --------------------------------------------------------------------------- #
# the per-(target, surrogate) measurement
# --------------------------------------------------------------------------- #

@torch.no_grad()
def measure(net, cand, x_t, y_adv, args, rng):
    core = net.module if isinstance(net, nn.DataParallel) else net
    if not hasattr(core, 'classifier') or not isinstance(core.classifier, nn.Linear):
        raise SystemExit('%s does not expose an nn.Linear .classifier; the Eq. (3) '
                         'split needs it' % type(core).__name__)
    was_training = core.training
    core.eval()
    try:
        logit_wrap = _LogitWrap(core).eval()
        embed_wrap = _EmbedWrap(core).eval()
        backbone, fixed, buffers = split_params(core, logit_wrap)

        W = core.classifier.weight.detach()
        has_bias = core.classifier.bias is not None
        C = W.shape[0]
        e_adv = F.one_hot(torch.tensor(int(y_adv), device=x_t.device), C).float()

        # ---- target side -------------------------------------------------- #
        xt = x_t.unsqueeze(0)
        h_t = core.embed(xt)[0]
        z_t = core(xt)[0]
        r_t = torch.softmax(z_t, 0) - e_adv
        u_t = W.t() @ r_t                                   # dL_t/dh_t
        y_t = torch.full((1,), int(y_adv), dtype=torch.long, device=xt.device)

        ce_one = lambda z: F.cross_entropy(
            z, torch.full((len(z),), int(y_adv), dtype=torch.long, device=z.device),
            reduction='none')

        # grad_phi L_t. Equal to J_t^T u_t, so it is also the tangent for jker.
        T_loss = backbone_grad(logit_wrap, backbone, fixed, buffers, xt, ce_one)

        # Hutchinson probes: tangent J_t^T v for Rademacher v in R^d.
        d = h_t.numel()
        probes, T_probe = [], []
        for _ in range(args.ntk_probes):
            v = (torch.randint(0, 2, (d,), generator=rng, device=x_t.device)
                 .float() * 2 - 1)
            probes.append(v)
            T_probe.append(backbone_grad(embed_wrap, backbone, fixed, buffers, xt,
                                         lambda h, v=v: h @ v))

        # ---- candidate side, batched -------------------------------------- #
        N = len(cand)
        out = {k: torch.empty(N, device=x_t.device)
               for k in ('hTh', 'cos', 'd_cos', 'd_l2', 'h_norm', 'margin',
                         'p_adv', 'r_norm', 'r_dot', 'A', 'jker')}
        ntk_raw = torch.empty(args.ntk_probes, N, device=x_t.device) \
            if args.ntk_probes else None
        backend = None

        for s in range(0, N, args.batch_size):
            b = cand[s:s + args.batch_size]
            sl = slice(s, s + len(b))
            h = core.embed(b)
            z = core(b)
            p = torch.softmax(z, 1)
            r = p - e_adv.unsqueeze(0)

            out['hTh'][sl] = h @ h_t
            out['h_norm'][sl] = h.norm(dim=1)
            out['cos'][sl] = F.cosine_similarity(h, h_t.unsqueeze(0).expand_as(h), dim=1)
            out['d_cos'][sl] = 1.0 - out['cos'][sl]
            out['d_l2'][sl] = ((h - h_t.unsqueeze(0)) ** 2).sum(1)
            zo = z.clone()
            zo[:, int(y_adv)] = float('-inf')
            out['margin'][sl] = z[:, int(y_adv)] - zo.max(1).values
            out['p_adv'][sl] = p[:, int(y_adv)]
            out['r_norm'][sl] = r.norm(dim=1)
            out['r_dot'][sl] = r @ r_t

            a, backend = per_sample_dirderiv(logit_wrap, backbone, fixed, buffers,
                                             b, T_loss, ce_one)
            out['A'][sl] = a
            j, _ = per_sample_dirderiv(embed_wrap, backbone, fixed, buffers, b,
                                       T_loss, lambda hh: hh @ u_t)
            out['jker'][sl] = j
            for pi, (v, Tv) in enumerate(zip(probes, T_probe)):
                q, _ = per_sample_dirderiv(embed_wrap, backbone, fixed, buffers, b,
                                           Tv, lambda hh, v=v: hh @ v)
                ntk_raw[pi, sl] = q

        # ---- Eq. (3) closed forms ----------------------------------------- #
        out['head_W'] = out['r_dot'] * out['hTh']
        out['head_b'] = out['r_dot'].clone() if has_bias else torch.zeros_like(out['r_dot'])
        out['g_full'] = out['head_W'] + out['head_b'] + out['A']
        if args.ntk_probes:
            out['ntk_tr'] = ntk_raw.mean(0)
            out['ntk_tr_se'] = (ntk_raw.std(0) / math.sqrt(args.ntk_probes)
                                if args.ntk_probes > 1
                                else torch.zeros(N, device=x_t.device))
        else:
            out['ntk_tr'] = torch.full((N,), float('nan'), device=x_t.device)
            out['ntk_tr_se'] = torch.full((N,), float('nan'), device=x_t.device)

        # ---- brute-force identity check, in float64 ------------------------ #
        #
        # Two things get checked, and they must be kept apart:
        #   (a) does the JVP reproduce <grad_phi l_i, grad_phi L_t>, and
        #   (b) does  g_full == head_W + head_b + A  hold at all.
        # Both are run at double precision on a subsample: in float32 the JVP
        # carries ~1e-4 relative noise on a conv backbone, which is harmless for
        # the rankings and correlations this script exists to measure but would
        # drown a machine-precision identity claim. Errors are reported against
        # the RMS of |g_full| over the checked subsample -- a per-element
        # relative error is meaningless for candidates whose alignment is ~0.
        checks = dict(verify_n=0, identity_max_abs_err=float('nan'),
                      identity_max_rel_err=float('nan'),
                      identity_scale=float('nan'),
                      A_f32_vs_f64_max_rel=float('nan'))
        if args.verify:
            import copy
            core64 = copy.deepcopy(core).double().eval()
            for p in core64.parameters():
                p.requires_grad_(True)
            lw64 = _LogitWrap(core64).eval()
            bb64, fx64, buf64 = split_params(core64, lw64)
            xt64 = xt.double()
            T64 = backbone_grad(lw64, bb64, fx64, buf64, xt64, ce_one)
            gt64 = full_param_grad(core64, xt64, y_t)

            k = min(args.verify, N)
            pick = torch.randperm(N, generator=rng, device=cand.device)[:k].tolist()
            exact_v, claim_v, a32_v, a64_v = [], [], [], []
            for i in pick:
                xi = cand[i:i + 1].double()
                yi = torch.full((1,), int(y_adv), dtype=torch.long, device=xi.device)
                a64, _ = per_sample_dirderiv(lw64, bb64, fx64, buf64, xi, T64, ce_one)
                with torch.no_grad():
                    hi = core64.embed(xi)[0]
                    zi = core64(xi)[0]
                    ri = torch.softmax(zi, 0) - e_adv.double()
                    ht64 = core64.embed(xt64)[0]
                    rt64 = torch.softmax(core64(xt64)[0], 0) - e_adv.double()
                    rdot = float(ri @ rt64)
                claim = rdot * float(hi @ ht64) + (rdot if has_bias else 0.0) + float(a64)
                exact_v.append(float(torch.dot(full_param_grad(core64, xi, yi), gt64)))
                claim_v.append(claim)
                a32_v.append(float(out['A'][i]))
                a64_v.append(float(a64))
            # keep the float64 reference values themselves, not only their error:
            # these are the exact numbers to quote for a machine-precision claim
            checks['verify_idx'] = np.asarray(pick, np.int64)
            checks['verify_A_f64'] = np.asarray(a64_v, np.float64)
            checks['verify_gfull_brute_f64'] = np.asarray(exact_v, np.float64)
            checks['verify_gfull_claim_f64'] = np.asarray(claim_v, np.float64)

            ex = np.asarray(exact_v)
            scale = float(np.sqrt(np.mean(ex ** 2))) if len(ex) else float('nan')
            denom = max(scale, 1e-30)
            checks.update(
                verify_n=len(pick),
                identity_scale=scale,
                identity_max_abs_err=float(np.max(np.abs(ex - np.asarray(claim_v))))
                if len(ex) else float('nan'),
                identity_max_rel_err=float(np.max(np.abs(ex - np.asarray(claim_v))))
                / denom if len(ex) else float('nan'),
                A_f32_vs_f64_max_rel=float(np.max(
                    np.abs(np.asarray(a32_v) - np.asarray(a64_v)))) / denom
                if len(ex) else float('nan'))
            del core64, lw64, bb64, fx64, buf64, T64, gt64

        scalars = {
            'h_t_norm': float(h_t.norm()), 'r_t_norm': float(r_t.norm()),
            'p_adv_t': float(torch.softmax(z_t, 0)[int(y_adv)]),
            'loss_t': float(F.cross_entropy(z_t.unsqueeze(0), y_t)),
            'feat_dim': int(d), 'has_bias': bool(has_bias),
            'backend': backend or 'n/a',
        }
        scalars.update(checks)
        return {k: v.detach().float().cpu().numpy() for k, v in out.items()}, scalars
    finally:
        core.train(was_training)


# --------------------------------------------------------------------------- #
# driver
# --------------------------------------------------------------------------- #

def corr(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 3 or a[ok].std() == 0 or b[ok].std() == 0:
        return float('nan'), float('nan')
    p = float(np.corrcoef(a[ok], b[ok])[0, 1])
    s = float(spearmanr(a[ok], b[ok]).statistic) if spearmanr else float('nan')
    return p, s


def topm_overlap(x, y, m, x_lower_better, y_lower_better):
    if m <= 0 or m > len(x):
        return float('nan')
    xa = np.argsort(x if x_lower_better else -x)[:m]
    ya = np.argsort(y if y_lower_better else -y)[:m]
    return len(set(xa.tolist()) & set(ya.tolist())) / float(m)


def main(args):
    device = torch.device(args.device if torch.cuda.is_available()
                          or args.device == 'cpu' else 'cpu')
    if device.type == 'cpu' and not args.allow_cpu:
        raise SystemExit('no CUDA device visible -- refusing to start (pass '
                         '--allow_cpu to override; this is very slow)')
    # cuDNN runs convolutions in TF32 by default on Ampere and later, which costs
    # ~1e-2 relative accuracy on A -- measured against the float64 reference, TF32
    # gives 2.3e-2 where true float32 gives 3.5e-4. Harmless for a ranking, not
    # harmless for a table that claims to report an exact quantity.
    if not args.allow_tf32:
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cuda.matmul.allow_tf32 = False
    log('device %s | torch %s | tf32 %s'
        % (device, torch.__version__, 'on' if args.allow_tf32 else 'off'))

    channel, im_size, num_classes, class_names, _, _, dst_train, dst_test, _ = \
        get_dataset(args.dataset, args.data_path)
    y_adv, target_class = parse_pair(args.class_pair, class_names, args.pair_order)
    log('%s / %s / %s -> poison class %d (%s), target class %d (%s)'
        % (args.dataset, args.model, args.class_pair, y_adv, class_names[y_adv],
           target_class, class_names[target_class]))

    train_labs = labels_of(dst_train)
    pool = (train_labs == y_adv).nonzero(as_tuple=True)[0]
    rng = torch.Generator(device='cpu').manual_seed(args.seed)
    if args.max_candidates and len(pool) > args.max_candidates:
        keep = torch.randperm(len(pool), generator=rng)[:args.max_candidates]
        pool = pool[keep.sort().values]
    cand_idx = pool.tolist()
    log('candidate pool: %d images labelled %s' % (len(cand_idx), class_names[y_adv]))

    tidx, tsrc = load_targets(args)
    log('targets (%d) from %s: %s' % (len(tidx), tsrc, tidx))

    # Surrogates: loaded straight out of the shared cache, same helper and same
    # hyper-parameters as the attack runs, so these are literally the nets the
    # paper's selections used.
    sur_args = argparse.Namespace(
        dataset=args.dataset, model=args.model, cache_dir=args.cache_dir,
        seed=args.seed, num_surrogates=args.num_surrogates,
        surrogate_epochs=args.surrogate_epochs, surrogate_lr=args.surrogate_lr,
        surrogate_bs=args.surrogate_bs, surrogate_decay=args.surrogate_decay,
        surrogate_wd=args.surrogate_wd, surrogate_aug=False,
        dsa_strategy='color_crop_cutout_flip_scale_rotate')
    from final_update import surrogate_dir
    sdir = surrogate_dir(sur_args)
    missing = [i for i in range(args.num_surrogates)
               if not os.path.isfile(os.path.join(sdir, 'net_%d.pt' % i))]
    log('surrogate pool %s: %d/%d cached' %
        (sdir, args.num_surrogates - len(missing), args.num_surrogates))
    if missing and not args.train_missing:
        raise SystemExit('missing surrogates %s in %s; pass --train_missing to train '
                         'them (needs the full training set on device)'
                         % (missing[:8], sdir))

    if missing:
        from final_update import stack_dataset
        tr_i, tr_l = stack_dataset(dst_train, device)
        te_i, te_l = stack_dataset(dst_test, device)
    else:
        tr_i = tr_l = te_i = te_l = None
    nets = get_surrogates(sur_args, tr_i, tr_l, te_i, te_l, channel, num_classes,
                          im_size, device, ParamDiffAug())
    del tr_i, tr_l, te_i, te_l
    if args.surrogate_ids:
        nets = [nets[i] for i in args.surrogate_ids]
    K = len(nets)
    for n in nets:
        n.eval()
        for p in n.parameters():
            p.requires_grad_(False)
    log('loaded %d surrogates' % K)

    combo = '%s_%s_%s' % (args.dataset, args.model, args.class_pair)
    outdir = os.path.join(args.out_dir, combo)
    os.makedirs(outdir, exist_ok=True)

    cand = images_at(dst_train, cand_idx, device)
    N = len(cand_idx)
    m_top = args.top_m or max(1, int(round(args.budget * len(train_labs))))
    log('top-m for the overlap columns: m = %d' % m_top)

    FIELDS = ['hTh', 'cos', 'd_cos', 'd_l2', 'h_norm', 'margin', 'p_adv', 'r_norm',
              'r_dot', 'A', 'jker', 'ntk_tr', 'ntk_tr_se', 'head_W', 'head_b',
              'g_full']
    summary_rows = []

    for t_pos, ti in enumerate(tidx):
        x_t = images_at(dst_test, [ti], device)[0]
        store = {f: np.empty((K, N), np.float32) for f in FIELDS}
        zc = {f: np.empty((K, N), np.float32) for f in ('z_dcos', 'z_margin', 'z_A',
                                                        'z_hTh', 'z_gfull')}
        per_net = []
        for k, net in enumerate(nets):
            t1 = time.time()
            vals, sc = measure(net, cand, x_t, y_adv, args,
                               torch.Generator(device=device).manual_seed(
                                   args.seed + 7919 * k + ti))
            for f in FIELDS:
                store[f][k] = vals[f]
            for src, dst_key in (('d_cos', 'z_dcos'), ('margin', 'z_margin'),
                                 ('A', 'z_A'), ('hTh', 'z_hTh'), ('g_full', 'z_gfull')):
                v = torch.from_numpy(vals[src])
                zc[dst_key][k] = standardize(v).numpy()
            sc.update(surrogate=k, target=ti, seconds=round(time.time() - t1, 2))
            per_net.append(sc)
            log('  target %d (%d/%d) surrogate %2d/%d | %s | identity err %.2e '
                '(rel to rms %.3g over n=%d) | float32 A err %.2e | %.1fs'
                % (ti, t_pos + 1, len(tidx), k + 1, K, sc['backend'],
                   sc['identity_max_abs_err'], sc['identity_max_rel_err'],
                   sc['verify_n'], sc['A_f32_vs_f64_max_rel'], sc['seconds']))

        # ensemble scores -------------------------------------------------- #
        cost_b0 = (zc['z_dcos'] + args.lam * zc['z_margin']).mean(0)
        cost_b1 = (zc['z_dcos'] + args.lam * zc['z_margin']
                   - args.beta * zc['z_A']).mean(0)
        s_paper_b0 = (zc['z_hTh'] - zc['z_margin']).mean(0)
        s_paper_b1 = (zc['z_hTh'] - zc['z_margin'] + args.beta * zc['z_A']).mean(0)
        s_eq3 = zc['z_gfull'].mean(0)

        v64 = {}
        for name in ('verify_idx', 'verify_A_f64', 'verify_gfull_brute_f64',
                     'verify_gfull_claim_f64'):
            if name in per_net[0]:
                v64[name] = np.stack([p[name] for p in per_net])

        np.savez_compressed(
            os.path.join(outdir, 'target_%d.npz' % ti),
            cand_idx=np.asarray(cand_idx, np.int64), **v64,
            target_idx=np.int64(ti), y_adv=np.int64(y_adv),
            target_class=np.int64(target_class),
            cost_prod_b0=cost_b0.astype(np.float32),
            cost_prod_b1=cost_b1.astype(np.float32),
            s_paper_b0=s_paper_b0.astype(np.float32),
            s_paper_b1=s_paper_b1.astype(np.float32),
            s_eq3=s_eq3.astype(np.float32),
            **store, **zc)

        if args.csv:
            import gzip
            path = os.path.join(outdir, 'target_%d.csv.gz' % ti)
            with gzip.open(path, 'wt', newline='') as fh:
                w = csv.writer(fh)
                w.writerow(['target_idx', 'surrogate', 'cand_idx'] + FIELDS)
                for k in range(K):
                    for i in range(N):
                        w.writerow([ti, k, cand_idx[i]] +
                                   ['%.8g' % store[f][k, i] for f in FIELDS])

        # per-(target, surrogate) diagnostics ------------------------------- #
        for k in range(K):
            row = dict(dataset=args.dataset, model=args.model,
                       class_pair=args.class_pair, target_idx=ti, surrogate=k,
                       n_candidates=N, **{kk: per_net[k][kk] for kk in
                                          ('identity_max_abs_err',
                                           'identity_max_rel_err',
                                           'identity_scale', 'verify_n',
                                           'A_f32_vs_f64_max_rel', 'feat_dim',
                                           'has_bias', 'backend', 'h_t_norm',
                                           'r_t_norm', 'p_adv_t', 'loss_t')})
            A, gf = store['A'][k], store['g_full'][k]
            hw, hth, ntk = store['head_W'][k], store['hTh'][k], store['ntk_tr'][k]
            mg, rn, rd = store['margin'][k], store['r_norm'][k], store['r_dot'][k]
            row['frac_A_of_gfull'] = float(np.mean(np.abs(A) /
                                                   np.maximum(np.abs(gf), 1e-30)))
            row['frac_head_of_gfull'] = float(np.mean(np.abs(hw) /
                                                      np.maximum(np.abs(gf), 1e-30)))
            for name, (a, b) in {
                    'A_vs_gfull': (A, gf), 'headW_vs_gfull': (hw, gf),
                    'hTh_vs_gfull': (hth, gf), 'hTh_vs_A': (hth, A),
                    'hTh_vs_ntktr': (hth, ntk), 'ntktr_vs_A': (ntk, A),
                    'margin_vs_rnorm': (mg, rn), 'margin_vs_rdot': (mg, rd),
                    'dcos_vs_hTh': (store['d_cos'][k], hth),
            }.items():
                p, s = corr(a, b)
                row['pearson_' + name] = p
                row['spearman_' + name] = s
            summary_rows.append(row)

        # top-m agreement between the three rankings ------------------------ #
        summary_rows.append(dict(
            dataset=args.dataset, model=args.model, class_pair=args.class_pair,
            target_idx=ti, surrogate='ENSEMBLE', n_candidates=N, top_m=m_top,
            overlap_prod_b1_vs_eq3=topm_overlap(cost_b1, s_eq3, m_top, True, False),
            overlap_prod_b0_vs_eq3=topm_overlap(cost_b0, s_eq3, m_top, True, False),
            overlap_paper_b1_vs_eq3=topm_overlap(s_paper_b1, s_eq3, m_top, False, False),
            overlap_prod_b1_vs_paper_b1=topm_overlap(cost_b1, s_paper_b1, m_top,
                                                     True, False),
            overlap_prod_b0_vs_prod_b1=topm_overlap(cost_b0, cost_b1, m_top,
                                                    True, True),
            pearson_costb1_vs_eq3=corr(-cost_b1, s_eq3)[0],
            spearman_costb1_vs_eq3=corr(-cost_b1, s_eq3)[1],
        ))

    keys = sorted({k for r in summary_rows for k in r})
    with open(os.path.join(outdir, 'summary.csv'), 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        for r in summary_rows:
            w.writerow(r)

    meta = dict(vars(args))
    meta.update(y_adv=int(y_adv), target_class=int(target_class),
                class_names_pair=[class_names[y_adv], class_names[target_class]],
                targets=tidx, target_source=tsrc, n_candidates=N,
                n_surrogates=K, surrogate_dir=sdir, top_m=m_top,
                torch=torch.__version__, finished=time.strftime('%Y-%m-%d %H:%M:%S'))
    with open(os.path.join(outdir, 'meta.json'), 'w') as fh:
        json.dump(meta, fh, indent=1, default=str)
    os.makedirs(args.out_dir, exist_ok=True)
    with open(os.path.join(args.out_dir, 'manifest.jsonl'), 'a') as fh:
        fh.write(json.dumps({k: meta[k] for k in
                             ('dataset', 'model', 'class_pair', 'targets',
                              'n_candidates', 'n_surrogates', 'finished')},
                            default=str) + '\n')
    log('wrote %s (%d targets x %d surrogates x %d candidates)'
        % (outdir, len(tidx), K, N))


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--dataset', default='CIFAR10')
    p.add_argument('--data_path', default='/home/mmoslem3/scratch/data')
    p.add_argument('--model', default='ConvNetBN')
    p.add_argument('--class_pair', default='dog-bird',
                   help="'<adversarial>-<target>' under the default --pair_order")
    p.add_argument('--pair_order', default='poison-target',
                   choices=['poison-target', 'target-poison'])
    p.add_argument('--attack', default='fc',
                   help='only used to pick the pinned target set for CIFAR-10 '
                        '(fc and gradmatch have different target sets; sapa shares '
                        "gradmatch's)")

    p.add_argument('--cache_dir', default='./cache')
    p.add_argument('--target_sets', default='./target_sets')
    p.add_argument('--target_idx_file', default=None)
    p.add_argument('--target_idx', nargs='*', type=int, default=None)
    p.add_argument('--num_targets', type=int, default=0,
                   help='0 = every target in the pinned file')
    p.add_argument('--out_dir', default='./alignment_probe_result')

    p.add_argument('--num_surrogates', type=int, default=20)
    p.add_argument('--surrogate_ids', nargs='*', type=int, default=None)
    p.add_argument('--surrogate_epochs', type=int, default=60)
    p.add_argument('--surrogate_lr', type=float, default=0.1)
    p.add_argument('--surrogate_bs', type=int, default=128)
    p.add_argument('--surrogate_decay', nargs='+', type=int, default=[35, 45])
    p.add_argument('--surrogate_wd', type=float, default=0.0)
    p.add_argument('--train_missing', action='store_true',
                   help='train and cache any surrogate that is not on disk yet')

    p.add_argument('--batch_size', type=int, default=64,
                   help='candidate batch for the JVP passes; memory only')
    p.add_argument('--ntk_probes', type=int, default=8,
                   help='Rademacher probes for the tr(J_i J_t^T) estimate; 0 skips it')
    p.add_argument('--verify', type=int, default=8,
                   help='candidates per (target, surrogate) re-done at float64 and '
                        'checked against a brute-force full-parameter gradient dot; '
                        '0 skips it')
    p.add_argument('--max_candidates', type=int, default=0,
                   help='subsample the candidate pool; 0 = all of it')

    p.add_argument('--lam', type=float, default=1.0, help='lambda_margin')
    p.add_argument('--beta', type=float, default=1.0, help='Jacobian weight')
    p.add_argument('--budget', type=float, default=0.005,
                   help='poison ratio used to derive top-m for the overlap columns')
    p.add_argument('--top_m', type=int, default=0, help='overrides --budget')

    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--device', default='cuda')
    p.add_argument('--allow_cpu', action='store_true')
    p.add_argument('--allow_tf32', action='store_true',
                   help='keep cuDNN/cuBLAS TF32 on. Off by default: TF32 costs ~1e-2 '
                        'relative accuracy on A, which the float64 check will report')
    p.add_argument('--csv', action='store_true',
                   help='also emit long-format target_<idx>.csv.gz')
    return p.parse_args()


if __name__ == '__main__':
    main(parse_args())
