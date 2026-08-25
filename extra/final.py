#!/usr/bin/env python
"""
poison_eval.py

Targeted clean-label poisoning evaluated in the standard train-from-scratch
setting, under the MetaPoison victim protocol. No dataset distillation.

Crafting objectives (--attack):
  fc         feature collision. Default --fc_mode sample is the PER-SAMPLE
             collision (Poison Frogs style), which is what actually transfers to
             a from-scratch victim. --fc_mode bullseye is the mean-pooled
             Bullseye Polytope objective (target-norm normalized), kept only for
             comparison; it dilutes the per-sample signal by 1/m and is meant
             for frozen-backbone transfer learning, not from-scratch training.
  gradmatch  gradient matching (Witches' Brew, Geiping et al. 2020):
             1 - cos(grad_theta CE(x_t, y_adv), grad_theta CE(poisons, y_adv)),
             ensemble averaged, signed Adam on delta, DiffAugment per step,
             R random restarts, best delta over all (restart, step) pairs.

Base selection (--base):
  random     uniform over the poison class.
  ours       standardized d(x) + lambda * M(x), lowest first, ensemble averaged.
             d = feature distance to the target (l2 or cosine), M = logit margin
             toward y_adv. Low score means close to the target in feature space
             AND sitting near the y_adv decision boundary.

Class pair naming follows MetaPoison: 'dog-bird' means poisons are drawn from
'dog' (y_adv) and the target image is a 'bird'. Use --pair_order target-poison
to flip that if you need to reproduce numbers from a codebase that reads it the
other way.

Everything is GPU resident and manually batched, no DataLoader in the hot loop.

Expects, next to this file:
  utils.py     get_dataset, DiffAugment, ParamDiffAug (get_time optional)
  networks.py  ConvNet, VGG, ResNet20BN (or utils.get_network as a fallback)

Example:
  python poison_eval.py --model ResNet20BN --attack gradmatch --base ours \
      --class_pair dog-bird --budget 0.01 --epsilon 0.0627451 \
      --craft_steps 250 --craft_alpha 0.0039216 --restarts 8 \
      --num_surrogates 5 --num_targets 10 --num_victims 6 \
      --victim_epochs 60 --victim_decay 35 45 --clean_baseline \
      --cache_dir ./cache --out_dir ./ours_result
"""

import argparse
import csv
import json
import os
import time
import warnings
from collections import defaultdict

warnings.filterwarnings('ignore', category=UserWarning)

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from _old_.utils import get_dataset, DiffAugment, ParamDiffAug

try:
    from _old_.utils import get_time
except ImportError:
    def get_time():
        return time.strftime('[%Y-%m-%d %H:%M:%S]')

try:
    from _old_.utils import get_network as _get_network
except ImportError:
    _get_network = None

try:
    from _old_.networks import ConvNet as _ConvNet
except Exception:
    _ConvNet = None
try:
    from _old_.networks import VGG as _VGG
except Exception:
    _VGG = None
try:
    from _old_.networks import ResNet20BN as _ResNet20BN
except Exception:
    _ResNet20BN = None


SUPPORTED_MODELS = ['ConvNetBN', 'VGG13BN', 'ResNet20BN']
CLASS_PAIRS = ['dog-bird', 'frog-airplane']

_LOG_PATH = None


def log(msg):
    line = '%s %s' % (get_time(), msg)
    print(line, flush=True)
    if _LOG_PATH:
        with open(_LOG_PATH, 'a') as f:
            f.write(line + '\n')


# --------------------------------------------------------------------------- #
# small helpers
# --------------------------------------------------------------------------- #

def embed_of(net):
    return net.module.embed if isinstance(net, nn.DataParallel) else net.embed


def standardize(v, eps=1e-8):
    return (v - v.mean()) / (v.std() + eps)


def flat_grad(grads):
    return torch.cat([g.reshape(-1) for g in grads])


def cosine(a, b, eps=1e-8):
    return torch.dot(a, b) / (a.norm() * b.norm() + eps)


def stack_dataset(dst, device):
    """Materialize a torchvision-style dataset as GPU tensors (already normalized)."""
    imgs = torch.stack([dst[i][0] for i in range(len(dst))]).to(device)
    labs = torch.tensor([dst[i][1] for i in range(len(dst))],
                        dtype=torch.long, device=device)
    return imgs, labs


def parse_pair(pair, class_names, order='poison-target'):
    a, b = pair.split('-')
    if order == 'poison-target':
        return class_names.index(a), class_names.index(b)   # y_adv, target_class
    return class_names.index(b), class_names.index(a)


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_network(name, channel, num_classes, im_size, device, seed=None):
    if seed is not None:
        set_seed(seed)
    if name == 'ConvNetBN' and _ConvNet is not None:
        net = _ConvNet(channel=channel, num_classes=num_classes, net_width=128,
                       net_depth=3, net_act='relu', net_norm='batchnorm',
                       net_pooling='avgpooling', im_size=im_size)
    elif name == 'VGG13BN' and _VGG is not None:
        net = _VGG('VGG13', channel=channel, num_classes=num_classes, norm='batchnorm')
    elif name == 'ResNet20BN' and _ResNet20BN is not None:
        net = _ResNet20BN(channel=channel, num_classes=num_classes)
    elif _get_network is not None:
        net = _get_network(name, channel, num_classes, im_size)
    else:
        raise ValueError('cannot build network %s: no factory available' % name)
    return net.to(device)


def set_requires_grad(nets, flag):
    for n in nets:
        for p in n.parameters():
            p.requires_grad_(flag)


# --------------------------------------------------------------------------- #
# training / evaluation (GPU resident, manual batching)
# --------------------------------------------------------------------------- #

def train_from_scratch(net, images, labels, epochs, lr, bs, decay_at, device,
                       weight_decay=0.0, aug=False, dsa_strategy=None, dsa_param=None):
    net.train()
    opt = torch.optim.SGD(net.parameters(), lr=lr, momentum=0.9,
                          weight_decay=weight_decay)
    crit = nn.CrossEntropyLoss().to(device)
    N = images.shape[0]
    cur_lr = lr
    decay_at = set(decay_at or [])
    for ep in range(epochs):
        if ep in decay_at:
            cur_lr *= 0.1
            for g in opt.param_groups:
                g['lr'] = cur_lr
        perm = torch.randperm(N, device=device)
        for i in range(0, N, bs):
            idx = perm[i:i + bs]
            img = images[idx]
            lab = labels[idx]
            if aug and dsa_strategy:
                img = DiffAugment(img, dsa_strategy, param=dsa_param)
            opt.zero_grad(set_to_none=True)
            loss = crit(net(img), lab)
            loss.backward()
            opt.step()
    net.eval()
    return net


@torch.no_grad()
def test_acc(net, images, labels, bs=1024):
    net.eval()
    c = 0
    for i in range(0, len(images), bs):
        c += (net(images[i:i + bs]).argmax(1) == labels[i:i + bs]).sum().item()
    return c / len(images)


@torch.no_grad()
def predict_target(net, x_t_norm):
    net.eval()
    return int(net(x_t_norm.unsqueeze(0)).argmax(1).item())


# --------------------------------------------------------------------------- #
# cached model pools (surrogates + clean victims), both on the real full set
# --------------------------------------------------------------------------- #

def surrogate_dir(args):
    return os.path.join(args.cache_dir, 'surrogates',
                        '%s_%dep_lr%g_bs%d_seed%d'
                        % (args.model, args.surrogate_epochs, args.surrogate_lr,
                           args.surrogate_bs, args.seed))


def victim_dir(args):
    return os.path.join(args.cache_dir, 'clean_victims',
                        '%s_%dep_lr%g_bs%d_wd%g_seed%d'
                        % (args.model, args.victim_epochs, args.victim_lr,
                           args.victim_bs, args.victim_wd, args.seed))


def _load_or_train(path, model_name, seed, train_imgs, train_labs, test_imgs, test_labs,
                   channel, num_classes, im_size, device, epochs, lr, bs, decay, wd,
                   aug, dsa_strategy, dsa_param, tag):
    net = build_network(model_name, channel, num_classes, im_size, device, seed=seed)
    if os.path.exists(path):
        net.load_state_dict(torch.load(path, map_location=device))
        net.eval()
        return net, None
    t0 = time.time()
    net = train_from_scratch(net, train_imgs, train_labs, epochs, lr, bs, decay,
                             device, weight_decay=wd, aug=aug,
                             dsa_strategy=dsa_strategy, dsa_param=dsa_param)
    acc = test_acc(net, test_imgs, test_labs)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(net.state_dict(), path)
    log('  trained %s: test acc = %.4f (%.0f s)' % (tag, acc, time.time() - t0))
    return net, acc


def get_surrogates(args, train_imgs, train_labs, test_imgs, test_labs,
                   channel, num_classes, im_size, device, dsa_param, only_id=None):
    d = surrogate_dir(args)
    ids = [only_id] if only_id is not None else range(args.num_surrogates)
    nets = []
    for i in ids:
        path = os.path.join(d, 'net_%d.pt' % i)
        net, _ = _load_or_train(
            path, args.model, args.seed + 1000 + i, train_imgs, train_labs,
            test_imgs, test_labs, channel, num_classes, im_size, device,
            args.surrogate_epochs, args.surrogate_lr, args.surrogate_bs,
            args.surrogate_decay, args.surrogate_wd, args.surrogate_aug,
            args.dsa_strategy, dsa_param, 'surrogate %d (%s)' % (i, args.model))
        net.eval()
        nets.append(net)
    return nets


def get_clean_victims(args, train_imgs, train_labs, test_imgs, test_labs,
                      channel, num_classes, im_size, device, dsa_param, only_id=None):
    d = victim_dir(args)
    ids = [only_id] if only_id is not None else range(args.num_victims)
    nets = []
    for i in ids:
        path = os.path.join(d, 'net_%d.pt' % i)
        net, _ = _load_or_train(
            path, args.model, args.seed + 900000 + i, train_imgs, train_labs,
            test_imgs, test_labs, channel, num_classes, im_size, device,
            args.victim_epochs, args.victim_lr, args.victim_bs,
            args.victim_decay, args.victim_wd, args.victim_aug,
            args.dsa_strategy, dsa_param, 'clean victim %d (%s)' % (i, args.model))
        net.eval()
        nets.append(net)
    return nets


# --------------------------------------------------------------------------- #
# target selection
# --------------------------------------------------------------------------- #

@torch.no_grad()
def ensemble_probs(nets, images, idx, bs=512):
    out = []
    for i in range(0, len(idx), bs):
        x = images[idx[i:i + bs]]
        s = None
        for n in nets:
            n.eval()
            p = F.softmax(n(x), dim=1)
            s = p if s is None else s + p
        out.append((s / len(nets)).cpu())
    return torch.cat(out)


def target_select_arg(s):
    """'easiest' | 'hardest' | 'random' | 'first', or a difficulty degree 0..100."""
    s = str(s).strip().lower()
    if s in ('easiest', 'hardest', 'random', 'first'):
        return s
    try:
        v = int(s)
    except ValueError:
        raise argparse.ArgumentTypeError(
            "target_select must be easiest/hardest/random/first or an integer 0..100")
    if not 0 <= v <= 100:
        raise argparse.ArgumentTypeError('target_select degree must be in [0, 100]')
    return v


def difficulty_degree(target_select):
    """Degree on the 0 (easiest) .. 100 (hardest) scale, or None for random/first."""
    if isinstance(target_select, int):
        return target_select
    return {'easiest': 0, 'hardest': 100}.get(target_select)


def select_targets(args, nets, test_imgs, test_labs, y_adv, target_class, gen):
    pool = (test_labs == target_class).nonzero(as_tuple=True)[0].cpu()

    # Eligibility is enforced in every mode: a target the clean ensemble already
    # predicts as y_adv is a free win and must never be selected.
    probs = ensemble_probs(nets, test_imgs, pool)
    p_adv = probs[:, y_adv]
    pred = probs.argmax(1)

    keep = (pred != y_adv)
    if args.require_correct_target:
        keep &= (pred == target_class)
    kept = keep.nonzero(as_tuple=True)[0]          # positions into pool
    if len(kept) == 0:
        raise RuntimeError('no eligible targets left; relax --require_correct_target')
    log('  target pool %d -> eligible %d (%d already predicted as y_adv or dropped)'
        % (len(pool), len(kept), len(pool) - len(kept)))

    def finish(order, how):
        chosen = pool[order].tolist()
        scores = {int(pool[i]): float(p_adv[i]) for i in order}
        log('  chosen %d targets (%s), p_adv range %.4f..%.4f'
            % (len(chosen), how, min(scores.values()), max(scores.values())))
        return chosen, scores

    if args.target_idx_file:
        with open(args.target_idx_file) as f:
            blob = json.load(f)
        key = args.class_pair
        want = (blob['pairs'][key]['indices'] if 'pairs' in blob else blob[key])
        pos = {int(pool[i]): int(i) for i in kept}
        order = [pos[int(i)] for i in want if int(i) in pos][:args.num_targets]
        if not order:
            raise RuntimeError('no eligible targets in %s for pair %s'
                               % (args.target_idx_file, key))
        if len(order) < min(len(want), args.num_targets):
            log('  warning: dropped %d free-win targets from %s'
                % (min(len(want), args.num_targets) - len(order), args.target_idx_file))
        return finish(torch.tensor(order), 'file')

    if args.target_select == 'first':
        return finish(kept[:args.num_targets], 'first')
    if args.target_select == 'random':
        perm = torch.randperm(len(kept), generator=gen)[:args.num_targets]
        return finish(kept[perm], 'random')

    # difficulty degree: 0 = easiest (highest clean p_adv) .. 100 = hardest.
    # Rank the eligible pool easiest-first and slide a window of num_targets
    # across it, so the degree is the window's percentile position.
    deg = difficulty_degree(args.target_select)
    ranked = kept[torch.argsort(p_adv[kept], descending=True)]
    k = min(args.num_targets, len(ranked))
    start = int(round(deg / 100.0 * (len(ranked) - k)))
    return finish(ranked[start:start + k],
                  'degree %d, rank %d-%d of %d' % (deg, start, start + k - 1, len(ranked)))


# --------------------------------------------------------------------------- #
# base selection
# --------------------------------------------------------------------------- #

@torch.no_grad()
def select_base_ours(nets, images_norm, labels, x_t_norm, y_adv, N_p, lam, device,
                     base_dist='l2', bs=512):
    cls_idx = (labels == y_adv).nonzero(as_tuple=True)[0]
    if len(cls_idx) < N_p:
        raise ValueError('class %d has %d images < N_p=%d' % (y_adv, len(cls_idx), N_p))
    cand = images_norm[cls_idx]
    score = torch.zeros(len(cls_idx), device=device)
    for net in nets:
        net.eval()
        emb = embed_of(net)
        f_t = emb(x_t_norm.unsqueeze(0))
        ds, ms = [], []
        for i in range(0, len(cand), bs):
            b = cand[i:i + bs]
            fb = emb(b)
            if base_dist == 'cosine':
                d = 1.0 - F.cosine_similarity(fb, f_t.expand(len(b), -1), dim=1)
            else:
                d = ((fb - f_t) ** 2).sum(dim=1)
            z = net(b)
            z_adv = z[:, y_adv].clone()
            z_o = z.clone()
            z_o[:, y_adv] = float('-inf')
            m = z_adv - z_o.max(dim=1).values
            ds.append(d)
            ms.append(m)
        score += standardize(torch.cat(ds)) + lam * standardize(torch.cat(ms))
    score /= len(nets)
    sel = torch.topk(score, k=N_p, largest=False).indices
    return cls_idx[sel]


def select_base_random(labels, y_adv, N_p, device, gen):
    cls_idx = (labels == y_adv).nonzero(as_tuple=True)[0]
    if len(cls_idx) < N_p:
        raise ValueError('class %d has %d images < N_p=%d' % (y_adv, len(cls_idx), N_p))
    perm = torch.randperm(len(cls_idx), generator=gen)[:N_p].to(device)
    return cls_idx[perm]


# --------------------------------------------------------------------------- #
# crafting: feature collision
# --------------------------------------------------------------------------- #

def craft_fc(nets, base01, x_t_norm, norm, eps, steps, alpha, device,
             restarts=1, mode='sample'):
    set_requires_grad(nets, False)
    for n in nets:
        n.eval()
    base01 = base01.detach()
    with torch.no_grad():
        f_tgts = [embed_of(n)(x_t_norm.unsqueeze(0)).detach() for n in nets]

    best_delta, best_obj = None, float('inf')
    for _ in range(max(1, restarts)):
        delta = torch.empty_like(base01).uniform_(-eps, eps)
        delta = (torch.clamp(base01 + delta, 0.0, 1.0) - base01).detach().requires_grad_(True)
        for _t in range(steps):
            x_adv_norm = norm(torch.clamp(base01 + delta, 0.0, 1.0))
            loss = 0.0
            for n, f_t in zip(nets, f_tgts):
                f = embed_of(n)(x_adv_norm)
                if mode == 'bullseye':
                    num = ((f.mean(dim=0, keepdim=True) - f_t) ** 2).sum()
                    loss = loss + num / (f_t ** 2).sum().clamp_min(1e-12)
                else:
                    loss = loss + F.mse_loss(f, f_t.expand_as(f))
            loss = loss / len(nets)
            obj_val = loss.item()
            if obj_val < best_obj:
                best_obj = obj_val
                best_delta = delta.detach().clone()
            grad = torch.autograd.grad(loss, delta)[0]
            with torch.no_grad():
                delta = delta - alpha * grad.sign()
                delta = delta.clamp_(-eps, eps)
                delta = torch.clamp(base01 + delta, 0.0, 1.0) - base01
            delta = delta.detach().requires_grad_(True)
        with torch.no_grad():
            x_adv_norm = norm(torch.clamp(base01 + delta, 0.0, 1.0))
            loss = 0.0
            for n, f_t in zip(nets, f_tgts):
                f = embed_of(n)(x_adv_norm)
                if mode == 'bullseye':
                    num = ((f.mean(dim=0, keepdim=True) - f_t) ** 2).sum()
                    loss = loss + num / (f_t ** 2).sum().clamp_min(1e-12)
                else:
                    loss = loss + F.mse_loss(f, f_t.expand_as(f))
            loss = (loss / len(nets)).item()
        if loss < best_obj:
            best_obj = loss
            best_delta = delta.detach().clone()

    return torch.clamp(base01 + best_delta, 0.0, 1.0), best_obj


# --------------------------------------------------------------------------- #
# crafting: gradient matching (Witches' Brew)
# --------------------------------------------------------------------------- #

def _gradmatch_net_grad(net, g_t, base01, delta, y_p, norm, crit, chunk,
                        use_dsa, dsa_strategy, dsa_param, seed):
    """d/d(delta) of (1 - cos(g_p, g_t)) for one surrogate, in poison micro-batches.

    Same value as the full-batch path, computed without ever holding the whole
    second-order graph.  g_p is the gradient of the *mean* poison loss, so each
    micro-batch is weighted by its share of the poison set.  The split is exact
    because the nets are in eval() (BN uses running stats, no cross-sample coupling)
    and a fixed DiffAugment seed makes the augmentation Siamese, i.e. identical for
    every image, hence identical for every micro-batch.
    """
    params = [p for p in net.parameters()]
    N = base01.shape[0]

    def chunk_loss(i0, i1, d):
        x = norm(torch.clamp(base01[i0:i1] + d, 0.0, 1.0))
        if use_dsa:
            x = DiffAugment(x, dsa_strategy, seed=seed, param=dsa_param)
        return crit(net(x), y_p[i0:i1]) * ((i1 - i0) / N)

    if chunk >= N:
        g_p = flat_grad(torch.autograd.grad(chunk_loss(0, N, delta), params,
                                            create_graph=True))
        obj = 1.0 - cosine(g_p, g_t)
        return torch.autograd.grad(obj, delta)[0], obj.item()

    # DiffAugment's Siamese branch (randb[:] = randb[0]) cannot handle a batch of 1,
    # so fold a trailing singleton into the chunk before it
    edges = list(range(0, N, chunk)) + [N]
    if edges[-1] - edges[-2] == 1:
        edges.pop(-2)
    spans = list(zip(edges[:-1], edges[1:]))

    # pass 1: accumulate g_p with no second-order graph
    g_p = None
    for i0, i1 in spans:
        g = flat_grad([gg.detach() for gg in
                       torch.autograd.grad(chunk_loss(i0, i1, delta[i0:i1].detach()),
                                           params)])
        g_p = g if g_p is None else g_p + g

    # freeze v = d(obj)/d(g_p) so pass 2 differentiates one scalar per micro-batch
    g_leaf = g_p.detach().requires_grad_(True)
    obj = 1.0 - cosine(g_leaf, g_t)
    v = torch.autograd.grad(obj, g_leaf)[0].detach()
    obj_val = obj.item()
    del g_p, g_leaf, obj

    # pass 2: <d g_p / d delta, v>, one micro-batch of second-order graph at a time
    grad = torch.zeros_like(delta)
    for i0, i1 in spans:
        d_c = delta[i0:i1]
        g = flat_grad(torch.autograd.grad(chunk_loss(i0, i1, d_c), params,
                                          create_graph=True))
        grad[i0:i1] = torch.autograd.grad(torch.dot(g, v), d_c)[0]
    return grad, obj_val


def craft_gradmatch(nets, base01, x_t_norm, y_adv, norm, eps, step, iters, restarts,
                    device, dsa_strategy=None, dsa_param=None, fast=False,
                    schedule=False, lowmem=False, chunk=0):
    set_requires_grad(nets, True)
    for n in nets:
        n.eval()
    crit = nn.CrossEntropyLoss().to(device)
    y_t = torch.full((1,), y_adv, dtype=torch.long, device=device)
    y_p = torch.full((base01.shape[0],), y_adv, dtype=torch.long, device=device)

    g_targets = []
    for net in nets:
        params = [p for p in net.parameters()]
        loss_t = crit(net(x_t_norm.unsqueeze(0)), y_t)
        g_t = torch.autograd.grad(loss_t, params)
        g_targets.append(flat_grad([g.detach() for g in g_t]))

    base01 = base01.detach()
    use_dsa = dsa_strategy not in (None, '', 'none', 'None')
    if chunk <= 0:
        chunk = base01.shape[0]
    best_delta, best_obj = None, float('inf')

    for _r in range(restarts):
        delta = torch.empty_like(base01).uniform_(-eps, eps)
        delta = (torch.clamp(base01 + delta, 0.0, 1.0) - base01).detach().requires_grad_(True)
        opt = torch.optim.Adam([delta], lr=step)
        sched = None
        if schedule:
            ms = [int(iters * 0.375), int(iters * 0.625), int(iters * 0.875)]
            sched = torch.optim.lr_scheduler.MultiStepLR(opt, milestones=ms, gamma=0.1)

        for _t in range(iters):
            if lowmem:
                # exact objective, but one surrogate and one micro-batch of poisons
                # at a time so the second-order graph never covers the whole set
                seed = int(torch.randint(0, 100000, (1,)).item()) if use_dsa else -1
                grad, obj_val = None, 0.0
                for net, g_t in zip(nets, g_targets):
                    g, o = _gradmatch_net_grad(net, g_t, base01, delta, y_p, norm,
                                               crit, chunk, use_dsa, dsa_strategy,
                                               dsa_param, seed)
                    grad = g if grad is None else grad + g
                    obj_val += o
                grad /= len(nets)
                obj_val /= len(nets)

                if obj_val < best_obj:
                    best_obj = obj_val
                    best_delta = delta.detach().clone()

                opt.zero_grad(set_to_none=True)
                delta.grad = grad.sign()
                opt.step()
                if sched is not None:
                    sched.step()
                with torch.no_grad():
                    delta.clamp_(-eps, eps)
                    delta.data = torch.clamp(base01 + delta, 0.0, 1.0) - base01
                continue

            x_adv_norm = norm(torch.clamp(base01 + delta, 0.0, 1.0))
            if use_dsa:
                seed = int(torch.randint(0, 100000, (1,)).item())
                x_adv_norm = DiffAugment(x_adv_norm, dsa_strategy, seed=seed,
                                         param=dsa_param)
            if fast:
                obj_val = 0.0
                grad_accum = torch.zeros_like(delta)
                for net, g_t in zip(nets, g_targets):
                    params = [p for p in net.parameters() if p.requires_grad]
                    loss_p = crit(net(x_adv_norm), y_p)
                    all_grads = torch.autograd.grad(loss_p, params + [delta])
                    g_p = flat_grad(list(all_grads[:-1])).detach()
                    grad_accum = grad_accum + all_grads[-1].detach()
                    obj_val += (1.0 - cosine(g_p, g_t)).item()
                obj_val /= len(nets)
                grad_accum /= len(nets)
                grad = grad_accum
            else:
                obj = 0.0
                for net, g_t in zip(nets, g_targets):
                    params = [p for p in net.parameters()]
                    loss_p = crit(net(x_adv_norm), y_p)
                    g_p = flat_grad(torch.autograd.grad(loss_p, params, create_graph=True))
                    obj = obj + (1.0 - cosine(g_p, g_t))
                obj = obj / len(nets)
                grad = torch.autograd.grad(obj, delta)[0]
                obj_val = obj.item()

            if obj_val < best_obj:
                best_obj = obj_val
                best_delta = delta.detach().clone()

            opt.zero_grad(set_to_none=True)
            delta.grad = grad.sign()
            opt.step()
            if sched is not None:
                sched.step()
            with torch.no_grad():
                delta.clamp_(-eps, eps)
                delta.data = torch.clamp(base01 + delta, 0.0, 1.0) - base01

    return torch.clamp(base01 + best_delta, 0.0, 1.0), best_obj


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #

def build_run_name(args):
    name = ('%s_%s_%s_%s_%s_b%g_eps%d_seed%d'
            % (args.dataset, args.model, args.attack, args.base, args.class_pair,
               args.budget, round(args.epsilon * 255), args.seed))
    if args.base == 'ours':
        name += '_lam%g_%s' % (args.lambda_margin, args.base_dist)
    if args.attack == 'fc' and args.fc_mode != 'sample':
        name += '_%s' % args.fc_mode
    if args.craft_ensemble:
        name += '_ce%d' % args.craft_ensemble
    if isinstance(args.target_select, int):
        name += '_tgt%d' % args.target_select
    return name


def main(args):
    global _LOG_PATH
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    set_seed(args.seed)
    dsa_param = ParamDiffAug()

    channel, im_size, num_classes, class_names, mean, std, dst_train, dst_test, _ = \
        get_dataset(args.dataset, args.data_path)

    train_imgs, train_labs = stack_dataset(dst_train, device)
    test_imgs, test_labs = stack_dataset(dst_test, device)
    N_total = train_imgs.shape[0]

    m = torch.tensor(mean, device=device).view(1, channel, 1, 1)
    s = torch.tensor(std, device=device).view(1, channel, 1, 1)
    norm = lambda x01: (x01 - m) / s
    denorm = lambda xn: xn * s + m

    # ---- precompute mode: train one cached net and exit -------------------
    if args.precompute_only:
        log('precompute: %s part=%s id=%s on %s'
            % (args.model, args.precompute_part, args.precompute_id, device))
        if args.precompute_part in ('surrogate', 'both'):
            get_surrogates(args, train_imgs, train_labs, test_imgs, test_labs,
                           channel, num_classes, im_size, device, dsa_param,
                           only_id=args.precompute_id)
        if args.precompute_part in ('victim', 'both'):
            get_clean_victims(args, train_imgs, train_labs, test_imgs, test_labs,
                              channel, num_classes, im_size, device, dsa_param,
                              only_id=args.precompute_id)
        log('precompute done.')
        return

    run_dir = os.path.join(args.out_dir, build_run_name(args))
    os.makedirs(run_dir, exist_ok=True)
    _LOG_PATH = os.path.join(run_dir, 'log.txt')

    log('=== run start: %s on %s ===' % (build_run_name(args), device))
    log('args: %s' % json.dumps(vars(args), sort_keys=True))

    y_adv, target_class = parse_pair(args.class_pair, class_names, args.pair_order)
    N_p = int(round(args.budget * N_total)) if args.budget else args.num_poisons
    log('N_total=%d budget=%g -> N_p=%d poisons, y_adv=%d(%s) target_class=%d(%s)'
        % (N_total, args.budget or 0, N_p, y_adv, class_names[y_adv],
           target_class, class_names[target_class]))

    log('=== surrogates (%d x %s, trained on the full real set) ==='
        % (args.num_surrogates, args.model))
    surrogates = get_surrogates(args, train_imgs, train_labs, test_imgs, test_labs,
                                channel, num_classes, im_size, device, dsa_param)
    craft_nets = surrogates[:args.craft_ensemble] if args.craft_ensemble else surrogates
    log('  crafting on %d/%d surrogates' % (len(craft_nets), len(surrogates)))

    clean_victims, cta_baseline_mean, cta_baseline_std = [], None, None
    if args.clean_baseline:
        log('=== clean victims (%d x %s) ===' % (args.num_victims, args.model))
        clean_victims = get_clean_victims(args, train_imgs, train_labs, test_imgs,
                                          test_labs, channel, num_classes, im_size,
                                          device, dsa_param)
        accs = [test_acc(n, test_imgs, test_labs) for n in clean_victims]
        cta_baseline_mean = float(np.mean(accs))
        cta_baseline_std = float(np.std(accs))
        log('  clean baseline CTA = %.4f +/- %.4f' % (cta_baseline_mean, cta_baseline_std))

    gen = torch.Generator(device='cpu').manual_seed(args.seed)
    rank_nets = clean_victims if (clean_victims and args.rank_on_victims) else surrogates
    targets, target_scores = select_targets(args, rank_nets, test_imgs, test_labs,
                                            y_adv, target_class, gen)
    log('  targets: %s' % targets)

    # ---- resume bookkeeping -----------------------------------------------
    results_path = os.path.join(run_dir, 'results.csv')
    fields = ['model', 'attack', 'base', 'class_pair', 'seed', 'budget', 'num_poisons',
              'epsilon', 'target_idx', 'target_score', 'victim_id', 'success',
              'clean_test_acc', 'clean_asr', 'craft_obj', 'realized_linf']
    completed = set()
    write_header = True
    if os.path.exists(results_path) and not args.no_resume:
        write_header = False
        with open(results_path, newline='') as f:
            for row in csv.DictReader(f):
                completed.add((int(row['target_idx']), int(row['victim_id'])))
        log('  resume: %d (target, victim) trials already done' % len(completed))
    rf = open(results_path, 'w' if (write_header or args.no_resume) else 'a', newline='')
    writer = csv.DictWriter(rf, fieldnames=fields)
    if write_header or args.no_resume:
        writer.writeheader()
        rf.flush()

    deltas_path = os.path.join(run_dir, 'deltas.pt')
    bases_path = os.path.join(run_dir, 'bases.json')
    cached_deltas = (torch.load(deltas_path, map_location='cpu')
                     if (os.path.exists(deltas_path) and not args.recompute_deltas) else {})
    cached_bases = (json.load(open(bases_path))
                    if (os.path.exists(bases_path) and not args.recompute_deltas) else {})

    tally = np.zeros(num_classes, dtype=np.int64)

    for ti, tidx in enumerate(targets):
        x_t_norm = test_imgs[tidx]
        log('=== target %d (%d/%d) ===' % (tidx, ti + 1, len(targets)))

        clean_asr = float('nan')
        if clean_victims:
            preds = [predict_target(n, x_t_norm) for n in clean_victims]
            clean_asr = 100.0 * sum(p == y_adv for p in preds) / len(clean_victims)

        # ---- base selection ------------------------------------------------
        if str(tidx) in cached_bases:
            base_idx = torch.tensor(cached_bases[str(tidx)], dtype=torch.long, device=device)
        elif args.base == 'random':
            base_idx = select_base_random(train_labs, y_adv, N_p, device, gen)
        else:
            base_idx = select_base_ours(surrogates, train_imgs, train_labs, x_t_norm,
                                        y_adv, N_p, args.lambda_margin, device,
                                        base_dist=args.base_dist)
        cached_bases[str(tidx)] = base_idx.cpu().tolist()

        # ---- crafting ------------------------------------------------------
        base01 = denorm(train_imgs[base_idx]).clamp(0.0, 1.0).detach()
        if str(tidx) in cached_deltas:
            x_adv01 = torch.clamp(base01 + cached_deltas[str(tidx)].to(device), 0.0, 1.0)
            obj = float('nan')
        else:
            t0 = time.time()
            if args.attack == 'gradmatch':
                x_adv01, obj = craft_gradmatch(
                    craft_nets, base01, x_t_norm, y_adv, norm, args.epsilon,
                    args.craft_alpha, args.craft_steps, args.restarts, device,
                    dsa_strategy=(args.dsa_strategy if args.craft_aug else None),
                    dsa_param=dsa_param, fast=args.fast_gradmatch,
                    schedule=args.craft_schedule, lowmem=args.craft_lowmem,
                    chunk=args.craft_batch)
            else:
                x_adv01, obj = craft_fc(
                    craft_nets, base01, x_t_norm, norm, args.epsilon, args.craft_steps,
                    args.craft_alpha, device, restarts=args.fc_restarts,
                    mode=args.fc_mode)
            cached_deltas[str(tidx)] = (x_adv01 - base01).detach().cpu()
            torch.save(cached_deltas, deltas_path)
            with open(bases_path, 'w') as f:
                json.dump(cached_bases, f)
            log('  crafted %d poisons in %.0f s, obj=%.5f' % (N_p, time.time() - t0, obj))

        linf = (x_adv01 - base01).abs().max().item()
        log('  realized linf = %.5f (%.2f/255), budget = %.2f/255'
            % (linf, linf * 255, args.epsilon * 255))

        # ---- inject in place, train victims, restore ------------------------
        clean_rows = train_imgs[base_idx].clone()
        train_imgs[base_idx] = norm(x_adv01)
        succ, ctas = [], []
        try:
            for vi in range(args.num_victims):
                if (tidx, vi) in completed:
                    continue
                seed_v = args.seed * 100000 + tidx * 100 + vi
                net = build_network(args.model, channel, num_classes, im_size,
                                    device, seed=seed_v)
                net = train_from_scratch(net, train_imgs, train_labs, args.victim_epochs,
                                         args.victim_lr, args.victim_bs, args.victim_decay,
                                         device, weight_decay=args.victim_wd,
                                         aug=args.victim_aug,
                                         dsa_strategy=args.dsa_strategy,
                                         dsa_param=dsa_param)
                pred = predict_target(net, x_t_norm)
                cta = test_acc(net, test_imgs, test_labs)
                ok = int(pred == y_adv)
                tally[pred] += 1
                succ.append(ok)
                ctas.append(cta)
                writer.writerow({
                    'model': args.model, 'attack': args.attack, 'base': args.base,
                    'class_pair': args.class_pair, 'seed': args.seed,
                    'budget': args.budget, 'num_poisons': N_p, 'epsilon': args.epsilon,
                    'target_idx': tidx, 'target_score': target_scores.get(tidx, ''),
                    'victim_id': vi, 'success': ok, 'clean_test_acc': cta,
                    'clean_asr': clean_asr, 'craft_obj': obj, 'realized_linf': linf})
                rf.flush()
                log('  [t%d v%d/%d] %s (pred=%s) CTA=%.4f'
                    % (tidx, vi + 1, args.num_victims,
                       'SUCCESS' if ok else 'fail', class_names[pred], cta))
                del net
                if device == 'cuda':
                    torch.cuda.empty_cache()
        finally:
            train_imgs[base_idx] = clean_rows

        if succ:
            log('  target %d: ASR=%.1f%% CTA=%.4f +/- %.4f'
                % (tidx, 100.0 * np.mean(succ), np.mean(ctas), np.std(ctas)))

    rf.close()

    # ---- summaries ---------------------------------------------------------
    per_target = defaultdict(list)
    all_cta = []
    with open(results_path, newline='') as f:
        for row in csv.DictReader(f):
            per_target[int(row['target_idx'])].append(int(row['success']))
            all_cta.append(float(row['clean_test_acc']))
    per_target_asr = [float(np.mean(v)) for v in per_target.values()]

    stats = {
        'model': args.model, 'attack': args.attack, 'base': args.base,
        'class_pair': args.class_pair, 'pair_order': args.pair_order,
        'seed': args.seed, 'budget': args.budget, 'num_poisons': N_p,
        'epsilon': args.epsilon, 'fc_mode': args.fc_mode,
        'lambda_margin': args.lambda_margin, 'base_dist': args.base_dist,
        'num_surrogates': args.num_surrogates,
        'craft_ensemble': args.craft_ensemble or args.num_surrogates,
        'restarts': args.restarts, 'craft_steps': args.craft_steps,
        'craft_alpha': args.craft_alpha, 'target_select': args.target_select,
        'num_targets': len(per_target), 'num_trials': len(all_cta),
        'asr_mean': float(np.mean(per_target_asr)) if per_target_asr else None,
        'asr_std': float(np.std(per_target_asr)) if per_target_asr else None,
        'cta_post_mean': float(np.mean(all_cta)) if all_cta else None,
        'cta_post_std': float(np.std(all_cta)) if all_cta else None,
        'cta_baseline_mean': cta_baseline_mean, 'cta_baseline_std': cta_baseline_std,
        'tally': tally.tolist(),
    }
    stats['cta_drop_mean'] = (None if (stats['cta_post_mean'] is None or
                                       cta_baseline_mean is None)
                              else stats['cta_post_mean'] - cta_baseline_mean)

    with open(os.path.join(run_dir, 'summary.json'), 'w') as f:
        json.dump(stats, f, indent=2)

    gpath = os.path.join(args.out_dir, 'summary_all.csv')
    gfields = [k for k in stats.keys() if k != 'tally']
    need_header = not os.path.exists(gpath)
    if not need_header:
        with open(gpath, newline='') as f:
            if next(csv.reader(f), []) != gfields:
                os.replace(gpath, '%s.%s.bak' % (gpath, time.strftime('%Y%m%d-%H%M%S')))
                need_header = True
    with open(gpath, 'a', newline='') as f:
        gw = csv.DictWriter(f, fieldnames=gfields)
        if need_header:
            gw.writeheader()
        gw.writerow({k: stats[k] for k in gfields})

    log('==== %s : ASR = %.1f%% +/- %.1f%% | CTA = %.4f (baseline %s) ===='
        % (build_run_name(args),
           100.0 * (stats['asr_mean'] or 0.0), 100.0 * (stats['asr_std'] or 0.0),
           stats['cta_post_mean'] or float('nan'),
           ('%.4f' % cta_baseline_mean) if cta_baseline_mean else 'n/a'))
    log('  target-prediction tally %s: %s' % (class_names, tally.tolist()))


def parse_args():
    p = argparse.ArgumentParser(description='Clean-label poisoning: FC / gradmatch '
                                            'crafting x random / ours base selection.')
    # data + model
    p.add_argument('--dataset', type=str, default='CIFAR10')
    p.add_argument('--data_path', type=str, default='./data')
    p.add_argument('--model', type=str, default='ConvNetBN', choices=SUPPORTED_MODELS)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--cache_dir', type=str, default='./cache')
    p.add_argument('--out_dir', type=str, default='./ours_result')
    p.add_argument('--dsa_strategy', type=str,
                   default='color_crop_cutout_flip_scale_rotate')

    # attack
    p.add_argument('--attack', type=str, default='fc', choices=['fc', 'gradmatch'])
    p.add_argument('--base', type=str, default='ours', choices=['random', 'ours'])
    p.add_argument('--class_pair', type=str, default='dog-bird', choices=CLASS_PAIRS)
    p.add_argument('--pair_order', type=str, default='poison-target',
                   choices=['poison-target', 'target-poison'],
                   help="'dog-bird' with poison-target means poisons are dogs and the "
                        "target image is a bird (MetaPoison convention).")
    p.add_argument('--budget', type=float, default=0.01,
                   help='fraction of the full training set turned into poisons')
    p.add_argument('--num_poisons', type=int, default=500,
                   help='used only when --budget is 0')
    p.add_argument('--epsilon', type=float, default=8.0 / 255.0)
    p.add_argument('--craft_steps', type=int, default=250)
    p.add_argument('--craft_alpha', type=float, default=1.0 / 255.0,
                   help='fc: PGD sign step. gradmatch: signed-Adam lr.')
    p.add_argument('--restarts', type=int, default=8, help='gradmatch restarts')
    p.add_argument('--fc_restarts', type=int, default=1)
    p.add_argument('--fc_mode', type=str, default='sample', choices=['sample', 'bullseye'],
                   help='sample = per-sample collision (works from scratch). '
                        'bullseye = mean-pooled Bullseye Polytope objective.')
    p.add_argument('--craft_ensemble', type=int, default=0,
                   help='number of surrogates used for crafting; 0 = all')
    p.add_argument('--craft_aug', action='store_true', default=True,
                   help='DiffAugment inside gradmatch crafting (Witches Brew default)')
    p.add_argument('--no_craft_aug', dest='craft_aug', action='store_false')
    p.add_argument('--craft_schedule', action='store_true', default=False,
                   help='decay the signed-Adam lr at 3/8, 5/8, 7/8 of the steps')
    p.add_argument('--fast_gradmatch', action='store_true', default=False,
                   help='first-order approximation, skips create_graph (2-3x faster, '
                        'use this for VGG13BN with large poison counts)')
    p.add_argument('--craft_lowmem', action='store_true', default=False,
                   help='gradmatch only: compute the same gradient one surrogate and '
                        'one --craft_batch slice of poisons at a time instead of all '
                        'at once. Off by default (identical to the old code path); '
                        'turn it on for large budgets that OOM. Overrides '
                        '--fast_gradmatch. Costs ~1.5-2x crafting time.')
    p.add_argument('--craft_batch', type=int, default=256,
                   help='poison micro-batch size used when --craft_lowmem is set; '
                        '0 = no splitting (per-surrogate savings only)')

    # base selection
    p.add_argument('--lambda_margin', type=float, default=1.0)
    p.add_argument('--base_dist', type=str, default='l2', choices=['l2', 'cosine'])

    # surrogates
    p.add_argument('--num_surrogates', type=int, default=5)
    p.add_argument('--surrogate_epochs', type=int, default=60)
    p.add_argument('--surrogate_lr', type=float, default=0.1)
    p.add_argument('--surrogate_bs', type=int, default=128)
    p.add_argument('--surrogate_decay', nargs='*', type=int, default=[35, 45])
    p.add_argument('--surrogate_wd', type=float, default=0.0)
    p.add_argument('--surrogate_aug', action='store_true', default=False)

    # targets
    p.add_argument('--num_targets', type=int, default=10)
    p.add_argument('--target_select', type=target_select_arg, default='easiest',
                   help='easiest | hardest | random | first, or a difficulty degree '
                        '0..100 (0 = easiest, 100 = hardest). Difficulty ranks test '
                        'images of the target class by the clean ensemble softmax '
                        'probability of y_adv; the degree slides the window of '
                        'num_targets across that ranking. Targets the clean ensemble '
                        'already predicts as y_adv are never selected.')
    p.add_argument('--target_idx_file', type=str, default=None)
    p.add_argument('--require_correct_target', action='store_true', default=False)
    p.add_argument('--rank_on_victims', action='store_true', default=True,
                   help='rank target easiness with the clean victims instead of the '
                        'surrogates (needs --clean_baseline)')
    p.add_argument('--rank_on_surrogates', dest='rank_on_victims', action='store_false')

    # victims (MetaPoison protocol: no augmentation, no weight decay)
    p.add_argument('--num_victims', type=int, default=6)
    p.add_argument('--victim_epochs', type=int, default=60)
    p.add_argument('--victim_lr', type=float, default=0.1)
    p.add_argument('--victim_bs', type=int, default=128)
    p.add_argument('--victim_decay', nargs='*', type=int, default=[35, 45])
    p.add_argument('--victim_wd', type=float, default=0.0,
                   help='keep this at 0. weight decay suppresses poison memorization '
                        'and will flatten your ASR.')
    p.add_argument('--victim_aug', action='store_true', default=False)
    p.add_argument('--clean_baseline', action='store_true', default=False)

    # bookkeeping
    p.add_argument('--no_resume', action='store_true', default=False)
    p.add_argument('--recompute_deltas', action='store_true', default=False)
    p.add_argument('--precompute_only', action='store_true', default=False)
    p.add_argument('--precompute_part', type=str, default='both',
                   choices=['surrogate', 'victim', 'both'])
    p.add_argument('--precompute_id', type=int, default=None)
    return p.parse_args()


if __name__ == '__main__':
    main(parse_args())