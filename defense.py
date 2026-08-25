#!/usr/bin/env python
"""
defense.py

Replay poisons that were ALREADY crafted and saved to disk through a DEFENDED
victim training. Nothing is selected and nothing is crafted here: the bases and
the deltas come straight out of the attack run's poison_cache, so the only thing
that differs between an attack run and its defense run is what the victim does
during training.

    attack run   ours_result/<run_name>/poison_cache/{delta_<t>.pt, base_<t>.json}
                 (or the legacy ours_result/<run_name>/{deltas.pt, bases.json})
    defense run  defense_result/<run_name>__def-<defense_tag>/{results.csv, ...}

That makes the base-selection comparison under defense exactly paired: same
targets, same poison indices, same perturbations, same victim seeds, same victim
hyperparameters. Only --defense changes.

The same harness answers the augmentation question (aug.sh), because victim-side
augmentation is just another thing the victim does during training:

    --victim_aug none | standard | randaug | cutout | dsa

The saved perturbation is written into the training image ONCE, before training
starts, and the augmentation is then resampled on top of it every epoch -- the
victim augments a poisoned dataset, which is the realistic threat model. Nothing
is re-crafted for it. See victim_aug.py for the recipes.

Defenses (--defense, '+'-separated to compose, e.g. `epic+friends`):

  none      the undefended MetaPoison victim protocol. With the same --seed this
            reproduces the attack run's numbers bit-for-bit (build_network()
            reseeds globally from seed*100000 + tidx*100 + victim_id, so the net
            init and the sgd shuffle order are fully determined by that seed),
            which is the control this whole script is checked against.

  epic      EPIC, Yang et al. ICML 2022, "Not All Poisons are Created Equal".
            Every --epic_freq epochs (from --epic_drop_after until the last lr
            decay) the last-layer gradient estimate g_i = softmax(f(x_i)) - e_yi
            is computed for every still-active training point, per-class greedy
            facility location picks B = --epic_subset_size * N medoids, every
            point is assigned to its most similar medoid, and every point whose
            medoid ends up with a cluster of size <= --epic_cluster_thresh is
            DROPPED from training for good. Effective poisons are isolated in
            gradient space, so that is what the singleton clusters are.

            submodlib is not installable here, so the facility location is a
            plain greedy on the same objective, with submodlib's default dense
            kernel: S_ij = exp(-||g_i - g_j||_2 / d), d = num_classes.

  friends   FRIENDS, Liu et al. NeurIPS 2022, "Friendly Noise against Adversarial
            Noise". At epoch --friendly_begin_epoch the current victim generates
            per-example friendly noise by maximizing the perturbation magnitude
            subject to leaving its own output alone
                min_eps  KL(f(x+eps) || f(x)) - mu * mean(eps^2),   |eps| <= clamp
            and from then on every image is trained on as
                clamp(x + friendly, 0, 1) + random_noise
            with the random component drawn fresh every step (--noise_type).

  noise     the random half of FRIENDS on its own (--noise_type without
            'friendly'), i.e. the paper's own ablation.

  advtrain  L_inf PGD adversarial training on the poisoned set, --adv_steps steps
            of size --adv_step_size inside a radius --adv_eps/255 ball.

Everything is GPU resident and manually batched, exactly like final_update.py,
and the (target, victim) trials are handed out over --gpus through one shared
queue. Trials are independent here (no crafting), so any gpu can run any trial.

Example:
  python defense.py --defense epic \
      --model ResNet20BN --attack fc --base ours --sel_dpp --sel_alpha 2.0 \
      --class_pair dog-bird --budget 0.005 --target_select 10 \
      --num_victims 6 --victim_epochs 50 --victim_bs 125 --victim_decay 40 \
      --clean_baseline --out_dir ours_result --defense_out_dir defense_result
"""

import argparse
import csv
import gc
import glob
import heapq
import json
import os
import time
import traceback
from collections import defaultdict

import numpy as np
import torch
import torch.multiprocessing as mp
import torch.nn as nn
import torch.nn.functional as TF

import atexit
import final_update as FU
import victim_aug as VA
from final_update import (build_network, log, predict_target, set_requires_grad,
                          set_seed, test_acc)

RESULT_FIELDS = ['model', 'attack', 'base', 'sel', 'class_pair', 'seed', 'budget',
                 'num_poisons', 'epsilon', 'defense', 'aug', 'target_idx',
                 'victim_id', 'success', 'clean_test_acc', 'clean_asr',
                 'realized_linf', 'n_kept', 'n_poison_kept',
                 'frac_poison_dropped', 'frac_clean_dropped']

NOISE_KINDS = ['uniform', 'gaussian', 'bernoulli', 'friendly']
DEFENSES = ['none', 'epic', 'friends', 'noise', 'advtrain']


# --------------------------------------------------------------------------- #
# naming
# --------------------------------------------------------------------------- #

def defense_parts(spec):
    """'epic+friends' -> ['epic', 'friends'], validated."""
    parts = [p for p in str(spec).strip().lower().split('+') if p]
    if not parts:
        parts = ['none']
    for p in parts:
        if p not in DEFENSES:
            raise ValueError('unknown defense %r (pick from %s, join with +)'
                             % (p, '/'.join(DEFENSES)))
    if 'none' in parts and len(parts) > 1:
        raise ValueError("'none' cannot be combined with another defense")
    return parts


def defense_tag(args):
    """Short, deterministic name for the defense CONFIG, not just its name.

    Two runs that share a tag were trained the same way, so the defended clean
    victim pool can be cached under it and shared across every attack config.
    """
    bits = []
    for part in args.defense_parts:
        if part == 'none':
            bits.append('none')
        elif part == 'epic':
            bits.append('epic-s%g-f%d-d%d%s'
                        % (args.epic_subset_size, args.epic_freq,
                           args.epic_drop_after,
                           '' if args.epic_cluster_thresh == 1.0
                           else '-c%g' % args.epic_cluster_thresh))
        elif part in ('friends', 'noise'):
            bits.append('%s-%s-e%g%s'
                        % (part, '.'.join(args.noise_type), args.noise_eps,
                           ('-p%d-clp%g' % (args.friendly_begin_epoch,
                                            args.friendly_clamp))
                           if 'friendly' in args.noise_type else ''))
        elif part == 'advtrain':
            bits.append('advtrain-e%g-k%d' % (args.adv_eps, args.adv_steps))
    # the augmentation is part of what the victim does during training, so it
    # belongs in the tag: it gets its own run directory and its own cached pool
    # of clean victims (a clean victim trained under randaug is not the clean
    # baseline for a cutout run).
    tag = VA.aug_tag(args)
    if tag:
        bits.append(tag)
    return '+'.join(bits)


def sel_tag(args):
    """How the bases were picked, for the results.csv 'sel' column."""
    if args.base != 'ours':
        return args.base
    suffix = ('_jacw%g' % args.jacobian_weight
              if getattr(args, 'use_jacobian_score', False) else '')
    for flag, name, val in [(args.sel_filter, 'filter', args.sel_pool),
                            (args.sel_pca, 'pca', args.sel_pool),
                            (args.sel_mmr, 'mmr', args.sel_mu),
                            (args.sel_dpp, 'dpp', args.sel_alpha)]:
        if flag:
            return '%s%g%s' % (name, val, suffix)
    return 'ours%s' % suffix


def attack_run_dir(args):
    return os.path.join(args.out_dir, FU.build_run_name(args))


def defense_run_dir(args):
    return os.path.join(args.defense_out_dir,
                        '%s__def-%s' % (FU.build_run_name(args), defense_tag(args)))


# --------------------------------------------------------------------------- #
# csv shards (same scheme as final_update.py, our own field list)
# --------------------------------------------------------------------------- #

def _shard_paths(run_dir):
    return sorted(glob.glob(os.path.join(run_dir, 'results_rank*.csv')))


def migrate_header(path, fields=None):
    """Rewrite `path` under the current field list if it was written under an
    older one, and report whether the file now exists with a usable header.

    RESULT_FIELDS grows over time (the 'aug' column arrived with --victim_aug).
    Appending new-format rows under an old header would leave a file whose rows
    are WIDER than its header, and csv.DictReader assigns by position, so every
    column after the new one would silently read as its neighbour -- corrupting
    both the resume set and the summary. Migrating is cheap; these files are a
    few hundred rows.
    """
    fields = fields or RESULT_FIELDS
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        return False
    with open(path, newline='') as f:
        head = next(csv.reader(f), [])
    if head == list(fields):
        return True
    with open(path, newline='') as f:
        rows = list(csv.DictReader(f))
    tmp = path + '.tmp'
    with open(tmp, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, '') for k in fields})
    os.replace(tmp, path)
    log('  migrated %s to the current column list (+%s)'
        % (os.path.basename(path),
           ', '.join(k for k in fields if k not in head) or 'reordered'))
    return True


def open_shard(run_dir, rank):
    path = os.path.join(run_dir, 'results_rank%d.csv' % rank)
    fresh = not migrate_header(path)
    rf = open(path, 'a', newline='')
    w = csv.DictWriter(rf, fieldnames=RESULT_FIELDS)
    if fresh:
        w.writeheader()
        rf.flush()
    return rf, w


def merge_result_shards(run_dir, results_path):
    shards = _shard_paths(run_dir)
    if not shards:
        # still worth doing: main() reads results.csv for the resume set, and a
        # run left behind by an older column list has to be readable first
        migrate_header(results_path)
        return 0
    seen = set()
    if migrate_header(results_path):
        with open(results_path, newline='') as f:
            for row in csv.DictReader(f):
                if row.get('target_idx'):
                    seen.add((row['target_idx'], row['victim_id']))
    new_rows = []
    for p in shards:
        with open(p, newline='') as f:
            for row in csv.DictReader(f):
                if not row.get('target_idx'):
                    continue
                key = (row['target_idx'], row['victim_id'])
                if key in seen:
                    continue
                seen.add(key)
                new_rows.append(row)
    need_header = (not os.path.exists(results_path)
                   or os.path.getsize(results_path) == 0)
    with open(results_path, 'a', newline='') as f:
        w = csv.DictWriter(f, fieldnames=RESULT_FIELDS)
        if need_header:
            w.writeheader()
        for r in new_rows:
            w.writerow({k: r.get(k, '') for k in RESULT_FIELDS})
    for p in shards:
        os.remove(p)
    return len(new_rows)


# --------------------------------------------------------------------------- #
# EPIC: greedy facility location on the last-layer gradient estimate
# --------------------------------------------------------------------------- #

@torch.no_grad()
def _fl_kernel(X, metric, chunk=1024):
    """submodlib's default dense kernel (method='sklearn'). Symmetric."""
    if metric == 'cosine':
        Xn = X / X.norm(dim=1, keepdim=True).clamp_min(1e-12)
        return Xn @ Xn.t()
    # The mm-based cdist is worth ~3e-3 of absolute error at short range, which is
    # exactly where the clustering happens -- it does not even give a point
    # distance 0 to itself. So take the exact path, in row blocks so no
    # intermediate is bigger than the kernel itself; d is 10 here, it is cheap.
    n = X.shape[0]
    S = torch.empty((n, n), device=X.device, dtype=X.dtype)
    for a in range(0, n, chunk):
        d = torch.cdist(X[a:a + chunk], X,
                        compute_mode='donot_use_mm_for_euclid_dist')
        S[a:a + chunk] = torch.exp(-d / X.shape[1])   # gamma = 1 / num_features
    return S


@torch.no_grad()
def _fl_greedy(S, k):
    """Lazy greedy max of f(A) = sum_i max_{j in A} S_ij, S_ij >= 0 and symmetric.

    Facility location is submodular, so the accelerated (lazy) greedy of Minoux
    returns exactly what the naive greedy returns -- it just skips recomputing
    gains that cannot possibly be the largest. That is also what EPIC asks
    submodlib for (optimizer='LazyGreedy'). Naive greedy costs a full n x n pass
    per medoid, which is minutes per class at n = 5000, k = 500; this costs a
    handful of single-row passes instead.

    -> (order, cluster, sizes): the k medoids in the order they were picked, the
    medoid every point is assigned to, and how many points each medoid got.
    """
    n = S.shape[0]
    k = int(max(1, min(k, n)))
    cur = torch.zeros(n, device=S.device, dtype=S.dtype)
    # with A empty and S >= 0, the marginal gain of j is just its row sum
    heap = [(-v, j) for j, v in enumerate(S.sum(1).tolist())]
    heapq.heapify(heap)
    order = []
    for _ in range(k):
        while True:
            _stale, j = heapq.heappop(heap)
            gain = float(torch.clamp(S[j] - cur, min=0).sum().item())
            # nothing still in the heap can beat it -> its gain is the largest
            if not heap or gain >= -heap[0][0]:
                order.append(j)
                cur = torch.maximum(cur, S[j])
                break
            heapq.heappush(heap, (-gain, j))
        if not heap:
            break
    order = torch.tensor(order, device=S.device, dtype=torch.long)
    sub = S[:, order]                                  # n x k
    best = sub.max(dim=1)
    cluster = best.indices
    # EPIC leaves a point unassigned when it is similar to no medoid at all
    cluster = torch.where(best.values > 0, cluster, torch.full_like(cluster, -1))
    sizes = torch.bincount(cluster[cluster >= 0], minlength=k).to(S.dtype)
    sizes[sizes == 0] = 1                              # as in EPIC's coreset.py
    return order, cluster, sizes


@torch.no_grad()
def epic_select(net, train_imgs, train_labs, active, num_classes, args, bs=1024):
    """-> the subset of `active` EPIC keeps this round."""
    net.eval()
    g = torch.empty((len(active), num_classes), device=train_imgs.device)
    for i in range(0, len(active), bs):
        idx = active[i:i + bs]
        p = torch.softmax(net(train_imgs[idx]), dim=1)
        p[torch.arange(len(idx), device=p.device), train_labs[idx]] -= 1.0
        g[i:i + bs] = p
    y = train_labs[active]

    B = int(args.epic_subset_size * train_imgs.shape[0])
    keep = torch.zeros(len(active), dtype=torch.bool, device=train_imgs.device)
    for c in range(num_classes):
        pos = (y == c).nonzero(as_tuple=True)[0]
        if len(pos) == 0:
            continue
        # equal_num=False, exactly as EPIC's get_orders_and_weights does it
        k = int(np.ceil(len(pos) / float(len(active)) * B))
        S = _fl_kernel(g[pos], args.epic_metric)
        _order, cluster, sizes = _fl_greedy(S, k)
        big = sizes > args.epic_cluster_thresh
        kept_here = torch.zeros(len(pos), dtype=torch.bool, device=keep.device)
        ok = cluster >= 0
        kept_here[ok] = big[cluster[ok]]
        keep[pos] = kept_here
        del S
    return active[keep]


# --------------------------------------------------------------------------- #
# FRIENDS: friendly noise + random noise
# --------------------------------------------------------------------------- #

def generate_friendly_noise(net, train_imgs, ctx, args, active=None):
    """Per-example friendly noise for the whole training set, in [0,1] units.

    min_eps  KL(f(x+eps) || f(x)) - mu * mean(eps^2)  subject to |eps| <= clamp,
    optimized independently per example (the loss is separable, so the batch only
    sets the gradient scale). The reference optimizes batches of 128 with lr=100;
    the lr is rescaled by --friendly_bs/128 here so a bigger compute batch walks
    the exact same trajectory.
    """
    device, norm = ctx['device'], ctx['norm']
    denorm = ctx['denorm']
    N = train_imgs.shape[0]
    clamp = args.friendly_clamp / 255.0
    lr = args.friendly_lr * (args.friendly_bs / 128.0)
    steps = args.friendly_epochs
    milestones = [steps // 2, steps // 4 * 3]

    was_training = net.training
    net.eval()
    set_requires_grad([net], False)
    noise = torch.zeros((N,) + tuple(train_imgs.shape[1:]), device=device,
                        dtype=torch.float16)
    t0 = time.time()
    try:
        for i in range(0, N, args.friendly_bs):
            sl = slice(i, min(i + args.friendly_bs, N))
            x01 = denorm(train_imgs[sl]).clamp(0.0, 1.0).detach()
            with torch.no_grad():
                out0 = TF.log_softmax(net(norm(x01)), dim=1)
            eps = ((torch.rand_like(x01) - 0.5) * 2 * (8.0 / 255.0)
                   ).detach().requires_grad_(True)
            opt = torch.optim.SGD([eps], lr=lr, momentum=args.friendly_momentum,
                                  nesterov=True)
            sched = torch.optim.lr_scheduler.MultiStepLR(opt, milestones)
            for _ in range(steps):
                e = eps.clamp(-clamp, clamp)
                out = TF.log_softmax(net(norm((x01 + e).clamp(0.0, 1.0))), dim=1)
                kl = TF.kl_div(out, out0, reduction='batchmean', log_target=True)
                loss = kl - args.friendly_mu * (e ** 2).mean()
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
                sched.step()
            noise[sl] = eps.detach().clamp(-clamp, clamp).half()
    finally:
        set_requires_grad([net], True)
        if was_training:
            net.train()
    log('  friendly noise: %d images in %.0f s, mean|eps| = %.4f (%.2f/255)'
        % (N, time.time() - t0, noise.abs().float().mean().item(),
           noise.abs().float().mean().item() * 255))
    return noise


def _add_random_noise(x01, kinds, eps):
    """The FRIENDS random component. Not clamped, exactly as in the reference."""
    if 'uniform' in kinds:
        x01 = x01 + (torch.rand_like(x01) * 2.0 - 1.0) * eps
    if 'gaussian' in kinds:
        x01 = x01 + torch.randn_like(x01) * eps
    if 'bernoulli' in kinds:
        x01 = x01 + ((torch.rand_like(x01) > 0.5).float() * 2.0 - 1.0) * eps
    return x01


# --------------------------------------------------------------------------- #
# adversarial training
# --------------------------------------------------------------------------- #

def pgd_perturb(net, x01, y, norm, eps, step, steps, crit):
    """L_inf PGD on the training batch, in [0,1] units. Attack in eval mode so the
    batchnorm running stats are not updated by the inner steps."""
    was_training = net.training
    net.eval()
    set_requires_grad([net], False)
    try:
        delta = (torch.rand_like(x01) * 2.0 - 1.0) * eps
        delta = ((x01 + delta).clamp(0.0, 1.0) - x01).detach()
        for _ in range(steps):
            delta.requires_grad_(True)
            loss = crit(net(norm((x01 + delta).clamp(0.0, 1.0))), y)
            g = torch.autograd.grad(loss, delta)[0]
            delta = (delta.detach() + step * g.sign()).clamp(-eps, eps)
            delta = ((x01 + delta).clamp(0.0, 1.0) - x01).detach()
    finally:
        set_requires_grad([net], True)
        if was_training:
            net.train()
    return (x01 + delta).clamp(0.0, 1.0)


# --------------------------------------------------------------------------- #
# the defended victim training
# --------------------------------------------------------------------------- #

def train_victim_defended(args, ctx, net, poison_mask=None, tag=''):
    """Train one victim from scratch on ctx['train_imgs'] under --defense.

    ctx['train_imgs'] already has the poisons written into it (or is clean, for
    the defended baseline pool), so --victim_aug augments the ALREADY-POISONED
    image, redrawn every epoch. -> (net, info) where info carries the EPIC drop
    statistics, which are '' for every other defense.
    """
    device = ctx['device']
    train_imgs, train_labs = ctx['train_imgs'], ctx['train_labs']
    norm, denorm = ctx['norm'], ctx['denorm']
    parts = args.defense_parts
    kinds = args.noise_type if ({'friends', 'noise'} & set(parts)) else []
    rand_kinds = [k for k in kinds if k != 'friendly']
    use01 = bool(rand_kinds) or ('friendly' in kinds) or ('advtrain' in parts)

    N = train_imgs.shape[0]
    num_classes = ctx['num_classes']
    info = {'n_kept': N, 'n_poison_kept': '', 'frac_poison_dropped': '',
            'frac_clean_dropped': ''}

    # one augmenter per process, reused across victims (it holds no state, but
    # building the RandAugment table is not free)
    if 'augmenter' not in ctx:
        ctx['augmenter'] = VA.make_augmenter(args, ctx)
        if ctx['augmenter'] is not None:
            log('  victim augmentation: %s' % ctx['augmenter'])
    augmenter = ctx['augmenter']

    # nothing to do beyond the plain MetaPoison protocol -> use the very function
    # the attack runs used, so --defense none reproduces them bit-for-bit
    if not use01 and 'epic' not in parts:
        return FU.train_from_scratch(
            net, train_imgs, train_labs, args.victim_epochs, args.victim_lr,
            args.victim_bs, args.victim_decay, device,
            weight_decay=args.victim_wd, augmenter=augmenter), info

    n_poison = int(poison_mask.sum().item()) if poison_mask is not None else 0
    opt = torch.optim.SGD(net.parameters(), lr=args.victim_lr, momentum=0.9,
                          weight_decay=args.victim_wd)
    crit = nn.CrossEntropyLoss().to(device)
    decay_at = set(args.victim_decay or [])
    cur_lr = args.victim_lr
    active = torch.arange(N, device=device)
    friendly = None
    noise_eps = args.noise_eps / 255.0
    adv_eps = args.adv_eps / 255.0
    adv_step = (args.adv_step_size / 255.0 if args.adv_step_size > 0
                else 2.5 * adv_eps / max(1, args.adv_steps))
    epic_stop = (args.epic_stop_after if args.epic_stop_after >= 0
                 else (max(args.victim_decay) if args.victim_decay
                       else args.victim_epochs))

    net.train()
    for ep in range(args.victim_epochs):
        if ep in decay_at:
            cur_lr *= 0.1
            for g in opt.param_groups:
                g['lr'] = cur_lr

        if ('epic' in parts and ep % args.epic_freq == 0
                and args.epic_drop_after <= ep < epic_stop):
            before = len(active)
            active = epic_select(net, train_imgs, train_labs, active,
                                 num_classes, args)
            net.train()
            if poison_mask is not None and n_poison:
                kept_p = int(poison_mask[active].sum().item())
                info['n_poison_kept'] = kept_p
                info['frac_poison_dropped'] = 1.0 - kept_p / float(n_poison)
                info['frac_clean_dropped'] = 1.0 - ((len(active) - kept_p)
                                                    / float(N - n_poison))
            #     log('  %sepic ep%d: %d -> %d kept, poisons %d/%d (%.1f%% dropped)'
            #         % (tag, ep, before, len(active), kept_p, n_poison,
            #            100.0 * info['frac_poison_dropped']))
            # else:
            #     log('  %sepic ep%d: %d -> %d kept' % (tag, ep, before, len(active)))
            info['n_kept'] = len(active)

        if 'friendly' in kinds and ep == args.friendly_begin_epoch:
            friendly = generate_friendly_noise(net, train_imgs, ctx, args)
            net.train()

        perm = active[torch.randperm(len(active), device=device)]
        for i in range(0, len(perm), args.victim_bs):
            idx = perm[i:i + args.victim_bs]
            img = train_imgs[idx]
            lab = train_labs[idx]
            if use01:
                x01 = denorm(img).clamp(0.0, 1.0)
                if friendly is not None:
                    x01 = (x01 + friendly[idx].float()).clamp(0.0, 1.0)
                if rand_kinds:
                    x01 = _add_random_noise(x01, rand_kinds, noise_eps)
                if 'advtrain' in parts:
                    x01 = pgd_perturb(net, x01.detach(), lab, norm, adv_eps,
                                      adv_step, args.adv_steps, crit)
                img = norm(x01)
            # augmentation last: the defense perturbs the poisoned image, the
            # victim's augmentation then acts on what it is about to train on
            if augmenter is not None:
                img = augmenter(img)
            opt.zero_grad(set_to_none=True)
            loss = crit(net(img), lab)
            loss.backward()
            opt.step()

    net.eval()
    del friendly
    return net, info


# --------------------------------------------------------------------------- #
# defended clean victim pool (shared by every attack config with the same tag)
# --------------------------------------------------------------------------- #

def defended_victim_dir(args):
    return os.path.join(args.cache_dir, 'defended_victims',
                        '%s%s_%s_%dep_lr%g_bs%d_wd%g_seed%d'
                        % (FU.dataset_tag(args), args.model, defense_tag(args),
                           args.victim_epochs,
                           args.victim_lr, args.victim_bs, args.victim_wd,
                           args.seed))


def get_defended_clean_victims(args, ctx, only_id=None):
    """The clean-data baseline for THIS defense. Depends on nothing but the model,
    the defense config and the victim hyperparameters, so it is cached once and
    reused across every budget / class pair / base selection."""
    d = defended_victim_dir(args)
    ids = [only_id] if only_id is not None else range(args.num_victims)
    nets = []
    for i in ids:
        path = os.path.join(d, 'net_%d.pt' % i)
        net = build_network(args.model, ctx['channel'], ctx['num_classes'],
                            ctx['im_size'], ctx['device'], seed=args.seed + 900000 + i)
        if os.path.exists(path):
            net.load_state_dict(torch.load(path, map_location=ctx['device']))
            net.eval()
            nets.append(net)
            continue
        t0 = time.time()
        net, _info = train_victim_defended(args, ctx, net, poison_mask=None,
                                           tag='clean%d ' % i)
        acc = test_acc(net, ctx['test_imgs'], ctx['test_labs'])
        os.makedirs(d, exist_ok=True)
        # write-then-rename: this pool is keyed only on the model, the defense
        # tag and the victim hyperparameters, so parallel sweeps (aug1..aug4)
        # share it and can reach this line for the same net_%d.pt at the same
        # moment. A plain torch.save would let one process read another's
        # half-written file. os.replace is atomic, so the loser just overwrites
        # with an identical checkpoint.
        tmp = '%s.%d.tmp' % (path, os.getpid())
        torch.save(net.state_dict(), tmp)
        os.replace(tmp, path)
        log('  trained defended clean victim %d (%s, %s): test acc = %.4f (%.0f s)'
            % (i, args.model, defense_tag(args), acc, time.time() - t0))
        net.eval()
        nets.append(net)
    return nets


# --------------------------------------------------------------------------- #
# the saved poisons
# --------------------------------------------------------------------------- #

def cached_targets(run_dir, legacy=None):
    """Target indices this attack run has usable (delta, bases) on disk for."""
    ts = set()
    d = os.path.join(run_dir, 'poison_cache')
    for p in glob.glob(os.path.join(d, 'delta_*.pt')):
        t = os.path.basename(p)[len('delta_'):-len('.pt')]
        if t.isdigit() and os.path.exists(os.path.join(d, 'base_%s.json' % t)):
            ts.add(int(t))
    if legacy is None:
        legacy = FU.load_legacy_cache(run_dir, False)
    ld, lb = legacy
    for k in ld:
        if str(k) in lb or k in lb:
            ts.add(int(k))
    return sorted(ts)


def verify_against_attack(attack_dir, results_path):
    """--defense none is a control: the same trial, seeded the same way, trained
    the same way. Say out loud how well it reproduced the undefended run."""
    ap = os.path.join(attack_dir, 'results.csv')
    if not (os.path.exists(ap) and os.path.exists(results_path)):
        return
    ref = {}
    with open(ap, newline='') as f:
        for row in csv.DictReader(f):
            if row.get('target_idx') and row.get('victim_id'):
                ref[(int(row['target_idx']), int(row['victim_id']))] = (
                    int(row['success']), float(row['clean_test_acc']))
    same, tot, dcta = 0, 0, []
    with open(results_path, newline='') as f:
        for row in csv.DictReader(f):
            key = (int(row['target_idx']), int(row['victim_id']))
            if key not in ref:
                continue
            tot += 1
            same += int(int(row['success']) == ref[key][0])
            dcta.append(abs(float(row['clean_test_acc']) - ref[key][1]))
    if tot:
        log('  control check vs the undefended run: %d/%d trials agree, '
            'max |dCTA| = %.5f (cudnn is not bit-deterministic, so a stray '
            'flip near the decision boundary is expected)'
            % (same, tot, max(dcta)))


def load_saved_poisons(ctx, run_dir, tidx, legacy):
    """(base_idx, x_adv01, linf) for one target, rebuilt exactly the way
    prepare_poisons() rebuilds it from the cache, or None."""
    delta, base = FU.load_poison_cache(run_dir, tidx, legacy, recompute=False)
    if delta is None or base is None:
        return None
    device = ctx['device']
    base_idx = torch.tensor(base, dtype=torch.long, device=device)
    base01 = ctx['denorm'](ctx['train_imgs'][base_idx]).clamp(0.0, 1.0).detach()
    x_adv01 = torch.clamp(base01 + delta.to(device), 0.0, 1.0)
    linf = (x_adv01 - base01).abs().max().item()
    return base_idx, x_adv01, linf


# --------------------------------------------------------------------------- #
# one (target, victim) trial
# --------------------------------------------------------------------------- #

def run_trial(args, ctx, tidx, vi, y_adv, prep, clean_asr, emit):
    device = ctx['device']
    train_imgs = ctx['train_imgs']
    base_idx, x_adv01, linf = prep
    x_t_norm = ctx['test_imgs'][tidx]

    poison_mask = torch.zeros(train_imgs.shape[0], dtype=torch.bool, device=device)
    poison_mask[base_idx] = True
    clean_rows = train_imgs[base_idx].clone()
    train_imgs[base_idx] = ctx['norm'](x_adv01)
    try:
        seed_v = args.seed * 100000 + tidx * 100 + vi
        net = build_network(args.model, ctx['channel'], ctx['num_classes'],
                            ctx['im_size'], device, seed=seed_v)
        t0 = time.time()
        net, info = train_victim_defended(args, ctx, net, poison_mask=poison_mask,
                                          tag='[t%d v%d] ' % (tidx, vi))
        pred = predict_target(net, x_t_norm)
        cta = test_acc(net, ctx['test_imgs'], ctx['test_labs'])
        ok = int(pred == y_adv)
        emit({
            'model': args.model, 'attack': args.attack, 'base': args.base,
            'sel': sel_tag(args), 'class_pair': args.class_pair, 'seed': args.seed,
            'budget': args.budget, 'num_poisons': len(base_idx),
            'epsilon': args.epsilon, 'defense': defense_tag(args),
            'aug': args.victim_aug,
            'target_idx': tidx, 'victim_id': vi, 'success': ok,
            'clean_test_acc': cta, 'clean_asr': clean_asr, 'realized_linf': linf,
            'n_kept': info['n_kept'], 'n_poison_kept': info['n_poison_kept'],
            'frac_poison_dropped': info['frac_poison_dropped'],
            'frac_clean_dropped': info['frac_clean_dropped']})
        log('  [t%d v%d/%d] %s (pred=%s) CTA=%.4f (%.0f s)'
            % (tidx, vi + 1, args.num_victims, 'SUCCESS' if ok else 'fail',
               ctx['class_names'][pred], cta, time.time() - t0))
        del net
        if str(device).startswith('cuda'):
            torch.cuda.empty_cache()
        return pred
    finally:
        train_imgs[base_idx] = clean_rows


# --------------------------------------------------------------------------- #
# multi-gpu: the trials are fully independent here, so one flat queue is enough
# --------------------------------------------------------------------------- #

def _trial_worker(rank, gpu, job_q, out_q, args, attack_dir, def_dir, log_path,
                  y_adv, clean_asrs):
    FU._LOG_PATH, FU._LOG_TAG = log_path, '[gpu%d]' % gpu
    rf = None
    try:
        torch.cuda.set_device(gpu)
        device = 'cuda:%d' % gpu
        set_seed(args.seed)
        ctx = FU.build_context(args, device)
        legacy = FU.load_legacy_cache(attack_dir, False)
        rf, writer = open_shard(def_dir, rank)
        lock_path = os.path.join(def_dir, '.lock')

        def emit(row):
            writer.writerow(row)
            rf.flush()
            # the parent is blocked in _run_pool and cannot refresh; a RandAugment
            # run is ~4.5 h against a 2 h staleness threshold, so without this the
            # lock would expire mid-run and a second process could take it over
            FU.touch_run_lock(lock_path)

        tally = np.zeros(ctx['num_classes'], dtype=np.int64)
        cur_tidx, prep = None, None
        while True:
            job = job_q.get()
            if job is None:
                break
            tidx, vi = job
            if prep is None or cur_tidx != tidx:
                prep = load_saved_poisons(ctx, attack_dir, tidx, legacy)
                cur_tidx = tidx
                if prep is None:
                    log('  target %d has no saved poisons, skipping' % tidx)
                    continue
                log('=== target %d: %d saved poisons, realized linf %.2f/255 ==='
                    % (tidx, len(prep[0]), prep[2] * 255))
            if prep is None:
                continue
            pred = run_trial(args, ctx, tidx, vi, y_adv, prep,
                             clean_asrs.get(tidx, float('nan')), emit)
            tally[pred] += 1
        out_q.put(('ok', rank, tally.tolist()))
    except Exception:
        log('WORKER FAILED:\n%s' % traceback.format_exc())
        out_q.put(('err', rank, traceback.format_exc()))
    finally:
        if rf is not None:
            rf.close()


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #

def main(args):
    gpus = FU.resolve_gpus(args.gpus)
    device = ('cuda:%d' % gpus[0]) if gpus else 'cpu'
    if gpus:
        torch.cuda.set_device(gpus[0])
    set_seed(args.seed)

    attack_dir = attack_run_dir(args)
    if not os.path.isdir(attack_dir):
        raise SystemExit('no attack run at %s -- craft it first (ours.sh / '
                         'sel_dpp.sh), or fix the flags so the run name matches'
                         % attack_dir)
    legacy = FU.load_legacy_cache(attack_dir, False)
    have = cached_targets(attack_dir, legacy)
    if not have:
        raise SystemExit('%s has no saved poisons (neither poison_cache/ nor '
                         'deltas.pt), nothing to replay' % attack_dir)

    if args.list_cached:
        print(json.dumps({'run': FU.build_run_name(args), 'dir': attack_dir,
                          'targets': have}))
        return

    def_dir = defense_run_dir(args)
    os.makedirs(def_dir, exist_ok=True)
    FU._LOG_PATH = os.path.join(def_dir, 'log.txt')

    # Same guard final_update.py uses, for the same reason: two processes in one run
    # dir merge and then delete each other's results_rank*.csv, and the survivor dies
    # on a stale file handle mid-write. defense_result/ needs it just as much as
    # ours_result/ -- several appendix scripts legitimately name the same defended run
    # (the aug shards and the older full ra_* sweeps overlap exactly), and until this
    # was here they would collide instead of one of them standing down.
    lock = FU.acquire_run_lock(def_dir)
    if lock is None:
        return
    atexit.register(FU.release_run_lock, lock)

    log('=== defense run: %s ===' % os.path.basename(def_dir))
    log('    poisons replayed from %s' % attack_dir)
    log('    defense = %s  (%s)' % (args.defense, defense_tag(args)))
    log('    victim augmentation = %s  (applied to the poisoned image, '
        'resampled every epoch)' % args.victim_aug)
    log('args: %s' % json.dumps({k: v for k, v in vars(args).items()},
                                sort_keys=True, default=str))

    ctx = FU.build_context(args, device)
    class_names = ctx['class_names']
    y_adv, target_class = FU.parse_pair(args.class_pair, class_names,
                                        args.pair_order)

    # ---- targets: whatever is on disk, optionally restricted by a pin file ----
    targets = list(have)
    if args.target_idx_file:
        with open(args.target_idx_file) as f:
            blob = json.load(f)
        want = (blob['pairs'][args.class_pair]['indices'] if 'pairs' in blob
                else blob[args.class_pair])
        missing = [t for t in want if int(t) not in set(have)]
        targets = [int(t) for t in want if int(t) in set(have)]
        if missing:
            log('  %d of %d pinned targets have no saved poisons here: %s'
                % (len(missing), len(want), missing))
    targets = targets[:args.num_targets] if args.num_targets else targets
    if not targets:
        raise SystemExit('no targets left to run')
    log('  %d target(s) with saved poisons: %s' % (len(targets), targets))

    # ---- defended clean baseline --------------------------------------------
    clean_asrs = {t: float('nan') for t in targets}
    cta_baseline_mean = cta_baseline_std = None
    if args.clean_baseline:
        log('=== defended clean victims (%d x %s, defense %s) ==='
            % (args.num_victims, args.model, defense_tag(args)))
        cvs = get_defended_clean_victims(args, ctx)
        accs = [test_acc(n, ctx['test_imgs'], ctx['test_labs']) for n in cvs]
        cta_baseline_mean = float(np.mean(accs))
        cta_baseline_std = float(np.std(accs))
        log('  defended clean CTA = %.4f +/- %.4f'
            % (cta_baseline_mean, cta_baseline_std))
        for t in targets:
            preds = [predict_target(n, ctx['test_imgs'][t]) for n in cvs]
            clean_asrs[t] = 100.0 * sum(p == y_adv for p in preds) / len(cvs)
        del cvs
        if gpus:
            torch.cuda.empty_cache()

    # ---- resume --------------------------------------------------------------
    results_path = os.path.join(def_dir, 'results.csv')
    if args.no_resume:
        for p in _shard_paths(def_dir):
            os.remove(p)
        if os.path.exists(results_path):
            os.remove(results_path)
    merged = merge_result_shards(def_dir, results_path)
    if merged:
        log('  merged %d rows left behind by an interrupted run' % merged)
    completed = set()
    if os.path.exists(results_path):
        with open(results_path, newline='') as f:
            for row in csv.DictReader(f):
                if row.get('target_idx') and row.get('victim_id'):
                    completed.add((int(row['target_idx']), int(row['victim_id'])))
        if completed:
            log('  resume: %d (target, victim) trials already done' % len(completed))

    jobs = [(t, vi) for t in targets for vi in range(args.num_victims)
            if (t, vi) not in completed]
    if not jobs:
        log('  everything already done')
    tally = np.zeros(ctx['num_classes'], dtype=np.int64)
    errors = []

    # ---- run -----------------------------------------------------------------
    if jobs and len(gpus) > 1 and args.parallel_trials and len(jobs) > 1:
        n = min(len(gpus), len(jobs))
        log('=== %d trial(s) over %d gpus %s ===' % (len(jobs), n, gpus[:n]))
        # every worker builds its own dataset on its own device, so drop the
        # parent's copy first
        del ctx
        gc.collect()
        torch.cuda.empty_cache()
        t0 = time.time()
        results, errors = FU._run_pool(
            _trial_worker, gpus[:n], jobs,
            (args, attack_dir, def_dir, FU._LOG_PATH, y_adv, clean_asrs))
        for _rank, part in results:
            tally += np.array(part, dtype=np.int64)
        for rank, err in errors:
            log('  !! worker %d failed:\n%s' % (rank, err))
        log('=== all workers finished in %.0f s ===' % (time.time() - t0))
    elif jobs:
        rf, writer = open_shard(def_dir, 0)

        def emit(row):
            writer.writerow(row)
            rf.flush()
            FU.touch_run_lock(lock)

        cur_tidx, prep = None, None
        try:
            for tidx, vi in jobs:
                if prep is None or cur_tidx != tidx:
                    prep = load_saved_poisons(ctx, attack_dir, tidx, legacy)
                    cur_tidx = tidx
                    if prep is None:
                        log('  target %d has no saved poisons, skipping' % tidx)
                        continue
                    log('=== target %d: %d saved poisons, realized linf %.2f/255 ==='
                        % (tidx, len(prep[0]), prep[2] * 255))
                if prep is None:
                    continue
                tally[run_trial(args, ctx, tidx, vi, y_adv, prep,
                                clean_asrs.get(tidx, float('nan')), emit)] += 1
        finally:
            rf.close()

    merge_result_shards(def_dir, results_path)
    if args.defense_parts == ['none'] and args.victim_aug == 'none':
        verify_against_attack(attack_dir, results_path)

    # ---- summary -------------------------------------------------------------
    per_target = defaultdict(list)
    all_cta, drop_p, drop_c = [], [], []
    if os.path.exists(results_path):
        with open(results_path, newline='') as f:
            for row in csv.DictReader(f):
                if not (row.get('target_idx') and row.get('success')):
                    continue
                per_target[int(row['target_idx'])].append(int(row['success']))
                all_cta.append(float(row['clean_test_acc']))
                if row.get('frac_poison_dropped'):
                    drop_p.append(float(row['frac_poison_dropped']))
                if row.get('frac_clean_dropped'):
                    drop_c.append(float(row['frac_clean_dropped']))
    per_target_asr = [float(np.mean(v)) for v in per_target.values()]
    stats = {
        'run': FU.build_run_name(args), 'attack_dir': attack_dir,
        'defense': args.defense, 'defense_tag': defense_tag(args),
        'aug': args.victim_aug,
        'model': args.model, 'attack': args.attack, 'base': args.base,
        'sel': sel_tag(args), 'class_pair': args.class_pair, 'seed': args.seed,
        'budget': args.budget, 'epsilon': args.epsilon,
        'use_jacobian_score': args.use_jacobian_score,
        'jacobian_weight': args.jacobian_weight,
        'jacobian_batch_size': args.jacobian_batch_size,
        'num_targets': len(per_target), 'num_trials': len(all_cta),
        'asr_mean': float(np.mean(per_target_asr)) if per_target_asr else None,
        'asr_std': float(np.std(per_target_asr)) if per_target_asr else None,
        'cta_post_mean': float(np.mean(all_cta)) if all_cta else None,
        'cta_post_std': float(np.std(all_cta)) if all_cta else None,
        'cta_baseline_mean': cta_baseline_mean,
        'cta_baseline_std': cta_baseline_std,
        'frac_poison_dropped_mean': float(np.mean(drop_p)) if drop_p else None,
        'frac_clean_dropped_mean': float(np.mean(drop_c)) if drop_c else None,
        'tally': tally.tolist(),
    }
    stats['cta_drop_mean'] = (None if (stats['cta_post_mean'] is None
                                       or cta_baseline_mean is None)
                              else stats['cta_post_mean'] - cta_baseline_mean)
    with open(os.path.join(def_dir, 'summary.json'), 'w') as f:
        json.dump(stats, f, indent=2)

    gpath = os.path.join(args.defense_out_dir, 'summary_all.csv')
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

    log('==== %s | defense %s : ASR = %.1f%% +/- %.1f%% | CTA = %.4f '
        '(defended clean %s)%s ===='
        % (FU.build_run_name(args), defense_tag(args),
           100.0 * (stats['asr_mean'] or 0.0), 100.0 * (stats['asr_std'] or 0.0),
           stats['cta_post_mean'] or float('nan'),
           ('%.4f' % cta_baseline_mean) if cta_baseline_mean else 'n/a',
           ('  poisons dropped %.1f%% vs clean %.1f%%'
            % (100.0 * stats['frac_poison_dropped_mean'],
               100.0 * stats['frac_clean_dropped_mean'])) if drop_p else ''))
    log('  target-prediction tally %s: %s' % (class_names, tally.tolist()))

    if errors:
        raise RuntimeError('%d gpu worker(s) failed; the summary above only covers '
                           'the trials that did finish. Re-run to resume.'
                           % len(errors))


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description='Replay saved poisons through a defended victim training.')

    # ---- which attack run to replay. Same names / defaults as final_update.py,
    # because FU.build_run_name(args) is what locates the poison cache. ----
    p.add_argument('--dataset', type=str, default='CIFAR10')
    p.add_argument('--data_path', type=str, default='./data')
    p.add_argument('--model', type=str, default='ConvNetBN',
                   choices=FU.SUPPORTED_MODELS)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--cache_dir', type=str, default='./cache')
    p.add_argument('--out_dir', type=str, default='./ours_result',
                   help='where the ATTACK runs live (read-only here)')
    p.add_argument('--defense_out_dir', type=str, default='./defense_result',
                   help='where this run writes its results')
    p.add_argument('--craft_aug', action='store_true', default=True,
                   help='naming only: which craft-time augmentation the poisons being '
                        'replayed were built with, so the right run dir is found')
    p.add_argument('--no_craft_aug', dest='craft_aug', action='store_false')
    p.add_argument('--dsa_strategy', type=str,
                   default='color_crop_cutout_flip_scale_rotate')
    p.add_argument('--attack', type=str, default='fc',
                   choices=['fc', 'gradmatch', 'sapa'])
    p.add_argument('--base', type=str, default='ours', choices=['random', 'ours'])
    p.add_argument('--class_pair', type=str, default='dog-bird',
                   help="'<adversarial>-<target>' class names; validated against the "
                        'dataset once it is loaded, as in final_update.py')
    p.add_argument('--pair_order', type=str, default='poison-target',
                   choices=['poison-target', 'target-poison'])
    p.add_argument('--budget', type=float, default=0.01)
    p.add_argument('--epsilon', type=float, default=8.0 / 255.0)
    p.add_argument('--fc_mode', type=str, default='sample',
                   choices=['sample', 'bullseye'])
    p.add_argument('--craft_ensemble', type=int, default=0)
    p.add_argument('--sharp_mode', type=str, default='worst',
                   choices=['worst', 'avg'])
    p.add_argument('--sharp_sigma', type=float, default=0.05)
    p.add_argument('--sharp_samples', type=int, default=20)
    p.add_argument('--lambda_margin', type=float, default=1.0)
    p.add_argument('--base_dist', type=str, default='l2', choices=['l2', 'cosine'])
    p.add_argument('--sel_filter', action='store_true', default=False)
    p.add_argument('--sel_pca', action='store_true', default=False)
    p.add_argument('--sel_pool', type=float, default=3.0)
    p.add_argument('--sel_mmr', action='store_true', default=False)
    p.add_argument('--sel_mu', type=float, default=1.0)
    p.add_argument('--sel_dpp', action='store_true', default=False)
    p.add_argument('--sel_alpha', type=float, default=1.0)
    p.add_argument('--use_jacobian_score', action='store_true', default=False,
                   help='naming only: replay the poison cache selected with the '
                        'exact Jacobian-aware pointwise score')
    p.add_argument('--jacobian_weight', type=float, default=1.0,
                   help='Jacobian score weight used by the attack run being replayed')
    p.add_argument('--jacobian_batch_size', type=int, default=64,
                   help='recorded attack-time Jacobian batch size; it does not '
                        'change run identity or defense computation')
    p.add_argument('--target_select', type=FU.target_select_arg, default='easiest',
                   help='label only here: it just has to match the attack run so '
                        'the _tgt<N> suffix of the run name lines up')

    # ---- which of the cached targets to replay ----
    p.add_argument('--num_targets', type=int, default=0,
                   help='0 = every target with saved poisons')
    p.add_argument('--target_idx_file', type=str, default=None,
                   help='restrict to these targets (same format as ours.sh writes)')

    # ---- victims. Keep these identical to the attack run. ----
    p.add_argument('--num_victims', type=int, default=6)
    p.add_argument('--victim_epochs', type=int, default=60)
    p.add_argument('--victim_lr', type=float, default=0.1)
    p.add_argument('--victim_bs', type=int, default=128)
    p.add_argument('--victim_decay', nargs='*', type=int, default=[35, 45])
    p.add_argument('--victim_wd', type=float, default=0.0)
    # none | standard | randaug | cutout | dsa -- applied to the already-poisoned
    # image, resampled every epoch; composes with any --defense
    VA.add_args(p)
    p.add_argument('--clean_baseline', action='store_true', default=False,
                   help='train/load a DEFENDED clean victim pool for the CTA '
                        'baseline and the clean ASR of each target')

    # ---- the defense ----
    p.add_argument('--defense', type=str, default='none',
                   help="none | epic | friends | noise | advtrain, '+'-joined to "
                        "compose, e.g. epic+friends")

    p.add_argument('--epic_subset_size', type=float, default=0.1,
                   help='B = this * N medoids per selection round (EPIC CIFAR-10 '
                        'from-scratch default)')
    p.add_argument('--epic_freq', type=int, default=2,
                   help='select every this many epochs (EPIC 40-epoch recipe)')
    p.add_argument('--epic_drop_after', type=int, default=10,
                   help='no dropping before this epoch')
    p.add_argument('--epic_stop_after', type=int, default=-1,
                   help='no dropping from this epoch on; -1 = the last lr decay')
    p.add_argument('--epic_cluster_thresh', type=float, default=1.0,
                   help='drop points whose medoid cluster is <= this big')
    p.add_argument('--epic_metric', type=str, default='euclidean',
                   choices=['euclidean', 'cosine'])

    p.add_argument('--noise_type', type=str, nargs='*', default=None,
                   choices=NOISE_KINDS,
                   help="default: friendly+bernoulli for --defense friends, "
                        "bernoulli for --defense noise")
    p.add_argument('--noise_eps', type=float, default=8.0,
                   help='random noise strength, in /255')
    p.add_argument('--friendly_begin_epoch', type=int, default=5)
    p.add_argument('--friendly_epochs', type=int, default=30)
    p.add_argument('--friendly_lr', type=float, default=100.0)
    p.add_argument('--friendly_momentum', type=float, default=0.9)
    p.add_argument('--friendly_mu', type=float, default=1.0)
    p.add_argument('--friendly_clamp', type=float, default=16.0,
                   help='friendly noise is clamped to +/- this /255')
    p.add_argument('--friendly_bs', type=int, default=512,
                   help='compute batch for the noise generation; the lr is '
                        'rescaled by bs/128 so the trajectory matches the '
                        "reference's batch of 128")

    p.add_argument('--adv_eps', type=float, default=8.0, help='in /255')
    p.add_argument('--adv_steps', type=int, default=5)
    p.add_argument('--adv_step_size', type=float, default=0.0,
                   help='in /255; 0 = 2.5 * eps / steps')

    # ---- bookkeeping ----
    p.add_argument('--gpus', type=str, default='all')
    p.add_argument('--no_parallel_trials', dest='parallel_trials',
                   action='store_false', default=True)
    p.add_argument('--no_resume', action='store_true', default=False)
    p.add_argument('--list_cached', action='store_true', default=False,
                   help='print the targets this attack run has poisons for, as '
                        'json, and exit (used by defense.sh to pair the runs)')

    args = p.parse_args(argv)

    on = [n for n, v in [('--sel_filter', args.sel_filter), ('--sel_mmr', args.sel_mmr),
                         ('--sel_dpp', args.sel_dpp), ('--sel_pca', args.sel_pca)] if v]
    if len(on) > 1:
        p.error('%s are mutually exclusive -- pick one' % ' / '.join(on))
    if on and args.base != 'ours':
        p.error('%s only affects --base ours (got --base %s)' % (on[0], args.base))
    if args.jacobian_weight < 0:
        p.error('--jacobian_weight must be nonnegative')
    if args.jacobian_batch_size <= 0:
        p.error('--jacobian_batch_size must be positive')
    if args.use_jacobian_score and args.base != 'ours':
        p.error('--use_jacobian_score only applies to --base ours')
    args.sel_mode = ({'--sel_filter': 'filter', '--sel_mmr': 'mmr',
                      '--sel_dpp': 'dpp', '--sel_pca': 'pca'}[on[0]] if on else None)

    try:
        args.defense_parts = defense_parts(args.defense)
    except ValueError as e:
        p.error(str(e))
    if args.noise_type is None:
        args.noise_type = (['friendly', 'bernoulli'] if 'friends' in args.defense_parts
                           else (['bernoulli'] if 'noise' in args.defense_parts else []))
    args.noise_type = list(dict.fromkeys(args.noise_type))     # dedup, keep order
    if 'friends' in args.defense_parts and 'friendly' not in args.noise_type:
        p.error("--defense friends without 'friendly' in --noise_type is just "
                "--defense noise; say that instead")
    if 'noise' in args.defense_parts and not args.noise_type:
        p.error('--defense noise needs --noise_type')
    if 'friends' in args.defense_parts and 'noise' in args.defense_parts:
        p.error("--defense friends already covers 'noise'; put every component "
                "in --noise_type instead")
    # --recompute_deltas / --FORCE never exist here: this script never crafts.
    args.recompute_deltas = False
    return args


if __name__ == '__main__':
    mp.set_start_method('spawn', force=True)
    main(parse_args())
