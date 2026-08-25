"""
main.py

Clean-label poisoning experiment driver.

Compares two downstream perturbation-crafting objectives (Feature Collision /
"FC" and Gradient Matching / "GRAD", i.e. Witches' Brew-style) crossed with two
base-selection strategies ("random" and "ours", the target-conditioned
s_i = B_i + coef * R_i score from the paper).

Highly Optimized Crafting Engine:
- Uses `_flat_grad` for instant GradMatch cosine similarity (avoids graph fragmentation).
- Dynamically freezes surrogate parameters during FC to prevent VRAM bloat.
- Uses pure PGD for FC and Signed Adam for GRAD.
- Fast first-order approximation via --fast_gradmatch.
"""

import argparse
import csv
import json
import logging
import os
import random
import sys
import time
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from _old_.networks import ConvNet, VGG, ResNet20BN
from _old_.utils import get_dataset


SUPPORTED_MODELS = ['ConvNetBN', 'VGG13BN', 'ResNet20BN']
SUPPORTED_ATTACKS = ['FC', 'GRAD']
SUPPORTED_BASES = ['random', 'ours']

CLASS_PAIRS = {
    'dog-bird': ('dog', 'bird'),
    'frog-airplane': ('frog', 'airplane'),
}


# --------------------------------------------------------------------------- #
# Fast Math Helpers
# --------------------------------------------------------------------------- #

def _flat_grad(grads):
    """Flattens a list of gradient tensors into a single 1D vector."""
    return torch.cat([g.reshape(-1) for g in grads])

def _cosine(a, b, eps=1e-8):
    """Computes cosine similarity between two 1D vectors."""
    return torch.dot(a, b) / (a.norm() * b.norm() + eps)


# --------------------------------------------------------------------------- #
# Setup / args
# --------------------------------------------------------------------------- #

def parse_args():
    p = argparse.ArgumentParser()

    p.add_argument('--dataset', type=str, default='CIFAR10')
    p.add_argument('--data_path', type=str, default='./data')
    p.add_argument('--data_frac', type=float, default=1.0)

    p.add_argument('--model', type=str, required=True, choices=SUPPORTED_MODELS)
    p.add_argument('--attack', type=str, default=None, choices=SUPPORTED_ATTACKS)
    p.add_argument('--base', type=str, default=None, choices=SUPPORTED_BASES)
    p.add_argument('--class_pair', type=str, default=None, choices=list(CLASS_PAIRS.keys()))

    # 'ours' base-selection hyperparameters
    p.add_argument('--num_surrogate', type=int, default=5)
    p.add_argument('--rel_metric', type=str, default='cosine', choices=['cosine', 'l2'])
    p.add_argument('--score_combine', type=str, default='product',
                   choices=['product', 'sum', 'rank_product'])
    p.add_argument('--coef', type=float, default=1.0)

    # poisoning / crafting hyperparameters
    p.add_argument('--budget', type=float, default=None)
    p.add_argument('--num_poisons', type=int, default=25)
    p.add_argument('--craft_ensemble', type=int, default=1)
    p.add_argument('--epsilon', type=float, default=8 / 255)
    p.add_argument('--craft_steps', type=int, default=250)
    p.add_argument('--craft_lr', type=float, default=0.01)
    p.add_argument('--restarts', type=int, default=8,
                   help="Number of random restarts for GRAD crafting.")
    p.add_argument('--fast_gradmatch', action='store_true', default=False,
                   help="First-order approximation for GRAD. ~3x faster per iteration.")

    # target / victim protocol
    p.add_argument('--num_targets', type=int, default=10)
    p.add_argument('--num_victims', type=int, default=6)
    p.add_argument('--num_baselines', type=int, default=None,
                   help="How many clean (un-poisoned) baseline models to train. These are only "
                        "used for the CTA reference number, the target-eligibility veto and the "
                        "free-win audit -- they are NOT the poisoned victims, so there is no "
                        "reason to keep the two counts locked together. Default: fall back to "
                        "--num_victims (the old behaviour). Changing it changes the veto "
                        "ensemble, hence the target-selection cache key.")
    p.add_argument('--victim_epochs', type=int, default=60)
    p.add_argument('--victim_lr', type=float, default=0.1)
    p.add_argument('--victim_bs', type=int, default=125)
    p.add_argument('--victim_decay', type=int, nargs='+', default=[35, 45])
    p.add_argument('--no_augmentation', action='store_true', default=True)

    # Target easiness
    p.add_argument('--ref_model', type=str, default='ResNet20BN', choices=SUPPORTED_MODELS)
    p.add_argument('--target_easiness', type=float, default=None)
    p.add_argument('--target_difficulty', type=float, default=None)
    p.add_argument('--target_pool', type=str, default='all', choices=['all', 'pair'])
    p.add_argument('--target_min_margin', type=float, default=0.05)
    p.add_argument('--target_eligibility_ensemble', type=int, default=3)
    p.add_argument('--target_margin_low', type=float, default=0.05)
    p.add_argument('--target_margin_high', type=float, default=0.5)

    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--cache_dir', type=str, default='./cache')
    p.add_argument('--out_dir', type=str, default='./runs')

    # Progress logging. Every stage here (training one net, crafting one target's poisons)
    # runs for minutes and used to print nothing until it finished, which is
    # indistinguishable from a hang. These control how chatty the two inner loops are.
    p.add_argument('--log_progress_secs', type=float, default=120.0,
                   help="Wall-clock throttle for the inner-loop progress lines (per-epoch "
                        "training lines, per-step crafting lines): at most one line per net "
                        "/ per craft every N seconds. Counting epochs instead was fine for "
                        "ONE net, but a run trains num_targets x num_victims of them and a "
                        "line every 5 epochs buried the actual results under ~700 identical "
                        "progress lines. Time-throttling keeps the 'not hung' signal at a "
                        "fixed cost per minute however many nets get trained. 0 = no "
                        "progress lines at all (stage start/finish only).")
    p.add_argument('--log_every_epochs', type=int, default=0,
                   help="Optional EXTRA condition on the training-progress line: never print "
                        "more often than every N epochs. 0 (default) = no epoch condition, "
                        "--log_progress_secs alone decides.")
    p.add_argument('--log_every_craft_steps', type=int, default=0,
                   help="Optional EXTRA condition on the crafting-progress line: never print "
                        "more often than every N optimization steps. 0 (default) = no step "
                        "condition, --log_progress_secs alone decides.")
    p.add_argument('--log_every_victim', action='store_true', default=False,
                   help="Log a start line for every single victim retrain. Off by default: "
                        "with num_targets x num_victims retrains those lines say nothing the "
                        "per-victim result line does not already say.")

    p.add_argument('--save_victim_ckpts', action='store_true', default=False)
    p.add_argument('--recompute_baseline', action='store_true', default=False,
                   help="Retrain the clean-baseline checkpoints even when a cache entry for "
                        "this exact training protocol exists. --no_resume deliberately does "
                        "NOT do this any more: it governs per-run work (results rows, bases, "
                        "deltas), not pre-trained checkpoints, and retraining num_victims "
                        "baselines at every budget was costing ~25 min per run for models "
                        "that are bit-identical to the ones already on disk.")
    p.add_argument('--recompute_reference', action='store_true', default=False)
    p.add_argument('--recompute_targets', action='store_true', default=False)
    p.add_argument('--recompute_surrogates', action='store_true', default=False)
    p.add_argument('--recompute_deltas', action='store_true', default=False)
    p.add_argument('--no_resume', action='store_true', default=False)
    p.add_argument('--baseline_only', action='store_true', default=False)
    p.add_argument('--baseline_victim_id', type=int, default=None)

    args = p.parse_args()
    if not args.baseline_only:
        missing = [name for name, val in [('--attack', args.attack), ('--base', args.base),
                                          ('--class_pair', args.class_pair),
                                          ('--budget', args.budget)] if val is None]
        if missing:
            p.error(f"the following arguments are required unless --baseline_only is set: "
                    f"{', '.join(missing)}")
        if args.craft_ensemble < 1:
            p.error('--craft_ensemble must be >= 1')
    if not 0.0 < args.data_frac <= 1.0:
        p.error('--data_frac must be in (0, 1]')
    if args.target_difficulty is not None:
        args.target_easiness = 1.0 - args.target_difficulty
    if args.num_baselines is None:
        args.num_baselines = args.num_victims
    elif args.num_baselines < 1:
        p.error('--num_baselines must be >= 1')
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    return args


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# --------------------------------------------------------------------------- #
# Networks
# --------------------------------------------------------------------------- #

def build_network(model_name, channel, num_classes, im_size, seed):
    set_seed(seed)
    if model_name == 'ConvNetBN':
        net = ConvNet(channel=channel, num_classes=num_classes, net_width=128, net_depth=3,
                      net_act='relu', net_norm='batchnorm', net_pooling='avgpooling', im_size=im_size)
    elif model_name == 'VGG13BN':
        net = VGG('VGG13', channel=channel, num_classes=num_classes, norm='batchnorm')
    elif model_name == 'ResNet20BN':
        net = ResNet20BN(channel=channel, num_classes=num_classes)
    else:
        raise ValueError(f'Unsupported model {model_name}')
    return net


# --------------------------------------------------------------------------- #
# Data materialization (Fast In-Memory Tensors)
# --------------------------------------------------------------------------- #

def normalize(x, mean, std, device):
    """Normalize into `device`. x is moved there first: the callers use this both to
    normalize a tensor already on the GPU (crafting) and to normalize the raw CPU training
    set straight onto the GPU, and only building mean/std on `device` made the second case
    a cross-device subtraction. .to() is a no-op when x is already there, and stays
    differentiable, so the crafting path is unaffected."""
    x = x.to(device)
    mean_t = torch.tensor(mean, device=device).view(1, -1, 1, 1)
    std_t = torch.tensor(std, device=device).view(1, -1, 1, 1)
    return (x - mean_t) / std_t


def materialize_raw_train(dst_train):
    if hasattr(dst_train, 'data'):
        data = dst_train.data
        imgs = torch.from_numpy(np.asarray(data)).float()
        if imgs.dim() == 3:
            imgs = imgs.unsqueeze(1)
        elif imgs.dim() == 4 and imgs.shape[-1] in (1, 3):
            imgs = imgs.permute(0, 3, 1, 2)
        imgs = imgs / 255.0
        if hasattr(dst_train, 'targets'):
            labels = torch.tensor(dst_train.targets).long()
        elif hasattr(dst_train, 'labels'):
            labels = torch.tensor(dst_train.labels).long()
        return imgs, labels
    imgs, labels = [], []
    for i in range(len(dst_train)):
        x, y = dst_train[i]
        imgs.append(x)
        labels.append(y)
    return torch.stack(imgs), torch.tensor(labels).long()


def frac_tag(args):
    return '' if args.data_frac >= 1.0 else f'_frac{args.data_frac:g}'


def proto_tag(args):
    """Cache-key suffix for the victim TRAINING PROTOCOL.

    Checkpoints trained under different victim hyperparameters are not interchangeable: a
    60-epoch/bs256 clean baseline compared against 50-epoch/bs125 victims produces a
    NEGATIVE CTA drop (the poisoned models come out more accurate than the "clean"
    reference), and the veto / free-win check run on those same mismatched models certifies
    nothing about the victims actually being attacked. So the protocol goes in every
    checkpoint filename: changing --victim_epochs / --victim_lr / --victim_bs /
    --victim_decay lands on a fresh cache entry instead of silently reusing the old one."""
    decay = '-'.join(str(d) for d in args.victim_decay)
    return f'_ep{args.victim_epochs}bs{args.victim_bs}lr{args.victim_lr:g}dec{decay}'


def load_first_cached(net, paths, device):
    """Load the first checkpoint in `paths` that exists and actually fits `net`.

    Returns the path that was loaded, or None if nothing usable was on disk. Legacy cache
    names are listed after the canonical one so that tightening a cache key never costs a
    retrain of a network that is already saved; a stale name that turns out to hold a
    different architecture just fails to load and we fall through to training.

    A hit on a legacy name is copied to the canonical one (paths[0]), so the next run finds
    it there directly -- without that, a checkpoint saved under an over-specific old key
    would still be retrained the first time the run uses a different --class_pair."""
    for p in paths:
        if not os.path.exists(p):
            continue
        try:
            net.load_state_dict(torch.load(p, map_location=device))
        except RuntimeError as e:
            logging.warning(f'[cache] {os.path.basename(p)} exists but does not fit this '
                            f'architecture, ignoring it ({e})')
            continue
        if p != paths[0] and not os.path.exists(paths[0]):
            os.makedirs(os.path.dirname(paths[0]), exist_ok=True)
            torch.save(net.state_dict(), paths[0])
            logging.info(f'[cache] promoted {os.path.basename(p)} -> '
                         f'{os.path.basename(paths[0])} (pair-independent name)')
        return p
    return None


class ProgressThrottle:
    """Rate-limits an inner-loop progress line to one per `secs` of wall clock.

    The point of those lines is to prove the run is not hung, and that costs one line a
    minute -- it does not cost one line per N epochs multiplied by every net in the run."""

    def __init__(self, secs):
        self.secs = float(secs or 0)
        self.t_last = time.time()

    def ready(self):
        if self.secs <= 0:
            return False
        now = time.time()
        if now - self.t_last < self.secs:
            return False
        self.t_last = now
        return True


def fmt_dur(seconds):
    """m:ss under an hour, h:mm:ss above it."""
    seconds = int(round(seconds))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f'{h}:{m:02d}:{s:02d}' if h else f'{m}:{s:02d}'


def subsample_train(images_raw, labels, frac, seed):
    if frac >= 1.0:
        return images_raw, labels
    rng = np.random.RandomState(seed)
    keep = []
    for c in labels.unique().tolist():
        idx_c = (labels == c).nonzero(as_tuple=True)[0].numpy()
        k = max(1, int(round(frac * len(idx_c))))
        keep.append(rng.choice(idx_c, size=k, replace=False))
    keep = np.sort(np.concatenate(keep))
    keep_t = torch.from_numpy(keep).long()
    return images_raw[keep_t], labels[keep_t]


def materialize_normalized_test(dst_test):
    imgs, labels = [], []
    for i in range(len(dst_test)):
        x, y = dst_test[i]
        imgs.append(x)
        labels.append(y)
    return torch.stack(imgs), torch.tensor(labels).long()


# --------------------------------------------------------------------------- #
# FAST Training loop (Bypasses CPU DataLoader entirely)
# --------------------------------------------------------------------------- #

def train_from_scratch(net, images_norm, labels, test_images, test_labels, epochs, lr, batch_size, decay_epochs, args,
                       tag=None, announce=True):
    """Train `net` from scratch. `tag` names this net in the progress log ('victim 2/6' etc.);
    pass None to train silently. `announce=False` drops the start/finish pair for callers
    that already log their own (the victim loop), keeping the throttled progress lines.

    Running loss/accuracy are accumulated as GPU tensors and only pulled to the host on the
    epochs that actually print, so the progress line costs one sync per printed line rather
    than one per batch."""
    net = net.to(args.device)
    optimizer = torch.optim.SGD(net.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
    criterion = nn.CrossEntropyLoss()

    N = images_norm.shape[0]
    decay_epochs = set(decay_epochs)
    cur_lr = lr
    every = args.log_every_epochs if tag else 0
    throttle = ProgressThrottle(args.log_progress_secs if tag else 0)
    t_start = time.time()

    if tag and announce:
        logging.info(f'  [{tag}] training: {epochs} epochs, {N} examples, bs={batch_size}, '
                     f'lr={lr:g}, decay at {sorted(decay_epochs)}')

    for ep in range(epochs):
        if ep in decay_epochs:
            cur_lr *= 0.1
            for g in optimizer.param_groups:
                g['lr'] = cur_lr

        net.train()
        run_loss = torch.zeros((), device=args.device)
        run_correct = torch.zeros((), device=args.device)
        perm = torch.randperm(N, device=args.device)
        for i in range(0, N, batch_size):
            idx = perm[i:i+batch_size]
            x = images_norm[idx]
            y = labels[idx]

            optimizer.zero_grad()
            out = net(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()

            run_loss += loss.detach() * y.numel()
            run_correct += (out.detach().argmax(dim=1) == y).sum()

        # No forced first/last epoch line: the start and finish lines already bracket the
        # run, so those two only ever duplicated them.
        if tag and (not every or (ep + 1) % every == 0) and throttle.ready():
            el = time.time() - t_start
            eta = el / (ep + 1) * (epochs - ep - 1)
            logging.info(f'  [{tag}] epoch {ep + 1}/{epochs} '
                         f'loss={run_loss.item() / N:.4f} acc={run_correct.item() / N:.4f} '
                         f'lr={cur_lr:g} | {fmt_dur(el)} elapsed, ~{fmt_dur(eta)} left')

    net.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for i in range(0, len(test_images), 512):
            x = test_images[i:i+512]
            y = test_labels[i:i+512]
            pred = net(x).argmax(dim=1)
            correct += (pred == y).sum().item()
            total += len(y)

    acc = correct / max(total, 1)
    if tag and announce:
        logging.info(f'  [{tag}] done in {fmt_dur(time.time() - t_start)} | test acc = {acc:.4f}')
    return net, acc


def evaluate_clean_accuracy(net, images_norm, labels, exclude_idx, device, batch_size=512):
    net.eval()
    keep = [i for i in range(images_norm.shape[0]) if i not in exclude_idx]
    correct, total = 0, 0
    with torch.no_grad():
        for start in range(0, len(keep), batch_size):
            idx = keep[start:start + batch_size]
            x = images_norm[idx].to(device)
            y = labels[idx].to(device)
            pred = net(x).argmax(dim=1)
            correct += (pred == y).sum().item()
            total += len(idx)
    return correct / max(total, 1)


# --------------------------------------------------------------------------- #
# Reference model + target selection
# --------------------------------------------------------------------------- #

def reference_member_path(args, member):
    """Canonical checkpoint path for reference member `member`.

    The reference models are trained on the whole clean training set, so NOTHING about them
    depends on which class pair the attack targets -- the pair used to be in this filename
    and bought nothing but a retrain of identical networks every time --class_pair changed.
    The architecture, which the old key omitted, does matter and is in the name now."""
    base = os.path.join(args.cache_dir, 'reference_model',
                        f'{args.dataset}_{args.ref_model}_seed{args.seed}'
                        f'{proto_tag(args)}{frac_tag(args)}')
    return base + ('.pt' if member == 0 else f'_e{member}.pt')


def reference_member_paths(args, member):
    """Canonical path first, then the pre-rename name, so old caches are still reused."""
    legacy = os.path.join(args.cache_dir, 'reference_model',
                          f'{args.dataset}_{args.class_pair}_seed{args.seed}'
                          f'{proto_tag(args)}{frac_tag(args)}')
    legacy += '.pt' if member == 0 else f'_e{member}.pt'
    return [reference_member_path(args, member), legacy]


def get_reference_model(args, channel, num_classes, im_size, images_train_norm, labels_train,
                        test_imgs, test_labs, member=0):
    cache_path = reference_member_path(args, member)
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)

    seed = args.seed if member == 0 else args.seed + 20000 + member
    name = f'reference {member + 1}/{max(1, args.target_eligibility_ensemble)}'
    net = build_network(args.ref_model, channel, num_classes, im_size, seed=seed)
    loaded = None if args.recompute_reference else load_first_cached(
        net, reference_member_paths(args, member), args.device)
    if loaded is not None:
        logging.info(f'[{name}] loaded {os.path.basename(loaded)}')
        return net.to(args.device).eval()

    logging.info(f'[{name}] no cache for this protocol, training -> {os.path.basename(cache_path)}')
    net, acc_test = train_from_scratch(net, images_train_norm, labels_train, test_imgs, test_labs,
                                       epochs=args.victim_epochs, lr=args.victim_lr,
                                       batch_size=args.victim_bs, decay_epochs=args.victim_decay,
                                       args=args, tag=name)
    logging.info(f'[{name}] test acc = {acc_test:.4f}')
    torch.save(net.state_dict(), cache_path)
    return net.eval()


def get_reference_ensemble(args, channel, num_classes, im_size, images_train_norm, labels_train,
                           test_imgs, test_labs):
    n = max(1, args.target_eligibility_ensemble)
    logging.info(f'[stage 3/4] reference ensemble: {n}x {args.ref_model} '
                 f'(scores target easiness + votes on eligibility)')
    t0 = time.time()
    nets = [get_reference_model(args, channel, num_classes, im_size, images_train_norm,
                                labels_train, test_imgs, test_labs, member=j)
            for j in range(n)]
    logging.info(f'[stage 3/4] reference ensemble ready in {fmt_dur(time.time() - t0)}')
    return nets


def is_eligible_target(ens_pred_idx, worst_margin, poison_class_idx, min_margin):
    return (ens_pred_idx != poison_class_idx
            and worst_margin > 0.0
            and worst_margin >= min_margin)


TARGET_METRIC_VERSION = 'ens_padv_easiness_vetomargin_v6'

def score_candidates(ref_nets, veto_nets, images_test_norm, candidates, poison_class_idx,
                     device, batch_size=512):
    scoring_ids = {id(n) for n in ref_nets}
    extra_nets = [n for n in veto_nets if id(n) not in scoring_ids]
    for net in list(ref_nets) + extra_nets:
        net.eval()

    easiness, ens_pred, worst_margin = {}, {}, {}
    with torch.no_grad():
        for start in range(0, len(candidates), batch_size):
            idx = candidates[start:start + batch_size]
            x = images_test_norm[idx].to(device)

            sum_logits, margin = None, None
            for net in list(ref_nets) + extra_nets:
                logits = net(x)
                if id(net) in scoring_ids:
                    sum_logits = logits if sum_logits is None else sum_logits + logits
                probs = torch.softmax(logits, dim=1)
                m = probs.max(dim=1).values - probs[:, poison_class_idx]
                margin = m if margin is None else torch.minimum(margin, m)

            probs = torch.softmax(sum_logits / len(ref_nets), dim=1)
            p_adv = probs[:, poison_class_idx].cpu().tolist()
            pred = probs.argmax(dim=1).cpu().tolist()
            margin = margin.cpu().tolist()
            for i, pa, pr, mg in zip(idx, p_adv, pred, margin):
                easiness[i] = pa
                ens_pred[i] = pr
                worst_margin[i] = mg
    return easiness, ens_pred, worst_margin


def select_targets(args, ref_nets, veto_nets, veto_desc, images_test_norm, labels_test, class_names):
    ref_nets = list(ref_nets)
    veto_nets = list(veto_nets)
    spectrum = args.target_easiness is not None
    key = (f'_ease{args.target_easiness:g}' if spectrum
           else f'_margin{args.target_margin_low:g}-{args.target_margin_high:g}')
    if len(ref_nets) > 1:
        key += f'_ens{len(ref_nets)}'
    if args.target_pool == 'pair':
        key += '_pair'
    key += f'_veto{args.model}{len(veto_nets)}m{args.target_min_margin:g}'
    cache_path = os.path.join(
        args.cache_dir, 'targets',
        f'{args.dataset}_{args.class_pair}_seed{args.seed}{frac_tag(args)}{key}.json')
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    
    if os.path.exists(cache_path) and not args.recompute_targets:
        with open(cache_path) as f:
            cached = json.load(f)
        cached_ids = cached['target_ids']
        stale = spectrum and cached.get('metric_version') != TARGET_METRIC_VERSION
        if stale:
            logging.info(f'[select_targets] {cache_path} stale; reselecting targets.')
        cached_scores = {int(k): v for k, v in cached.get('easiness', {}).items()}
        if len(cached_ids) >= args.num_targets and not stale:
            ids = cached_ids[:args.num_targets]
            return ids, {i: cached_scores.get(i) for i in ids}

    target_class_name, poison_class_name = CLASS_PAIRS[args.class_pair]
    target_class_idx = class_names.index(target_class_name)
    poison_class_idx = class_names.index(poison_class_name)

    if args.target_pool == 'pair':
        pool_mask = labels_test == target_class_idx
    else:
        pool_mask = labels_test != poison_class_idx
    candidates = pool_mask.nonzero(as_tuple=True)[0].tolist()

    easiness, ens_pred, worst_margin = score_candidates(
        ref_nets, veto_nets, images_test_norm, candidates, poison_class_idx, args.device)

    min_margin = max(args.target_min_margin, 0.0)

    if spectrum:
        eligible = [i for i in candidates
                    if is_eligible_target(ens_pred[i], worst_margin[i], poison_class_idx, min_margin)]
        if not eligible:
            raise RuntimeError('No eligible targets available. Lower --target_min_margin.')
        ordered = sorted(eligible, key=lambda i: (-easiness[i], i))
        n = len(ordered)
        k = min(args.num_targets, n)
        e = min(max(args.target_easiness, 0.0), 1.0)
        start = int(round((1.0 - e) * (n - k)))
        chosen = ordered[start:start + k]
    else:
        eligible = [i for i in candidates
                    if is_eligible_target(ens_pred[i], worst_margin[i], poison_class_idx, min_margin)]
        margin = {i: 1.0 - easiness[i] for i in eligible}
        filtered = [i for i in eligible if args.target_margin_low <= margin[i] <= args.target_margin_high]
        if len(filtered) >= args.num_targets:
            chosen = sorted(filtered, key=lambda i: (margin[i], i))[:args.num_targets]
        else:
            mid = (args.target_margin_low + args.target_margin_high) / 2.0
            chosen = sorted(eligible, key=lambda i: (abs(margin[i] - mid), i))[:args.num_targets]

    with open(cache_path, 'w') as f:
        json.dump({'metric_version': TARGET_METRIC_VERSION,
                   'target_easiness_requested': args.target_easiness,
                   'target_ids': chosen,
                   'easiness': {str(k): easiness[k] for k in chosen},
                   'worst_margin': {str(k): worst_margin[k] for k in chosen}}, f, indent=2)
    return chosen, {i: easiness[i] for i in chosen}


# --------------------------------------------------------------------------- #
# Clean baseline
# --------------------------------------------------------------------------- #

def train_or_load_one_baseline(args, model_name, v, channel, num_classes, im_size,
                               images_train_norm, labels_train, test_imgs, test_labs):
    cache_path = os.path.join(args.cache_dir, 'clean_baseline',
                              f'{args.dataset}_{model_name}_seed{args.seed}'
                              f'{proto_tag(args)}{frac_tag(args)}_v{v}.pt')
    sidecar_path = cache_path[:-3] + '.json'
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    baseline_seed = args.seed * 10000 + 999900 + v

    net = build_network(model_name, channel, num_classes, im_size, seed=baseline_seed)
    if os.path.exists(cache_path) and not args.recompute_baseline:
        net.load_state_dict(torch.load(cache_path, map_location=args.device))
        net = net.to(args.device)
        net.eval()

        # The accuracy of a checkpoint that has not changed cannot have changed either, and
        # it was written to the sidecar the first time round. Re-running the full test set
        # per baseline per budget just to reprint the same number is pure waste; only fall
        # back to measuring it when the sidecar is missing or unreadable (older caches).
        acc_test = None
        if os.path.exists(sidecar_path):
            try:
                with open(sidecar_path) as f:
                    acc_test = float(json.load(f)['test_acc'])
            except (ValueError, KeyError, OSError):
                acc_test = None

        if acc_test is None:
            correct, total = 0, 0
            with torch.no_grad():
                for i in range(0, len(test_imgs), 512):
                    x = test_imgs[i:i+512]
                    y = test_labs[i:i+512]
                    pred = net(x).argmax(dim=1)
                    correct += (pred == y).sum().item()
                    total += len(y)
            acc_test = correct / max(total, 1)
            src = 'measured'
        else:
            src = 'cached'
        logging.info(f'[clean baseline] v{v + 1}/{args.num_baselines}: loaded {os.path.basename(cache_path)} '
                     f'| test acc = {acc_test:.4f} ({src})')
    else:
        logging.info(f'[clean baseline] v{v + 1}/{args.num_baselines}: no cache for this protocol, '
                     f'training -> {os.path.basename(cache_path)}')
        net, acc_test = train_from_scratch(net, images_train_norm, labels_train, test_imgs, test_labs,
                                           epochs=args.victim_epochs, lr=args.victim_lr,
                                           batch_size=args.victim_bs, decay_epochs=args.victim_decay,
                                           args=args, tag=f'clean baseline v{v + 1}/{args.num_baselines}')
        torch.save(net.state_dict(), cache_path)

    with open(sidecar_path, 'w') as f:
        json.dump({'dataset': args.dataset, 'model': model_name, 'seed': args.seed,
                   'victim_id': v, 'test_acc': acc_test}, f)

    return net.eval(), acc_test


def get_clean_baseline(args, model_name, channel, num_classes, im_size,
                       images_train_norm, labels_train, test_imgs, test_labs):
    logging.info(f'[stage 1/4] clean baselines: {args.num_baselines}x {model_name} '
                 f'(protocol{proto_tag(args)})')
    t0 = time.time()
    nets, accs = [], []
    for v in range(args.num_baselines):
        net, acc = train_or_load_one_baseline(args, model_name, v, channel, num_classes, im_size,
                                              images_train_norm, labels_train, test_imgs, test_labs)
        nets.append(net)
        accs.append(acc)
    logging.info(f'[stage 1/4] clean baselines ready in {fmt_dur(time.time() - t0)}')
    return float(np.mean(accs)), float(np.std(accs)), nets


def audit_free_wins(nets, images_test_norm, target_ids, poison_class_idx, class_names, device):
    counts = {}
    with torch.no_grad():
        x = images_test_norm[list(target_ids)].to(device)
        for net in nets:
            net.eval()
            pred = net(x).argmax(dim=1).cpu().tolist()
            for t, p in zip(target_ids, pred):
                counts[t] = counts.get(t, 0) + int(p == poison_class_idx)
    return counts


def summarize_results(rows):
    per_target_success = defaultdict(list)
    all_cta = []
    for r in rows:
        per_target_success[r['target_id']].append(r['success'])
        all_cta.append(r['clean_test_acc'])
    per_target_asr = [float(np.mean(v)) for v in per_target_success.values()]
    return {
        'asr_mean': float(np.mean(per_target_asr)) if per_target_asr else None,
        'asr_std': float(np.std(per_target_asr)) if per_target_asr else None,
        'cta_post_mean': float(np.mean(all_cta)) if all_cta else None,
        'cta_post_std': float(np.std(all_cta)) if all_cta else None,
        'num_targets': len(per_target_success),
        'num_trials': len(rows),
    }


# --------------------------------------------------------------------------- #
# Surrogate ensemble
# --------------------------------------------------------------------------- #

def num_scoring_surrogates(args):
    return args.num_surrogate if args.base == 'ours' else 1


def get_surrogate_ensemble(args, channel, num_classes, im_size, images_train_norm, labels_train, test_imgs, test_labs):
    K = max(num_scoring_surrogates(args), args.craft_ensemble)
    logging.info(f'[stage 2/4] surrogate ensemble: {K}x {args.model} (protocol{proto_tag(args)})')
    t0 = time.time()
    nets = []
    for k in range(K):
        # Surrogates are trained on the whole clean training set with a pair-independent
        # seed, so the class pair that used to sit in this filename only ever caused the
        # same K networks to be retrained once per pair. It is gone from the key; the old
        # name is still checked so nothing already on disk gets retrained.
        cache_path = os.path.join(
            args.cache_dir, 'surrogates',
            f'{args.dataset}_{args.model}_seed{args.seed}'
            f'{proto_tag(args)}{frac_tag(args)}_k{k}.pt')
        legacy_path = os.path.join(
            args.cache_dir, 'surrogates',
            f'{args.dataset}_{args.model}_{args.class_pair}_seed{args.seed}'
            f'{proto_tag(args)}{frac_tag(args)}_k{k}.pt')
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)

        net = build_network(args.model, channel, num_classes, im_size, seed=args.seed + 1000 + k)
        loaded = None if args.recompute_surrogates else load_first_cached(
            net, [cache_path, legacy_path], args.device)
        if loaded is not None:
            net = net.to(args.device)
            logging.info(f'[surrogate {k + 1}/{K}] loaded {os.path.basename(loaded)}')
        else:
            logging.info(f'[surrogate {k + 1}/{K}] no cache for this protocol, '
                         f'training -> {os.path.basename(cache_path)}')
            net, acc_test = train_from_scratch(net, images_train_norm, labels_train, test_imgs, test_labs,
                                               epochs=args.victim_epochs, lr=args.victim_lr,
                                               batch_size=args.victim_bs, decay_epochs=args.victim_decay,
                                               args=args, tag=f'surrogate {k + 1}/{K}')
            logging.info(f'[surrogate {k + 1}/{K}] test acc = {acc_test:.4f}')
            torch.save(net.state_dict(), cache_path)
        net.eval()
        nets.append(net)
    logging.info(f'[stage 2/4] surrogate ensemble ready in {fmt_dur(time.time() - t0)}')
    return nets


# --------------------------------------------------------------------------- #
# Base selection
# --------------------------------------------------------------------------- #

def select_bases_random(candidate_idx, num_poisons, seed):
    rng = np.random.RandomState(seed)
    return rng.choice(candidate_idx, size=min(num_poisons, len(candidate_idx)), replace=False).tolist()


def rank_normalize(v):
    n = v.numel()
    if n <= 1:
        return torch.zeros_like(v)
    order = torch.argsort(v)
    ranks = torch.empty_like(v)
    ranks[order] = torch.arange(n, dtype=v.dtype, device=v.device)
    return ranks / (n - 1)


def select_bases_ours(candidate_idx, images_train_norm, target_norm, surrogate_nets,
                      poison_class_idx, coef, num_poisons, device,
                      rel_metric='cosine', combine='product', batch_size=256):
    K = len(surrogate_nets)
    n = len(candidate_idx)
    b_all = torch.zeros(K, n)
    r_all = torch.zeros(K, n)

    with torch.no_grad():
        target_embeds = [net.embed(target_norm.unsqueeze(0).to(device)) for net in surrogate_nets]

        for start in range(0, n, batch_size):
            chunk = candidate_idx[start:start + batch_size]
            batch = images_train_norm[chunk].to(device)
            sl = slice(start, start + len(chunk))

            for k, (net, t_embed) in enumerate(zip(surrogate_nets, target_embeds)):
                probs = torch.softmax(net(batch), dim=1)
                p_a = probs[:, poison_class_idx]
                probs_masked = probs.clone()
                probs_masked[:, poison_class_idx] = -1.0
                p_max_other, _ = probs_masked.max(dim=1)
                b_k = 1.0 - (p_a - p_max_other)

                feats = net.embed(batch)
                if rel_metric == 'cosine':
                    r_k = F.cosine_similarity(feats, t_embed.expand_as(feats), dim=1)
                elif rel_metric == 'l2':
                    d = torch.norm(feats - t_embed.expand_as(feats), p=2, dim=1)
                    r_k = 1.0 / (1.0 + d) if combine == 'product' else -d
                else:
                    raise ValueError(f'Unknown rel_metric {rel_metric}')

                b_all[k, sl] = b_k.cpu()
                r_all[k, sl] = r_k.cpu()

    scores = torch.zeros(n)
    for k in range(K):
        if combine == 'product':
            s_k = b_all[k] * r_all[k]
        elif combine == 'sum':
            s_k = b_all[k] + coef * r_all[k]
        elif combine == 'rank_product':
            s_k = rank_normalize(b_all[k]) * rank_normalize(r_all[k]).pow(coef)
        scores += s_k / K

    order = torch.argsort(scores, descending=True).tolist()
    top_m = order[:num_poisons]
    return [candidate_idx[i] for i in top_m]


# --------------------------------------------------------------------------- #
# Perturbation crafting: Ultra-Fast
# --------------------------------------------------------------------------- #

def craft_poison(attack, base_imgs_raw, target_norm, surrogate_nets, poison_class_idx,
                 epsilon, craft_steps, craft_lr, restarts, mean, std, device, fast_gradmatch=False,
                 tag='', log_every=0, log_secs=120.0):
    """
    Optimized crafting with Fast Graph Management.
    - FC uses pure PGD with frozen surrogate parameters (stops massive VRAM overhead).
    - GRAD uses `_flat_grad` mapping to stop Python broadcasting fragmentation.
    """
    m = base_imgs_raw.shape[0]
    base_imgs_raw = base_imgs_raw.detach().to(device)
    target_norm = target_norm.detach().to(device)
    criterion = nn.CrossEntropyLoss().to(device)
    poison_label = torch.full((m,), poison_class_idx, dtype=torch.long, device=device)

    # VERY IMPORTANT SPEED FIX:
    # If doing FC, we don't need parameter gradients. Freezing them stops PyTorch 
    # from storing massive intermediate activations during the forward pass.
    requires_grad = (attack == 'GRAD')
    for net in surrogate_nets:
        net.eval()
        for p in net.parameters():
            p.requires_grad_(requires_grad)

    # Precompute target variables
    if attack == 'FC':
        with torch.no_grad():
            target_feats = [net.embed(target_norm.unsqueeze(0)).detach() for net in surrogate_nets]
    else:
        target_grads = []
        for net in surrogate_nets:
            net.zero_grad()
            out = net(target_norm.unsqueeze(0))
            loss_t = criterion(out, torch.tensor([poison_class_idx], device=device))
            params = [p for p in net.parameters()]
            g = torch.autograd.grad(loss_t, params)
            target_grads.append(_flat_grad([gi.detach() for gi in g]))

    best_loss = float('inf')
    best_delta = None

    actual_restarts = restarts if attack == 'GRAD' else 1

    # The crafting objective is the only readout on whether the attack itself is working:
    # for GRAD it is 1 - cos(poison grad, target grad), so 1.0 means "no alignment at all"
    # and lower is better; for FC it is the squared distance between the pooled poison
    # feature and the target feature. A budget sweep whose ASR does not move is a very
    # different story depending on whether this number converges at every m or only at
    # small m, and until it is logged there is no way to tell.
    t_craft = time.time()
    # One throttle for the whole craft, not one per restart -- otherwise 8 restarts buy back
    # 8x the progress lines the throttle was meant to remove.
    throttle = ProgressThrottle(log_secs)
    logging.info(f'[craft{tag}] {attack}: m={m} poisons, {actual_restarts} restart(s) x '
                 f'{craft_steps} steps, eps={epsilon:g}, lr={craft_lr:g}, '
                 f'{len(surrogate_nets)} surrogate(s)'
                 + (', fast (first-order)' if (attack == 'GRAD' and fast_gradmatch) else ''))

    for r in range(actual_restarts):
        t_restart = time.time()
        first_step_loss = None
        delta = torch.empty_like(base_imgs_raw).uniform_(-epsilon, epsilon)
        delta = (torch.clamp(base_imgs_raw + delta, 0.0, 1.0) - base_imgs_raw).detach().requires_grad_(True)
        
        # FC uses raw PGD (no Adam overhead)
        if attack == 'GRAD':
            optimizer = torch.optim.Adam([delta], lr=craft_lr)

        final_step_loss = float('inf')
        
        for step in range(craft_steps):
            if attack == 'GRAD':
                optimizer.zero_grad()
                
            x_adv_raw = torch.clamp(base_imgs_raw + delta, 0.0, 1.0)
            x_adv_norm = normalize(x_adv_raw, mean, std, device)

            if attack == 'FC':
                loss = 0.0
                for k, net in enumerate(surrogate_nets):
                    f = net.embed(x_adv_norm)
                    pooled = f.mean(dim=0, keepdim=True)
                    loss = loss + ((pooled - target_feats[k]) ** 2).sum()
                    
                loss = loss / len(surrogate_nets)
                final_step_loss = loss.item()
                
                grad_delta, = torch.autograd.grad(loss, [delta])
                
                # Raw PGD optimization exactly like the old script
                with torch.no_grad():
                    delta = delta - craft_lr * grad_delta.sign()
                    delta.clamp_(-epsilon, epsilon)
                    delta = torch.clamp(base_imgs_raw + delta, 0.0, 1.0) - base_imgs_raw
                delta = delta.detach().requires_grad_(True)
                
            else: # GRAD
                total_obj = 0.0
                grad_accum = torch.zeros_like(delta)
                
                if fast_gradmatch:
                    for k, net in enumerate(surrogate_nets):
                        params = [p for p in net.parameters() if p.requires_grad]
                        out = net(x_adv_norm)
                        loss_p = criterion(out, poison_label)
                        
                        all_grads = torch.autograd.grad(loss_p, params + [delta])
                        # FAST MATH FIX: Flatten the param gradients BEFORE cosine
                        g_p = _flat_grad(list(all_grads[:-1])).detach()
                        grad_accum = grad_accum + all_grads[-1].detach()
                        total_obj += (1.0 - _cosine(g_p, target_grads[k])).item()
                        
                    total_obj /= len(surrogate_nets)
                    delta.grad = (grad_accum / len(surrogate_nets)).sign()
                    final_step_loss = total_obj
                else:
                    for k, net in enumerate(surrogate_nets):
                        params = [p for p in net.parameters() if p.requires_grad]
                        out = net(x_adv_norm)
                        loss_p = criterion(out, poison_label)
                        
                        # FAST MATH FIX: Flatten the param gradients BEFORE cosine
                        g_p = _flat_grad(torch.autograd.grad(loss_p, params, create_graph=True))
                        total_obj += (1.0 - _cosine(g_p, target_grads[k]))
                        
                    total_obj /= len(surrogate_nets)
                    grad_delta, = torch.autograd.grad(total_obj, [delta])
                    delta.grad = grad_delta.sign()
                    final_step_loss = total_obj.item()

                optimizer.step()
                # Projection for GRAD
                with torch.no_grad():
                    delta.clamp_(-epsilon, epsilon)
                    delta.data = torch.clamp(base_imgs_raw + delta.data, 0.0, 1.0) - base_imgs_raw

            if first_step_loss is None:
                first_step_loss = final_step_loss
            if (not log_every or (step + 1) % log_every == 0) and throttle.ready():
                el = time.time() - t_restart
                eta = el / (step + 1) * (craft_steps - step - 1)
                extra = (f' align={1.0 - final_step_loss:+.4f}' if attack == 'GRAD' else '')
                logging.info(f'  [craft{tag} restart {r + 1}/{actual_restarts}] '
                             f'step {step + 1}/{craft_steps} loss={final_step_loss:.5f}{extra} '
                             f'| {fmt_dur(el)} elapsed, ~{fmt_dur(eta)} left')
        
        improved = final_step_loss < best_loss
        # A restart that did not beat the incumbent changes nothing downstream; only the
        # ones that move `best_loss` (and the last one, which closes out the craft) are
        # worth a line. The rest were pure volume -- restarts x targets of them per run.
        if improved or r == actual_restarts - 1:
            logging.info(f'  [craft{tag} restart {r + 1}/{actual_restarts}] finished in '
                         f'{fmt_dur(time.time() - t_restart)}: loss {first_step_loss:.5f} -> '
                         f'{final_step_loss:.5f}{" (new best)" if improved else ""}')
        if improved:
            best_loss = final_step_loss
            best_delta = delta.detach().clone()

    sat = (best_delta.abs() > 0.99 * epsilon).float().mean().item()
    logging.info(f'[craft{tag}] done in {fmt_dur(time.time() - t_craft)} | best loss '
                 f'{best_loss:.5f} | mean|delta|={best_delta.abs().mean().item():.4f} '
                 f'({sat:.1%} of pixels at the eps bound)')
    return best_delta.cpu()


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def main():
    args = parse_args()
    set_seed(args.seed)

    # Logging is configured properly further down (once run_dir exists, so the file handler
    # can be attached); do it once here too so the dataset load and any early failure are
    # visible instead of the process sitting silent on a bare terminal for ~30s.
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')
    t_boot = time.time()
    logging.info(f'[data] loading {args.dataset} from {args.data_path} and pushing to {args.device} ...')

    # Fast in-memory data push (Solves 10x CPU bottleneck)
    channel, im_size, num_classes, class_names, mean, std, dst_train, dst_test, _ = \
        get_dataset(args.dataset, args.data_path)

    images_train_raw, labels_train = materialize_raw_train(dst_train)
    images_train_raw, labels_train = subsample_train(images_train_raw, labels_train, args.data_frac, args.seed)
    
    # Store raw on CPU (small memory), but push normalized instantly to GPU to turbocharge victim training
    images_train_norm = normalize(images_train_raw, mean, std, args.device)
    labels_train = labels_train.to(args.device)
    
    images_test_norm, labels_test = materialize_normalized_test(dst_test)
    images_test_norm = images_test_norm.to(args.device)
    labels_test = labels_test.to(args.device)

    logging.info(f'[data] ready in {fmt_dur(time.time() - t_boot)}: '
                 f'train={len(labels_train)} test={len(labels_test)} on {args.device}')


    if args.baseline_only:
        logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')
        _, acc = train_or_load_one_baseline(args, args.model, args.baseline_victim_id, channel, num_classes, im_size,
                                            images_train_norm, labels_train, images_test_norm, labels_test)
        logging.info(f'[baseline_only done] {args.model} v{args.baseline_victim_id} acc={acc:.4f}')
        return

    run_name = (f'{args.dataset}_{args.model}_{args.attack}_{args.base}_{args.class_pair}'
                f'_budget{args.budget:g}_seed{args.seed}')
    if args.base == 'ours':
        run_name += (f'_K{args.num_surrogate}_coef{args.coef}'
                     f'_{args.rel_metric}_{args.score_combine}')
    if args.craft_ensemble != 1:
        run_name += f'_ce{args.craft_ensemble}'
    if args.data_frac < 1.0:
        run_name += f'_frac{args.data_frac:g}'
    run_dir = os.path.join(args.out_dir, run_name)
    os.makedirs(run_dir, exist_ok=True)

    log_path = os.path.join(run_dir, 'log.txt')
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    for h in list(root_logger.handlers):
        root_logger.removeHandler(h)
    file_handler = logging.FileHandler(log_path, mode='a')
    stream_handler = logging.StreamHandler()
    fmt = logging.Formatter('%(asctime)s %(message)s')
    file_handler.setFormatter(fmt)
    stream_handler.setFormatter(fmt)
    root_logger.addHandler(file_handler)
    root_logger.addHandler(stream_handler)

    logging.info(f'=== run start: {run_name} on {args.device} ===')
    # Dump the full configuration. Reusing a checkpoint trained under different victim
    # hyperparameters is the kind of bug that only shows up as a number that makes no sense
    # weeks later, so the protocol that produced every run is recorded in its own log.
    logging.info('[config] ' + ' '.join(f'{k}={v}' for k, v in sorted(vars(args).items())
                                        if k not in ('device',)))

    train_len = len(labels_train)
    args.num_poisons = round(args.budget * train_len)
    logging.info(f'[budget] budget={args.budget} -> num_poisons={args.num_poisons} (train_len={train_len})')

    target_class_name, poison_class_name = CLASS_PAIRS[args.class_pair]
    target_class_idx = class_names.index(target_class_name)
    poison_class_idx = class_names.index(poison_class_name)

    cta_baseline_mean, cta_baseline_std, baseline_nets = get_clean_baseline(
        args, args.model, channel, num_classes, im_size, images_train_norm, labels_train, images_test_norm, labels_test)
    logging.info(f'[clean baseline] {args.model} CTA = {cta_baseline_mean:.4f} +/- {cta_baseline_std:.4f}')

    surrogate_nets = get_surrogate_ensemble(args, channel, num_classes, im_size,
                                            images_train_norm, labels_train, images_test_norm, labels_test)
    scoring_nets = surrogate_nets[:num_scoring_surrogates(args)]
    crafting_nets = surrogate_nets[:args.craft_ensemble]

    ref_nets = get_reference_ensemble(args, channel, num_classes, im_size,
                                      images_train_norm, labels_train, images_test_norm, labels_test)
    veto_nets = ref_nets + surrogate_nets + baseline_nets
    veto_desc = (f'{len(ref_nets)}x{args.ref_model} ref, '
                 f'{len(surrogate_nets)}x{args.model} sur, '
                 f'{len(baseline_nets)}x{args.model} base')
    
    logging.info(f'[stage 4/4] selecting {args.num_targets} targets '
                 f'(easiness={args.target_easiness}, pool={args.target_pool}, '
                 f'veto = {len(veto_nets)} clean models: {veto_desc}, '
                 f'min margin {args.target_min_margin:g}) ...')
    t0 = time.time()
    target_ids, target_scores = select_targets(args, ref_nets, veto_nets, veto_desc,
                                               images_test_norm, labels_test, class_names)
    del ref_nets, veto_nets
    shown = ', '.join(f'{i}(e={target_scores[i]:.3f})' if target_scores.get(i) is not None else str(i)
                      for i in target_ids)
    logging.info(f'[stage 4/4] {len(target_ids)} targets in {fmt_dur(time.time() - t0)} '
                 f'(easiness = clean ensemble p_{poison_class_name}, 1=easiest .. 0=hardest): {shown}')

    # Free-win audit: a target that a CLEAN victim already calls the adversarial class is
    # scored as a success with zero contribution from the poisons, which pins ASR at a
    # budget-independent constant and hides whatever the attack is actually doing.
    free_wins = audit_free_wins(baseline_nets, images_test_norm, target_ids, poison_class_idx,
                                class_names, args.device)
    n_hit = sum(1 for c in free_wins.values() if c > 0)
    if n_hit:
        logging.info(f'[targets] WARNING: {n_hit}/{len(target_ids)} chosen targets are ALREADY '
                     f'predicted "{poison_class_name}" by at least one clean, un-poisoned victim: '
                     + ', '.join(f'{t}({c} of {args.num_baselines})'
                                 for t, c in free_wins.items() if c > 0)
                     + '. Those trials are free wins, not attacks -- raise --target_min_margin.')
    else:
        logging.info(f'[targets] free-win audit: no chosen target is predicted '
                     f'"{poison_class_name}" by any of the {args.num_baselines} clean baselines.')
    del baseline_nets

    candidate_idx = (labels_train == poison_class_idx).nonzero(as_tuple=True)[0].tolist()

    results_path = os.path.join(run_dir, 'results.csv')
    completed = set()
    if args.no_resume:
        write_header = True
    else:
        write_header = not os.path.exists(results_path)
        if not write_header:
            with open(results_path, newline='') as f:
                for row in csv.DictReader(f):
                    completed.add((int(row['target_id']), int(row['victim_id'])))

    results_file = open(results_path, 'w' if args.no_resume else 'a', newline='')
    writer = csv.writer(results_file)
    if write_header:
        writer.writerow(['dataset', 'model', 'attack', 'base', 'class_pair', 'seed',
                         'num_surrogate', 'coef', 'craft_ensemble', 'budget', 'num_poisons',
                         'epsilon', 'target_easiness', 'target_margin_low', 'target_margin_high',
                         'target_id', 'target_score', 'victim_id', 'success', 'clean_test_acc'])

    deltas_path = os.path.join(run_dir, 'deltas.pt')
    bases_path = os.path.join(run_dir, 'bases.json')
    reuse_crafted = not (args.recompute_deltas or args.no_resume)
    cached_deltas = torch.load(deltas_path) if (os.path.exists(deltas_path) and reuse_crafted) else {}
    cached_bases = json.load(open(bases_path)) if (os.path.exists(bases_path) and reuse_crafted) else {}

    t_loop = time.time()
    for t_pos, target_id in enumerate(target_ids):
        target_norm = images_test_norm[target_id]
        done_eta = ''
        if t_pos:
            per_target = (time.time() - t_loop) / t_pos
            done_eta = (f' | ~{fmt_dur(per_target * (len(target_ids) - t_pos))} left for the '
                        f'remaining {len(target_ids) - t_pos} target(s)')
        logging.info(f'=== target {target_id} ({t_pos + 1}/{len(target_ids)}) : selecting bases '
                     f'+ crafting {args.num_poisons} poisons{done_eta} ===')

        if str(target_id) in cached_bases:
            base_ids = cached_bases[str(target_id)]
        elif args.base == 'random':
            base_ids = select_bases_random(candidate_idx, args.num_poisons, seed=args.seed + target_id)
        else:
            base_ids = select_bases_ours(candidate_idx, images_train_norm, target_norm, scoring_nets,
                                         poison_class_idx, args.coef, args.num_poisons, args.device,
                                         rel_metric=args.rel_metric, combine=args.score_combine)
        cached_bases[str(target_id)] = base_ids

        if target_id in cached_deltas:
            delta = cached_deltas[target_id]
        else:
            base_imgs_raw = images_train_raw[base_ids]
            
            delta = craft_poison(args.attack, base_imgs_raw, target_norm, crafting_nets,
                                 poison_class_idx, args.epsilon, args.craft_steps, args.craft_lr,
                                 args.restarts, mean, std, args.device, fast_gradmatch=args.fast_gradmatch,
                                 tag=f' {target_id} ({t_pos + 1}/{len(target_ids)})',
                                 log_every=args.log_every_craft_steps,
                                 log_secs=args.log_progress_secs)
                                 
            cached_deltas[target_id] = delta
            torch.save(cached_deltas, deltas_path)
            with open(bases_path, 'w') as f:
                json.dump(cached_bases, f)

        base_idx_t = torch.as_tensor(base_ids, dtype=torch.long, device=args.device)
        clean_norm_rows = images_train_norm[base_idx_t].clone()
        poisoned_rows_raw = torch.clamp(images_train_raw[base_ids].to(args.device) + delta.to(args.device), 0.0, 1.0)
        images_train_norm[base_idx_t] = normalize(poisoned_rows_raw, mean, std, args.device)
        del poisoned_rows_raw

        target_successes, target_ctas = [], []
        try:
            for victim_id in range(args.num_victims):
                if (target_id, victim_id) in completed:
                    continue

                victim_seed = args.seed * 10000 + target_id * 100 + victim_id
                vtag = (f'target {target_id} ({t_pos + 1}/{len(target_ids)}) | '
                        f'victim {victim_id + 1}/{args.num_victims}')
                if args.log_every_victim:
                    logging.info(f'[{vtag}] training on the poisoned set '
                                 f'({args.num_poisons} poisons, seed {victim_seed})')
                victim_net = build_network(args.model, channel, num_classes, im_size, seed=victim_seed)
                # announce=False: the SUCCESS/FAIL line below reports this retrain's outcome
                # already, and the start/finish pair repeated the same fixed protocol once
                # per victim per target.
                victim_net, _ = train_from_scratch(victim_net, images_train_norm, labels_train, images_test_norm, labels_test,
                                                   epochs=args.victim_epochs, lr=args.victim_lr,
                                                   batch_size=args.victim_bs, decay_epochs=args.victim_decay,
                                                   args=args, tag=vtag, announce=args.log_every_victim)

                victim_net.eval()
                with torch.no_grad():
                    pred = victim_net(target_norm.unsqueeze(0)).argmax(dim=1).item()
                success = int(pred == poison_class_idx)

                cta = evaluate_clean_accuracy(victim_net, images_test_norm, labels_test,
                                              exclude_idx=set(target_ids), device=args.device)

                writer.writerow([args.dataset, args.model, args.attack, args.base, args.class_pair,
                                 args.seed, args.num_surrogate if args.base == 'ours' else '',
                                 args.coef if args.base == 'ours' else '', args.craft_ensemble,
                                 args.budget, args.num_poisons, args.epsilon,
                                 args.target_easiness if args.target_easiness is not None else '',
                                 args.target_margin_low, args.target_margin_high,
                                 target_id, target_scores.get(target_id, ''), victim_id, success, cta])
                results_file.flush()
                target_successes.append(success)
                target_ctas.append(cta)
                logging.info(f'[{vtag}] {"SUCCESS" if success else "FAIL"} '
                             f'(pred={class_names[pred]}, adv={poison_class_name}) | '
                             f'CTA={cta:.4f} (baseline {cta_baseline_mean:.4f}, '
                             f'drop {cta_baseline_mean - cta:+.4f})')

                if args.save_victim_ckpts:
                    ckpt_dir = os.path.join(run_dir, 'victims')
                    os.makedirs(ckpt_dir, exist_ok=True)
                    torch.save(victim_net.state_dict(),
                               os.path.join(ckpt_dir, f'target{target_id}_victim{victim_id}.pt'))
                del victim_net
        finally:
            images_train_norm[base_idx_t] = clean_norm_rows

        if target_successes:
            logging.info(f'=== target {target_id} FINISHED: ASR={np.mean(target_successes):.4f} | '
                         f'CTA={np.mean(target_ctas):.4f} +/- {np.std(target_ctas):.4f} ===')

    results_file.close()

    with open(results_path, newline='') as f:
        all_rows = [{'target_id': int(r['target_id']), 'victim_id': int(r['victim_id']),
                     'success': int(r['success']), 'clean_test_acc': float(r['clean_test_acc'])}
                    for r in csv.DictReader(f)]
    stats = summarize_results(all_rows)
    stats['cta_drop_mean'] = (stats['cta_post_mean'] - cta_baseline_mean
                              if stats['cta_post_mean'] is not None else None)
    stats.update({
        'dataset': args.dataset, 'model': args.model, 'attack': args.attack, 'base': args.base,
        'class_pair': args.class_pair, 'seed': args.seed,
        'num_surrogate': args.num_surrogate if args.base == 'ours' else None,
        'coef': args.coef if args.base == 'ours' else None,
        'rel_metric': args.rel_metric if args.base == 'ours' else None,
        'score_combine': args.score_combine if args.base == 'ours' else None,
        'craft_ensemble': args.craft_ensemble, 'restarts': args.restarts,
        'data_frac': args.data_frac, 'train_len': train_len,
        'budget': args.budget, 'num_poisons': args.num_poisons, 'epsilon': args.epsilon,
        'target_easiness': args.target_easiness, 'target_pool': args.target_pool,
        'target_min_margin': args.target_min_margin,
        'target_score_mean': (float(np.mean([d for d in target_scores.values() if d is not None]))
                              if any(d is not None for d in target_scores.values()) else None),
        'target_margin_low': args.target_margin_low, 'target_margin_high': args.target_margin_high,
        'target_eligibility_ensemble': args.target_eligibility_ensemble,
        'targets_already_adv_clean_victims': sum(1 for c in free_wins.values() if c > 0),
        'free_win_trials': int(sum(free_wins.values())),
        'cta_baseline_mean': cta_baseline_mean, 'cta_baseline_std': cta_baseline_std,
    })

    summary_path = os.path.join(run_dir, 'summary.json')
    with open(summary_path, 'w') as f:
        json.dump(stats, f, indent=2)

    global_summary_path = os.path.join(args.out_dir, 'summary_all.csv')
    fieldnames = ['dataset', 'model', 'attack', 'base', 'class_pair', 'seed',
                  'num_surrogate', 'coef', 'rel_metric', 'score_combine',
                  'craft_ensemble', 'restarts', 'data_frac', 'train_len',
                  'budget', 'num_poisons', 'epsilon',
                  'target_easiness', 'target_pool', 'target_score_mean',
                  'target_eligibility_ensemble', 'target_min_margin', 'free_win_trials',
                  'target_margin_low', 'target_margin_high',
                  'num_targets', 'num_trials',
                  'cta_baseline_mean', 'cta_baseline_std',
                  'asr_mean', 'asr_std', 'cta_post_mean', 'cta_post_std', 'cta_drop_mean']
    
    write_global_header = not os.path.exists(global_summary_path)
    if not write_global_header:
        with open(global_summary_path, newline='') as f:
            existing_header = next(csv.reader(f), [])
        if existing_header != fieldnames:
            backup = f'{global_summary_path}.{time.strftime("%Y%m%d-%H%M%S")}.bak'
            os.replace(global_summary_path, backup)
            write_global_header = True
            
    with open(global_summary_path, 'a', newline='') as f:
        gw = csv.DictWriter(f, fieldnames=fieldnames)
        if write_global_header:
            gw.writeheader()
        gw.writerow({k: stats[k] for k in fieldnames})

if __name__ == '__main__':
    main()