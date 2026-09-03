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
  sapa       sharpness-aware data poisoning (He et al., ICLR 2024) on top of
             gradient matching. Identical to gradmatch except that the TARGET
             gradient is taken at a sharpness-perturbed parameter vector, so the
             poisons are matched to the worst re-trained model in a neighbourhood
             of theta rather than to theta itself:
               --sharp_mode worst  one SAM ascent step,
                                   eps = sigma * g / ||g||_2, g = grad_theta CE(x_t, y_adv),
                                   then g_t = grad_theta CE(x_t, y_adv; theta + eps)
               --sharp_mode avg    mean of grad_theta CE(x_t, y_adv; theta + xi) over
                                   --sharp_samples draws of xi ~ N(0, sigma^2 I)
             The poison-side gradient, the passenger loss, the optimizer, the
             restarts and the projection are all untouched, so --sharp_sigma 0
             with --sharp_mode worst reproduces --attack gradmatch exactly.
             NOTE sigma means different things in the two modes: in 'worst' it is
             a global l2 radius (SAM's rho, 0.05 is the paper default and is a
             ~0.1% perturbation of a trained net), in 'avg' it is a PER-ELEMENT
             gaussian std, where 0.05 destroys the network. Use ~1e-3 there.

Base selection (--base):
  random     uniform over the poison class.
  ours       standardized d(x) + lambda * M(x), lowest first, ensemble averaged.
             d = feature distance to the target (l2 or cosine), M = logit margin
             toward y_adv. Low score means close to the target in feature space
             AND sitting near the y_adv decision boundary. With
             --use_jacobian_score, subtract beta times the standardized exact
             candidate/target backbone-gradient interaction per surrogate.
             --sel_exact_alignment ranks by the exact full-parameter
             g_i^T g_t. --sel_a_minus_mr ranks by the backbone interaction A_i
             plus (-M_i) times R_i, using the paper's A, M, and R definitions.
             --sel_component selects one of the component ablations -M, R, A,
             A-M, A+R, or (-M)*R with the same component definitions/scaling.

Class pair naming follows MetaPoison: 'dog-bird' means poisons are drawn from
'dog' (y_adv) and the target image is a 'bird'. Use --pair_order target-poison
to flip that if you need to reproduce numbers from a codebase that reads it the
other way.

Everything is GPU resident and manually batched, no DataLoader in the hot loop.

Multi-GPU (--gpus, default 'all'): if more than one cuda device is visible the
work is run in parallel, one process per gpu (spawn), each with its own copy of
the dataset and its own surrogates loaded from the shared cache. The unit of work
is a (target, victim) trial and gpus claim them off a shared board:

  1. a gpu stays on the target it already holds the poisons for;
  2. otherwise it starts a target nobody has started -- target 1 on gpu 0,
     target 2 on gpu 1, ... so all gpus craft different targets at once;
  3. otherwise -- the tail, when fewer targets are left than there are gpus --
     it helps train the remaining victims of a target another gpu has already
     crafted, reading those poisons out of poison_cache. Nothing is crafted
     twice and no gpu sits idle waiting for the last target to finish.

Any surrogate / clean victim still missing from the cache is trained first, also
spread over the gpus, so the target workers only ever read that cache. Each
worker appends to its own results_rank<i>.csv, which the parent merges into
results.csv at the end; a run killed halfway resumes from those shards.
Use --no_parallel_targets to force the old single-gpu sequential behaviour.

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
import atexit
import csv
import gc
import glob
import json
import math
import os
import queue as _queue
import time
import traceback
import warnings
from collections import defaultdict

warnings.filterwarnings('ignore', category=UserWarning)

import numpy as np
import torch
import torch.multiprocessing as mp
import torch.nn as nn
import torch.nn.functional as F

from utils import get_dataset, DiffAugment, ParamDiffAug

try:
    from utils import get_time
except ImportError:
    def get_time():
        return time.strftime('[%Y-%m-%d %H:%M:%S]')

try:
    from utils import get_network as _get_network
except ImportError:
    _get_network = None

try:
    from networks import ConvNet as _ConvNet
except Exception:
    _ConvNet = None
try:
    from networks import VGG as _VGG
except Exception:
    _VGG = None
try:
    from networks import ResNet20BN as _ResNet20BN
except Exception:
    _ResNet20BN = None


DSA_DEFAULT = 'color_crop_cutout_flip_scale_rotate'
SUPPORTED_MODELS = ['ConvNetBN', 'VGG13BN', 'ResNet20BN', 'ResNet18', 'ResNet18BN']
CLASS_PAIRS = ['dog-bird', 'frog-airplane']   # the main sweep's pairs; --class_pair
                                             # accepts any '<adv>-<target>' whose two
                                             # names exist in the dataset

_LOG_PATH = None
_LOG_TAG = ''          # set to '[gpu3]' inside a worker process
_JACOBIAN_BACKENDS_LOGGED = set()
_EXACT_ALIGNMENT_BACKENDS_LOGGED = set()

RESULT_FIELDS = ['model', 'attack', 'base', 'class_pair', 'seed', 'budget',
                 'num_poisons', 'epsilon', 'target_idx', 'target_score', 'victim_id',
                 'success', 'clean_test_acc', 'clean_asr', 'craft_obj', 'realized_linf']


def log(msg):
    line = '%s%s %s' % (get_time(), (' ' + _LOG_TAG) if _LOG_TAG else '', msg)
    print(line, flush=True)
    if _LOG_PATH:
        # one short O_APPEND write per line, so several worker processes can share
        # the same log file without stepping on each other
        with open(_LOG_PATH, 'a') as f:
            f.write(line + '\n')


# --------------------------------------------------------------------------- #
# small helpers
# --------------------------------------------------------------------------- #

def embed_of(net):
    return net.module.embed if isinstance(net, nn.DataParallel) else net.embed


def standardize(v, eps=1e-8):
    return (v - v.mean()) / (v.std() + eps)


def rho_to_m(rho, training_set_size):
    """Map a poison ratio to its absolute budget exactly as attack runs do."""
    return int(round(rho * training_set_size))


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
    if pair.count('-') != 1:
        raise SystemExit("--class_pair must be '<adversarial>-<target>', got %r" % pair)
    a, b = pair.split('-')
    for n in (a, b):
        if n not in class_names:
            raise SystemExit('--class_pair: %r is not a class of this dataset. Known: %s'
                             % (n, ', '.join(map(str, class_names[:12]))
                                + (' ...' if len(class_names) > 12 else '')))
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
                       weight_decay=0.0, aug=False, dsa_strategy=None, dsa_param=None,
                       augmenter=None):
    """`augmenter`, when given, replaces the DiffAugment path: it is called on
    every minibatch (already normalized, poisons already written in) and returns
    the augmented batch. See victim_aug.py."""
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
            if augmenter is not None:
                img = augmenter(img)
            elif aug and dsa_strategy:
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

def dataset_tag(args):
    """Cache paths were keyed on the model alone, which is unambiguous only while
    there is one dataset: ConvNetBN on CIFAR-10 and ConvNetBN on TinyImageNet have
    different input sizes and class counts, so they cannot share a checkpoint.
    CIFAR-10 keeps the bare name so every net already in cache/ stays valid."""
    ds = getattr(args, 'dataset', 'CIFAR10')
    return '' if ds == 'CIFAR10' else ds + '_'


def surrogate_dir(args):
    return os.path.join(args.cache_dir, 'surrogates',
                        '%s%s_%dep_lr%g_bs%d_seed%d'
                        % (dataset_tag(args), args.model, args.surrogate_epochs,
                           args.surrogate_lr, args.surrogate_bs, args.seed))


def victim_dir(args):
    return os.path.join(args.cache_dir, 'clean_victims',
                        '%s%s_%dep_lr%g_bs%d_wd%g_seed%d'
                        % (dataset_tag(args), args.model, args.victim_epochs,
                           args.victim_lr, args.victim_bs, args.victim_wd, args.seed))


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


def get_sel_surrogates(args, *rest):
    """Surrogate pool used for base selection only, when --sel_model is set.

    Cross-architecture transfer: the bases are picked by one architecture (S) and
    the poisons are then crafted and evaluated on another (A = V). Same seeds and
    same surrogate hyper-parameters as a normal run of S, so these are literally
    the nets S's own run selected with -- they load straight out of the cache.
    """
    sel_args = argparse.Namespace(**vars(args))
    sel_args.model = args.sel_model
    return get_surrogates(sel_args, *rest)


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

def _restore_training_states(states):
    """Restore per-module train/eval flags without changing any other state."""
    for module, training in states:
        module.training = training


def _log_jacobian_backend(backend, reason=None):
    if backend in _JACOBIAN_BACKENDS_LOGGED:
        return
    _JACOBIAN_BACKENDS_LOGGED.add(backend)
    if reason:
        log('  Jacobian score exact backend: %s (functional JVP unavailable: %s)'
            % (backend, reason))
    else:
        log('  Jacobian score exact backend: %s' % backend)


def _jacobian_backend_metadata():
    backends = set(_JACOBIAN_BACKENDS_LOGGED)
    marker = 'Jacobian score exact backend: '
    if _LOG_PATH and os.path.exists(_LOG_PATH):
        with open(_LOG_PATH) as handle:
            for line in handle:
                if marker in line:
                    value = line.split(marker, 1)[1].split(
                        ' (functional JVP unavailable:', 1)[0].strip()
                    if value:
                        backends.add(value)
    return ', '.join(sorted(backends)) if backends else None


def _log_exact_alignment_backend(backend, reason=None):
    if backend in _EXACT_ALIGNMENT_BACKENDS_LOGGED:
        return
    _EXACT_ALIGNMENT_BACKENDS_LOGGED.add(backend)
    if reason:
        log('  Exact gi/gt alignment backend: %s (functional JVP unavailable: %s)'
            % (backend, reason))
    else:
        log('  Exact gi/gt alignment backend: %s' % backend)


def _backbone_gradient_interactions(net, candidates, x_t_norm, y_adv,
                                    batch_size=64):
    """Return exact candidate/target CE-gradient dots over non-head parameters.

    The production path computes one target reverse-mode gradient and then one
    functional JVP per candidate batch.  No parameter gradients are accumulated
    and no candidate-by-parameter tensor is constructed.  If forward AD is not
    supported by an operator, the exact dummy-loss-weight double-backward
    identity computes the same batch vector.
    """
    if batch_size <= 0:
        raise ValueError('jacobian batch size must be positive, got %d' % batch_size)
    core = net.module if isinstance(net, nn.DataParallel) else net
    if not hasattr(core, 'classifier'):
        raise ValueError('Jacobian scoring requires the model to expose its final '
                         'linear head as .classifier; got %s' % type(core).__name__)
    if not isinstance(core.classifier, nn.Linear):
        raise ValueError('Jacobian scoring requires .classifier to be nn.Linear; got %s'
                         % type(core.classifier).__name__)

    named_params = dict(core.named_parameters())
    head_ids = {id(p) for p in core.classifier.parameters()}
    backbone = {name: p for name, p in named_params.items() if id(p) not in head_ids}
    fixed = {name: p for name, p in named_params.items() if id(p) in head_ids}
    if not backbone:
        raise ValueError('Jacobian scoring found no parameters outside .classifier')
    buffers = dict(core.named_buffers())
    states = [(module, module.training) for module in core.modules()]
    core.eval()
    target = x_t_norm.unsqueeze(0) if x_t_norm.ndim == candidates.ndim - 1 else x_t_norm
    label_target = torch.full((target.shape[0],), int(y_adv), dtype=torch.long,
                              device=target.device)

    def functional_logits(backbone_params, x):
        params = dict(fixed)
        params.update(backbone_params)
        return torch.func.functional_call(core, (params, buffers), (x,))

    target_grad = None
    jvp_error = None
    try:
        with torch.enable_grad():
            try:
                if not all(hasattr(torch.func, name)
                           for name in ('functional_call', 'grad', 'jvp')):
                    raise RuntimeError('torch.func functional_call/grad/jvp is unavailable')

                def target_loss(backbone_params):
                    return F.cross_entropy(functional_logits(backbone_params, target),
                                           label_target)

                target_grad = torch.func.grad(target_loss)(backbone)
                target_grad = {name: grad.detach() for name, grad in target_grad.items()}

                values = []
                for start in range(0, len(candidates), batch_size):
                    batch = candidates[start:start + batch_size]

                    def candidate_losses(backbone_params):
                        logits = functional_logits(backbone_params, batch)
                        labels = torch.full((len(batch),), int(y_adv), dtype=torch.long,
                                            device=batch.device)
                        return F.cross_entropy(logits, labels, reduction='none')

                    _, tangent = torch.func.jvp(candidate_losses, (backbone,),
                                                (target_grad,))
                    values.append(tangent.detach())
                result = torch.cat(values)
                backend = 'torch.func JVP'
            except Exception as exc:
                jvp_error = exc
                # Discard any partial JVP output and recompute every batch with the
                # exact fallback, so a run never mixes or silently approximates.
                fallback_backbone = {
                    name: value.detach().requires_grad_(True)
                    for name, value in backbone.items()
                }

                def fallback_logits(x):
                    params = dict(fixed)
                    params.update(fallback_backbone)
                    return torch.func.functional_call(core, (params, buffers), (x,))

                values = []
                try:
                    if target_grad is None:
                        loss_t = F.cross_entropy(fallback_logits(target), label_target)
                        grads_t = torch.autograd.grad(
                            loss_t, tuple(fallback_backbone.values()))
                        target_grad = {
                            name: grad.detach()
                            for name, grad in zip(fallback_backbone, grads_t)
                        }
                    backbone_values = tuple(fallback_backbone.values())
                    target_values = tuple(target_grad.values())
                    for start in range(0, len(candidates), batch_size):
                        batch = candidates[start:start + batch_size]
                        weights = torch.zeros(len(batch), device=batch.device,
                                              dtype=batch.dtype, requires_grad=True)
                        labels = torch.full((len(batch),), int(y_adv), dtype=torch.long,
                                            device=batch.device)
                        losses = F.cross_entropy(
                            fallback_logits(batch), labels, reduction='none')
                        mixed = torch.autograd.grad(
                            torch.dot(weights, losses), backbone_values,
                            create_graph=True)
                        directional = sum((g * v).sum()
                                          for g, v in zip(mixed, target_values))
                        interaction = torch.autograd.grad(directional, weights)[0]
                        values.append(interaction.detach())
                    result = torch.cat(values)
                    backend = 'dummy-weight double backward'
                except Exception as fallback_error:
                    raise RuntimeError(
                        'exact Jacobian scoring failed with both batched backends; '
                        'torch.func JVP error: %s; double-backward error: %s'
                        % (jvp_error, fallback_error)) from fallback_error
    finally:
        _restore_training_states(states)

    reason = None
    if jvp_error is not None:
        reason = '%s: %s' % (type(jvp_error).__name__, str(jvp_error).split('\n')[0])
    _log_jacobian_backend(backend, reason)
    return result, backend


def _full_gradient_interactions(net, candidates, x_t_norm, y_adv,
                                batch_size=64):
    """Return exact <g_i, g_t> over every model parameter for each candidate.

    This is the full gradient alignment in Equation (2), including the final
    classifier. As in the backbone-only implementation above, the primary path
    uses one target reverse-mode gradient and batched functional JVPs. The
    double-backward fallback is algebraically exact and never materializes a
    candidate-by-parameter Jacobian.
    """
    if batch_size <= 0:
        raise ValueError('exact-alignment batch size must be positive, got %d'
                         % batch_size)
    core = net.module if isinstance(net, nn.DataParallel) else net
    parameters = dict(core.named_parameters())
    if not parameters:
        raise ValueError('exact gi/gt alignment requires a model with parameters')
    buffers = dict(core.named_buffers())
    states = [(module, module.training) for module in core.modules()]
    core.eval()
    target = x_t_norm.unsqueeze(0) if x_t_norm.ndim == candidates.ndim - 1 else x_t_norm
    label_target = torch.full((target.shape[0],), int(y_adv), dtype=torch.long,
                              device=target.device)

    def functional_logits(model_params, x):
        return torch.func.functional_call(core, (model_params, buffers), (x,))

    target_grad = None
    jvp_error = None
    try:
        with torch.enable_grad():
            try:
                if not all(hasattr(torch.func, name)
                           for name in ('functional_call', 'grad', 'jvp')):
                    raise RuntimeError('torch.func functional_call/grad/jvp is unavailable')

                def target_loss(model_params):
                    return F.cross_entropy(functional_logits(model_params, target),
                                           label_target)

                target_grad = torch.func.grad(target_loss)(parameters)
                target_grad = {name: grad.detach()
                               for name, grad in target_grad.items()}

                values = []
                for start in range(0, len(candidates), batch_size):
                    batch = candidates[start:start + batch_size]

                    def candidate_losses(model_params):
                        logits = functional_logits(model_params, batch)
                        labels = torch.full((len(batch),), int(y_adv), dtype=torch.long,
                                            device=batch.device)
                        return F.cross_entropy(logits, labels, reduction='none')

                    # For every candidate i, this directional derivative is
                    # sum_p (d ell_i / d p) * (d L_t / d p) = g_i^T g_t.
                    _, tangent = torch.func.jvp(candidate_losses, (parameters,),
                                                (target_grad,))
                    values.append(tangent.detach())
                result = torch.cat(values)
                backend = 'torch.func JVP'
            except Exception as exc:
                jvp_error = exc
                fallback_params = {
                    name: value.detach().requires_grad_(True)
                    for name, value in parameters.items()
                }

                def fallback_logits(x):
                    return torch.func.functional_call(
                        core, (fallback_params, buffers), (x,))

                values = []
                try:
                    if target_grad is None:
                        loss_t = F.cross_entropy(fallback_logits(target), label_target)
                        grads_t = torch.autograd.grad(
                            loss_t, tuple(fallback_params.values()))
                        target_grad = {
                            name: grad.detach()
                            for name, grad in zip(fallback_params, grads_t)
                        }
                    parameter_values = tuple(fallback_params.values())
                    target_values = tuple(target_grad.values())
                    for start in range(0, len(candidates), batch_size):
                        batch = candidates[start:start + batch_size]
                        weights = torch.zeros(len(batch), device=batch.device,
                                              dtype=batch.dtype, requires_grad=True)
                        labels = torch.full((len(batch),), int(y_adv), dtype=torch.long,
                                            device=batch.device)
                        losses = F.cross_entropy(
                            fallback_logits(batch), labels, reduction='none')
                        mixed = torch.autograd.grad(
                            torch.dot(weights, losses), parameter_values,
                            create_graph=True)
                        directional = sum((g * v).sum()
                                          for g, v in zip(mixed, target_values))
                        interaction = torch.autograd.grad(directional, weights)[0]
                        values.append(interaction.detach())
                    result = torch.cat(values)
                    backend = 'dummy-weight double backward'
                except Exception as fallback_error:
                    raise RuntimeError(
                        'exact full-parameter gi/gt alignment failed with both '
                        'batched backends; torch.func JVP error: %s; '
                        'double-backward error: %s'
                        % (jvp_error, fallback_error)) from fallback_error
    finally:
        _restore_training_states(states)

    reason = None
    if jvp_error is not None:
        reason = '%s: %s' % (type(jvp_error).__name__, str(jvp_error).split('\n')[0])
    _log_exact_alignment_backend(backend, reason)
    return result, backend


def _log_interaction_diagnostics(surrogate_idx, interaction):
    finite = torch.isfinite(interaction)
    vals = interaction[finite]
    if len(vals):
        mean = float(vals.mean())
        std = float(vals.std()) if len(vals) > 1 else 0.0
        lo, hi = float(vals.min()), float(vals.max())
        frac_pos = float((vals > 0).float().mean())
    else:
        mean = std = lo = hi = frac_pos = float('nan')
    # log('  Jacobian A surrogate %d: finite=%s mean=%g std=%g min=%g max=%g positive=%g'
    #     % (surrogate_idx, bool(finite.all()), mean, std, lo, hi, frac_pos))


@torch.no_grad()
def _ours_pointwise_score(nets, images_norm, labels, x_t_norm, y_adv, lam, device,
                          base_dist='l2', bs=512, collect_feats=False,
                          use_jacobian_score=False, jacobian_weight=1.0,
                          jacobian_batch_size=64):
    """Shared per-surrogate proposed score, optionally with exact Jacobian A."""
    cls_idx = (labels == y_adv).nonzero(as_tuple=True)[0]
    cand = images_norm[cls_idx]
    score = torch.zeros(len(cls_idx), device=device)
    blocks = []
    for surrogate_idx, net in enumerate(nets):
        states = [(module, module.training) for module in net.modules()]
        net.eval()
        try:
            emb = embed_of(net)
            f_t = emb(x_t_norm.unsqueeze(0))
            ds, ms, fs = [], [], []
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
                if collect_feats:
                    fs.append(F.normalize(fb.detach().flatten(1), dim=1))
            component = standardize(torch.cat(ds)) + lam * standardize(torch.cat(ms))
            if use_jacobian_score:
                interaction, _ = _backbone_gradient_interactions(
                    net, cand, x_t_norm, y_adv, jacobian_batch_size)
                _log_interaction_diagnostics(surrogate_idx, interaction)
                # Smaller costs are selected, while positive interaction is good.
                # Avoid even a zero-times-NaN perturbation when beta is exactly 0.
                if jacobian_weight != 0:
                    component = component - jacobian_weight * standardize(interaction)
                del interaction
            score += component
            if collect_feats:
                blocks.append(torch.cat(fs))
        finally:
            _restore_training_states(states)
    score /= len(nets)
    feats = (torch.cat(blocks, dim=1) / math.sqrt(len(nets))
             if collect_feats else None)
    return cls_idx, score, feats


@torch.no_grad()
def select_base_ours(nets, images_norm, labels, x_t_norm, y_adv, N_p, lam, device,
                     base_dist='l2', bs=512, use_jacobian_score=False,
                     jacobian_weight=1.0, jacobian_batch_size=64):
    cls_idx = (labels == y_adv).nonzero(as_tuple=True)[0]
    if len(cls_idx) < N_p:
        raise ValueError('class %d has %d images < N_p=%d' % (y_adv, len(cls_idx), N_p))
    cls_idx, score, _ = _ours_pointwise_score(
        nets, images_norm, labels, x_t_norm, y_adv, lam, device, base_dist, bs,
        use_jacobian_score=use_jacobian_score, jacobian_weight=jacobian_weight,
        jacobian_batch_size=jacobian_batch_size)
    sel = torch.topk(score, k=N_p, largest=False).indices
    return cls_idx[sel]


@torch.no_grad()
def select_base_exact_alignment(nets, images_norm, labels, x_t_norm, y_adv, N_p,
                                device, batch_size=64):
    """Select by the exact full-parameter g_i^T g_t.

    The raw dot-product scale can differ substantially across independently
    trained surrogates. Standardizing within each surrogate's candidate pool
    preserves that surrogate's ranking while preventing its gradient magnitude
    from dominating the ensemble average.
    """
    cls_idx = (labels == y_adv).nonzero(as_tuple=True)[0]
    if len(cls_idx) < N_p:
        raise ValueError('class %d has %d images < N_p=%d'
                         % (y_adv, len(cls_idx), N_p))
    candidates = images_norm[cls_idx]
    alignment = torch.zeros(len(cls_idx), device=device)
    for net in nets:
        interaction, _ = _full_gradient_interactions(
            net, candidates, x_t_norm, y_adv, batch_size)
        alignment += standardize(interaction)
        del interaction
    alignment /= len(nets)
    selected = torch.topk(alignment, k=N_p, largest=True).indices
    return cls_idx[selected]


COMPONENT_SELECTOR_LABELS = {
    'minus-m': '-M',
    'r': 'R',
    'a': 'A',
    'a-minus-m': 'A-M',
    'a-plus-r': 'A+R',
    'minus-m-times-r': '(-M)*R',
}

COMPONENT_SELECTOR_SUFFIXES = {
    'minus-m': 'MinusM',
    'r': 'R',
    'a': 'A',
    'a-minus-m': 'AminusM',
    'a-plus-r': 'AplusR',
    'minus-m-times-r': 'MinusMtimesR',
}


@torch.no_grad()
def select_base_components(nets, images_norm, labels, x_t_norm, y_adv, N_p,
                           device, formula, batch_size=64):
    """Select using a scaled expression of the paper's A, M, and R components.

    A_i is <grad_phi ell_i, grad_phi L_adv,t>, M_i is the adversarial-class
    logit margin, and R_i is the raw representation inner product <h_i, h_t>.
    Each component used by ``formula`` is averaged over surrogates in its raw
    scale and then standardized across the candidate pool before the requested
    expression is evaluated. Components absent from the expression are not
    computed; in particular, formulas without A avoid the expensive backbone
    gradient interaction.
    """
    if formula not in COMPONENT_SELECTOR_LABELS:
        raise ValueError('unknown component selector %r' % formula)
    cls_idx = (labels == y_adv).nonzero(as_tuple=True)[0]
    if len(cls_idx) < N_p:
        raise ValueError('class %d has %d images < N_p=%d'
                         % (y_adv, len(cls_idx), N_p))
    candidates = images_norm[cls_idx]
    need_margin = formula in ('minus-m', 'a-minus-m', 'minus-m-times-r')
    need_relevance = formula in ('r', 'a-plus-r', 'minus-m-times-r')
    need_interaction = formula in ('a', 'a-minus-m', 'a-plus-r')
    margin = torch.zeros(len(cls_idx), device=device) if need_margin else None
    relevance = (torch.zeros(len(cls_idx), device=device)
                 if need_relevance else None)
    interaction = (torch.zeros(len(cls_idx), device=device)
                   if need_interaction else None)
    for net in nets:
        if need_margin or need_relevance:
            states = [(module, module.training) for module in net.modules()]
            net.eval()
            try:
                embed = embed_of(net)
                target_feature = (embed(x_t_norm.unsqueeze(0)).flatten(1)
                                  if need_relevance else None)
                margins, relevances = [], []
                for start in range(0, len(candidates), batch_size):
                    batch = candidates[start:start + batch_size]
                    if need_relevance:
                        candidate_feature = embed(batch).flatten(1)
                        relevances.append(
                            (candidate_feature * target_feature).sum(dim=1))
                    if need_margin:
                        logits = net(batch)
                        adversarial_logit = logits[:, y_adv].clone()
                        other_logits = logits.clone()
                        other_logits[:, y_adv] = float('-inf')
                        margins.append(
                            adversarial_logit - other_logits.max(dim=1).values)
                if need_margin:
                    margin += torch.cat(margins)
                if need_relevance:
                    relevance += torch.cat(relevances)
            finally:
                _restore_training_states(states)

        if need_interaction:
            backbone_interaction, _ = _backbone_gradient_interactions(
                net, candidates, x_t_norm, y_adv, batch_size)
            interaction += backbone_interaction
            del backbone_interaction

    if need_margin:
        margin = standardize(margin / len(nets))
    if need_relevance:
        relevance = standardize(relevance / len(nets))
    if need_interaction:
        interaction = standardize(interaction / len(nets))

    if formula == 'minus-m':
        score = -margin
    elif formula == 'r':
        score = relevance
    elif formula == 'a':
        score = interaction
    elif formula == 'a-minus-m':
        score = interaction - margin
    elif formula == 'a-plus-r':
        score = interaction + relevance
    else:  # minus-m-times-r
        score = (-margin) * relevance
    selected = torch.topk(score, k=N_p, largest=True).indices
    return cls_idx[selected]


@torch.no_grad()
def select_base_a_minus_mr(nets, images_norm, labels, x_t_norm, y_adv, N_p,
                           device, batch_size=64):
    """Select by A_i + (-M_i) * R_i using the paper's scaled components."""
    cls_idx = (labels == y_adv).nonzero(as_tuple=True)[0]
    if len(cls_idx) < N_p:
        raise ValueError('class %d has %d images < N_p=%d'
                         % (y_adv, len(cls_idx), N_p))
    candidates = images_norm[cls_idx]
    margin = torch.zeros(len(cls_idx), device=device)
    relevance = torch.zeros(len(cls_idx), device=device)
    interaction = torch.zeros(len(cls_idx), device=device)
    for net in nets:
        states = [(module, module.training) for module in net.modules()]
        net.eval()
        try:
            embed = embed_of(net)
            target_feature = embed(x_t_norm.unsqueeze(0)).flatten(1)
            margins, relevances = [], []
            for start in range(0, len(candidates), batch_size):
                batch = candidates[start:start + batch_size]
                candidate_feature = embed(batch).flatten(1)
                logits = net(batch)
                adversarial_logit = logits[:, y_adv].clone()
                other_logits = logits.clone()
                other_logits[:, y_adv] = float('-inf')
                margins.append(adversarial_logit - other_logits.max(dim=1).values)
                relevances.append((candidate_feature * target_feature).sum(dim=1))
            margin += torch.cat(margins)
            relevance += torch.cat(relevances)
        finally:
            _restore_training_states(states)
        backbone_interaction, _ = _backbone_gradient_interactions(
            net, candidates, x_t_norm, y_adv, batch_size)
        interaction += backbone_interaction
        del backbone_interaction
    margin = standardize(margin / len(nets))
    relevance = standardize(relevance / len(nets))
    interaction = standardize(interaction / len(nets))
    score = interaction + (-margin) * relevance
    selected = torch.topk(score, k=N_p, largest=True).indices
    return cls_idx[selected]


# --------------------------------------------------------------------------- #
# diversity-aware variants of the base selection
#
# All three keep score_i EXACTLY as select_base_ours computes it. They differ
# only in how a set of N_p is assembled from those scores, because taking the
# N_p smallest is a modular objective: every candidate near the target is also
# near every other such candidate, so the chosen set is a tight cluster whose
# feature matrix is low effective rank.
#
#   --sel_filter  gate      keep the best  c*N_p by score, then farthest-point
#   --sel_mmr     additive  greedy on  score_i + mu * max_{j in S} sim(i, j)
#   --sel_dpp     product   greedy log-det of  L_ij = q_i q_j K_ij,  q = exp(-a*score)
#
# Each collapses to plain select_base_ours at one end of its knob (c=1, mu=0,
# alpha -> inf), which is the cheapest correctness test.
# --------------------------------------------------------------------------- #

@torch.no_grad()          # selection is pure inference -- without this the graph
                          # over 5000 candidates x 5 surrogates exhausts the GPU
def _ours_score_and_feats(nets, images_norm, labels, x_t_norm, y_adv, lam, device,
                          base_dist='l2', bs=512, use_jacobian_score=False,
                          jacobian_weight=1.0, jacobian_batch_size=64):
    """score_i identical to select_base_ours, plus the candidate features needed
    for a pairwise similarity. Per-surrogate features are L2-normalised and
    concatenated, so a single cosine on the result equals the MEAN of the
    per-surrogate cosines -- the same ensemble averaging the score already uses."""
    return _ours_pointwise_score(
        nets, images_norm, labels, x_t_norm, y_adv, lam, device, base_dist, bs,
        collect_feats=True, use_jacobian_score=use_jacobian_score,
        jacobian_weight=jacobian_weight,
        jacobian_batch_size=jacobian_batch_size)


@torch.no_grad()
def select_base_topr(nets, images_norm, labels, x_t_norm, y_adv, N_p, r, lam, device,
                     base_dist='l2', use_jacobian_score=False,
                     jacobian_weight=1.0, jacobian_batch_size=64):
    """Concentrate the poison budget into r feature-space neighbourhoods.

    Same budget, fewer distinct regions: take the r best-scoring candidates as
    seeds, then fill the remaining N_p - r slots with each seed's nearest
    neighbours in the candidate pool, round-robin so the seeds stay balanced.

    Every returned index is DISTINCT. Literally replicating one base is not
    expressible in this threat model: a poison replaces a specific training
    example, so m copies of one index collapse to a single poisoned image and the
    budget would silently drop from N_p to r. Concentrating into neighbourhoods
    keeps N_p poisoned images while still varying how many separate favourable
    regions the budget is spread over, which is the question the ablation asks.

    r >= N_p reduces to plain greedy selection.
    """
    cls_idx, score, feats = _ours_score_and_feats(nets, images_norm, labels, x_t_norm,
        y_adv, lam, device, base_dist=base_dist,
        use_jacobian_score=use_jacobian_score, jacobian_weight=jacobian_weight,
        jacobian_batch_size=jacobian_batch_size)
    if len(cls_idx) < N_p:
        raise ValueError('class %d has %d candidates < N_p=%d' % (y_adv, len(cls_idx), N_p))
    if r >= N_p:
        return cls_idx[torch.topk(score, k=N_p, largest=False).indices]
    seeds = torch.topk(score, k=r, largest=False).indices
    chosen, taken = seeds.tolist(), set(seeds.tolist())
    order = [torch.argsort(feats @ feats[s], descending=True).tolist() for s in seeds]
    ptr = [0] * r
    while len(chosen) < N_p:
        moved = False
        for k in range(r):
            if len(chosen) >= N_p:
                break
            while ptr[k] < len(order[k]) and order[k][ptr[k]] in taken:
                ptr[k] += 1
            if ptr[k] < len(order[k]):
                j = order[k][ptr[k]]
                ptr[k] += 1
                chosen.append(j)
                taken.add(j)
                moved = True
        if not moved:
            raise ValueError('candidate pool exhausted before reaching N_p')
    return cls_idx[torch.tensor(chosen, device=device)]


def _sim_to(feats, j):
    return feats @ feats[j]


@torch.no_grad()
def select_base_ours_div(nets, images_norm, labels, x_t_norm, y_adv, N_p, lam, device,
                         base_dist='l2', bs=512, mode='filter', pool=3.0, mu=0.5,
                         alpha=1.0, use_jacobian_score=False,
                         jacobian_weight=1.0, jacobian_batch_size=64):
    cls_idx = (labels == y_adv).nonzero(as_tuple=True)[0]
    if len(cls_idx) < N_p:
        raise ValueError('class %d has %d images < N_p=%d' % (y_adv, len(cls_idx), N_p))
    cls_idx, score, feats = _ours_score_and_feats(
        nets, images_norm, labels, x_t_norm, y_adv, lam, device, base_dist, bs,
        use_jacobian_score=use_jacobian_score, jacobian_weight=jacobian_weight,
        jacobian_batch_size=jacobian_batch_size)
    N = len(cls_idx)
    score = standardize(score)          # monotone: does not change plain ranking

    if mode == 'filter':
        k = min(N, max(N_p, int(round(pool * N_p))))
        cand = torch.topk(score, k=k, largest=False).indices          # quality gate
        if k <= N_p:
            return cls_idx[cand[:N_p]]
        fp = feats[cand]
        sel = [int(torch.argmin(score[cand]).item())]                 # best-score seed
        maxsim = _sim_to(fp, sel[0]).clone()
        for _ in range(N_p - 1):
            maxsim[torch.tensor(sel, device=device)] = float('inf')
            j = int(torch.argmin(maxsim).item())                      # farthest point
            sel.append(j)
            maxsim = torch.maximum(maxsim, _sim_to(fp, j))
        return cls_idx[cand[torch.tensor(sel, device=device)]]

    if mode == 'mmr':
        # score is z-scored, so the similarity term is z-scored too (against the
        # distribution of pairwise sims among the candidates) -- otherwise mu is
        # not scale free and its useful value swings by orders of magnitude with
        # the embedding. Estimated from a random subsample to avoid an N x N.
        g = torch.Generator(device='cpu').manual_seed(0)
        p = feats[torch.randperm(N, generator=g)[:min(N, 1024)].to(device)]
        off = (p @ p.T).flatten()
        s_mu, s_sd = off.mean(), off.std().clamp_min(1e-8)
        sel = [int(torch.argmin(score).item())]
        maxsim = _sim_to(feats, sel[0]).clone()
        for _ in range(N_p - 1):
            obj = score + mu * ((maxsim - s_mu) / s_sd)
            obj[torch.tensor(sel, device=device)] = float('inf')
            j = int(torch.argmin(obj).item())
            sel.append(j)
            maxsim = torch.maximum(maxsim, _sim_to(feats, j))
        return cls_idx[torch.tensor(sel, device=device)]


    if mode == 'pca':
        # 1. Quality gate: limit to top k by score to ensure baseline effectiveness
        k = min(N, max(N_p, int(round(pool * N_p))))
        cand = torch.topk(score, k=k, largest=False).indices
        if k <= N_p:
            return cls_idx[cand[:N_p]]
            
        fp = feats[cand]
        
        # 2. PCA: Center features and compute eigendecomposition of covariance matrix
        fp_centered = fp - fp.mean(dim=0, keepdim=True)
        cov = (fp_centered.T @ fp_centered) / (k - 1)
        
        # L = eigenvalues (ascending), V = eigenvectors
        L, V = torch.linalg.eigh(cov)
        
        # Get top principal components (up to N_p). Flip to make descending.
        num_pcs = min(N_p, V.shape[1])
        V_top = V[:, -num_pcs:].flip(dims=[1])
        
        # 3. Projection: Absolute projection of candidates onto top PCs
        proj = (fp_centered @ V_top).abs()
        
        sel = []
        available = torch.ones(k, dtype=torch.bool, device=device)
        
        # 4. Selection: Pick the most aligned candidate for each principal component
        for i in range(num_pcs):
            p_c = proj[:, i].clone()
            p_c[~available] = -float('inf')
            best = int(torch.argmax(p_c).item())
            sel.append(best)
            available[best] = False
            
        # 5. Fallback: If N_p > feature dimension, fill remainder using best remaining scores
        if num_pcs < N_p:
            rem_scores = score[cand].clone()
            rem_scores[~available] = float('inf')
            for _ in range(N_p - num_pcs):
                best = int(torch.argmin(rem_scores).item())
                sel.append(best)
                available[best] = False
                rem_scores[best] = float('inf')
                
        return cls_idx[cand[torch.tensor(sel, device=device)]]


    if mode == 'dpp':
        # fast greedy MAP for a DPP (Chen et al., NeurIPS 2018).
        # L = q q^T . K, K = feats feats^T (PSD, unit diagonal), q = exp(-alpha*score)
        # score is z-scored, so exp(-alpha*score) overflows for alpha >~ 20; shifting
        # by the best score is a positive rescaling of L (it cannot change the argmax
        # of the ratio between candidates) and keeps q in (0, 1].
        score = score.double()
        feats = feats.double()
        q = torch.exp(-alpha * (score - score.min()))
        di2 = q * q                                   # diag(L), since K_ii = 1
        cis = torch.zeros(N_p, N, device=device, dtype=torch.float64)
        j = int(torch.argmax(di2).item())
        sel = [j]
        for k in range(1, N_p):
            Lj = q * q[j] * _sim_to(feats, j)
            ei = (Lj - cis[:k - 1, :].T @ cis[:k - 1, j]) / torch.sqrt(di2[j].clamp_min(1e-12))
            cis[k - 1, :] = ei
            di2 = (di2 - ei * ei).clamp_min(0.0)
            di2[torch.tensor(sel, device=device)] = -1.0
            j = int(torch.argmax(di2).item())
            sel.append(j)
        return cls_idx[torch.tensor(sel, device=device)]

    raise ValueError('unknown selection mode %r' % mode)


def select_base_random(labels, y_adv, N_p, device, gen):
    cls_idx = (labels == y_adv).nonzero(as_tuple=True)[0]
    if len(cls_idx) < N_p:
        raise ValueError('class %d has %d images < N_p=%d' % (y_adv, len(cls_idx), N_p))
    perm = torch.randperm(len(cls_idx), generator=gen)[:N_p].to(device)
    return cls_idx[perm]


# --------------------------------------------------------------------------- #
# the selection ladder of app-base.tex, tab:selection-ladder
#
# Every rule below picks N_p bases from the SAME adversarial-class pool that
# select_base_ours draws from, so the rows of that table differ only in the
# scoring rule. They fall into three groups:
#
#   uninformed            first, bottom
#   target-independent    grand, el2n, boundary
#   target-conditioned    pixel, featsim, relevance
#
# 'bottom' is the diagnostic reference: it is select_base_ours run backwards,
# i.e. the N_p WORST-scoring candidates, and is the only rule here meant to
# under-perform random.
#
# GraNd and EL2N follow Paul et al. (2021). EL2N is exact -- ||softmax(z) - y||_2.
# GraNd is the standard last-layer approximation, ||softmax(z) - y||_2 * ||f(x)||_2,
# which is the norm of the loss gradient w.r.t. the final linear layer; the full
# parameter-gradient norm would cost a backward pass per candidate per surrogate.
# Both score the candidate's own difficulty and never look at the target, which is
# exactly the point of including them.
#
# Direction matters and differs per rule. GraNd/EL2N take the LARGEST scores (the
# hard, high-influence examples those papers advocate); every distance-like rule
# takes the SMALLEST (closest to the target).
# --------------------------------------------------------------------------- #

SEL_CRITERIA = ['first', 'bottom', 'grand', 'el2n', 'boundary',
                'pixel', 'featsim', 'relevance']


@torch.no_grad()
def select_base_criterion(criterion, nets, images_norm, labels, x_t_norm, y_adv,
                          N_p, lam, device, base_dist='l2', bs=512):
    cls_idx = (labels == y_adv).nonzero(as_tuple=True)[0]
    if len(cls_idx) < N_p:
        raise ValueError('class %d has %d images < N_p=%d' % (y_adv, len(cls_idx), N_p))
    cand = images_norm[cls_idx]

    if criterion == 'first':
        return cls_idx[:N_p]

    if criterion == 'pixel':                      # pixel-space L2 to the target
        d = torch.empty(len(cand), device=device)
        flat_t = x_t_norm.reshape(-1)
        for i in range(0, len(cand), bs):
            b = cand[i:i + bs]
            d[i:i + len(b)] = ((b.reshape(len(b), -1) - flat_t) ** 2).sum(dim=1)
        return cls_idx[torch.topk(d, k=N_p, largest=False).indices]

    if criterion == 'featsim':                    # cosine to the target, ONE surrogate
        net = nets[0]
        net.eval()
        emb = embed_of(net)
        f_t = emb(x_t_norm.unsqueeze(0))
        s = torch.empty(len(cand), device=device)
        for i in range(0, len(cand), bs):
            b = cand[i:i + bs]
            s[i:i + len(b)] = F.cosine_similarity(emb(b), f_t.expand(len(b), -1), dim=1)
        return cls_idx[torch.topk(s, k=N_p, largest=True).indices]

    # ---- the remaining rules average a per-surrogate score over the ensemble --
    score = torch.zeros(len(cls_idx), device=device)
    for net in nets:
        net.eval()
        emb = embed_of(net)
        f_t = emb(x_t_norm.unsqueeze(0)) if criterion in ('relevance', 'bottom') else None
        parts = []
        for i in range(0, len(cand), bs):
            b = cand[i:i + bs]
            z = net(b)
            if criterion in ('grand', 'el2n'):
                p = F.softmax(z, dim=1)
                p[:, y_adv] -= 1.0                # candidates are all true class y_adv
                e = p.norm(dim=1)
                parts.append(e * emb(b).flatten(1).norm(dim=1) if criterion == 'grand' else e)
                continue
            z_adv = z[:, y_adv].clone()
            z_o = z.clone()
            z_o[:, y_adv] = float('-inf')
            m = z_adv - z_o.max(dim=1).values
            if criterion == 'boundary':
                parts.append(m)
                continue
            fb = emb(b)
            if base_dist == 'cosine':
                d = 1.0 - F.cosine_similarity(fb, f_t.expand(len(b), -1), dim=1)
            else:
                d = ((fb - f_t) ** 2).sum(dim=1)
            if criterion == 'relevance':
                parts.append(d)
            else:                                 # 'bottom' -- the full ours score
                parts.append(standardize(d) + lam * standardize(m))
        score += torch.cat(parts) if criterion == 'bottom' else standardize(torch.cat(parts))
    score /= len(nets)

    largest = criterion in ('grand', 'el2n', 'bottom')
    return cls_idx[torch.topk(score, k=N_p, largest=largest).indices]


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
# crafting: gradient matching (Witches' Brew) and its sharpness-aware variant
# --------------------------------------------------------------------------- #

def _perturbed_target_grad(net, x_t_norm, y_t, crit, mode, sigma, samples):
    """grad_theta CE(x_t, y_adv) taken at a sharpness-perturbed theta (SAPA).

    The perturbation is applied to a deepcopy, so the surrogate itself is never
    touched -- restoring it by subtracting the perturbation back off would leave
    float drift in weights that are shared by every later target.

    Only the parameters are perturbed, never the buffers: the nets are in eval()
    so BN uses its running stats, and named/plain parameters() skips
    running_mean / running_var, which is also what the reference does.
    """
    import copy

    params = [p for p in net.parameters()]
    loss_t = crit(net(x_t_norm.unsqueeze(0)), y_t)
    g0 = [g.detach() for g in torch.autograd.grad(loss_t, params)]

    if mode == 'worst':
        # SAM's one-step inner maximization: ascend the ADVERSARIAL target loss,
        # i.e. move to the model in the sigma-ball that the attack works worst on.
        # (The released SAPA code guards this step with `if p.grad is None:
        # continue` on a fresh deepcopy whose .grad is never populated, so the
        # step is skipped and worstsharp silently degenerates to plain gradient
        # matching. This is the step the paper actually specifies.)
        gnorm = torch.sqrt(sum((g ** 2).sum() for g in g0))
        scale = sigma / (gnorm + 1e-12)
        clone = copy.deepcopy(net)
        with torch.no_grad():
            for p, g in zip(clone.parameters(), g0):
                p.add_(g * scale)
        cp = [p for p in clone.parameters()]
        loss_s = crit(clone(x_t_norm.unsqueeze(0)), y_t)
        g_out = flat_grad([g.detach() for g in torch.autograd.grad(loss_s, cp)])
        del clone
        return g_out

    if mode == 'avg':
        # average-case sharpness: E_xi[ grad CE(theta + xi) ], xi ~ N(0, sigma^2 I)
        acc = None
        for _ in range(max(1, samples)):
            clone = copy.deepcopy(net)
            with torch.no_grad():
                for p in clone.parameters():
                    p.add_(torch.randn_like(p) * sigma)
            cp = [p for p in clone.parameters()]
            loss_s = crit(clone(x_t_norm.unsqueeze(0)), y_t)
            g = flat_grad([gg.detach() for gg in torch.autograd.grad(loss_s, cp)])
            acc = g if acc is None else acc + g
            del clone
        return acc / max(1, samples)

    raise ValueError('unknown sharpness mode %r' % mode)


def _target_grads(nets, x_t_norm, y_t, crit, sharp_mode, sharp_sigma, sharp_samples):
    """Per-surrogate target gradient. sharp_mode None is the plain gradmatch one."""
    g_targets = []
    for net in nets:
        if sharp_mode is None:
            params = [p for p in net.parameters()]
            loss_t = crit(net(x_t_norm.unsqueeze(0)), y_t)
            g_t = torch.autograd.grad(loss_t, params)
            g_targets.append(flat_grad([g.detach() for g in g_t]))
        else:
            g_targets.append(_perturbed_target_grad(
                net, x_t_norm, y_t, crit, sharp_mode, sharp_sigma, sharp_samples))
    return g_targets


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
                    schedule=False, lowmem=False, chunk=0,
                    sharp_mode=None, sharp_sigma=0.0, sharp_samples=20):
    set_requires_grad(nets, True)
    for n in nets:
        n.eval()
    crit = nn.CrossEntropyLoss().to(device)
    y_t = torch.full((1,), y_adv, dtype=torch.long, device=device)
    y_p = torch.full((base01.shape[0],), y_adv, dtype=torch.long, device=device)

    # The ONLY thing SAPA changes. sharp_mode None is the plain Witches' Brew
    # target gradient; everything below this line is shared by both attacks and
    # only ever sees g_targets as a frozen flat vector.
    g_targets = _target_grads(nets, x_t_norm, y_t, crit, sharp_mode, sharp_sigma,
                              sharp_samples)

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
                for _i, (net, g_t) in enumerate(zip(nets, g_targets)):
                    params = [p for p in net.parameters() if p.requires_grad]
                    loss_p = crit(net(x_adv_norm), y_p)
                    # x_adv_norm is built once and shared by every net, so all but
                    # the last backward must keep that subgraph alive
                    all_grads = torch.autograd.grad(
                        loss_p, params + [delta],
                        retain_graph=(_i < len(nets) - 1))
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
# per-run context, on-disk caches, csv shards
# --------------------------------------------------------------------------- #

def resolve_gpus(spec):
    """--gpus all | none | '0,2,3'  ->  list of cuda ordinals ([] means cpu)."""
    if not torch.cuda.is_available():
        return []
    n = torch.cuda.device_count()
    s = str(spec or 'all').strip().lower()
    if s in ('none', 'cpu', '-1', ''):
        return []
    if s == 'all':
        return list(range(n))
    ids = [int(x) for x in s.replace(',', ' ').split()]
    bad = [i for i in ids if i < 0 or i >= n]
    if bad:
        raise ValueError('--gpus asks for device(s) %s but only %d cuda devices exist'
                         % (bad, n))
    return ids


def build_context(args, device):
    """Everything that is per-process: the dataset on THIS device + norm helpers."""
    channel, im_size, num_classes, class_names, mean, std, dst_train, dst_test, _ = \
        get_dataset(args.dataset, args.data_path)
    train_imgs, train_labs = stack_dataset(dst_train, device)
    test_imgs, test_labs = stack_dataset(dst_test, device)
    m = torch.tensor(mean, device=device).view(1, channel, 1, 1)
    s = torch.tensor(std, device=device).view(1, channel, 1, 1)
    return {
        'device': device, 'channel': channel, 'im_size': im_size,
        'num_classes': num_classes, 'class_names': class_names,
        'train_imgs': train_imgs, 'train_labs': train_labs,
        'test_imgs': test_imgs, 'test_labs': test_labs,
        'norm': (lambda x01: (x01 - m) / s),
        'denorm': (lambda xn: xn * s + m),
        'dsa_param': ParamDiffAug(),
    }


def _poison_cache_dir(run_dir):
    return os.path.join(run_dir, 'poison_cache')


def load_legacy_cache(run_dir, recompute):
    """The old single-file deltas.pt / bases.json, kept read-only for old runs."""
    if recompute:
        return {}, {}
    dp = os.path.join(run_dir, 'deltas.pt')
    bp = os.path.join(run_dir, 'bases.json')
    d = torch.load(dp, map_location='cpu') if os.path.exists(dp) else {}
    b = json.load(open(bp)) if os.path.exists(bp) else {}
    return d, b


def load_poison_cache(run_dir, tidx, legacy, recompute):
    """(delta_cpu, base_list) for one target, or (None, None).

    Per-target files instead of one big dict, so several processes can write the
    cache at the same time. Delta and bases are only usable together -- a delta
    was crafted from a specific base set.
    """
    if recompute:
        return None, None
    legacy_d, legacy_b = legacy
    d = _poison_cache_dir(run_dir)
    dp = os.path.join(d, 'delta_%d.pt' % tidx)
    bp = os.path.join(d, 'base_%d.json' % tidx)
    delta = (torch.load(dp, map_location='cpu') if os.path.exists(dp)
             else legacy_d.get(str(tidx)))
    base = (json.load(open(bp)) if os.path.exists(bp) else legacy_b.get(str(tidx)))
    if delta is None or base is None:
        return None, None
    return delta, base


def save_poison_cache(run_dir, tidx, delta, base_list):
    d = _poison_cache_dir(run_dir)
    os.makedirs(d, exist_ok=True)
    dp = os.path.join(d, 'delta_%d.pt' % tidx)
    bp = os.path.join(d, 'base_%d.json' % tidx)
    torch.save(delta, dp + '.tmp')
    os.replace(dp + '.tmp', dp)                     # atomic, no half-written cache
    with open(bp + '.tmp', 'w') as f:
        json.dump(base_list, f)
    os.replace(bp + '.tmp', bp)


LOCK_STALE_S = 7200      # a lock older than this is assumed to belong to a dead job.
                         # Must exceed the longest gap between trials: a b0.04 craft
                         # alone runs ~77 min.


def acquire_run_lock(run_dir):
    """Refuse to run two processes in one run dir.

    Several appendix tables legitimately share a configuration -- the ConvNetBN
    dog-bird b0.005 runs appear in the broad, matched-architecture and
    cross-dataset scripts -- so the same run dir can be requested twice. Sharing
    the RESULT is fine; running both at once is not: each process merges and then
    deletes the other's results_rank*.csv, and the survivor dies on a stale file
    handle mid-write.

    Returns the lock path, or None if someone else holds it (caller should skip).
    """
    path = os.path.join(run_dir, '.lock')
    me = '%s:%d' % (os.uname().nodename, os.getpid())
    try:
        fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        os.write(fd, me.encode())
        os.close(fd)
        return path
    except FileExistsError:
        pass
    try:
        age = time.time() - os.path.getmtime(path)
        who = open(path).read().strip()
    except OSError:
        return acquire_run_lock(run_dir)          # vanished between the two calls
    if age < LOCK_STALE_S:
        log('  another process holds this run (%s, %.0f min ago) -- skipping. '
            'Rerun this script once it is done.' % (who, age / 60))
        return None
    log('  taking over a stale lock from %s (%.0f min old)' % (who, age / 60))
    with open(path, 'w') as f:
        f.write(me)
    return path


def touch_run_lock(path):
    try:
        os.utime(path, None)
    except OSError:
        pass


def release_run_lock(path):
    try:
        if path:
            os.remove(path)
    except OSError:
        pass


def _shard_paths(run_dir):
    return sorted(glob.glob(os.path.join(run_dir, 'results_rank*.csv')))


def open_shard(run_dir, rank):
    """Per-worker csv. Rows are flushed as they happen; merged into results.csv
    by the parent once every worker is done."""
    path = os.path.join(run_dir, 'results_rank%d.csv' % rank)
    fresh = (not os.path.exists(path)) or os.path.getsize(path) == 0
    rf = open(path, 'a', newline='')
    w = csv.DictWriter(rf, fieldnames=RESULT_FIELDS)
    if fresh:
        w.writeheader()
        rf.flush()
    return rf, w


def merge_result_shards(run_dir, results_path):
    """Fold results_rank*.csv into results.csv, dropping (target, victim) dupes."""
    shards = _shard_paths(run_dir)
    if not shards:
        return 0
    seen = set()
    if os.path.exists(results_path):
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
# one target = one unit of work (base selection -> crafting -> victim training)
# --------------------------------------------------------------------------- #

def prepare_poisons(args, ctx, sel_nets, craft_nets, tidx, y_adv, N_p, run_dir,
                    legacy, recompute=None):
    """Base selection + crafting for one target -> (base_idx, x_adv01, obj, linf).

    sel_nets is the pool the bases are scored on, craft_nets the pool the poisons
    are optimized against. They are the same ensemble unless --sel_model asks for
    a cross-architecture run.

    Split out of the victim loop so that a gpu which has nothing left to craft can
    still help train the victims of a target another gpu already crafted: it just
    calls this, gets the poisons straight out of poison_cache, and trains.
    """
    device = ctx['device']
    train_imgs, train_labs = ctx['train_imgs'], ctx['train_labs']
    norm, denorm = ctx['norm'], ctx['denorm']
    if recompute is None:
        recompute = args.recompute_deltas

    # Per-target rng, so base selection and crafting are reproducible no matter
    # which gpu picks the target up or in what order.
    tseed = args.seed * 100003 + int(tidx)
    set_seed(tseed)
    gen = torch.Generator(device='cpu').manual_seed(tseed)

    x_t_norm = ctx['test_imgs'][tidx]
    cached_delta, cached_base = load_poison_cache(run_dir, tidx, legacy, recompute)
    jacobian_kwargs = {
        'use_jacobian_score': getattr(args, 'use_jacobian_score', False),
        'jacobian_weight': getattr(args, 'jacobian_weight', 1.0),
        'jacobian_batch_size': getattr(args, 'jacobian_batch_size', 64),
    }

    # ---- base selection ----------------------------------------------------
    if cached_base is not None:
        base_idx = torch.tensor(cached_base, dtype=torch.long, device=device)
    elif args.base == 'random':
        base_idx = select_base_random(train_labs, y_adv, N_p, device, gen)
    elif getattr(args, 'sel_exact_alignment', False):
        base_idx = select_base_exact_alignment(
            sel_nets[:args.sel_K] if args.sel_K else sel_nets,
            train_imgs, train_labs, x_t_norm, y_adv, N_p, device,
            batch_size=getattr(args, 'jacobian_batch_size', 64))
    elif getattr(args, 'sel_a_minus_mr', False):
        base_idx = select_base_a_minus_mr(
            sel_nets[:args.sel_K] if args.sel_K else sel_nets,
            train_imgs, train_labs, x_t_norm, y_adv, N_p, device,
            batch_size=getattr(args, 'jacobian_batch_size', 64))
    elif getattr(args, 'sel_component', None):
        base_idx = select_base_components(
            sel_nets[:args.sel_K] if args.sel_K else sel_nets,
            train_imgs, train_labs, x_t_norm, y_adv, N_p, device,
            formula=args.sel_component,
            batch_size=getattr(args, 'jacobian_batch_size', 64))
    elif getattr(args, 'sel_criterion', None):
        base_idx = select_base_criterion(
            args.sel_criterion,
            sel_nets[:args.sel_K] if args.sel_K else sel_nets,
            train_imgs, train_labs, x_t_norm, y_adv, N_p, args.lambda_margin,
            device, base_dist=args.base_dist)
    elif args.base_topr:
        base_idx = select_base_topr(sel_nets[:args.sel_K] if args.sel_K else sel_nets,
                                    train_imgs, train_labs, x_t_norm, y_adv, N_p,
                                    args.base_topr, args.lambda_margin, device,
                                    base_dist=args.base_dist, **jacobian_kwargs)
        log('  target %d: budget concentrated in %d neighbourhood(s), %d distinct bases'
            % (tidx, args.base_topr, len(base_idx)))
    elif args.sel_mode:
        base_idx = select_base_ours_div(sel_nets[:args.sel_K] if args.sel_K else sel_nets,
                                        train_imgs, train_labs,
                                        x_t_norm, y_adv, N_p, args.lambda_margin,
                                        device, base_dist=args.base_dist,
                                        mode=args.sel_mode, pool=args.sel_pool,
                                        mu=args.sel_mu, alpha=args.sel_alpha,
                                        **jacobian_kwargs)
    else:
        base_idx = select_base_ours(sel_nets[:args.sel_K] if args.sel_K else sel_nets,
                                    train_imgs, train_labs, x_t_norm,
                                    y_adv, N_p, args.lambda_margin, device,
                                    base_dist=args.base_dist, **jacobian_kwargs)

    # ---- crafting ----------------------------------------------------------
    base01 = denorm(train_imgs[base_idx]).clamp(0.0, 1.0).detach()
    if cached_delta is not None:
        x_adv01 = torch.clamp(base01 + cached_delta.to(device), 0.0, 1.0)
        obj = float('nan')
    else:
        t0 = time.time()
        if args.attack in ('gradmatch', 'sapa'):
            # sharp_mode is None for gradmatch, so this is the exact same call the
            # gradmatch runs have always made
            x_adv01, obj = craft_gradmatch(
                craft_nets, base01, x_t_norm, y_adv, norm, args.epsilon,
                args.craft_alpha, args.craft_steps, args.restarts, device,
                dsa_strategy=(args.dsa_strategy if args.craft_aug else None),
                dsa_param=ctx['dsa_param'], fast=args.fast_gradmatch,
                schedule=args.craft_schedule, lowmem=args.craft_lowmem,
                chunk=args.craft_batch,
                sharp_mode=(args.sharp_mode if args.attack == 'sapa' else None),
                sharp_sigma=args.sharp_sigma, sharp_samples=args.sharp_samples)
        else:
            x_adv01, obj = craft_fc(
                craft_nets, base01, x_t_norm, norm, args.epsilon, args.craft_steps,
                args.craft_alpha, device, restarts=args.fc_restarts,
                mode=args.fc_mode)
        save_poison_cache(run_dir, tidx, (x_adv01 - base01).detach().cpu(),
                          base_idx.cpu().tolist())
        log('  target %d: crafted %d poisons in %.0f s, obj=%.5f'
            % (tidx, N_p, time.time() - t0, obj))

    linf = (x_adv01 - base01).abs().max().item()
    log('  target %d: realized linf = %.5f (%.2f/255), budget = %.2f/255'
        % (tidx, linf, linf * 255, args.epsilon * 255))
    return base_idx, x_adv01, obj, linf


def run_victims(args, ctx, tidx, y_adv, N_p, victim_ids, prep, target_score,
                clean_asr, emit):
    """Inject the poisons, train the listed victims from scratch, restore."""
    device = ctx['device']
    train_imgs, train_labs = ctx['train_imgs'], ctx['train_labs']
    class_names, num_classes = ctx['class_names'], ctx['num_classes']
    base_idx, x_adv01, obj, linf = prep
    x_t_norm = ctx['test_imgs'][tidx]
    tally = np.zeros(num_classes, dtype=np.int64)
    succ, ctas = [], []

    clean_rows = train_imgs[base_idx].clone()
    train_imgs[base_idx] = ctx['norm'](x_adv01)
    try:
        for vi in victim_ids:
            # victim init and sgd order depend only on (seed, target, victim), so a
            # trial gives the same answer whichever gpu happens to run it
            seed_v = args.seed * 100000 + tidx * 100 + vi
            net = build_network(args.model, ctx['channel'], num_classes,
                                ctx['im_size'], device, seed=seed_v)
            net = train_from_scratch(net, train_imgs, train_labs, args.victim_epochs,
                                     args.victim_lr, args.victim_bs, args.victim_decay,
                                     device, weight_decay=args.victim_wd,
                                     aug=args.victim_aug,
                                     dsa_strategy=args.dsa_strategy,
                                     dsa_param=ctx['dsa_param'])
            pred = predict_target(net, x_t_norm)
            cta = test_acc(net, ctx['test_imgs'], ctx['test_labs'])
            ok = int(pred == y_adv)
            tally[pred] += 1
            succ.append(ok)
            ctas.append(cta)
            emit({
                'model': args.model, 'attack': args.attack, 'base': args.base,
                'class_pair': args.class_pair, 'seed': args.seed,
                'budget': args.budget, 'num_poisons': N_p, 'epsilon': args.epsilon,
                'target_idx': tidx, 'target_score': target_score,
                'victim_id': vi, 'success': ok, 'clean_test_acc': cta,
                'clean_asr': clean_asr, 'craft_obj': obj, 'realized_linf': linf})
            log('  [t%d v%d/%d] %s (pred=%s) CTA=%.4f'
                % (tidx, vi + 1, args.num_victims,
                   'SUCCESS' if ok else 'fail', class_names[pred], cta))
            del net
            if str(device).startswith('cuda'):
                torch.cuda.empty_cache()
    finally:
        train_imgs[base_idx] = clean_rows

    if len(succ) > 1:
        log('  target %d, %d victims on this gpu: ASR=%.1f%% CTA=%.4f +/- %.4f'
            % (tidx, len(succ), 100.0 * np.mean(succ), np.mean(ctas), np.std(ctas)))
    return tally


def run_one_target(args, ctx, sel_nets, craft_nets, tidx, y_adv, N_p,
                   target_score, clean_asr, run_dir, completed, legacy, emit,
                   pos=1, total=1):
    """Whole target on one device: used by the single-gpu / sequential path."""
    log('=== target %d (%d/%d) ===' % (tidx, pos, total))
    todo = [vi for vi in range(args.num_victims) if (tidx, vi) not in completed]
    if not todo:
        log('  all %d victims already done, skipping' % args.num_victims)
        return np.zeros(ctx['num_classes'], dtype=np.int64)
    prep = prepare_poisons(args, ctx, sel_nets, craft_nets, tidx, y_adv, N_p,
                           run_dir, legacy)
    return run_victims(args, ctx, tidx, y_adv, N_p, todo, prep, target_score,
                       clean_asr, emit)


# --------------------------------------------------------------------------- #
# multi-gpu process pool: one worker per gpu, shared job queue
# --------------------------------------------------------------------------- #

def _run_pool(worker, gpus, jobs, extra):
    """Start one process per gpu, feed them `jobs` through a shared queue, and
    collect one ('ok'|'err', rank, payload) message per worker.

    The queue is what makes this a work queue rather than a static split: a gpu
    that finishes a job early immediately pulls the next one.
    """
    mpctx = mp.get_context('spawn')
    job_q, out_q = mpctx.Queue(), mpctx.Queue()
    for j in jobs:
        job_q.put(j)
    for _ in gpus:
        job_q.put(None)                     # one sentinel per worker
    procs = []
    for rank, gpu in enumerate(gpus):
        p = mpctx.Process(target=worker, args=(rank, gpu, job_q, out_q) + tuple(extra))
        p.start()
        procs.append(p)
    return _collect(procs, out_q, gpus)


def _collect(procs, out_q, gpus):
    """One ('ok'|'err', rank, payload) message per worker, without hanging if a
    worker dies without sending one (cuda OOM kill, segfault)."""
    results, errors, waiting = [], [], set(range(len(procs)))
    while waiting:
        try:
            status, rank, payload = out_q.get(timeout=10.0)
        except _queue.Empty:
            if all(procs[r].is_alive() for r in waiting):
                continue
            time.sleep(2.0)                 # let any in-flight message land first
            while True:
                try:
                    status, rank, payload = out_q.get_nowait()
                except _queue.Empty:
                    break
                waiting.discard(rank)
                (results if status == 'ok' else errors).append((rank, payload))
            for r in sorted(waiting):
                if not procs[r].is_alive():
                    errors.append((r, 'worker on gpu %d died, exit code %s'
                                   % (gpus[r], procs[r].exitcode)))
                    waiting.discard(r)
            continue
        waiting.discard(rank)
        (results if status == 'ok' else errors).append((rank, payload))
    for p in procs:
        p.join()
    return results, errors


# --- the target/victim scheduler ------------------------------------------- #
#
# The unit of work is a (target, victim) trial, but the trials of one target are
# NOT independent: whoever runs the first one has to craft that target's poisons.
# So the claim policy is three-tiered, in this order:
#
#   1. stay on the target this gpu already has poisons for  (no reload, no craft)
#   2. otherwise start a target nobody has started yet      (one target per gpu,
#      so all gpus craft different targets at the same time)
#   3. otherwise -- the tail, when there are fewer unstarted targets left than
#      gpus -- help with the victims of a target another gpu has already crafted
#
# Rule 3 is what keeps every gpu busy on the last one or two targets. It only
# takes from targets marked ready, i.e. whose poisons are already in
# poison_cache, so a helper never re-crafts anything.

def claim_next(state, lock, current_tidx):
    """-> ((tidx, victim_id), False) | (None, True = wait) | (None, False = done)"""
    with lock:
        rem = state['remaining']
        if not rem:
            return None, False
        started, ready = state['started'], state['ready']

        def take(t):
            vs = rem[t]
            vi = vs.pop(0)
            if vs:
                rem[t] = vs
            else:
                del rem[t]
            state['remaining'] = rem
            if t not in started:
                started[t] = True
                state['started'] = started
            return (t, vi), False

        if current_tidx in rem:                                   # 1. stay put
            return take(current_tidx)
        fresh = sorted(t for t in rem if t not in started)
        if fresh:                                                 # 2. new target
            return take(fresh[0])
        helpable = [t for t in rem if ready.get(t)]
        if helpable:                                              # 3. help out
            return take(max(helpable, key=lambda t: len(rem[t])))
        # work is left, but every remaining target is still being crafted
        return None, True


def _mark_ready(state, lock, tidx):
    with lock:
        ready = state['ready']
        ready[tidx] = True
        state['ready'] = ready


def _leave(state, lock, owned):
    """Last thing a worker does. Any target it had started crafting but never
    finished is un-started, so a gpu waiting on it crafts it instead of waiting
    out --steal_timeout for poisons that are never coming.

    Runs in a finally, so it must never raise and mask the real error.
    """
    try:
        with lock:
            state['active'] = state['active'] - 1
            ready, started, rem = state['ready'], state['started'], state['remaining']
            dropped = [t for t in owned if not ready.get(t) and t in rem]
            if dropped:
                for t in dropped:
                    started.pop(t, None)
                state['started'] = started
    except Exception:
        pass


def _target_worker(rank, gpu, out_q, state, lock, args, run_dir, log_path,
                   y_adv, N_p, target_scores, clean_asrs):
    global _LOG_PATH, _LOG_TAG
    _LOG_PATH, _LOG_TAG = log_path, '[gpu%d]' % gpu
    rf = None
    owned = set()              # targets this gpu took responsibility for crafting
    try:
        torch.cuda.set_device(gpu)
        device = 'cuda:%d' % gpu
        set_seed(args.seed)
        ctx = build_context(args, device)
        surrogates = get_surrogates(args, ctx['train_imgs'], ctx['train_labs'],
                                    ctx['test_imgs'], ctx['test_labs'], ctx['channel'],
                                    ctx['num_classes'], ctx['im_size'], device,
                                    ctx['dsa_param'])
        craft_nets = surrogates[:args.craft_ensemble] if args.craft_ensemble else surrogates
        sel_nets = surrogates
        if args.sel_model and args.sel_model != args.model and args.base == 'ours':
            sel_nets = get_sel_surrogates(args, ctx['train_imgs'], ctx['train_labs'],
                                          ctx['test_imgs'], ctx['test_labs'],
                                          ctx['channel'], ctx['num_classes'],
                                          ctx['im_size'], device, ctx['dsa_param'])
        legacy = load_legacy_cache(run_dir, args.recompute_deltas)
        rf, writer = open_shard(run_dir, rank)

        def emit(row):
            writer.writerow(row)
            rf.flush()

        tally = np.zeros(ctx['num_classes'], dtype=np.int64)
        cur_tidx, prep, waited = None, None, 0.0
        while True:
            job, wait = claim_next(state, lock, cur_tidx)
            if job is None:
                if not wait:
                    break
                with lock:
                    alone = state['active'] <= 1
                if alone:
                    log('  trials left but no other gpu is alive to craft them, '
                        'stopping (re-run to resume)')
                    break
                if waited >= args.steal_timeout:
                    log('  waited %.0f s for another gpu to finish crafting, stopping'
                        % waited)
                    break
                if waited == 0.0:
                    log('  no target left to start; waiting to help with the victims '
                        'of a target another gpu is still crafting')
                time.sleep(10.0)
                waited += 10.0
                continue
            waited = 0.0
            tidx, vi = job
            if prep is None or cur_tidx != tidx:
                with lock:
                    already = bool(state['ready'].get(tidx))
                log('=== target %d %s ==='
                    % (tidx, 'victim %d (helping out)' % vi if already
                       else '(crafting here)'))
                if not already:
                    owned.add(tidx)
                # `already` means another gpu crafted it this run, so read the cache
                # even under --recompute_deltas instead of crafting it twice
                prep = prepare_poisons(args, ctx, sel_nets, craft_nets, tidx, y_adv,
                                       N_p, run_dir, legacy,
                                       recompute=(args.recompute_deltas and not already))
                cur_tidx = tidx
                _mark_ready(state, lock, tidx)
            tally += run_victims(args, ctx, tidx, y_adv, N_p, [vi], prep,
                                 target_scores.get(tidx, ''),
                                 clean_asrs.get(tidx, float('nan')), emit)
        out_q.put(('ok', rank, tally.tolist()))
    except Exception:
        log('WORKER FAILED:\n%s' % traceback.format_exc())
        out_q.put(('err', rank, traceback.format_exc()))
    finally:
        _leave(state, lock, owned)
        if rf is not None:
            rf.close()


def run_targets_parallel(args, gpus, pending, run_dir, log_path, y_adv, N_p,
                         target_scores, clean_asrs, completed, num_classes):
    mpctx = mp.get_context('spawn')
    mgr = mpctx.Manager()
    lock = mgr.Lock()
    state = mgr.dict()
    state['remaining'] = {t: [vi for vi in range(args.num_victims)
                              if (t, vi) not in completed] for t in pending}
    state['started'] = {}
    state['ready'] = {}
    state['active'] = len(gpus)
    out_q = mpctx.Queue()
    procs = []
    for rank, gpu in enumerate(gpus):
        p = mpctx.Process(target=_target_worker,
                          args=(rank, gpu, out_q, state, lock, args, run_dir,
                                log_path, y_adv, N_p, target_scores, clean_asrs))
        p.start()
        procs.append(p)
    results, errors = _collect(procs, out_q, gpus)
    tally = np.zeros(num_classes, dtype=np.int64)
    for _rank, part in results:
        tally += np.array(part, dtype=np.int64)
    left = sum(len(v) for v in state['remaining'].values())
    mgr.shutdown()
    if left:
        errors.append((-1, '%d (target, victim) trials were never run' % left))
    return tally, errors


def _pretrain_worker(rank, gpu, job_q, out_q, args, log_path):
    """Train one missing cached net (surrogate or clean victim) per job."""
    global _LOG_PATH, _LOG_TAG
    _LOG_PATH, _LOG_TAG = log_path, '[gpu%d]' % gpu
    try:
        torch.cuda.set_device(gpu)
        device = 'cuda:%d' % gpu
        set_seed(args.seed)
        ctx = build_context(args, device)
        common = (ctx['train_imgs'], ctx['train_labs'], ctx['test_imgs'],
                  ctx['test_labs'], ctx['channel'], ctx['num_classes'],
                  ctx['im_size'], device, ctx['dsa_param'])
        while True:
            job = job_q.get()
            if job is None:
                break
            part, i = job
            if part == 'surrogate':
                get_surrogates(args, *common, only_id=i)
            else:
                get_clean_victims(args, *common, only_id=i)
        out_q.put(('ok', rank, None))
    except Exception:
        log('PRETRAIN WORKER FAILED:\n%s' % traceback.format_exc())
        out_q.put(('err', rank, traceback.format_exc()))


def pretrain_pools_parallel(args, gpus, log_path):
    """Fan the missing surrogate / clean-victim trainings out over the gpus, so
    the target stage below finds every net in the cache and never races on it."""
    jobs = []
    d = surrogate_dir(args)
    for i in range(args.num_surrogates):
        if not os.path.exists(os.path.join(d, 'net_%d.pt' % i)):
            jobs.append(('surrogate', i))
    if args.clean_baseline:
        d = victim_dir(args)
        for i in range(args.num_victims):
            if not os.path.exists(os.path.join(d, 'net_%d.pt' % i)):
                jobs.append(('victim', i))
    if len(jobs) < 2 or len(gpus) < 2:
        return
    n = min(len(gpus), len(jobs))
    log('=== pre-training %d missing cached nets over %d gpus %s ==='
        % (len(jobs), n, gpus[:n]))
    t0 = time.time()
    _, errors = _run_pool(_pretrain_worker, gpus[:n], jobs, (args, log_path))
    for rank, err in errors:
        log('  pretrain worker %d failed:\n%s' % (rank, err))
    if errors:
        raise RuntimeError('%d pretrain worker(s) failed, see the log' % len(errors))
    log('  pre-training done in %.0f s' % (time.time() - t0))


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #

def build_run_name(args):
    # eps%d alone rounds eps*255 to an integer, so any two radii inside the same
    # 1/255 bucket -- 2e-3 and 5e-3 are both 'eps1' -- would share a run directory
    # and cross-contaminate each other's poison_cache. Sub-1/255 radii therefore get
    # a precise tag. Radii that ARE whole 1/255 steps keep the old spelling, so every
    # run already on disk (eps8, eps16) resolves unchanged.
    e255 = args.epsilon * 255.0
    eps_tag = ('%d' % round(e255) if abs(e255 - round(e255)) < 1e-3
               else ('%g' % e255).replace('.', 'p'))
    name = ('%s_%s_%s_%s_%s_b%g_eps%s_seed%d'
            % (args.dataset, args.model, args.attack, args.base, args.class_pair,
               args.budget, eps_tag, args.seed))
    if args.base == 'ours':
        name += '_lam%g_%s' % (args.lambda_margin, args.base_dist)
        # a diversity mode is a DIFFERENT selection, so it must not share a run
        # directory with the plain --base ours runs
        if args.sel_filter:
            name += '_selfilter%g' % args.sel_pool
        elif args.sel_pca:
            name += '_selpca%g' % args.sel_pool
        elif args.sel_mmr:
            name += '_selmmr%g' % args.sel_mu
        elif args.sel_dpp:
            name += '_seldpp%g' % args.sel_alpha
        elif getattr(args, 'sel_exact_alignment', False):
            name += '_selexactgigt'
        elif getattr(args, 'sel_a_minus_mr', False):
            name += '_selAminusMR'
        elif getattr(args, 'sel_component', None):
            name += '_sel%s' % COMPONENT_SELECTOR_SUFFIXES[args.sel_component]
    # The exact interaction changes selected bases and must never reuse a baseline
    # poison cache.  Batch size/backend affect performance only, not run identity.
    if getattr(args, 'use_jacobian_score', False):
        name += '_jacw%g' % getattr(args, 'jacobian_weight', 1.0)
    # bases picked by another architecture are a different selection, so the run
    # must not share a directory (or a poison_cache) with the S = A run. Outside
    # the --base ours block on purpose: --base random ignores sel_model, but an
    # S != A random run still needs its own directory, or it would overwrite the
    # diagonal run it is meant to reproduce. For --base ours the suffix lands in
    # exactly the same place it always did, so no existing run dir changes name.
    # getattr, not args.sel_model: defense.py builds its own Namespace from a
    # different parser and calls this to find the attack run it replays.
    sel_model = getattr(args, 'sel_model', None)
    if sel_model and sel_model != args.model:
        name += '_selarch%s' % sel_model
    # a different selector ensemble size is a different selection, so it needs its
    # own run dir. Only when asked for explicitly -- runs that never pass --sel_K
    # keep the name they have always had.
    crit = getattr(args, 'sel_criterion', None)
    if crit:
        name += '_sel%s' % crit
    sel_K = getattr(args, 'sel_K', None)
    if sel_K:
        name += '_K%d' % sel_K
    # Craft-time augmentation changes the poisons themselves, so an unmatched and a
    # matched craft must not share a run dir (or a poison_cache) -- without this the
    # second one silently resumes off the first one's cache. Only non-default
    # settings get a suffix, so every existing run dir keeps its name.
    if not getattr(args, 'craft_aug', True):
        name += '_craftnoaug'
    elif getattr(args, 'dsa_strategy', DSA_DEFAULT) != DSA_DEFAULT:
        name += '_craft%s' % args.dsa_strategy.replace('_', '')
    topr = getattr(args, 'base_topr', None)
    if topr:
        name += '_top%d' % topr
    if args.attack == 'fc' and args.fc_mode != 'sample':
        name += '_%s' % args.fc_mode
    if args.attack == 'sapa':
        # a different sigma / mode is a different attack, so it must not share a
        # run directory (and therefore a poison_cache) with another sapa config
        name += '_%s%g' % (args.sharp_mode, args.sharp_sigma)
        if args.sharp_mode == 'avg':
            name += 'x%d' % args.sharp_samples
    if args.craft_ensemble:
        name += '_ce%d' % args.craft_ensemble
    if isinstance(args.target_select, int):
        name += '_tgt%d' % args.target_select
    return name


# --------------------------------------------------------------------------- #
# --time_mode: base-selection overhead only, no crafting, no victim training.
#
# select_base_ours / _ours_div('dpp') / _exact_alignment / _a_minus_mr /
# _components all end the same way: some per-candidate quantities are computed
# (feature distance d, margin M, the Jacobian backbone interaction A, the raw
# representation inner product R -- whichever the selector needs, one or more
# forward/backward passes over the WHOLE candidate pool per surrogate), then
# those quantities are combined into one score and N_p candidates are picked
# (torch.topk, or the greedy DPP log-det loop). The functions below are
# timing-instrumented duplicates of that dispatch, split into ONE phase PER
# COMPONENT (d / M / A / R / gi_gt, whichever apply) plus a final 'sort' phase
# for the combine + topk / greedy step. Where the original computes two
# components in one interleaved per-batch loop (--base ours's d and M; a-mr's
# M and R), the timed duplicate does each as its own full pass over every
# surrogate instead -- same total forward/backward work, same final score,
# only reordered, which is immaterial since these components never depend on
# one another. All of them run under torch.no_grad(), same as the originals
# they duplicate (_ours_pointwise_score, select_base_exact_alignment,
# select_base_a_minus_mr, select_base_components are themselves decorated
# @torch.no_grad()) -- WITHOUT this every forward pass here would build and
# retain a full autograd graph with nothing to ever call .backward() and free
# it, so memory grows across the whole candidate pool x every surrogate until
# the GPU OOMs. The original select_base_* functions above are untouched and
# still define the real run's behavior whenever --time_mode is not passed.
# --------------------------------------------------------------------------- #

def _is_cuda(device):
    return isinstance(device, str) and device.startswith('cuda')


class _Stopwatch:
    """Named checkpoints for both elapsed time and cumulative peak GPU memory.

    Call .start() once, then .mark(name) after each phase. .elapsed[name] is
    the wall-clock time of THAT phase alone (CUDA-synced, so async kernels are
    timed correctly). .mem_mb[name] is torch.cuda.max_memory_allocated() read
    (never reset) at that checkpoint, i.e. the cumulative peak GPU memory used
    by the call SO FAR, since whatever reset_peak_memory_stats() the caller did
    before .start(). The last mark's mem_mb is therefore the true peak for the
    whole call.
    """

    def __init__(self, device):
        self.device = device
        self._t = None
        self.elapsed = {}
        self.mem_mb = {}

    def _sync(self):
        if _is_cuda(self.device):
            torch.cuda.synchronize(self.device)

    def start(self):
        self._sync()
        self._t = time.perf_counter()
        return self

    def mark(self, name):
        self._sync()
        now = time.perf_counter()
        self.elapsed[name] = now - self._t
        self._t = now
        self.mem_mb[name] = (torch.cuda.max_memory_allocated(self.device) / (1024.0 ** 2)
                             if _is_cuda(self.device) else 0.0)
        return self


@torch.no_grad()
def timed_ours_components(nets, images_norm, labels, x_t_norm, y_adv, lam, device,
                          sw, base_dist='l2', bs=512, collect_feats=False,
                          use_jacobian_score=False, jacobian_weight=1.0,
                          jacobian_batch_size=64):
    """Timed duplicate of _ours_pointwise_score, split into a 'd' phase, an 'M'
    phase, and (if enabled) an 'A' phase, each a full pass over every surrogate,
    instead of one per-batch loop that computes d and M together. Marks 'd',
    'M', and (if use_jacobian_score) 'A' on ``sw``. Returns (cls_idx, score,
    feats) exactly like _ours_pointwise_score / _ours_score_and_feats.
    """
    cls_idx = (labels == y_adv).nonzero(as_tuple=True)[0]
    cand = images_norm[cls_idx]
    n_nets = len(nets)

    # ---- d: feature distance to the target, standardized per surrogate ------
    d_total = torch.zeros(len(cls_idx), device=device)
    feat_blocks = [] if collect_feats else None
    for net in nets:
        states = [(module, module.training) for module in net.modules()]
        net.eval()
        try:
            emb = embed_of(net)
            f_t = emb(x_t_norm.unsqueeze(0))
            ds, fs = [], []
            for i in range(0, len(cand), bs):
                b = cand[i:i + bs]
                fb = emb(b)
                if base_dist == 'cosine':
                    d = 1.0 - F.cosine_similarity(fb, f_t.expand(len(b), -1), dim=1)
                else:
                    d = ((fb - f_t) ** 2).sum(dim=1)
                ds.append(d)
                if collect_feats:
                    fs.append(F.normalize(fb.detach().flatten(1), dim=1))
            d_total += standardize(torch.cat(ds))
            if collect_feats:
                feat_blocks.append(torch.cat(fs))
        finally:
            _restore_training_states(states)
    sw.mark('d')

    # ---- M: adversarial-class logit margin, standardized per surrogate ------
    m_total = torch.zeros(len(cls_idx), device=device)
    for net in nets:
        states = [(module, module.training) for module in net.modules()]
        net.eval()
        try:
            ms = []
            for i in range(0, len(cand), bs):
                b = cand[i:i + bs]
                z = net(b)
                z_adv = z[:, y_adv].clone()
                z_o = z.clone()
                z_o[:, y_adv] = float('-inf')
                ms.append(z_adv - z_o.max(dim=1).values)
            m_total += standardize(torch.cat(ms))
        finally:
            _restore_training_states(states)
    sw.mark('M')

    # ---- A (optional): exact backbone-gradient interaction -------------------
    a_total = None
    if use_jacobian_score:
        a_total = torch.zeros(len(cls_idx), device=device)
        for surrogate_idx, net in enumerate(nets):
            interaction, _ = _backbone_gradient_interactions(
                net, cand, x_t_norm, y_adv, jacobian_batch_size)
            _log_interaction_diagnostics(surrogate_idx, interaction)
            a_total += standardize(interaction)
            del interaction
        sw.mark('A')

    score = d_total + lam * m_total
    if use_jacobian_score and jacobian_weight != 0:
        score = score - jacobian_weight * a_total
    score = score / n_nets

    feats = (torch.cat(feat_blocks, dim=1) / math.sqrt(n_nets)
             if collect_feats else None)
    return cls_idx, score, feats


@torch.no_grad()
def timed_select_base_ours(nets, images_norm, labels, x_t_norm, y_adv, N_p, lam,
                           device, base_dist='l2', bs=512, use_jacobian_score=False,
                           jacobian_weight=1.0, jacobian_batch_size=64):
    """Timed duplicate of select_base_ours. Returns (base_idx, stopwatch) with
    phases 'd', 'M', ('A' if enabled), 'sort'."""
    cls_idx = (labels == y_adv).nonzero(as_tuple=True)[0]
    if len(cls_idx) < N_p:
        raise ValueError('class %d has %d images < N_p=%d' % (y_adv, len(cls_idx), N_p))
    sw = _Stopwatch(device).start()
    cls_idx, score, _ = timed_ours_components(
        nets, images_norm, labels, x_t_norm, y_adv, lam, device, sw,
        base_dist=base_dist, bs=bs, collect_feats=False,
        use_jacobian_score=use_jacobian_score, jacobian_weight=jacobian_weight,
        jacobian_batch_size=jacobian_batch_size)
    sel = torch.topk(score, k=N_p, largest=False).indices
    base_idx = cls_idx[sel]
    sw.mark('sort')
    return base_idx, sw


@torch.no_grad()
def timed_select_base_dpp(nets, images_norm, labels, x_t_norm, y_adv, N_p, lam,
                          device, base_dist='l2', bs=512, alpha=1.0,
                          use_jacobian_score=False, jacobian_weight=1.0,
                          jacobian_batch_size=64):
    """Timed duplicate of select_base_ours_div(mode='dpp'). Returns (base_idx,
    stopwatch) with phases 'd', 'M', ('A' if enabled), 'sort' (the final
    standardize + greedy log-det loop is folded into 'sort', matching the
    original, where it happens right before the mode branch)."""
    cls_idx = (labels == y_adv).nonzero(as_tuple=True)[0]
    if len(cls_idx) < N_p:
        raise ValueError('class %d has %d images < N_p=%d' % (y_adv, len(cls_idx), N_p))
    sw = _Stopwatch(device).start()
    cls_idx, score, feats = timed_ours_components(
        nets, images_norm, labels, x_t_norm, y_adv, lam, device, sw,
        base_dist=base_dist, bs=bs, collect_feats=True,
        use_jacobian_score=use_jacobian_score, jacobian_weight=jacobian_weight,
        jacobian_batch_size=jacobian_batch_size)
    N = len(cls_idx)
    score = standardize(score)          # monotone: does not change plain ranking

    # ---- sort: greedy log-det MAP for a DPP, verbatim from
    # select_base_ours_div's 'dpp' branch (Chen et al., NeurIPS 2018) ---------
    score = score.double()
    feats = feats.double()
    q = torch.exp(-alpha * (score - score.min()))
    di2 = q * q                                   # diag(L), since K_ii = 1
    cis = torch.zeros(N_p, N, device=device, dtype=torch.float64)
    j = int(torch.argmax(di2).item())
    sel = [j]
    for k in range(1, N_p):
        Lj = q * q[j] * _sim_to(feats, j)
        ei = (Lj - cis[:k - 1, :].T @ cis[:k - 1, j]) / torch.sqrt(di2[j].clamp_min(1e-12))
        cis[k - 1, :] = ei
        di2 = (di2 - ei * ei).clamp_min(0.0)
        di2[torch.tensor(sel, device=device)] = -1.0
        j = int(torch.argmax(di2).item())
        sel.append(j)
    base_idx = cls_idx[torch.tensor(sel, device=device)]
    sw.mark('sort')
    return base_idx, sw


@torch.no_grad()
def timed_select_base_exact_alignment(nets, images_norm, labels, x_t_norm, y_adv,
                                      N_p, device, batch_size=64):
    """Timed duplicate of select_base_exact_alignment (the exact g_i^T g_t).
    Returns (base_idx, stopwatch) with phases 'gi_gt', 'sort'."""
    cls_idx = (labels == y_adv).nonzero(as_tuple=True)[0]
    if len(cls_idx) < N_p:
        raise ValueError('class %d has %d images < N_p=%d'
                         % (y_adv, len(cls_idx), N_p))
    candidates = images_norm[cls_idx]
    sw = _Stopwatch(device).start()
    alignment = torch.zeros(len(cls_idx), device=device)
    for net in nets:
        interaction, _ = _full_gradient_interactions(
            net, candidates, x_t_norm, y_adv, batch_size)
        alignment += standardize(interaction)
        del interaction
    alignment /= len(nets)
    sw.mark('gi_gt')
    selected = torch.topk(alignment, k=N_p, largest=True).indices
    base_idx = cls_idx[selected]
    sw.mark('sort')
    return base_idx, sw


@torch.no_grad()
def timed_select_base_a_minus_mr(nets, images_norm, labels, x_t_norm, y_adv, N_p,
                                 device, batch_size=64):
    """Timed duplicate of select_base_a_minus_mr: A_i + (-M_i) * R_i. Returns
    (base_idx, stopwatch) with phases 'A', 'M', 'R', 'sort'.

    The original computes M and R together (one loop per net, both from the
    same batch) and A separately. Here A, M, and R are each a full, separate
    pass over every surrogate, so each can be timed/memory-marked on its own;
    the values are identical since none of the three depends on another.
    """
    cls_idx = (labels == y_adv).nonzero(as_tuple=True)[0]
    if len(cls_idx) < N_p:
        raise ValueError('class %d has %d images < N_p=%d'
                         % (y_adv, len(cls_idx), N_p))
    candidates = images_norm[cls_idx]
    sw = _Stopwatch(device).start()

    # ---- A: backbone gradient interaction ------------------------------------
    interaction = torch.zeros(len(cls_idx), device=device)
    for net in nets:
        backbone_interaction, _ = _backbone_gradient_interactions(
            net, candidates, x_t_norm, y_adv, batch_size)
        interaction += backbone_interaction
        del backbone_interaction
    interaction = standardize(interaction / len(nets))
    sw.mark('A')

    # ---- M: adversarial-class logit margin -----------------------------------
    margin = torch.zeros(len(cls_idx), device=device)
    for net in nets:
        states = [(module, module.training) for module in net.modules()]
        net.eval()
        try:
            margins = []
            for start in range(0, len(candidates), batch_size):
                batch = candidates[start:start + batch_size]
                logits = net(batch)
                adversarial_logit = logits[:, y_adv].clone()
                other_logits = logits.clone()
                other_logits[:, y_adv] = float('-inf')
                margins.append(adversarial_logit - other_logits.max(dim=1).values)
            margin += torch.cat(margins)
        finally:
            _restore_training_states(states)
    margin = standardize(margin / len(nets))
    sw.mark('M')

    # ---- R: raw representation inner product with the target ----------------
    relevance = torch.zeros(len(cls_idx), device=device)
    for net in nets:
        states = [(module, module.training) for module in net.modules()]
        net.eval()
        try:
            embed = embed_of(net)
            target_feature = embed(x_t_norm.unsqueeze(0)).flatten(1)
            relevances = []
            for start in range(0, len(candidates), batch_size):
                batch = candidates[start:start + batch_size]
                candidate_feature = embed(batch).flatten(1)
                relevances.append((candidate_feature * target_feature).sum(dim=1))
            relevance += torch.cat(relevances)
        finally:
            _restore_training_states(states)
    relevance = standardize(relevance / len(nets))
    sw.mark('R')

    # ---- sort: A + (-M)*R, then top-N_p --------------------------------------
    score = interaction + (-margin) * relevance
    selected = torch.topk(score, k=N_p, largest=True).indices
    base_idx = cls_idx[selected]
    sw.mark('sort')
    return base_idx, sw


@torch.no_grad()
def timed_select_base_components(formula, nets, images_norm, labels, x_t_norm,
                                 y_adv, N_p, device, batch_size=64):
    """Timed duplicate of select_base_components (the six -M/R/A/... ablations).
    Returns (base_idx, stopwatch) with phases for whichever of 'A'/'M'/'R' the
    formula needs, plus 'sort'."""
    if formula not in COMPONENT_SELECTOR_LABELS:
        raise ValueError('unknown component selector %r' % formula)
    cls_idx = (labels == y_adv).nonzero(as_tuple=True)[0]
    if len(cls_idx) < N_p:
        raise ValueError('class %d has %d images < N_p=%d'
                         % (y_adv, len(cls_idx), N_p))
    candidates = images_norm[cls_idx]
    need_margin = formula in ('minus-m', 'a-minus-m', 'minus-m-times-r')
    need_relevance = formula in ('r', 'a-plus-r', 'minus-m-times-r')
    need_interaction = formula in ('a', 'a-minus-m', 'a-plus-r')

    sw = _Stopwatch(device).start()

    interaction = None
    if need_interaction:
        interaction = torch.zeros(len(cls_idx), device=device)
        for net in nets:
            backbone_interaction, _ = _backbone_gradient_interactions(
                net, candidates, x_t_norm, y_adv, batch_size)
            interaction += backbone_interaction
            del backbone_interaction
        interaction = standardize(interaction / len(nets))
        sw.mark('A')

    margin = None
    if need_margin:
        margin = torch.zeros(len(cls_idx), device=device)
        for net in nets:
            states = [(module, module.training) for module in net.modules()]
            net.eval()
            try:
                margins = []
                for start in range(0, len(candidates), batch_size):
                    batch = candidates[start:start + batch_size]
                    logits = net(batch)
                    adversarial_logit = logits[:, y_adv].clone()
                    other_logits = logits.clone()
                    other_logits[:, y_adv] = float('-inf')
                    margins.append(adversarial_logit - other_logits.max(dim=1).values)
                margin += torch.cat(margins)
            finally:
                _restore_training_states(states)
        margin = standardize(margin / len(nets))
        sw.mark('M')

    relevance = None
    if need_relevance:
        relevance = torch.zeros(len(cls_idx), device=device)
        for net in nets:
            states = [(module, module.training) for module in net.modules()]
            net.eval()
            try:
                embed = embed_of(net)
                target_feature = embed(x_t_norm.unsqueeze(0)).flatten(1)
                relevances = []
                for start in range(0, len(candidates), batch_size):
                    batch = candidates[start:start + batch_size]
                    candidate_feature = embed(batch).flatten(1)
                    relevances.append((candidate_feature * target_feature).sum(dim=1))
                relevance += torch.cat(relevances)
            finally:
                _restore_training_states(states)
        relevance = standardize(relevance / len(nets))
        sw.mark('R')

    if formula == 'minus-m':
        score = -margin
    elif formula == 'r':
        score = relevance
    elif formula == 'a':
        score = interaction
    elif formula == 'a-minus-m':
        score = interaction - margin
    elif formula == 'a-plus-r':
        score = interaction + relevance
    else:  # minus-m-times-r
        score = (-margin) * relevance
    selected = torch.topk(score, k=N_p, largest=True).indices
    base_idx = cls_idx[selected]
    sw.mark('sort')
    return base_idx, sw


def time_main(args):
    """--time_mode entry point: base-selection overhead only.

    No poison crafting, no victim training. Loads the dataset and trains/loads
    the real surrogate ensemble ONCE (surrogate_dir's cache key never depends
    on class_pair -- see surrogate_dir -- so this is correct and shared across
    every class pair below, not just an optimization).

    If --time_pairs_file is given (a JSON list of {class_pair, target_idx_file,
    target_select, note}), it loops those class pairs ONE AT A TIME -- full
    target selection + --time_repeats timed calls per target for pair 1, then
    the same for pair 2, etc. -- and pools EVERY (pair, target, repeat)
    measurement into ONE combined breakdown, not a separate average per pair.
    Without --time_pairs_file it falls back to the single pair from
    --class_pair/--target_idx_file/--target_select, unchanged from before.

    Every measurement is pooled per COMPONENT -- e.g. for a-mr, 'A', 'M', and
    'R' each get their own mean/std, not one lumped "compute" number -- plus a
    'sort' phase and a 'combined' end-to-end total. Time is each phase's own
    duration; memory is the cumulative peak GPU memory reached by the end of
    that phase, so 'sort' is the true peak for the whole call. Everything is
    logged and written to <out_dir>/TIME_<run name>/timing.json.
    """
    global _LOG_PATH
    gpus = resolve_gpus(args.gpus)
    device = ('cuda:%d' % gpus[0]) if gpus else 'cpu'
    if gpus:
        torch.cuda.set_device(gpus[0])
    set_seed(args.seed)

    if args.time_pairs_file:
        with open(args.time_pairs_file) as f:
            pair_configs = json.load(f)
        for pc in pair_configs:
            ts = pc.get('target_select')
            if isinstance(ts, float):
                pc['target_select'] = int(ts)
    else:
        pair_configs = [{'class_pair': args.class_pair,
                        'target_idx_file': args.target_idx_file,
                        'target_select': args.target_select, 'note': None}]
    if not pair_configs:
        raise ValueError('--time_pairs_file resolved to zero usable class pairs')

    pair_label = '+'.join(pc['class_pair'] for pc in pair_configs)
    name_args = argparse.Namespace(**vars(args))
    name_args.class_pair = pair_label
    run_dir = os.path.join(args.out_dir, 'TIME_' + build_run_name(name_args))
    os.makedirs(run_dir, exist_ok=True)
    _LOG_PATH = os.path.join(run_dir, 'log.txt')

    log('=== [time_mode] run start: %s on %s (no crafting, no victim training) ==='
        % (build_run_name(name_args), ('cuda %s' % gpus) if gpus else 'cpu'))
    log('args: %s' % json.dumps(vars(args), sort_keys=True))
    log('  %d class pair(s): %s' % (len(pair_configs), pair_label))

    ctx = build_context(args, device)
    channel, im_size = ctx['channel'], ctx['im_size']
    num_classes, class_names = ctx['num_classes'], ctx['class_names']
    train_imgs, train_labs = ctx['train_imgs'], ctx['train_labs']
    test_imgs, test_labs = ctx['test_imgs'], ctx['test_labs']
    dsa_param = ctx['dsa_param']
    N_total = train_imgs.shape[0]

    N_p = rho_to_m(args.budget, N_total) if args.budget else args.num_poisons
    log('N_total=%d budget=%g -> N_p=%d poisons' % (N_total, args.budget or 0, N_p))

    log('=== surrogates (%d x %s, trained on the full real set) ==='
        % (args.num_surrogates, args.model))
    surrogates = get_surrogates(args, train_imgs, train_labs, test_imgs, test_labs,
                                channel, num_classes, im_size, device, dsa_param)
    sel_nets = surrogates
    if args.sel_model and args.sel_model != args.model and args.base == 'ours':
        log('=== selection surrogates (%d x %s, cross-architecture) ==='
            % (args.num_surrogates, args.sel_model))
        sel_nets = get_sel_surrogates(args, train_imgs, train_labs, test_imgs,
                                      test_labs, channel, num_classes, im_size,
                                      device, dsa_param)
    if args.sel_K:
        sel_nets = sel_nets[:args.sel_K]

    jacobian_kwargs = {
        'use_jacobian_score': getattr(args, 'use_jacobian_score', False),
        'jacobian_weight': getattr(args, 'jacobian_weight', 1.0),
        'jacobian_batch_size': getattr(args, 'jacobian_batch_size', 64),
    }

    def run_once(tidx, y_adv):
        x_t_norm = test_imgs[tidx]
        # same per-target reseed prepare_poisons uses, so a timed run selects
        # exactly the bases a real run would
        set_seed(args.seed * 100003 + int(tidx))
        if _is_cuda(device):
            torch.cuda.reset_peak_memory_stats(device)

        if args.base == 'random':
            sw = _Stopwatch(device).start()
            rgen = torch.Generator(device='cpu').manual_seed(
                args.seed * 100003 + int(tidx))
            base_idx = select_base_random(train_labs, y_adv, N_p, device, rgen)
            sw.mark('draw')
            sw.mark('sort')     # a uniform draw has no separate ranking step
        elif getattr(args, 'sel_exact_alignment', False):
            base_idx, sw = timed_select_base_exact_alignment(
                sel_nets, train_imgs, train_labs, x_t_norm, y_adv, N_p, device,
                batch_size=args.jacobian_batch_size)
        elif getattr(args, 'sel_a_minus_mr', False):
            base_idx, sw = timed_select_base_a_minus_mr(
                sel_nets, train_imgs, train_labs, x_t_norm, y_adv, N_p, device,
                batch_size=args.jacobian_batch_size)
        elif getattr(args, 'sel_component', None):
            base_idx, sw = timed_select_base_components(
                args.sel_component, sel_nets, train_imgs, train_labs, x_t_norm,
                y_adv, N_p, device, batch_size=args.jacobian_batch_size)
        elif args.sel_dpp:
            base_idx, sw = timed_select_base_dpp(
                sel_nets, train_imgs, train_labs, x_t_norm, y_adv, N_p,
                args.lambda_margin, device, base_dist=args.base_dist,
                alpha=args.sel_alpha, **jacobian_kwargs)
        elif args.sel_filter or args.sel_mmr or args.sel_pca or args.base_topr \
                or args.sel_criterion:
            raise NotImplementedError(
                '--time_mode does not instrument --sel_filter/--sel_mmr/--sel_pca/'
                '--base_topr/--sel_criterion; sel_dpp_time.sh never sets these, so '
                'this only fires if you passed one by hand. Add a timed_* variant '
                'above (mirroring the pattern for the other selectors) if you need '
                'to time one of them.')
        else:
            base_idx, sw = timed_select_base_ours(
                sel_nets, train_imgs, train_labs, x_t_norm, y_adv, N_p,
                args.lambda_margin, device, base_dist=args.base_dist,
                **jacobian_kwargs)

        return sw, len(base_idx)

    # one untimed warm-up call (first pair's first target), so lazy CUDA
    # context init, cuDNN algorithm search, and torch.func's first-call
    # tracing overhead never land inside a measured repeat
    warm_pc = pair_configs[0]
    args.class_pair = warm_pc['class_pair']
    args.target_idx_file = warm_pc.get('target_idx_file')
    args.target_select = warm_pc['target_select']
    warm_y_adv, warm_target_class = parse_pair(args.class_pair, class_names,
                                               args.pair_order)
    warm_gen = torch.Generator(device='cpu').manual_seed(args.seed)
    warm_targets, _ = select_targets(args, surrogates, test_imgs, test_labs,
                                     warm_y_adv, warm_target_class, warm_gen)
    log('  warm-up call (untimed)...')
    run_once(warm_targets[0], warm_y_adv)

    pooled = defaultdict(lambda: {'time': [], 'mem': []})
    end_to_end_time, end_to_end_mem, all_n_bases = [], [], []

    for pc in pair_configs:
        args.class_pair = pc['class_pair']
        args.target_idx_file = pc.get('target_idx_file')
        args.target_select = pc['target_select']
        y_adv, target_class = parse_pair(args.class_pair, class_names, args.pair_order)
        log('=== class pair %s: y_adv=%d(%s) target_class=%d(%s) ==='
            % (args.class_pair, y_adv, class_names[y_adv],
               target_class, class_names[target_class]))
        if pc.get('note'):
            log('    %s' % pc['note'])
        gen = torch.Generator(device='cpu').manual_seed(args.seed)
        targets, target_scores = select_targets(args, surrogates, test_imgs, test_labs,
                                                y_adv, target_class, gen)
        log('    targets: %s' % targets)

        for tidx in targets:
            for r in range(args.time_repeats):
                sw, n_sel = run_once(tidx, y_adv)
                for name, t in sw.elapsed.items():
                    pooled[name]['time'].append(t)
                    pooled[name]['mem'].append(sw.mem_mb[name])
                e2e_t = sum(sw.elapsed.values())
                e2e_m = sw.mem_mb['sort']
                end_to_end_time.append(e2e_t)
                end_to_end_mem.append(e2e_m)
                all_n_bases.append(n_sel)
                phase_str = ' '.join('%s=%.4fs' % (k, v) for k, v in sw.elapsed.items())
                log('    pair %s target %d repeat %d/%d: %s end2end=%.4fs '
                    'peak_mem=%.1fMB (%d bases)'
                    % (pc['class_pair'], tidx, r + 1, args.time_repeats,
                       phase_str, e2e_t, e2e_m, n_sel))

    def _stats(v):
        a = np.array(v, dtype=np.float64)
        return float(a.mean()), float(a.std())

    phase_stats = {}
    for name, vals in pooled.items():
        t_mean, t_std = _stats(vals['time'])
        m_mean, m_std = _stats(vals['mem'])
        phase_stats[name] = {'time_mean_s': t_mean, 'time_std_s': t_std,
                             'mem_mean_mb': m_mean, 'mem_std_mb': m_std,
                             'n': len(vals['time'])}
    e2e_t_mean, e2e_t_std = _stats(end_to_end_time)
    e2e_m_mean, e2e_m_std = _stats(end_to_end_mem)
    total_n = len(end_to_end_time)

    log('==== TIMING SUMMARY: %s | %s / %s | %d class pair(s), %d repeat(s)/target '
        '= %d measured selections (pooled across pairs) ===='
        % (build_run_name(name_args), args.dataset, args.model, len(pair_configs),
           args.time_repeats, total_n))
    for name, s in phase_stats.items():
        log('  %-8s: time %.4fs +/- %.4fs | mem %8.1fMB +/- %6.1fMB  (n=%d)'
            % (name, s['time_mean_s'], s['time_std_s'],
               s['mem_mean_mb'], s['mem_std_mb'], s['n']))
    log('  %-8s: time %.4fs +/- %.4fs | mem %8.1fMB +/- %6.1fMB  (n=%d)'
        % ('combined', e2e_t_mean, e2e_t_std, e2e_m_mean, e2e_m_std, total_n))

    out = {
        'dataset': args.dataset, 'model': args.model, 'attack': args.attack,
        'class_pairs': [pc['class_pair'] for pc in pair_configs], 'base': args.base,
        'sel_dpp': bool(args.sel_dpp), 'sel_alpha': args.sel_alpha,
        'sel_exact_alignment': bool(getattr(args, 'sel_exact_alignment', False)),
        'sel_a_minus_mr': bool(getattr(args, 'sel_a_minus_mr', False)),
        'sel_component': getattr(args, 'sel_component', None),
        'use_jacobian_score': bool(getattr(args, 'use_jacobian_score', False)),
        'jacobian_weight': getattr(args, 'jacobian_weight', 1.0),
        'jacobian_batch_size': getattr(args, 'jacobian_batch_size', 64),
        'budget': args.budget, 'num_poisons': N_p,
        'num_surrogates': args.num_surrogates,
        'repeats_per_target': args.time_repeats,
        'total_measured_selections': total_n,
        'phase_breakdown': phase_stats,
        'combined': {'time_mean_s': e2e_t_mean, 'time_std_s': e2e_t_std,
                    'mem_mean_mb': e2e_m_mean, 'mem_std_mb': e2e_m_std},
        'per_run': {
            'phases': {name: {'time_s': vals['time'], 'mem_mb': vals['mem']}
                      for name, vals in pooled.items()},
            'end_to_end_s': end_to_end_time, 'peak_mem_mb': end_to_end_mem,
            'n_bases': all_n_bases,
        },
    }
    timing_path = os.path.join(run_dir, 'timing.json')
    with open(timing_path, 'w') as f:
        json.dump(out, f, indent=2)
    log('  wrote %s' % timing_path)


def main(args):
    global _LOG_PATH
    gpus = resolve_gpus(args.gpus)
    device = ('cuda:%d' % gpus[0]) if gpus else 'cpu'
    if gpus:
        torch.cuda.set_device(gpus[0])
    set_seed(args.seed)

    ctx = build_context(args, device)
    channel, im_size = ctx['channel'], ctx['im_size']
    num_classes, class_names = ctx['num_classes'], ctx['class_names']
    train_imgs, train_labs = ctx['train_imgs'], ctx['train_labs']
    test_imgs, test_labs = ctx['test_imgs'], ctx['test_labs']
    dsa_param = ctx['dsa_param']
    N_total = train_imgs.shape[0]

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

    lock = acquire_run_lock(run_dir)
    if lock is None:
        return
    atexit.register(release_run_lock, lock)

    log('=== run start: %s on %s ==='
        % (build_run_name(args), ('cuda %s' % gpus) if gpus else 'cpu'))
    log('args: %s' % json.dumps(vars(args), sort_keys=True))
    if getattr(args, 'use_jacobian_score', False):
        log('Jacobian score: enabled, weight=%g, batch_size=%d'
            % (getattr(args, 'jacobian_weight', 1.0),
               getattr(args, 'jacobian_batch_size', 64)))
    else:
        log('Jacobian score: disabled')
    if getattr(args, 'sel_exact_alignment', False):
        log('Exact full-parameter gi^T gt selector: enabled, batch_size=%d; '
            'per-surrogate alignment standardization enabled'
            % getattr(args, 'jacobian_batch_size', 64))
    if getattr(args, 'sel_a_minus_mr', False):
        log('A - MR selector: standardized A + (-M)*R, batch_size=%d; '
            'components averaged over surrogates before standardization'
            % getattr(args, 'jacobian_batch_size', 64))
    if getattr(args, 'sel_component', None):
        log('%s component selector: raw components averaged over surrogates, '
            'then each used component standardized across candidates; batch_size=%d'
            % (COMPONENT_SELECTOR_LABELS[args.sel_component],
               getattr(args, 'jacobian_batch_size', 64)))

    y_adv, target_class = parse_pair(args.class_pair, class_names, args.pair_order)
    N_p = rho_to_m(args.budget, N_total) if args.budget else args.num_poisons
    log('N_total=%d budget=%g -> N_p=%d poisons, y_adv=%d(%s) target_class=%d(%s)'
        % (N_total, args.budget or 0, N_p, y_adv, class_names[y_adv],
           target_class, class_names[target_class]))

    # Any net still missing from the cache is trained here, spread over the gpus.
    # After this every worker below only ever LOADS from the cache, so no two
    # processes can race to write the same net_*.pt.
    if len(gpus) > 1 and args.parallel_pretrain:
        pretrain_pools_parallel(args, gpus, _LOG_PATH)

    log('=== surrogates (%d x %s, trained on the full real set) ==='
        % (args.num_surrogates, args.model))
    surrogates = get_surrogates(args, train_imgs, train_labs, test_imgs, test_labs,
                                channel, num_classes, im_size, device, dsa_param)
    craft_nets = surrogates[:args.craft_ensemble] if args.craft_ensemble else surrogates
    log('  crafting on %d/%d surrogates' % (len(craft_nets), len(surrogates)))

    sel_nets = surrogates
    if args.sel_model and args.sel_model != args.model and args.base == 'ours':
        log('=== selection surrogates (%d x %s, cross-architecture) ==='
            % (args.num_surrogates, args.sel_model))
        sel_nets = get_sel_surrogates(args, train_imgs, train_labs, test_imgs,
                                      test_labs, channel, num_classes, im_size,
                                      device, dsa_param)

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

    # clean-victim ASR per target, computed once here so the workers below never
    # need to load the clean victim pool at all
    clean_asrs = {}
    for tidx in targets:
        if clean_victims:
            preds = [predict_target(n, test_imgs[tidx]) for n in clean_victims]
            clean_asrs[tidx] = 100.0 * sum(p == y_adv for p in preds) / len(clean_victims)
        else:
            clean_asrs[tidx] = float('nan')

    # ---- resume bookkeeping -----------------------------------------------
    results_path = os.path.join(run_dir, 'results.csv')
    if args.no_resume:
        for p in _shard_paths(run_dir):
            os.remove(p)
        if os.path.exists(results_path):
            os.remove(results_path)
    merged = merge_result_shards(run_dir, results_path)
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

    pending = [t for t in targets
               if any((t, vi) not in completed for vi in range(args.num_victims))]
    tally = np.zeros(num_classes, dtype=np.int64)
    worker_errors = []

    # ---- run the targets ---------------------------------------------------
    n_trials = sum(1 for t in pending for vi in range(args.num_victims)
                   if (t, vi) not in completed)
    if len(gpus) > 1 and args.parallel_targets and n_trials > 1:
        n = min(len(gpus), n_trials)
        log('=== %d target(s), %d trials over %d gpus %s: one target per gpu while '
            'there are targets left to start, then idle gpus help with the victims '
            'of the targets still running ===' % (len(pending), n_trials, n, gpus[:n]))
        # drop the parent's copy of the dataset and the model pools first: every
        # worker builds its own on its own device
        del ctx, surrogates, craft_nets, clean_victims, rank_nets
        del train_imgs, train_labs, test_imgs, test_labs
        gc.collect()
        torch.cuda.empty_cache()
        t0 = time.time()
        tally, worker_errors = run_targets_parallel(
            args, gpus[:n], pending, run_dir, _LOG_PATH, y_adv, N_p,
            target_scores, clean_asrs, completed, num_classes)
        for rank, err in worker_errors:
            log('  !! %s:\n%s' % ('worker %d failed' % rank if rank >= 0
                                  else 'incomplete', err))
        log('=== all workers finished in %.0f s ===' % (time.time() - t0))
    else:
        legacy = load_legacy_cache(run_dir, args.recompute_deltas)
        rf, writer = open_shard(run_dir, 0)

        def emit(row):
            writer.writerow(row)
            rf.flush()
            touch_run_lock(lock)

        try:
            for i, tidx in enumerate(pending):
                tally += run_one_target(args, ctx, sel_nets, craft_nets, tidx, y_adv,
                                        N_p, target_scores.get(tidx, ''),
                                        clean_asrs.get(tidx, float('nan')), run_dir,
                                        completed, legacy, emit, i + 1, len(pending))
        finally:
            rf.close()

    merge_result_shards(run_dir, results_path)

    # ---- summaries ---------------------------------------------------------
    per_target = defaultdict(list)
    all_cta = []
    # with open(results_path, newline='') as f:
    #     for row in csv.DictReader(f):
    #         per_target[int(row['target_idx'])].append(int(row['success']))
            # all_cta.append(float(row['clean_test_acc']))
    with open(results_path, newline='') as f:
        for row in csv.DictReader(f):
            if row.get('target_idx') and row.get('success'):
                per_target[int(row['target_idx'])].append(int(row['success']))
                all_cta.append(float(row['clean_test_acc']))
    per_target_asr = [float(np.mean(v)) for v in per_target.values()]

    stats = {
        'model': args.model, 'attack': args.attack, 'base': args.base,
        'class_pair': args.class_pair, 'pair_order': args.pair_order,
        'seed': args.seed, 'budget': args.budget, 'num_poisons': N_p,
        'epsilon': args.epsilon, 'fc_mode': args.fc_mode,
        'lambda_margin': args.lambda_margin, 'base_dist': args.base_dist,
        'use_jacobian_score': getattr(args, 'use_jacobian_score', False),
        'jacobian_weight': getattr(args, 'jacobian_weight', 1.0),
        'jacobian_batch_size': getattr(args, 'jacobian_batch_size', 64),
        'jacobian_backend': (_jacobian_backend_metadata()
                             if getattr(args, 'use_jacobian_score', False) else None),
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

    if worker_errors:
        raise RuntimeError('%d gpu worker(s) failed; the summary above only covers '
                           'the trials that did finish. Re-run to resume.'
                           % len(worker_errors))


def parse_args(argv=None):
    p = argparse.ArgumentParser(description='Clean-label poisoning: FC / gradmatch '
                                            'crafting x random / ours base selection.')
    # data + model
    p.add_argument('--dataset', type=str, default='CIFAR10')
    p.add_argument('--data_path', type=str, default='./data')
    p.add_argument('--model', type=str, default='ConvNetBN', choices=SUPPORTED_MODELS)
    p.add_argument('--base_topr', type=int, default=None,
                   help='concentrate the poison budget into r feature-space '
                        'neighbourhoods: r best-scoring seeds, then their nearest '
                        'neighbours until N_p DISTINCT bases are chosen. r >= N_p is '
                        'plain greedy. Only affects --base ours.')
    p.add_argument('--sel_K', type=int, default=None,
                   help='how many surrogates the SELECTOR averages over (K in the '
                        'ensemble-size ablation). Defaults to --num_surrogates; '
                        'crafting still uses --craft_ensemble either way')
    p.add_argument('--sel_model', type=str, default=None, choices=SUPPORTED_MODELS,
                   help='architecture whose surrogates pick the bases (S in the '
                        'cross-architecture table). Defaults to --model; crafting '
                        'and victim training always use --model')
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--cache_dir', type=str, default='./cache')
    p.add_argument('--out_dir', type=str, default='./ours_result')
    p.add_argument('--dsa_strategy', type=str,
                   default='color_crop_cutout_flip_scale_rotate')

    # multi-gpu
    p.add_argument('--gpus', type=str, default='all',
                   help="cuda devices to use: 'all' (default), 'none' for cpu, or a "
                        "list like '0,2,3'. With more than one device the targets run "
                        "in parallel, one target per gpu; whenever a gpu finishes a "
                        "target it takes the next one off the shared queue.")
    p.add_argument('--no_parallel_targets', dest='parallel_targets',
                   action='store_false', default=True,
                   help='run the targets one after another on the first gpu instead')
    p.add_argument('--steal_timeout', type=float, default=7200.0,
                   help='seconds an idle gpu waits for another gpu to finish crafting '
                        'before it gives up and exits, once there are no unstarted '
                        'targets left for it to take')
    p.add_argument('--no_parallel_pretrain', dest='parallel_pretrain',
                   action='store_false', default=True,
                   help='train the missing cached surrogates / clean victims one at a '
                        'time instead of spreading them over the gpus')

    # attack
    p.add_argument('--attack', type=str, default='fc',
                   choices=['fc', 'gradmatch', 'sapa'])
    p.add_argument('--base', type=str, default='ours', choices=['random', 'ours'])
    p.add_argument('--sel_criterion', type=str, default=None, choices=SEL_CRITERIA,
                   help='alternative base-selection rule for the selection ladder of app-base.tex. Only meaningful with --base ours; it replaces the pointwise score entirely, so --sel_dpp / --sel_mmr / --sel_filter do not apply.')
    p.add_argument('--class_pair', type=str, default='dog-bird',
                   help="'<adversarial>-<target>' class names, e.g. dog-bird. Any pair "
                        'the dataset defines is accepted; the names are validated '
                        'against the class list once the dataset is loaded.')
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

    # --- sharpness-aware crafting (--attack sapa only) -----------------------
    # These change the TARGET gradient only and are ignored for fc / gradmatch,
    # which therefore keep running exactly as they did before.
    p.add_argument('--sharp_mode', type=str, default='worst', choices=['worst', 'avg'],
                   help="worst = one SAM ascent step of radius --sharp_sigma on the "
                        "target loss (the ICLR 2024 SAPA default). avg = average the "
                        "target gradient over --sharp_samples gaussian draws.")
    p.add_argument('--sharp_sigma', type=float, default=0.05,
                   help='worst: global l2 radius of the parameter perturbation (SAM '
                        'rho; 0.05 is the paper default). avg: PER-ELEMENT gaussian '
                        'std, a completely different scale -- 0.05 there puts the net '
                        'at chance, use ~1e-3. 0 with --sharp_mode worst reproduces '
                        '--attack gradmatch exactly.')
    p.add_argument('--sharp_samples', type=int, default=20,
                   help='--sharp_mode avg only: gaussian draws averaged (SAPA uses 20)')

    # base selection
    p.add_argument('--lambda_margin', type=float, default=1.0)
    p.add_argument('--base_dist', type=str, default='l2', choices=['l2', 'cosine'])
    p.add_argument('--use_jacobian_score', action='store_true', default=False,
                   help='augment the proposed pointwise score with the exact '
                        'backbone-gradient interaction')
    p.add_argument('--jacobian_weight', type=float, default=1.0,
                   help='nonnegative beta multiplying the standardized Jacobian term')
    p.add_argument('--jacobian_batch_size', type=int, default=64,
                   help='positive candidate batch size for exact Jacobian or full '
                        'gi/gt interactions')
    p.add_argument('--sel_exact_alignment', action='store_true',
                   default=argparse.SUPPRESS,
                   help='select by exact full-parameter g_i^T g_t; each surrogate '
                        'is standardized before averaging')
    p.add_argument('--sel_a_minus_mr', action='store_true',
                   default=argparse.SUPPRESS,
                   help='select by standardized A_i + (-M_i)*R_i using the paper\'s '
                        'components, averaged over surrogates before scaling')
    p.add_argument('--sel_component', type=str, default=argparse.SUPPRESS,
                   choices=list(COMPONENT_SELECTOR_LABELS),
                   help='select by one paper-component expression: -M, R, A, A-M, '
                        'A+R, or (-M)*R. Raw components are averaged over surrogates '
                        'and each used component is standardized before combining')

    # --- diversity-aware base selection (--base ours only) --------------------
    # All three reuse the SAME per-candidate score as plain --base ours and only
    # change how the set of N_p is assembled from it. Mutually exclusive; passing
    # none of them reproduces plain --base ours exactly.
    p.add_argument('--sel_filter', action='store_true', default=False,
                   help='quality gate then farthest-point: keep the best '
                        '--sel_pool x N_p by score, then spread within that pool')
    p.add_argument('--sel_pca', action='store_true', default=False,
                   help='quality gate then farthest-point: keep the best '
                        '--sel_pool x N_p by score, then spread within that pool')
    p.add_argument('--sel_pool', type=float, default=3.0,
                   help='--sel_filter pool size as a multiple of N_p (1.0 = plain ours)')
    p.add_argument('--sel_mmr', action='store_true', default=False,
                   help='greedy on score_i + --sel_mu * max_{j in S} sim(i, j)')
    p.add_argument('--sel_mu', type=float, default=1.0,
                   help='--sel_mmr redundancy weight, in z-score units so it is '
                        'comparable to the score itself (0.0 = plain ours)')
    p.add_argument('--sel_dpp', action='store_true', default=False,
                   help='greedy log-det (DPP MAP) with quality exp(-alpha*score)')
    p.add_argument('--sel_alpha', type=float, default=1.0,
                   help='--sel_dpp quality sharpness (large = plain ours)')

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
    p.add_argument('--FORCE', action='store_true', default=False,
                   help='redo the whole run instead of picking it up from disk: '
                        'ignore results.csv, re-select the bases and re-craft every '
                        'delta, overwriting the poison cache. Implies --no_resume and '
                        '--recompute_deltas. The cached surrogates / clean victims in '
                        '--cache_dir are still reused (they are seed-determined and '
                        'shared across runs).')
    p.add_argument('--no_resume', action='store_true', default=False)
    p.add_argument('--recompute_deltas', action='store_true', default=False)
    p.add_argument('--precompute_only', action='store_true', default=False)
    p.add_argument('--precompute_part', type=str, default='both',
                   choices=['surrogate', 'victim', 'both'])
    p.add_argument('--precompute_id', type=int, default=None)

    # --- timing mode (final_update_time.py only) ------------------------------
    # Measures base-selection overhead in isolation: real dataset, real trained
    # surrogates, real targets, but no poison crafting and no victim training.
    p.add_argument('--time_mode', action='store_true', default=False,
                   help='measure base-selection compute/sort time and peak GPU '
                        'memory only; no crafting, no victim training')
    p.add_argument('--time_repeats', type=int, default=10,
                   help='--time_mode: how many timed repeats per target')
    p.add_argument('--time_pairs_file', type=str, default=None,
                   help='--time_mode: path to a JSON file listing multiple class '
                        'pairs to pool into ONE aggregate -- a list of objects '
                        'each with class_pair, target_idx_file (or null), '
                        'target_select, and an optional note. Overrides '
                        '--class_pair/--target_idx_file/--target_select.')
    args = p.parse_args(argv)

    if args.FORCE:
        args.no_resume = True
        args.recompute_deltas = True

    on = [n for n, v in [('--sel_filter', args.sel_filter), ('--sel_mmr', args.sel_mmr),
                         ('--sel_dpp', args.sel_dpp), ('--sel_pca', args.sel_pca)] if v]
    if len(on) > 1:
        p.error('%s are mutually exclusive -- pick one' % ' / '.join(on))
    if on and args.base != 'ours':
        p.error('%s only affects --base ours (got --base %s)' % (on[0], args.base))
    if not math.isfinite(args.jacobian_weight) or args.jacobian_weight < 0:
        p.error('--jacobian_weight must be a finite nonnegative value')
    if args.jacobian_batch_size <= 0:
        p.error('--jacobian_batch_size must be positive')
    if args.use_jacobian_score and args.base == 'random':
        p.error('--use_jacobian_score is not applicable to --base random')
    if args.use_jacobian_score and args.sel_criterion:
        p.error('--use_jacobian_score cannot be combined with --sel_criterion, '
                'which replaces the proposed pointwise score')
    exact_alignment = getattr(args, 'sel_exact_alignment', False)
    a_minus_mr = getattr(args, 'sel_a_minus_mr', False)
    sel_component = getattr(args, 'sel_component', None)
    if exact_alignment and args.base != 'ours':
        p.error('--sel_exact_alignment only affects --base ours')
    if exact_alignment and on:
        p.error('--sel_exact_alignment cannot be combined with %s' % on[0])
    if exact_alignment and args.sel_criterion:
        p.error('--sel_exact_alignment cannot be combined with --sel_criterion')
    if exact_alignment and args.base_topr:
        p.error('--sel_exact_alignment cannot be combined with --base_topr')
    if exact_alignment and args.use_jacobian_score:
        p.error('--sel_exact_alignment already uses exact gi/gt and cannot be '
                'combined with --use_jacobian_score')
    if a_minus_mr and args.base != 'ours':
        p.error('--sel_a_minus_mr only affects --base ours')
    if a_minus_mr and on:
        p.error('--sel_a_minus_mr cannot be combined with %s' % on[0])
    if a_minus_mr and args.sel_criterion:
        p.error('--sel_a_minus_mr cannot be combined with --sel_criterion')
    if a_minus_mr and args.base_topr:
        p.error('--sel_a_minus_mr cannot be combined with --base_topr')
    if a_minus_mr and args.use_jacobian_score:
        p.error('--sel_a_minus_mr already contains A and cannot be combined '
                'with --use_jacobian_score')
    if exact_alignment and a_minus_mr:
        p.error('--sel_exact_alignment and --sel_a_minus_mr are mutually exclusive')
    if sel_component and args.base != 'ours':
        p.error('--sel_component only affects --base ours')
    if sel_component and on:
        p.error('--sel_component cannot be combined with %s' % on[0])
    if sel_component and args.sel_criterion:
        p.error('--sel_component cannot be combined with --sel_criterion')
    if sel_component and args.base_topr:
        p.error('--sel_component cannot be combined with --base_topr')
    if sel_component and args.use_jacobian_score:
        p.error('--sel_component cannot be combined with --use_jacobian_score')
    if sel_component and (exact_alignment or a_minus_mr):
        p.error('--sel_component, --sel_exact_alignment, and --sel_a_minus_mr '
                'are mutually exclusive')
    # --sel_model with --base random is allowed but does nothing to the result:
    # select_base_random draws from the class pool with a per-target rng and
    # never touches a net, so such a run reproduces the S = A one exactly. It
    # gets its own run dir (see build_run_name) so it cannot overwrite it.
    args.sel_mode = ({'--sel_filter': 'filter', '--sel_mmr': 'mmr',
                      '--sel_dpp': 'dpp', '--sel_pca': 'pca'}[on[0]] if on else None)
    if args.time_repeats <= 0:
        p.error('--time_repeats must be positive')
    return args


if __name__ == '__main__':
    _args = parse_args()
    if _args.time_mode:
        time_main(_args)
    else:
        main(_args)