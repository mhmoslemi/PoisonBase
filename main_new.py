"""
main.py

Clean-label poisoning experiment driver.

Compares two downstream perturbation-crafting objectives (Feature Collision /
"FC" and Gradient Matching / "GRAD", i.e. Witches' Brew-style) crossed with two
base-selection strategies ("random" and "ours", the target-conditioned
s_i = B_i + coef * R_i score from the paper), across three architectures
(ConvNetBN, VGG13BN, ResNet20BN) and two standard CIFAR-10 class pairs
(dog-bird, frog-airplane).

This version includes Signed Adam and Random Restarts for aggressive L_inf
optimization, while maintaining the Bullseye FC objective and target-easiness metrics.
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

from networks import ConvNet, VGG, ResNet20BN
from utils import get_dataset, TensorDataset, epoch as run_epoch


SUPPORTED_MODELS = ['ConvNetBN', 'VGG13BN', 'ResNet20BN']
SUPPORTED_ATTACKS = ['FC', 'GRAD']
SUPPORTED_BASES = ['random', 'ours']

# target-class -> poison/adversarial-class.
CLASS_PAIRS = {
    'dog-bird': ('dog', 'bird'),
    'frog-airplane': ('frog', 'airplane'),
}


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
                   help="Number of random restarts for crafting (Witches' Brew style).")

    # target / victim protocol
    p.add_argument('--num_targets', type=int, default=10)
    p.add_argument('--num_victims', type=int, default=6)
    p.add_argument('--victim_epochs', type=int, default=60)
    p.add_argument('--victim_lr', type=float, default=0.1)
    p.add_argument('--victim_bs', type=int, default=125) # Changed default to match your bash script
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

    p.add_argument('--save_victim_ckpts', action='store_true', default=False)
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
        if args.target_easiness is not None:
            p.error('--target_difficulty is the deprecated, reversed spelling of '
                    '--target_easiness; give one or the other, not both.')
        args.target_easiness = 1.0 - args.target_difficulty
        print(f'[warn] --target_difficulty {args.target_difficulty:g} is deprecated; using '
              f'--target_easiness {args.target_easiness:g}.', file=sys.stderr)
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
# Data materialization
# --------------------------------------------------------------------------- #

def normalize(x, mean, std, device):
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
        else:
            raise AttributeError('Could not find labels on training dataset object.')
        return imgs, labels
    imgs, labels = [], []
    for i in range(len(dst_train)):
        x, y = dst_train[i]
        imgs.append(x)
        labels.append(y)
    return torch.stack(imgs), torch.tensor(labels).long()


def frac_tag(args):
    return '' if args.data_frac >= 1.0 else f'_frac{args.data_frac:g}'


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
# Training loop
# --------------------------------------------------------------------------- #

def train_from_scratch(net, images_norm, labels, testloader, epochs, lr, batch_size, decay_epochs, args):
    net = net.to(args.device)
    optimizer = torch.optim.SGD(net.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=decay_epochs, gamma=0.1)
    criterion = nn.CrossEntropyLoss()

    train_ds = TensorDataset(images_norm, labels)
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)

    for ep in range(epochs):
        run_epoch('train', train_loader, net, optimizer, criterion, args, aug=not args.no_augmentation)
        scheduler.step()

    _, acc_test = run_epoch('test', testloader, net, optimizer, criterion, args, aug=False)
    return net, acc_test


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


def evaluate_test_acc(net, testloader, device):
    net.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for img, lab in testloader:
            img = img.float().to(device)
            lab = lab.long().to(device)
            pred = net(img).argmax(dim=1)
            correct += (pred == lab).sum().item()
            total += lab.shape[0]
    return correct / max(total, 1)


# --------------------------------------------------------------------------- #
# Reference model + target selection
# --------------------------------------------------------------------------- #

def reference_member_path(args, member):
    base = os.path.join(args.cache_dir, 'reference_model',
                        f'{args.dataset}_{args.class_pair}_seed{args.seed}{frac_tag(args)}')
    return base + ('.pt' if member == 0 else f'_e{member}.pt')


def get_reference_model(args, channel, num_classes, im_size, images_train_norm, labels_train,
                        testloader, member=0):
    cache_path = reference_member_path(args, member)
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)

    seed = args.seed if member == 0 else args.seed + 20000 + member
    net = build_network(args.ref_model, channel, num_classes, im_size, seed=seed)
    if os.path.exists(cache_path) and not args.recompute_reference:
        net.load_state_dict(torch.load(cache_path, map_location=args.device))
        return net.to(args.device).eval()

    net, acc_test = train_from_scratch(net, images_train_norm, labels_train, testloader,
                                       epochs=args.victim_epochs, lr=args.victim_lr,
                                       batch_size=args.victim_bs, decay_epochs=args.victim_decay,
                                       args=args)
    logging.info(f'[reference model{"" if member == 0 else f" member {member}"}] '
                 f'test acc = {acc_test:.4f}')
    torch.save(net.state_dict(), cache_path)
    return net.eval()


def get_reference_ensemble(args, channel, num_classes, im_size, images_train_norm, labels_train,
                           testloader):
    n = max(1, args.target_eligibility_ensemble)
    return [get_reference_model(args, channel, num_classes, im_size, images_train_norm,
                                 labels_train, testloader, member=j)
            for j in range(n)]


def is_eligible_target(ens_pred_idx, worst_margin, poison_class_idx, min_margin):
    return (ens_pred_idx != poison_class_idx
            and worst_margin > 0.0
            and worst_margin >= min_margin)


TARGET_METRIC_VERSION = 'ens_padv_easiness_vetomargin_v6'
DEGENERATE_EASINESS = 1e-3


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


def select_targets(args, ref_nets, veto_nets, veto_desc, images_test_norm, labels_test,
                   class_names):
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
        if not stale:
            logging.info(f'[select_targets] cached set too short; reselecting.')

    target_class_name, poison_class_name = CLASS_PAIRS[args.class_pair]
    target_class_idx = class_names.index(target_class_name)
    poison_class_idx = class_names.index(poison_class_name)

    if args.target_pool == 'pair':
        pool_mask = labels_test == target_class_idx
        pool_desc = f'"{target_class_name}" test images'
    else:
        pool_mask = labels_test != poison_class_idx
        pool_desc = f'test images whose true label is not "{poison_class_name}"'
    candidates = pool_mask.nonzero(as_tuple=True)[0].tolist()

    easiness, ens_pred, worst_margin = score_candidates(
        ref_nets, veto_nets, images_test_norm, candidates, poison_class_idx, args.device)

    min_margin = max(args.target_min_margin, 0.0)

    if spectrum:
        eligible = [i for i in candidates
                    if is_eligible_target(ens_pred[i], worst_margin[i], poison_class_idx, min_margin)]
        n_already = sum(1 for i in candidates if worst_margin[i] <= 0.0 or ens_pred[i] == poison_class_idx)
        n_thin = len(candidates) - len(eligible) - n_already
        logging.info(f'[select_targets] pool: {len(candidates)} {pool_desc}. '
                     f'Eligibility: {len(eligible)} usable; dropped {n_already} already pred as adv, '
                     f'and {n_thin} below min margin.')
        
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
                               images_train_norm, labels_train, testloader):
    # Fixed cache key to include victim_bs to avoid stale cache bugs
    cache_path = os.path.join(args.cache_dir, 'clean_baseline',
                              f'{args.dataset}_{model_name}_seed{args.seed}_bs{args.victim_bs}{frac_tag(args)}_v{v}.pt')
    sidecar_path = cache_path[:-3] + '.json'
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    baseline_seed = args.seed * 10000 + 999900 + v

    net = build_network(model_name, channel, num_classes, im_size, seed=baseline_seed)
    if os.path.exists(cache_path) and not args.no_resume:
        net.load_state_dict(torch.load(cache_path, map_location=args.device))
        net = net.to(args.device)
        acc_test = evaluate_test_acc(net, testloader, args.device)
    else:
        net, acc_test = train_from_scratch(net, images_train_norm, labels_train, testloader,
                                           epochs=args.victim_epochs, lr=args.victim_lr,
                                           batch_size=args.victim_bs, decay_epochs=args.victim_decay,
                                           args=args)
        torch.save(net.state_dict(), cache_path)

    with open(sidecar_path, 'w') as f:
        json.dump({'dataset': args.dataset, 'model': model_name, 'seed': args.seed,
                   'victim_id': v, 'test_acc': acc_test}, f)

    return net.eval(), acc_test


def get_clean_baseline(args, model_name, channel, num_classes, im_size,
                       images_train_norm, labels_train, testloader):
    nets, accs = [], []
    for v in range(args.num_victims):
        net, acc = train_or_load_one_baseline(args, model_name, v, channel, num_classes, im_size,
                                              images_train_norm, labels_train, testloader)
        nets.append(net)
        accs.append(acc)
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


def get_surrogate_ensemble(args, channel, num_classes, im_size, images_train_norm, labels_train, testloader):
    K = max(num_scoring_surrogates(args), args.craft_ensemble)
    nets = []
    for k in range(K):
        cache_path = os.path.join(
            args.cache_dir, 'surrogates',
            f'{args.dataset}_{args.model}_{args.class_pair}_seed{args.seed}{frac_tag(args)}_k{k}.pt')
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)

        net = build_network(args.model, channel, num_classes, im_size, seed=args.seed + 1000 + k)
        if os.path.exists(cache_path) and not args.recompute_surrogates:
            net.load_state_dict(torch.load(cache_path, map_location=args.device))
            net = net.to(args.device)
        else:
            net, acc_test = train_from_scratch(net, images_train_norm, labels_train, testloader,
                                               epochs=args.victim_epochs, lr=args.victim_lr,
                                               batch_size=args.victim_bs, decay_epochs=args.victim_decay,
                                               args=args)
            logging.info(f'[surrogate {k}] test acc = {acc_test:.4f}')
            torch.save(net.state_dict(), cache_path)
        net.eval()
        nets.append(net)
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
# Perturbation crafting: FC and GRAD w/ Signed Adam & Random Restarts
# --------------------------------------------------------------------------- #

def craft_poison(attack, base_imgs_raw, target_norm, surrogate_nets, poison_class_idx,
                 epsilon, craft_steps, craft_lr, restarts, mean, std, device):
    """
    Optimized crafting with Signed Adam and Random Restarts.
    Solves the L_inf optimization failure at high budgets.
    """
    m = base_imgs_raw.shape[0]
    base_imgs_raw = base_imgs_raw.to(device)
    target_norm = target_norm.to(device)
    criterion = nn.CrossEntropyLoss()
    poison_label = torch.full((m,), poison_class_idx, dtype=torch.long, device=device)

    for net in surrogate_nets:
        net.eval()

    # Precompute target features/grads once (they don't change across restarts)
    if attack == 'FC':
        with torch.no_grad():
            target_feats = [net.embed(target_norm.unsqueeze(0)).detach() for net in surrogate_nets]
    else:
        target_grads = []
        for net in surrogate_nets:
            net.zero_grad()
            out = net(target_norm.unsqueeze(0))
            loss_t = criterion(out, torch.tensor([poison_class_idx], device=device))
            params = [p for p in net.parameters() if p.requires_grad]
            g = torch.autograd.grad(loss_t, params)
            target_grads.append([gi.detach() for gi in g])

    best_loss = float('inf')
    best_delta = None

    for r in range(restarts):
        # Random initialization uniformly in [-epsilon, epsilon] to escape local minima
        delta = torch.empty_like(base_imgs_raw).uniform_(-epsilon, epsilon)
        delta = (torch.clamp(base_imgs_raw + delta, 0.0, 1.0) - base_imgs_raw).detach().requires_grad_(True)
        optimizer = torch.optim.Adam([delta], lr=craft_lr)

        final_step_loss = float('inf')
        for step in range(craft_steps):
            optimizer.zero_grad()
            x_adv_raw = torch.clamp(base_imgs_raw + delta, 0.0, 1.0)
            x_adv_norm = normalize(x_adv_raw, mean, std, device)

            total_loss = 0.0
            for k, net in enumerate(surrogate_nets):
                if attack == 'FC':
                    # Bullseye FC objective: match the average poison feature to target
                    feats = net.embed(x_adv_norm)
                    pooled = feats.mean(dim=0, keepdim=True)
                    loss_k = ((pooled - target_feats[k]) ** 2).sum()
                else:
                    # Witches' Brew GRAD objective: match gradients using cosine similarity
                    net.zero_grad()
                    out = net(x_adv_norm)
                    loss_poison = criterion(out, poison_label)
                    params = [p for p in net.parameters() if p.requires_grad]
                    g_poison = torch.autograd.grad(loss_poison, params, create_graph=True)
                    num = sum((gp * gt).sum() for gp, gt in zip(g_poison, target_grads[k]))
                    denom = (torch.sqrt(sum((gp ** 2).sum() for gp in g_poison) + 1e-12) *
                             torch.sqrt(sum((gt ** 2).sum() for gt in target_grads[k]) + 1e-12))
                    loss_k = 1.0 - num / denom
                total_loss = total_loss + loss_k

            total_loss = total_loss / len(surrogate_nets)
            grad_delta, = torch.autograd.grad(total_loss, [delta])
            
            # --- THE FIX: Signed Adam ---
            # Replaces the raw gradient with its sign to force efficient L_inf traversal.
            delta.grad = grad_delta.sign()
            optimizer.step()

            # Projection step
            with torch.no_grad():
                delta.data = torch.clamp(delta.data, -epsilon, epsilon)
                x_proj = torch.clamp(base_imgs_raw + delta.data, 0.0, 1.0)
                delta.data = x_proj - base_imgs_raw
            
            final_step_loss = total_loss.item()
        
        # Keep the best restart delta
        if final_step_loss < best_loss:
            best_loss = final_step_loss
            best_delta = delta.detach().clone()

    return best_delta.cpu()


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def main():
    args = parse_args()
    set_seed(args.seed)

    if args.baseline_only:
        logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')
        channel, im_size, num_classes, class_names, mean, std, dst_train, dst_test, testloader = \
            get_dataset(args.dataset, args.data_path)
        images_train_raw, labels_train = materialize_raw_train(dst_train)
        images_train_raw, labels_train = subsample_train(images_train_raw, labels_train,
                                                         args.data_frac, args.seed)
        images_train_norm = normalize(images_train_raw, mean, std, 'cpu')
        _, acc = train_or_load_one_baseline(args, args.model, args.baseline_victim_id,
                                            channel, num_classes, im_size,
                                            images_train_norm, labels_train, testloader)
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

    channel, im_size, num_classes, class_names, mean, std, dst_train, dst_test, testloader = \
        get_dataset(args.dataset, args.data_path)

    images_train_raw, labels_train = materialize_raw_train(dst_train)
    full_train_len = len(labels_train)
    images_train_raw, labels_train = subsample_train(images_train_raw, labels_train,
                                                     args.data_frac, args.seed)
    images_train_norm = normalize(images_train_raw, mean, std, 'cpu')
    images_test_norm, labels_test = materialize_normalized_test(dst_test)

    train_len = len(labels_train)
    args.num_poisons = round(args.budget * train_len)
    logging.info(f'[budget] budget={args.budget} -> num_poisons={args.num_poisons} (train_len={train_len})')

    target_class_name, poison_class_name = CLASS_PAIRS[args.class_pair]
    target_class_idx = class_names.index(target_class_name)
    poison_class_idx = class_names.index(poison_class_name)

    cta_baseline_mean, cta_baseline_std, baseline_nets = get_clean_baseline(
        args, args.model, channel, num_classes, im_size, images_train_norm, labels_train, testloader)
    logging.info(f'[clean baseline] {args.model} CTA = {cta_baseline_mean:.4f} +/- {cta_baseline_std:.4f}')

    surrogate_nets = get_surrogate_ensemble(args, channel, num_classes, im_size,
                                            images_train_norm, labels_train, testloader)
    scoring_nets = surrogate_nets[:num_scoring_surrogates(args)]
    crafting_nets = surrogate_nets[:args.craft_ensemble]

    ref_nets = get_reference_ensemble(args, channel, num_classes, im_size,
                                      images_train_norm, labels_train, testloader)
    veto_nets = ref_nets + surrogate_nets + baseline_nets
    veto_desc = (f'{len(ref_nets)}x{args.ref_model} ref, '
                 f'{len(surrogate_nets)}x{args.model} sur, '
                 f'{len(baseline_nets)}x{args.model} vic')
    
    target_ids, target_scores = select_targets(args, ref_nets, veto_nets, veto_desc,
                                               images_test_norm, labels_test, class_names)
    del ref_nets, veto_nets

    free_wins = audit_free_wins(baseline_nets, images_test_norm, target_ids, poison_class_idx,
                                class_names, args.device)
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

    for t_pos, target_id in enumerate(target_ids):
        target_norm = images_test_norm[target_id]
        logging.info(f'=== target {target_id} ({t_pos + 1}/{len(target_ids)}) : selecting bases + crafting poisons ===')

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
            
            # --- Updated optimization call ---
            delta = craft_poison(args.attack, base_imgs_raw, target_norm, crafting_nets,
                                 poison_class_idx, args.epsilon, args.craft_steps, args.craft_lr,
                                 args.restarts, mean, std, args.device)
                                 
            cached_deltas[target_id] = delta
            torch.save(cached_deltas, deltas_path)
            with open(bases_path, 'w') as f:
                json.dump(cached_bases, f)

        base_idx_t = torch.as_tensor(base_ids, dtype=torch.long)
        clean_norm_rows = images_train_norm[base_idx_t].clone()
        poisoned_rows_raw = torch.clamp(images_train_raw[base_idx_t] + delta, 0.0, 1.0)
        images_train_norm[base_idx_t] = normalize(poisoned_rows_raw, mean, std, 'cpu')
        del poisoned_rows_raw

        target_successes, target_ctas = [], []
        try:
            for victim_id in range(args.num_victims):
                if (target_id, victim_id) in completed:
                    continue

                victim_seed = args.seed * 10000 + target_id * 100 + victim_id
                victim_net = build_network(args.model, channel, num_classes, im_size, seed=victim_seed)
                victim_net, _ = train_from_scratch(victim_net, images_train_norm, labels_train, testloader,
                                                   epochs=args.victim_epochs, lr=args.victim_lr,
                                                   batch_size=args.victim_bs, decay_epochs=args.victim_decay,
                                                   args=args)

                victim_net.eval()
                with torch.no_grad():
                    pred = victim_net(target_norm.unsqueeze(0).to(args.device)).argmax(dim=1).item()
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
                logging.info(f'[target {target_id} | victim {victim_id + 1}/{args.num_victims}] '
                             f'{"SUCCESS" if success else "FAIL"} '
                             f'(pred={class_names[pred]}, adv={poison_class_name}) | '
                             f'CTA={cta:.4f} (drop {cta - cta_baseline_mean:+.4f})')

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