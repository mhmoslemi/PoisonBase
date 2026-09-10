#!/usr/bin/env python3
"""Rerun only BP poison construction and record the failure diagnostics.

This deliberately never trains or evaluates a victim.  Target-level attack
success is read from an already completed ``results.csv`` so that the diagnostic
uses the same victim outcomes while recomputing only selection/crafting.
"""

import argparse
import csv
import glob
import json
import os

import numpy as np
import torch
import torch.nn.functional as F

import final_update as attack


METRIC_FIELDS = [
    'selection', 'budget', 'target_idx', 'num_poisons', 'asr_percent',
    'asr_trials', 'clean_base_bp_loss', 'initialization_bp_loss',
    'final_bp_loss', 'perturb_l2_mean', 'perturb_l2_std', 'perturb_l2_min',
    'perturb_l2_max', 'perturb_linf_mean', 'perturb_linf_std',
    'perturb_linf_min', 'perturb_linf_max', 'centroid_target_l2_mean',
    'centroid_target_l2_std', 'base_spread_l2_mean', 'base_spread_l2_std',
]

CURVE_FIELDS = ['selection', 'budget', 'target_idx', 'restart', 'step', 'bp_loss']


def parse_cli():
    parser = argparse.ArgumentParser(
        description='BP headroom diagnostic; poison construction only, no victims')
    parser.add_argument('--selection', choices=['random', 'basis'], required=True)
    parser.add_argument('--budget', type=float, choices=[0.005, 0.02], required=True)
    parser.add_argument('--data-path', required=True)
    parser.add_argument('--cache-dir', required=True)
    parser.add_argument('--target-file', required=True)
    parser.add_argument('--asr-run-dir', required=True)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--num-surrogates', type=int, default=20)
    parser.add_argument('--craft-ensemble', type=int, default=5)
    parser.add_argument('--craft-steps', type=int, default=250)
    parser.add_argument('--craft-alpha', type=float, default=1.0 / 255.0)
    parser.add_argument('--fc-restarts', type=int, default=1)
    parser.add_argument('--curve-every', type=int, default=25)
    parser.add_argument('--epsilon', type=float, default=8.0 / 255.0)
    parser.add_argument('--seed', type=int, default=42)
    return parser.parse_args()


def checkpoint_args(cli):
    """Build the same attack namespace used by the completed BP runs."""
    return attack.parse_args([
        '--dataset', 'CIFAR10', '--data_path', cli.data_path,
        '--cache_dir', cli.cache_dir, '--model', 'ResNet20BN',
        '--attack', 'fc', '--base', ('random' if cli.selection == 'random' else 'ours'),
        '--class_pair', 'dog-bird', '--pair_order', 'poison-target',
        '--budget', str(cli.budget), '--epsilon', str(cli.epsilon),
        '--craft_steps', str(cli.craft_steps), '--craft_alpha', str(cli.craft_alpha),
        '--fc_restarts', str(cli.fc_restarts), '--fc_mode', 'sample',
        '--craft_ensemble', str(cli.craft_ensemble),
        '--num_surrogates', str(cli.num_surrogates),
        '--surrogate_epochs', '60', '--surrogate_decay', '35', '45',
        '--seed', str(cli.seed), '--base_dist', 'cosine', '--lambda_margin', '1',
        '--num_targets', '10', '--num_victims', '6',
        '--victim_epochs', '50', '--victim_lr', '0.1', '--victim_bs', '125',
        '--victim_decay', '40', '--victim_wd', '0',
    ])


def require_surrogate_checkpoints(args):
    directory = attack.surrogate_dir(args)
    missing = [os.path.join(directory, 'net_%d.pt' % i)
               for i in range(args.num_surrogates)
               if not os.path.isfile(os.path.join(directory, 'net_%d.pt' % i))]
    if missing:
        raise SystemExit(
            'missing %d crafting/selection checkpoint(s); this diagnostic refuses '
            'to train replacements. First missing: %s' % (len(missing), missing[0]))


def target_indices(path):
    with open(path) as handle:
        blob = json.load(handle)
    values = blob['pairs']['dog-bird']['indices'] if 'pairs' in blob else blob['dog-bird']
    return [int(value) for value in values]


def load_target_asr(run_dir):
    """Read merged and interrupted shards, deduplicating target/victim trials."""
    rows = {}
    paths = [os.path.join(run_dir, 'results.csv')]
    paths.extend(sorted(glob.glob(os.path.join(run_dir, 'results_rank*.csv'))))
    for path in paths:
        if not os.path.isfile(path):
            continue
        with open(path, newline='') as handle:
            for row in csv.DictReader(handle):
                if not row.get('target_idx') or not row.get('victim_id'):
                    continue
                key = (int(row['target_idx']), int(row['victim_id']))
                rows[key] = int(float(row['success']))
    grouped = {}
    for (target_idx, _victim_id), success in rows.items():
        grouped.setdefault(target_idx, []).append(success)
    return {target_idx: (100.0 * float(np.mean(values)), len(values))
            for target_idx, values in grouped.items()}


def target_features(nets, x_t_norm):
    with torch.no_grad():
        return [attack.embed_of(net)(x_t_norm.unsqueeze(0)).detach() for net in nets]


def bp_loss(nets, f_targets, images_norm):
    loss = 0.0
    for net, f_target in zip(nets, f_targets):
        features = attack.embed_of(net)(images_norm)
        loss = loss + F.mse_loss(features, f_target.expand_as(features))
    return loss / len(nets)


def objective_value(nets, f_targets, base01, delta, norm):
    with torch.no_grad():
        images_norm = norm(torch.clamp(base01 + delta, 0.0, 1.0))
        return float(bp_loss(nets, f_targets, images_norm).item())


def craft_with_curve(nets, base01, x_t_norm, norm, epsilon, steps, alpha,
                     restarts, every):
    """The same sign-PGD BP optimizer as final_update.craft_fc, with logging."""
    attack.set_requires_grad(nets, False)
    for net in nets:
        net.eval()
    f_targets = target_features(nets, x_t_norm)
    best_delta, best_loss = None, float('inf')
    first_initial_loss = None
    curve = []

    for restart in range(max(1, restarts)):
        delta = torch.empty_like(base01).uniform_(-epsilon, epsilon)
        delta = (torch.clamp(base01 + delta, 0.0, 1.0) - base01).detach()
        initial_loss = objective_value(nets, f_targets, base01, delta, norm)
        if first_initial_loss is None:
            first_initial_loss = initial_loss
        curve.append((restart, 0, initial_loss))

        for step in range(1, steps + 1):
            delta = delta.detach().requires_grad_(True)
            images_norm = norm(torch.clamp(base01 + delta, 0.0, 1.0))
            loss = bp_loss(nets, f_targets, images_norm)
            value = float(loss.item())
            if value < best_loss:
                best_loss = value
                best_delta = delta.detach().clone()
            gradient = torch.autograd.grad(loss, delta)[0]
            with torch.no_grad():
                delta = delta - alpha * gradient.sign()
                delta = delta.clamp_(-epsilon, epsilon)
                delta = torch.clamp(base01 + delta, 0.0, 1.0) - base01
            if step % every == 0 or step == steps:
                value = objective_value(nets, f_targets, base01, delta, norm)
                curve.append((restart, step, value))

        value = objective_value(nets, f_targets, base01, delta, norm)
        if value < best_loss:
            best_loss = value
            best_delta = delta.detach().clone()

    return best_delta, first_initial_loss, best_loss, curve, f_targets


def geometry(nets, base_norm, f_targets):
    centroid_distances, spreads = [], []
    with torch.no_grad():
        for net, f_target in zip(nets, f_targets):
            features = attack.embed_of(net)(base_norm).flatten(1)
            target = f_target.flatten(1)
            centroid = features.mean(dim=0, keepdim=True)
            centroid_distances.append(float(torch.linalg.vector_norm(
                centroid - target).item()))
            spreads.append(float(torch.linalg.vector_norm(
                features - centroid, dim=1).mean().item()))
    return centroid_distances, spreads


def summary(values):
    array = np.asarray(values, dtype=float)
    return [float(array.mean()), float(array.std()), float(array.min()), float(array.max())]


def write_csv(path, fields, rows):
    with open(path, 'w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main():
    cli = parse_cli()
    if cli.curve_every <= 0:
        raise SystemExit('--curve-every must be positive')
    if not torch.cuda.is_available():
        raise SystemExit('CUDA is required for this diagnostic job')
    device = 'cuda:0'
    torch.cuda.set_device(0)

    args = checkpoint_args(cli)
    require_surrogate_checkpoints(args)
    attack.set_seed(cli.seed)
    context = attack.build_context(args, device)
    nets = attack.get_surrogates(
        args, context['train_imgs'], context['train_labs'], context['test_imgs'],
        context['test_labs'], context['channel'], context['num_classes'],
        context['im_size'], device, context['dsa_param'])
    craft_nets = nets[:cli.craft_ensemble]
    y_adv, _target_class = attack.parse_pair(
        'dog-bird', context['class_names'], 'poison-target')
    num_poisons = attack.rho_to_m(cli.budget, len(context['train_imgs']))
    targets = target_indices(cli.target_file)
    asr = load_target_asr(cli.asr_run_dir)

    os.makedirs(cli.output_dir, exist_ok=True)
    artifact_dir = os.path.join(cli.output_dir, 'artifacts')
    os.makedirs(artifact_dir, exist_ok=True)
    metric_rows, curve_rows = [], []

    for position, target_idx in enumerate(targets, 1):
        print('[%d/%d] %s budget=%g target=%d' % (
            position, len(targets), cli.selection, cli.budget, target_idx), flush=True)
        target_seed = cli.seed * 100003 + target_idx
        attack.set_seed(target_seed)
        generator = torch.Generator(device='cpu').manual_seed(target_seed)
        x_target_norm = context['test_imgs'][target_idx]
        if cli.selection == 'random':
            base_idx = attack.select_base_random(
                context['train_labs'], y_adv, num_poisons, device, generator)
        else:
            base_idx = attack.select_base_ours(
                nets, context['train_imgs'], context['train_labs'], x_target_norm,
                y_adv, num_poisons, 1.0, device, base_dist='cosine')
        base01 = context['denorm'](context['train_imgs'][base_idx]).clamp(0.0, 1.0).detach()

        # prepare_poisons resets this seed immediately before selection. Selection
        # is deterministic (RAND uses its separate CPU generator), so resetting it
        # here reproduces the attack's CUDA uniform initialization exactly.
        attack.set_seed(target_seed)
        best_delta, initial_loss, final_loss, curve, f_targets = craft_with_curve(
            craft_nets, base01, x_target_norm, context['norm'], cli.epsilon,
            cli.craft_steps, cli.craft_alpha, cli.fc_restarts, cli.curve_every)
        clean_loss = objective_value(
            craft_nets, f_targets, base01, torch.zeros_like(base01), context['norm'])

        perturbation = best_delta.flatten(1)
        l2 = torch.linalg.vector_norm(perturbation, dim=1).detach().cpu().numpy()
        linf = perturbation.abs().amax(dim=1).detach().cpu().numpy()
        centroid_distances, spreads = geometry(
            craft_nets, context['norm'](base01), f_targets)
        target_asr, target_trials = asr.get(target_idx, (float('nan'), 0))
        l2_stats, linf_stats = summary(l2), summary(linf)
        metric_rows.append(dict(zip(METRIC_FIELDS, [
            cli.selection, cli.budget, target_idx, num_poisons,
            target_asr, target_trials, clean_loss, initial_loss, final_loss,
            *l2_stats, *linf_stats,
            float(np.mean(centroid_distances)), float(np.std(centroid_distances)),
            float(np.mean(spreads)), float(np.std(spreads)),
        ])))
        for restart, step, loss in curve:
            curve_rows.append({
                'selection': cli.selection, 'budget': cli.budget,
                'target_idx': target_idx, 'restart': restart,
                'step': step, 'bp_loss': loss,
            })

        with open(os.path.join(artifact_dir, 'bases_%d.json' % target_idx), 'w') as handle:
            json.dump(base_idx.detach().cpu().tolist(), handle)
        torch.save(best_delta.detach().cpu(),
                   os.path.join(artifact_dir, 'delta_%d.pt' % target_idx))

    write_csv(os.path.join(cli.output_dir, 'target_metrics.csv'), METRIC_FIELDS, metric_rows)
    write_csv(os.path.join(cli.output_dir, 'optimization_curves.csv'), CURVE_FIELDS, curve_rows)
    metadata = {
        'model': 'ResNet20BN', 'attack': 'fc', 'class_pair': 'dog-bird',
        'selection': cli.selection, 'budget': cli.budget,
        'num_targets': len(targets), 'num_surrogates': cli.num_surrogates,
        'craft_ensemble': cli.craft_ensemble, 'craft_steps': cli.craft_steps,
        'craft_alpha': cli.craft_alpha, 'fc_restarts': cli.fc_restarts,
        'curve_every': cli.curve_every, 'epsilon': cli.epsilon, 'seed': cli.seed,
        'target_file': cli.target_file, 'asr_run_dir': cli.asr_run_dir,
        'victim_training_performed': False,
        'geometry': {
            'centroid_target_l2': 'mean across crafting checkpoints of ||mean(f(base))-f(target)||_2',
            'base_spread_l2': 'mean across crafting checkpoints of mean_i ||f(base_i)-centroid||_2',
        },
    }
    with open(os.path.join(cli.output_dir, 'metadata.json'), 'w') as handle:
        json.dump(metadata, handle, indent=2, sort_keys=True)
    print('wrote diagnostic results to %s' % cli.output_dir, flush=True)


if __name__ == '__main__':
    main()
