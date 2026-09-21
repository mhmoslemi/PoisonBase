#!/usr/bin/env python3
"""Pinned screenshot protocol: one new selector/attack/budget per invocation."""
import argparse
import csv
import json
import os
from pathlib import Path
import sys

SELECTORS = ('classifier', 'classifier-norm', 'target-margin',
             'similarity-target-margin', 'confidence')
SETTINGS = (('fc', '0.001'), ('fc', '0.002'),
            ('gradmatch', '0.001'), ('gradmatch', '0.002'), ('gradmatch', '0.005'),
            ('sapa', '0.001'), ('sapa', '0.002'), ('sapa', '0.005'))
TARGETS = {
    'fc': [2540, 8129, 3725, 5705, 7663, 3875, 5169, 6335],
    'gradmatch': [843, 1257, 8252, 8730, 8055, 5859, 8954, 821],
}


def cells():
    return [dict(id=f'{i:03d}', attack=attack, budget=budget, selector=selector)
            for i, (attack, budget, selector) in enumerate(
                ((attack, budget, selector) for attack, budget in SETTINGS
                 for selector in SELECTORS), 1)]


def attack_args(cell, data_dir, cache_dir, out_dir, target_file):
    """Explicit values bypass mutable defaults in sel_dpp.sh and final_update."""
    return [
        '--dataset', 'CIFAR10', '--model', 'ConvNetBN', '--seed', '42',
        '--data_path', str(data_dir), '--cache_dir', str(cache_dir),
        '--out_dir', str(out_dir), '--gpus', 'all',
        '--attack', cell['attack'], '--base', 'ours',
        '--class_pair', 'dog-bird', '--pair_order', 'poison-target',
        '--budget', cell['budget'], '--epsilon', '0.0313725',
        '--sel_component', cell['selector'], '--jacobian_batch_size', '64',
        '--base_dist', 'cosine', '--lambda_margin', '1',
        '--craft_steps', '250', '--craft_alpha', '0.0039216',
        '--restarts', '8', '--fc_restarts', '1', '--fc_mode', 'sample',
        '--craft_ensemble', '5', '--craft_aug',
        '--dsa_strategy', 'color_crop_cutout_flip_scale_rotate',
        '--sharp_mode', 'worst', '--sharp_sigma', '0.05',
        '--num_surrogates', '20', '--surrogate_epochs', '60',
        '--surrogate_lr', '0.1', '--surrogate_bs', '128',
        '--surrogate_decay', '35', '45', '--surrogate_wd', '0',
        '--num_targets', '8', '--target_idx_file', str(target_file),
        '--target_select', '50' if cell['attack'] == 'fc' else '70',
        '--keep_pinned_targets', '--rank_on_victims',
        '--num_victims', '6', '--victim_epochs', '50',
        '--victim_lr', '0.1', '--victim_bs', '125',
        '--victim_decay', '40', '--victim_wd', '0', '--clean_baseline',
    ]


def verify_results(path, targets):
    with Path(path).open(newline='') as handle:
        rows = list(csv.DictReader(handle))
    expected = {(target, victim) for target in targets for victim in range(6)}
    observed = [(int(row['target_idx']), int(row['victim_id'])) for row in rows]
    if len(rows) != 48 or set(observed) != expected:
        raise RuntimeError(f'incomplete results: expected 48 unique trials, got '
                           f'{len(rows)} rows / {len(set(observed))} unique pairs')
    if any(int(row['success']) not in (0, 1) for row in rows):
        raise RuntimeError('results contain non-binary success values')
    print('Verified 8 pinned targets x 6 victims = 48 unique evaluations.', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--id', required=True, choices=[c['id'] for c in cells()])
    parser.add_argument('--work-root', required=True, type=Path)
    args = parser.parse_args()
    cell = next(c for c in cells() if c['id'] == args.id)
    root = args.work_root.resolve()
    sys.path.insert(0, str(root))
    import final_update as fu

    targets = TARGETS['fc' if cell['attack'] == 'fc' else 'gradmatch']
    target_file = root / 'targets.json'
    target_file.write_text(json.dumps({'pairs': {'dog-bird': {'indices': targets}}},
                                     indent=2) + '\n')
    out_dir = root / 'results'
    out_dir.mkdir(parents=True, exist_ok=True)
    argv = attack_args(cell, root / 'data', root / 'cache', out_dir, target_file)
    parsed = fu.parse_args(argv)
    protocol = dict(vars(parsed))
    for key in ('data_path', 'cache_dir', 'out_dir', 'target_idx_file'):
        protocol.pop(key)
    protocol.update(cell=cell, target_indices=targets)
    protocol_file = out_dir / 'protocol.json'
    if protocol_file.exists() and json.loads(protocol_file.read_text()) != protocol:
        raise RuntimeError('saved protocol differs; use a separate result directory')
    temporary = protocol_file.with_suffix('.tmp')
    temporary.write_text(json.dumps(protocol, indent=2, sort_keys=True) + '\n')
    os.replace(temporary, protocol_file)
    print('One experiment:', json.dumps(cell), flush=True)
    fu.main(parsed)
    verify_results(out_dir / fu.build_run_name(parsed) / 'results.csv', targets)


if __name__ == '__main__':
    main()
