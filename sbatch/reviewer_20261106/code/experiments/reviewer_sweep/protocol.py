"""Fixed reviewer ablation; no existing result directory is reused."""
import hashlib
import json
import os
from pathlib import Path

MODELS = ('ConvNetBN', 'ResNet20BN')
BUDGETS = (0.001, 0.002, 0.005)
TARGETS = (843, 1257, 8252, 8730, 8055, 5859, 8954, 821, 1547, 4427)
SELECTORS = ('random', 'margin', 'similarity', 'basis', 'wo_pretrained', 'wo_ood')
REPEATS = (0, 1, 2)
VERSION = 'reviewer-20261106-v1'


def cells():
    rows = []
    def add(model, budget, selector, fraction=1., pool=1., defense='none'):
        rows.append(dict(id=len(rows), model=model, budget=budget,
                         selector=selector, fraction=fraction, pool=pool,
                         defense=defense))
    for model in MODELS:
        for budget in BUDGETS:
            for selector in SELECTORS:
                add(model, budget, selector)
    for model in MODELS:
        for fraction, pool in ((.5, 1.), (.2, 1.), (1., .5), (1., .2),
                               (.5, .5), (.2, .2)):
            for selector in ('random', 'margin', 'basis'):
                add(model, .002, selector, fraction, pool)
        for budget in (.001, .005):
            for selector in ('random', 'margin', 'basis'):
                add(model, budget, selector, .2, .2)
    for model in MODELS:
        for budget in BUDGETS:
            for selector in ('random', 'margin', 'basis'):
                for defense in ('asd', 'telltale'):
                    add(model, budget, selector, defense=defense)
    assert len(rows) == 120
    return rows


def attack_cell(cell):
    wanted = dict(cell, defense='none')
    return next(c for c in cells() if all(c[k] == v for k, v in wanted.items()
                                        if k != 'id'))


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f'.{os.getpid()}.tmp')
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n')
    os.replace(tmp, path)


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()[:16]


def subsets(labels, repeat):
    """Nested stratified attacker sets; defender seeds are fixed independently."""
    import numpy as np
    labels = np.asarray(labels)
    trusted_rng = np.random.default_rng(27011)
    rng = np.random.default_rng(7100 + repeat)
    trusted, orders = [], []
    for cls in range(10):
        idx = np.flatnonzero(labels == cls)
        trusted.extend(trusted_rng.permutation(idx)[:10].tolist())
        orders.append(rng.permutation(idx))
    visible = {f: np.sort(np.concatenate([o[:round(len(o)*f)] for o in orders]))
               for f in (1., .5, .2)}
    candidates = {f: np.setdiff1d(orders[5][:round(len(orders[5])*f)], trusted)
                  for f in (1., .5, .2)}
    return np.asarray(sorted(trusted)), visible, candidates


def attack_args(model, data_root):
    import final_update as fu
    return fu.parse_args([
        '--model', model, '--dataset', 'CIFAR10', '--data_path', str(data_root),
        '--seed', '42', '--attack', 'gradmatch', '--class_pair', 'dog-bird',
        '--num_surrogates', '20', '--craft_ensemble', '5',
        '--craft_steps', '250', '--restarts', '8', '--epsilon', str(8/255),
        '--craft_alpha', str(1/255), '--base_dist', 'cosine', '--lambda_margin', '1',
        '--surrogate_epochs', '60', '--surrogate_bs', '128',
        '--surrogate_decay', '35', '45', '--victim_epochs', '60',
        '--victim_bs', '128', '--victim_decay', '35', '45', '--num_victims', '3',
        '--num_targets', '10', '--gpus', '0', '--no_parallel_targets',
        '--no_parallel_pretrain'])


def source_fingerprint():
    here = Path(__file__).resolve().parent
    root = Path(os.environ.get('REVIEW_CODE', here.parents[1]))
    digest = hashlib.sha256()
    files = sorted(here.glob('*.py')) + [root/name for name in ('final_update.py','networks.py','utils.py')]
    for file in files:
        digest.update(file.name.encode()); digest.update(file.read_bytes())
    return digest.hexdigest()
