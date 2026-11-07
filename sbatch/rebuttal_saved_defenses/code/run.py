"""Replay saved GM poisons: ASD, or Telltale detection followed by retraining."""
import argparse
import fcntl
import gc
import hashlib
import json
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch

import final_update as fu
from common import (Paused, atomic_json, install_stop_handlers,
                    pause_if_requested, save_torch)
from methods import train_asd
from telltale import detect, plateau, train_sgd

WORKSPACE = Path(os.environ.get('DEFENSE_ROOT', '/scratch/mmoslem3/PoisonBase'))
BATCH = WORKSPACE/'sbatch/rebuttal_saved_defenses'
OUT = WORKSPACE/'rebuttal_saved_defenses_result'
VERSION = 'saved-poisons-asd-telltale-v1'


def make_context(model, plan):
    args = fu.parse_args(['--model', model, '--dataset', 'CIFAR10',
                          '--data_path', os.environ.get('DEFENSE_DATA', str(WORKSPACE/'data')),
                          '--gpus', '0', '--seed', '42'])
    ctx = fu.build_context(args, torch.device('cuda:0'))
    if len(ctx['train_labs']) != 50000 or len(ctx['test_labs']) != 10000:
        raise RuntimeError('Expected the original CIFAR-10 train/test sets.')
    targets = plan['targets'][model]
    if not all(int(ctx['test_labs'][t]) == 2 for t in targets):
        raise RuntimeError('Pinned targets must be CIFAR-10 test birds.')
    labels = ctx['test_labs'].cpu().numpy()
    generator = np.random.default_rng(71307)
    validation = np.concatenate([
        generator.permutation(np.setdiff1d(np.flatnonzero(labels == c), targets))[:100]
        for c in range(10)])
    ctx['validation'] = torch.tensor(validation, device=ctx['device'])
    return ctx


def build(model, ctx, seed):
    random.seed(seed)
    return fu.build_network(model, 3, 10, (32, 32), ctx['device'], seed=seed)


def saved_poison(cell, target, ctx, legacy_cache):
    """Load only. Never train surrogates, select BASIS samples, or craft deltas."""
    expected = round(50000*cell['budget'])
    candidates = [WORKSPACE/'ours_result'/cell['source_run'],
                  Path('/scratch/mmoslem3/attack_if/ours_result')/cell['source_run']]
    delta, base, provenance = None, None, {}
    for directory in candidates:
        pc = directory/'poison_cache'
        dp, bp = pc/f'delta_{target}.pt', pc/f'base_{target}.json'
        if delta is None and dp.is_file():
            delta = torch.load(dp, map_location='cpu', weights_only=False)
            provenance['delta'] = str(dp)
        if base is None and bp.is_file():
            base = json.loads(bp.read_text())
            provenance['base'] = str(bp)
        if delta is not None and base is not None:
            break
        dp, bp = directory/'deltas.pt', directory/'bases.json'
        if dp.is_file() and bp.is_file():
            if directory not in legacy_cache:
                legacy_cache.clear()  # Only the current cell's legacy bundle is needed.
                legacy_cache[directory] = (
                    torch.load(dp, map_location='cpu', weights_only=False),
                    json.loads(bp.read_text()))
            deltas, bases = legacy_cache[directory]
            if delta is None and (str(target) in deltas or target in deltas):
                delta = deltas[str(target)] if str(target) in deltas else deltas[target]
                provenance['delta'] = f'{dp} target={target}'
            if base is None and (str(target) in bases or target in bases):
                base = bases[str(target)] if str(target) in bases else bases[target]
                provenance['base'] = f'{bp} target={target}'
    if delta is None:
        raise FileNotFoundError(f'Missing SAVED delta for {cell["id"]} target {target}; '
                                'this runner never crafts poisons.')
    if base is None and cell['selection'] == 'random':
        # Exact historical final_update.prepare_poisons() random base draw.
        generator = torch.Generator(device='cpu').manual_seed(42*100003+target)
        base = fu.select_base_random(ctx['train_labs'].cpu(), 5, expected,
                                     'cpu', generator).tolist()
        provenance['base'] = f'reconstructed historical CPU generator seed={42*100003+target}'
    if base is None:
        raise FileNotFoundError(f'Missing SAVED BASIS base indices for {cell["id"]}/{target}')
    indices = torch.as_tensor(base, dtype=torch.long)
    delta = torch.as_tensor(delta, dtype=torch.float32).contiguous().cpu()
    if (len(indices) != expected or len(indices.unique()) != expected
            or tuple(delta.shape) != (expected, 3, 32, 32)
            or not torch.isfinite(delta).all()
            or float(delta.abs().max()) > 8/255+1e-6
            or not bool((ctx['train_labs'][indices.to(ctx['device'])] == 5).all())):
        raise RuntimeError(f'Invalid poison cache for {cell["id"]}/{target}')
    digest = hashlib.sha256()
    digest.update(indices.numpy().tobytes())
    digest.update(delta.numpy().tobytes())
    identity = dict(version=VERSION, cell=cell, target=target,
                    poison_sha256=digest.hexdigest(), source=provenance)
    path = OUT/'poison_records'/cell['id']/f'target_{target}.json'
    if path.exists():
        prior = json.loads(path.read_text())
        if prior['poison_sha256'] != identity['poison_sha256']:
            raise RuntimeError(f'Saved poison changed since the first defense run: {path}')
    else:
        atomic_json(path, identity)
    return indices.to(ctx['device']), delta.to(ctx['device']), identity


def trusted_samples(labels, poison_indices):
    """Official ASD input: 10 genuinely clean training seeds per class."""
    labels = labels.cpu().numpy()
    poison_indices = poison_indices.cpu().numpy()
    generator = np.random.default_rng(27011)
    trusted = np.concatenate([
        generator.choice(np.setdiff1d(np.flatnonzero(labels == c), poison_indices),
                         size=10, replace=False)
        for c in range(10)])
    return torch.as_tensor(trusted, dtype=torch.long)


def evaluate(net, ctx, target):
    net.eval()
    with torch.no_grad():
        prediction = int(net(ctx['test_imgs'][target:target+1]).argmax(1))
    return dict(success=int(prediction == 5), prediction=prediction,
                clean_accuracy=fu.test_acc(net, ctx['test_imgs'], ctx['test_labs']))


def trial_directory(defense, cell, target, victim):
    return OUT/defense/cell['id']/f'target_{target}'/f'victim_{victim}'


def check_identity(directory, identity):
    path = directory/'protocol.json'
    if path.exists():
        prior = json.loads(path.read_text())
        if prior != identity:
            raise RuntimeError(f'Checkpoint protocol mismatch: {path}')
    else:
        atomic_json(path, identity)


def defend_target(defense, cell, target, victims, ctx, legacy):
    if all((trial_directory(defense, cell, target, v)/'result.json').exists() for v in victims):
        print(f'SKIP complete: {defense} {cell["id"]} target={target} victims={victims}', flush=True)
        return
    pause_if_requested()
    indices, delta, poison_info = saved_poison(cell, target, ctx, legacy)
    trusted = trusted_samples(ctx['train_labs'], indices).to(ctx['device']) if defense == 'asd' else None
    original = ctx['train_imgs'][indices].clone()
    pixels = ctx['denorm'](original).clamp(0, 1)
    ctx['train_imgs'][indices] = ctx['norm']((pixels+delta).clamp(0, 1))
    del pixels, delta
    try:
        retained = None
        if defense == 'telltale':
            directory = OUT/'telltale'/cell['id']/f'target_{target}'/'detector'
            detector_seed = 6200000+target
            identity = dict(version=VERSION, poison_sha256=poison_info['poison_sha256'],
                            model=cell['model'], seed=detector_seed, method='telltale-paper')
            check_identity(directory, identity)
            if not (directory/'detection.npz').exists():
                net = build(cell['model'], ctx, detector_seed)
                v = ctx['validation']
                for epochs in (120, 160, 200, 240):
                    traces, validation = train_sgd(net, ctx['train_imgs'], ctx['train_labs'],
                        directory/'trajectory', epochs=epochs, bs=128, decay=(70, 90),
                        validation_data=(ctx['test_imgs'][v], ctx['test_labs'][v]),
                        record_losses=True)
                    if plateau(validation) is not None:
                        break
                del net
                gc.collect()
                torch.cuda.empty_cache()
                fu.set_seed(detector_seed)
                detect(traces, validation, ctx['device'], detector_seed, directory)
                del traces, validation
            with np.load(directory/'detection.npz') as data:
                retained = torch.as_tensor(np.flatnonzero(~data['suspicious']), device=ctx['device'])
            if len(retained) < 125:
                raise RuntimeError('Telltale retained too few training samples.')
            print(f'Telltale retained {len(retained)}/50000 images', flush=True)
        for victim in victims:
            directory = trial_directory(defense, cell, target, victim)
            if (directory/'result.json').exists():
                print(f'SKIP complete: {directory}', flush=True)
                continue
            pause_if_requested()
            seed = 42*100000+target*100+victim
            identity = dict(version=VERSION, poison_sha256=poison_info['poison_sha256'],
                            model=cell['model'], seed=seed, defense=defense,
                            epochs=120 if defense == 'asd' else 50,
                            zero_asr_shortcut=cell['zero_asr'])
            check_identity(directory, identity)
            began = time.monotonic()
            print(f'START {defense} {cell["id"]} target={target} victim={victim} seed={seed}', flush=True)
            net = build(cell['model'], ctx, seed)
            if defense == 'asd':
                train_asd(net, ctx['train_imgs'], ctx['train_labs'], trusted,
                          directory/'training', epochs=120, steps=1024)
            else:
                train_sgd(net, ctx['train_imgs'], ctx['train_labs'], directory/'training',
                          indices=retained, epochs=50, bs=125, decay=(40,))
            result = dict(evaluate(net, ctx, target), version=VERSION, defense=defense,
                          cell=cell['id'], model=cell['model'], budget=cell['budget'],
                          selection=cell['selection'], target=target, victim=victim,
                          seed=seed, zero_asr_shortcut=cell['zero_asr'],
                          poison_sha256=poison_info['poison_sha256'],
                          seconds_this_attempt=time.monotonic()-began)
            atomic_json(directory/'result.json', result)
            print('RESULT', json.dumps(result, sort_keys=True), flush=True)
            del net
            gc.collect()
            torch.cuda.empty_cache()
    finally:
        ctx['train_imgs'][indices] = original


def run_batch(index):
    plan = json.loads((BATCH/'plan.json').read_text())
    job = plan['jobs'][index]
    if job['index'] != index:
        raise RuntimeError('Invalid batch index.')
    OUT.mkdir(parents=True, exist_ok=True)
    directory = OUT/'batches'/f'{index:02d}'
    directory.mkdir(parents=True, exist_ok=True)
    with (directory/'running.lock').open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError(f'Batch {index} is already running; refusing a duplicate.')
        if (directory/'done.json').exists():
            print(f'Batch {index} is already complete.', flush=True)
            return
        print('BATCH', json.dumps(job, sort_keys=True), flush=True)
        ctx = make_context(job['model'], plan)
        cells = {cell['id']: cell for cell in plan['cells']}
        legacy = {}
        tasks = job['tasks']
        for pos, task in enumerate(tasks):
            print(f'Batch {index}: target task {pos+1}/{len(tasks)}', flush=True)
            defend_target(job['defense'], cells[task['cell']], task['target'], task['victims'], ctx, legacy)
        atomic_json(directory/'done.json', dict(version=VERSION, index=index,
                                               tasks=len(tasks), job=os.environ.get('SLURM_JOB_ID')))
        print(f'BATCH {index} COMPLETE', flush=True)


def aggregate():
    plan = json.loads((BATCH/'plan.json').read_text())
    rows = []
    for cell in plan['cells']:
        for defense in ('asd', 'telltale'):
            results = []
            for target in cell['targets']:
                for victim in cell['victims']:
                    path = trial_directory(defense, cell, target, victim)/'result.json'
                    if path.exists():
                        results.append(json.loads(path.read_text()))
            expected = len(cell['targets'])*len(cell['victims'])
            complete = len(results) == expected
            row = dict(cell=cell['id'], defense=defense, finished=len(results),
                       expected=expected, complete=complete)
            if complete:
                row.update(asr=0. if cell['zero_asr'] else 100*float(np.mean([r['success'] for r in results])),
                           cta=100*float(np.mean([r['clean_accuracy'] for r in results])),
                           zero_asr_shortcut=cell['zero_asr'])
            rows.append(row)
            print(json.dumps(row, sort_keys=True))
    atomic_json(OUT/'summary.json', rows)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch', type=int)
    parser.add_argument('--aggregate', action='store_true')
    cli = parser.parse_args()
    if cli.aggregate:
        aggregate()
    else:
        install_stop_handlers()
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        if not torch.cuda.is_available():
            raise SystemExit('A Slurm GPU allocation is required.')
        torch.cuda.set_device(0)
        print('GPU:', torch.cuda.get_device_name(0), flush=True)
        try:
            run_batch(cli.batch)
        except Paused as exc:
            print(str(exc), flush=True)
            sys.exit(75)
