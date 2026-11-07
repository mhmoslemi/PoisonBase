"""Replay existing W5 poisons; never select samples or craft perturbations."""
import argparse
import fcntl
import gc
import hashlib
import json
import os
from pathlib import Path
import sys
import time

CODE = Path(__file__).resolve().parents[2]
ROOT = Path(os.environ.get('W5_ROOT', str(CODE))).resolve()
os.environ.setdefault('DEFENSE_ROOT', str(ROOT))
sys.path.insert(0, str(CODE))
sys.path.insert(0, str(CODE/'experiments/rebuttal_saved_defenses'))

import numpy as np
import torch
import final_update as fu
from common import Paused, atomic_json, install_stop_handlers, pause_if_requested, save_torch
# The file itself is also named run.py, so load the saved-poison module explicitly.
import importlib.util
spec = importlib.util.spec_from_file_location('saved_poison_replay',
    CODE/'experiments/rebuttal_saved_defenses/run.py')
replay = importlib.util.module_from_spec(spec)
spec.loader.exec_module(replay)
from defenses import activation_clustering, deep_knn, train

OUT = ROOT/'rebuttal_w5_result'
VERSION = 'w5-v1-pgd10-knn5000-art-default'


def plan():
    path = Path(os.environ.get('W5_PLAN', str(ROOT/'sbatch/rebuttal_saved_defenses/plan.json')))
    return json.loads(path.read_text())


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def identity(path, value):
    path = Path(path)/'protocol.json'
    if path.exists():
        if json.loads(path.read_text()) != value:
            raise RuntimeError(f'Protocol mismatch: {path}')
    else:
        atomic_json(path, value)


def features(cell, target, ctx, poison_hash, output):
    """Reuse only Telltale's initial, unfiltered poisoned-data model."""
    source = (ROOT/'rebuttal_saved_defenses_result/telltale'/cell['id']/
              f'target_{target}'/'detector')
    previous = json.loads((source/'protocol.json').read_text())
    if (previous['poison_sha256'] != poison_hash or previous['model'] != cell['model']
            or previous['method'] != 'telltale-paper'
            or not (source/'detection.npz').exists()):
        raise RuntimeError('Feature source is incompatible or still being trained.')
    checkpoint = source/'trajectory/checkpoint.pt'
    frozen = dict(version=VERSION, poison_sha256=poison_hash,
                  source=str(checkpoint), source_sha256=sha256(checkpoint),
                  layer='penultimate-raw', model=cell['model'])
    identity(output, frozen)
    path = output/'features.pt'
    if path.exists():
        return torch.load(path, map_location='cpu', weights_only=True), frozen
    state = torch.load(checkpoint, map_location='cpu', weights_only=False)
    if not 120 <= state['epoch'] <= 240:
        raise RuntimeError('Feature model has not completed the declared training schedule.')
    net = replay.build(cell['model'], ctx, previous['seed'])
    net.load_state_dict(state['net'])
    net.eval()
    del state
    began = time.monotonic()
    with torch.no_grad():
        result = torch.cat([net.embed(batch).cpu()
                            for batch in ctx['train_imgs'].split(256)])
    if len(result) != 50000 or not torch.isfinite(result).all():
        raise RuntimeError('Invalid feature matrix.')
    save_torch(path, result)
    print(f'Features: {tuple(result.shape)}, {time.monotonic()-began:.2f}s', flush=True)
    del net
    gc.collect()
    torch.cuda.empty_cache()
    return result, frozen


def filter_data(defense, cell, target, ctx, poison_hash, output):
    directory = output/'filters'/cell['id']/f'target_{target}'
    directory.mkdir(parents=True, exist_ok=True)
    with (directory/'.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        embedding, source = features(cell, target, ctx, poison_hash, directory/'features')
        destination = directory/defense
        identity(destination, dict(version=VERSION, defense=defense, source=source,
                                  k=5000 if defense == 'deepknn' else None,
                                  seed=7300000+target))
        path = destination/'retained.pt'
        if path.exists():
            return torch.load(path, map_location=ctx['device'], weights_only=True)
        began = time.monotonic()
        if defense == 'deepknn':
            clean = deep_knn(embedding.to(ctx['device']), ctx['train_labs'], destination)
        elif defense == 'activation':
            clean = activation_clustering(embedding.numpy(), ctx['train_labs'].cpu().numpy(),
                                          destination, 7300000+target)
        else:
            raise ValueError(defense)
        retained = torch.as_tensor(np.flatnonzero(clean), dtype=torch.long)
        counts = torch.bincount(ctx['train_labs'][retained.to(ctx['device'])], minlength=10)
        if len(retained) < 125 or not bool((counts > 0).all()):
            raise RuntimeError('Filter removed all examples of a class or too much data.')
        save_torch(path, retained)
        atomic_json(destination/'summary.json', dict(retained=len(retained), total=50000,
            seconds=time.monotonic()-began, class_counts=counts.cpu().tolist()))
        return retained.to(ctx['device'])


def run_target(defense, cell, target, victims, ctx, output=OUT, stop_after=None):
    indices, delta, poison = replay.saved_poison(cell, target, ctx, {})
    original = ctx['train_imgs'][indices].clone()
    ctx['train_imgs'][indices] = ctx['norm']((ctx['denorm'](original).clamp(0, 1)+delta).clamp(0, 1))
    del delta
    try:
        retained = (None if defense == 'advtrain' else
                    filter_data(defense, cell, target, ctx, poison['poison_sha256'], output))
        for victim in victims:
            directory = output/defense/cell['id']/f'target_{target}'/f'victim_{victim}'
            directory.mkdir(parents=True, exist_ok=True)
            with (directory/'.lock').open('w') as lock:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                protocol = dict(version=VERSION, cell=cell['id'], defense=defense,
                                target=target, victim=victim, model=cell['model'],
                                seed=4200000+target*100+victim,
                                poison_sha256=poison['poison_sha256'],
                                zero_asr_shortcut=cell['zero_asr'])
                identity(directory, protocol)
                if (directory/'result.json').exists():
                    print(f'SKIP complete: {directory}', flush=True)
                    continue
                pause_if_requested()
                net = replay.build(cell['model'], ctx, protocol['seed'])
                history = train(net, ctx['train_imgs'], ctx['train_labs'], directory/'training',
                    protocol, indices=retained, norm=ctx['norm'], denorm=ctx['denorm'],
                    adversarial=defense == 'advtrain', stop_after=stop_after)
                if len(history) == 50:
                    result = dict(protocol, **replay.evaluate(net, ctx, target),
                                  training_seconds=sum(row['seconds'] for row in history),
                                  retained=50000 if retained is None else len(retained))
                    # Keep the measured prediction even for the agreed zero-ASR rows.
                    atomic_json(directory/'result.json', result)
                    print('RESULT', json.dumps(result), flush=True)
                del net
                gc.collect()
                torch.cuda.empty_cache()
    finally:
        ctx['train_imgs'][indices] = original


def aggregate():
    rows = []
    for cell in plan()['cells']:
        for defense in ('deepknn', 'activation', 'advtrain'):
            results = []
            for target in cell['targets']:
                for victim in cell['victims']:
                    path = OUT/defense/cell['id']/f'target_{target}'/f'victim_{victim}'/'result.json'
                    if not path.exists():
                        continue
                    result = json.loads(path.read_text())
                    if (result['version'] != VERSION or result['target'] != target
                            or result['victim'] != victim or result['cell'] != cell['id']
                            or result['defense'] != defense
                            or result['seed'] != 4200000+target*100+victim):
                        raise RuntimeError(f'Result identity mismatch: {path}')
                    record = ROOT/'rebuttal_saved_defenses_result/poison_records'/cell['id']/f'target_{target}.json'
                    if result['poison_sha256'] != json.loads(record.read_text())['poison_sha256']:
                        raise RuntimeError(f'Result poison hash mismatch: {path}')
                    results.append(result)
            expected = len(cell['targets'])*len(cell['victims'])
            row = dict(cell=cell['id'], defense=defense, completed=len(results), expected=expected)
            if len(results) == expected:
                row.update(asr=0. if cell['zero_asr'] else
                           100*sum(r['success'] for r in results)/expected,
                           cta=100*sum(r['clean_accuracy'] for r in results)/expected,
                           zero_asr_shortcut=cell['zero_asr'])
            rows.append(row)
    atomic_json(OUT/'summary.json', rows)
    print(json.dumps(rows, indent=2))


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--cell')
    parser.add_argument('--defense', choices=['deepknn', 'activation', 'advtrain'])
    parser.add_argument('--aggregate', action='store_true')
    parser.add_argument('--targets', type=int, nargs='+')
    parser.add_argument('--victims', type=int, nargs='+')
    args = parser.parse_args()
    if not os.environ.get('SLURM_JOB_ID'):
        raise RuntimeError('Run inside a Slurm allocation.')
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    install_stop_handlers()
    if args.aggregate:
        aggregate()
        return
    if args.cell is None or args.defense is None:
        parser.error('--cell and --defense are required unless using --aggregate')
    settings = plan()
    cell = next(cell for cell in settings['cells'] if cell['id'] == args.cell)
    targets, victims = args.targets or cell['targets'], args.victims or cell['victims']
    if not set(targets) <= set(cell['targets']) or not set(victims) <= set(cell['victims']):
        raise ValueError('Requested targets/victims are outside the agreed table grid.')
    ctx = replay.make_context(cell['model'], settings)
    for target in targets:
        run_target(args.defense, cell, target, victims, ctx)


if __name__ == '__main__':
    try:
        main()
    except Paused as error:
        print(str(error), flush=True)
        sys.exit(75)
