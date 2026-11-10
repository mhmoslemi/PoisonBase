"""Craft the five pinned Tiny ImageNet targets using the saved VGG13BN pool."""
import argparse
import copy
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import sys


VARIANTS = {
    'M': ('M', 1.0),
    'R': ('R', 1.0),
    'basis_lam0p1': ('BASIS', 0.1),
    'basis_lam10': ('BASIS', 10.0),
    'basis_lam1': ('BASIS', 1.0),
}


def configuration(plan, source_hash, variant):
    cell = copy.deepcopy(next(c for c in plan['cells']
                              if c['id'] == 'alligator_tarantula_b0005_basis'))
    component, weight = VARIANTS[variant]
    assert len(cell['targets']) == 5 and cell['args']['budget'] == 0.0005
    cell['component'] = component
    cell['args']['lambda_margin'] = weight
    if variant == 'basis_lam1':
        return cell, source_hash
    cell['id'] = 'alligator_tarantula_b0005_' + variant
    identity = json.dumps(dict(source_protocol_hash=source_hash, cell=cell), sort_keys=True)
    return cell, hashlib.sha256(identity.encode()).hexdigest()


def run(worker, root, batch, variant):
    replay = worker.replay
    plan = json.loads((batch / 'plan.json').read_text())
    source_hash = hashlib.sha256((batch / 'plan.json').read_bytes() +
                                 (batch / 'code.sha256').read_bytes()).hexdigest()
    cell, protocol_hash = configuration(plan, source_hash, variant)
    results = root / plan['results']
    ready = json.loads((results / 'preflight.json').read_text())
    assert ready['complete'] and ready['protocol_hash'] == source_hash
    assert (root / 'data/tinyimagenet.pt').stat().st_size == plan['data_bytes']
    for index in range(20):
        model = results / 'surrogates' / f'net_{index}.pt'
        metadata = json.loads(model.with_suffix('.json').read_text())
        assert model.stat().st_size > 0 and metadata['epochs'] == 150
        assert metadata['identity'] == dict(protocol_hash=source_hash,
                                          target='surrogate', victim=index)
    directory = results / 'cells' / cell['id']
    directory.mkdir(parents=True, exist_ok=True)
    config = dict(source_protocol_hash=source_hash, protocol_hash=protocol_hash, cell=cell)
    with (directory / 'craft_interactive.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        saved = directory / 'craft_interactive.json'
        if saved.exists():
            assert json.loads(saved.read_text()) == config, 'Incompatible saved configuration'
        replay.save(saved, config)
        print(f'{variant}: 5 targets, 50 poisons each; 20 saved selection / 5 crafting surrogates.', flush=True)
        _, ctx = replay.context(cell, str(root / 'data'))
        nets = None
        for number, target in enumerate(cell['targets'], 1):
            replay.check_time()
            folder = replay.target_directory(results, cell, target)
            print(f'[{number}/5] Target {target}: starting / checking saved poison', flush=True)
            if not (folder / 'poison.pt').exists() and nets is None:
                print('Loading the 20 saved surrogates...', flush=True)
                nets = replay.pool(cell, ctx, results, count=20)
            identity = dict(protocol_hash=protocol_hash, cell=cell['id'], target=target)
            replay.poison(cell, ctx, nets or [], target, folder, identity)
            print(f'[{number}/5] Target {target}: DONE ({folder / "poison.pt"})', flush=True)
        print(f'DONE: all 5 targets for {variant}.', flush=True)


def run_victim(worker, root, batch, variant, target, victim):
    replay = worker.replay
    plan = json.loads((batch / 'plan.json').read_text())
    source_hash = hashlib.sha256((batch / 'plan.json').read_bytes() +
                                 (batch / 'code.sha256').read_bytes()).hexdigest()
    cell, protocol_hash = configuration(plan, source_hash, variant)
    assert target in cell['targets'] and victim in range(cell['args']['num_victims'])
    assert cell['args']['victim_epochs'] == 120
    results = root / plan['results']
    config = dict(source_protocol_hash=source_hash, protocol_hash=protocol_hash, cell=cell)
    directory = replay.target_directory(results, cell, target)
    assert json.loads((directory.parent / 'craft_interactive.json').read_text()) == config
    assert (directory / 'poison.pt').is_file(), 'Saved poison is required; finish crafting first.'
    with (directory / f'victim_{victim}.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        replay.check_time()
        print(f'VICTIM {variant}: target {target}, victim {victim}, 120 epochs.', flush=True)
        _, ctx = replay.context(cell, str(root / 'data'))
        task = dict(targets=[target], victim=victim)
        worker.train_victim(task, cell, ctx, results, protocol_hash)
        replay.aggregate(dict(plan, cells=[cell]), results, protocol_hash)
        print(f'DONE: {variant}, target {target}, victim {victim}.', flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', required=True, type=Path)
    parser.add_argument('--variant', required=True, choices=VARIANTS)
    parser.add_argument('--target', type=int)
    parser.add_argument('--victim', type=int)
    options = parser.parse_args()
    if (options.target is None) != (options.victim is None):
        parser.error('--target and --victim must be supplied together')
    batch = options.root / 'sbatch/rebuttal_tiny_gm_vgg13_5x5'
    sys.path.insert(0, str(batch / 'code'))
    import replay
    component_select = replay.select
    import worker
    assert os.environ.get('SLURM_JOB_ID') and worker.torch.cuda.is_available(), 'Use an interactive GPU allocation.'
    worker.torch.set_num_threads(1)
    worker.torch.set_num_interop_threads(1)
    if options.variant in ('M', 'R'):
        replay.select = component_select
    for signum in (signal.SIGUSR1, signal.SIGTERM, signal.SIGINT):
        signal.signal(signum, replay.request_stop)
    try:
        if options.victim is None:
            run(worker, options.root, batch, options.variant)
        else:
            run_victim(worker, options.root, batch, options.variant, options.target, options.victim)
    except replay.Paused:
        print('CHECKPOINT SAVED. Run the same sh file again to resume.', flush=True)
        raise SystemExit(75)


if __name__ == '__main__':
    main()
