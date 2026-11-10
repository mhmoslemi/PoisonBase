"""Run one target using the unchanged frozen crafting protocol and caches."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import sys


def run_target(worker, root, batch, task_id, target):
    replay = worker.replay
    plan = json.loads((batch / 'plan.json').read_text())
    protocol_hash = hashlib.sha256((batch / 'plan.json').read_bytes() +
                                  (batch / 'code.sha256').read_bytes()).hexdigest()
    task = next(t for t in plan['tasks'] if t['id'] == task_id)
    assert task['stage'] == 'craft' and target in task['targets']
    cell = next(c for c in plan['cells'] if c['id'] == task['cell'])
    results = root / plan['results']
    ready = json.loads((results / 'preflight.json').read_text())
    assert ready['protocol_hash'] == protocol_hash and ready['complete']
    config = dict(protocol_hash=protocol_hash, task=task, target=target)
    job_dir = results / 'jobs' / f'{task_id}_target_{target}'
    job_dir.mkdir(parents=True, exist_ok=True)
    with (job_dir / 'run.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        for name in ('config.json', '.done'):
            saved = job_dir / name
            if saved.exists():
                assert json.loads(saved.read_text()) == config
        replay.save(job_dir / 'config.json', config)
        _, ctx = replay.context(cell, str(root / 'data'))
        directory = replay.target_directory(results, cell, target)
        identity = dict(protocol_hash=protocol_hash, cell=cell['id'], target=target)
        count = 5 if cell['component'] == 'random' else 20
        nets = [] if (directory / 'poison.pt').exists() else replay.pool(cell, ctx, results, count=count)
        replay.check_time()
        replay.poison(cell, ctx, nets, target, directory, identity)
        replay.save(job_dir / '.done', config)
        print('TARGET COMPLETE', task_id, target, flush=True)

        # The original task is complete only when every split target is saved.
        parent = results / 'jobs' / task_id
        parent.mkdir(parents=True, exist_ok=True)
        with (parent / 'run.lock').open('a') as parent_lock:
            fcntl.flock(parent_lock, fcntl.LOCK_EX)
            if all((replay.target_directory(results, cell, t) / 'poison.pt').exists()
                   for t in task['targets']):
                for t in task['targets']:
                    saved_identity = dict(protocol_hash=protocol_hash, cell=cell['id'], target=t)
                    replay.poison(cell, ctx, [], t, replay.target_directory(results, cell, t),
                                  saved_identity, require_saved=True)
                parent_config = dict(protocol_hash=protocol_hash, task=task)
                for name in ('config.json', '.done'):
                    saved = parent / name
                    if saved.exists():
                        assert json.loads(saved.read_text()) == parent_config
                replay.save(parent / 'config.json', parent_config)
                replay.save(parent / '.done', parent_config)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', required=True, type=Path)
    parser.add_argument('--batch', required=True, type=Path)
    parser.add_argument('--task', required=True)
    parser.add_argument('--target', required=True, type=int)
    options = parser.parse_args()
    sys.path.insert(0, str(options.batch / 'code'))
    import worker
    assert os.environ.get('SLURM_JOB_ID') and worker.torch.cuda.is_available(), 'Use a Slurm GPU job.'
    worker.torch.set_num_threads(1)
    worker.torch.set_num_interop_threads(1)
    signal.signal(signal.SIGUSR1, worker.replay.request_stop)
    signal.signal(signal.SIGTERM, worker.replay.request_stop)
    try:
        run_target(worker, options.root, options.batch, options.task, options.target)
    except worker.replay.Paused as error:
        print('Crafting checkpoint saved; resume this target.', str(error), flush=True)
        raise SystemExit(75)


if __name__ == '__main__':
    main()
