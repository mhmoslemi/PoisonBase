"""Run one bounded W5 bundle from the frozen Killarney plan."""
import argparse
import fcntl
import json
import os
import sys

import torch
import sklearn

import run as w5


def main():
    if not os.environ.get('SLURM_JOB_ID'):
        raise RuntimeError('Run this worker inside a Slurm allocation.')
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--task', required=True)
    args = parser.parse_args()
    settings = w5.plan()
    task = next(t for t in settings['tasks'] if t['id'] == args.task)
    directory = w5.OUT/'jobs'/task['id']
    directory.mkdir(parents=True, exist_ok=True)
    with (directory/'.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        w5.identity(directory, dict(version=w5.VERSION, task=task))
        if (directory/'.done').exists():
            print(f'SKIP complete: {task["id"]}', flush=True)
            return
        torch.set_num_threads(1)
        torch.backends.cuda.matmul.allow_tf32 = False
        w5.install_stop_handlers()
        print('Runtime', json.dumps(dict(torch=torch.__version__, numpy=w5.np.__version__,
              sklearn=sklearn.__version__, gpu=torch.cuda.get_device_name(0))), flush=True)
        cells = {cell['id']: cell for cell in settings['cells']}
        ctx = w5.replay.make_context(task['model'], settings)
        print(f'W5 {task["id"]}: {task["model"]}, {task["defense"]}, '
              f'{sum(len(unit["victims"]) for unit in task["units"])} victims; '
              f'estimate {task["estimated_seconds"]/60:.1f} min', flush=True)
        for unit in task['units']:
            cell = cells[unit['cell']]
            if (cell['model'] != task['model'] or unit['target'] not in cell['targets']
                    or not set(unit['victims']) <= set(cell['victims'])):
                raise RuntimeError('Task is outside the agreed table grid.')
            paths = [w5.OUT/task['defense']/cell['id']/f'target_{unit["target"]}'/
                     f'victim_{victim}'/'result.json' for victim in unit['victims']]
            if all(path.exists() for path in paths):
                print(f'SKIP complete: {unit}', flush=True)
                continue
            w5.run_target(task['defense'], cell, unit['target'], unit['victims'], ctx)
        if not all((w5.OUT/task['defense']/unit['cell']/f'target_{unit["target"]}'/
                    f'victim_{victim}'/'result.json').exists()
                   for unit in task['units'] for victim in unit['victims']):
            raise RuntimeError('Bundle returned with missing victim results.')
        w5.atomic_json(directory/'.done', dict(version=w5.VERSION, task=task['id']))
        print(f'DONE {task["id"]}', flush=True)


if __name__ == '__main__':
    try:
        main()
    except w5.Paused as error:
        print(str(error), flush=True)
        sys.exit(75)
