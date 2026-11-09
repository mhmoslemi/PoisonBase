"""Allocated-node regression for Rand draws and R/M selector dispatch."""
import importlib.util
import json
import os
from pathlib import Path
import sys

if not os.environ.get('SLURM_JOB_ID'):
    raise RuntimeError('Run inside Slurm, never on the login node.')

import torch

torch.set_num_threads(1)
batch = Path(sys.argv[1])
spec = importlib.util.spec_from_file_location('missing_cell_worker',
    batch / 'code/experiments/rebuttal_mr/run.py')
worker = importlib.util.module_from_spec(spec)
spec.loader.exec_module(worker)
for cell in json.loads((batch / 'plan.json').read_text())['cells']:
    args = cell['args']
    labels = torch.zeros(100000 if args['dataset'] == 'TinyImageNet' else 50000,
                         dtype=torch.long)
    labels[:1000] = 5
    target = cell['targets'][0]
    count = round(args['budget'] * len(labels))
    ctx = dict(device=torch.device('cpu'), train_labs=labels, y_adv=5,
               train_imgs=None, test_imgs={target: None})
    if cell['component'] == 'random':
        actual = worker.select(cell, ctx, [], target)
        generator = torch.Generator().manual_seed(args['seed'] * 100003 + target)
        expected = torch.arange(1000)[torch.randperm(1000, generator=generator)[:count]]
        assert torch.equal(actual, expected)
    else:
        called = []

        def select(*values, **kwargs):
            assert len(values[0]) == 20 and values[5] == count
            assert kwargs['distance_margin_coef'] == (1.0 if cell['component'] == 'R' else 0.0)
            called.append(True)
            return torch.arange(count)

        worker.fu.select_base_ours = select
        worker.select(cell, ctx, [None] * 20, target)
        assert called
print('PASS: all ten selector configs, deterministic Rand, correct R/M dispatch.', flush=True)
