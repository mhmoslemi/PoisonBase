"""Compute-node regression: plain GM must not require SAPA-only settings."""
import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile

if not os.environ.get('SLURM_JOB_ID'):
    raise RuntimeError('Run this check inside Slurm, never on the login node.')

import torch

batch = Path(sys.argv[1])
spec = importlib.util.spec_from_file_location('resnet_gm_worker',
    batch / 'code/experiments/rebuttal_mr/run.py')
worker = importlib.util.module_from_spec(spec)
spec.loader.exec_module(worker)


class CraftReached(Exception):
    pass


def craft(nets, *args, sharp_mode=None, sharp_sigma=0.0, sharp_samples=20, **kwargs):
    assert len(nets) == 5
    assert sharp_mode is None and sharp_sigma == 0.0 and sharp_samples == 20
    raise CraftReached()


worker.fu.craft_gradmatch = craft
worker.select = lambda *args: torch.arange(100)
images = torch.zeros(100, 3, 32, 32)
ctx = dict(device=torch.device('cpu'), train_imgs=images,
           test_imgs={5118: images[0]}, denorm=lambda x: x, norm=lambda x: x,
           y_adv=5, dsa_param=None)
for cell in json.loads((batch / 'plan.json').read_text())['cells']:
    assert cell['args']['attack'] == 'gradmatch'
    assert 'sharp_sigma' not in cell['args'] and 'sharp_samples' not in cell['args']
    with tempfile.TemporaryDirectory(dir=os.environ['SLURM_TMPDIR']) as directory:
        try:
            worker.poison(cell, ctx, [object()] * 5, 5118, Path(directory), {})
        except CraftReached:
            pass
        else:
            raise AssertionError('GM crafting was not reached')
print('PASS: both original GM configs reach crafting without SAPA-only arguments.', flush=True)
