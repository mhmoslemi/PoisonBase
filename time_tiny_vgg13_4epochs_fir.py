"""Time four VGG13BN training epochs; run directly in an interactive GPU shell."""
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'sbatch/rebuttal_tiny_gm_vgg13_5x5/code'))

import torch
import worker
import replay
import final_update as fu
from victim_aug import VictimAugmenter

torch.set_num_threads(1)
torch.set_num_interop_threads(1)
assert torch.cuda.is_available(), 'Run this on the GPU node.'
data = ROOT / 'data'
if not (data / 'tinyimagenet.pt').is_file():
    data = Path('/home/mmoslem3/scratch/data')
plan = json.loads((ROOT / 'sbatch/rebuttal_tiny_gm_vgg13_5x5/plan.json').read_text())
print('GPU:', torch.cuda.get_device_name(0), flush=True)
print('Loading Tiny ImageNet...', flush=True)
_, ctx = replay.context(plan['cells'][0], str(data))
assert ctx['train_imgs'].shape == (100000, 3, 64, 64)
net = fu.build_network('VGG13BN', 3, 200, (64, 64), ctx['device'], seed=42)
augmenter = VictimAugmenter('standard', ctx)
torch.cuda.synchronize()
times = []
previous = time.monotonic()

def progress(epoch, total):
    global previous
    torch.cuda.synchronize()
    now = time.monotonic()
    times.append(now - previous)
    previous = now
    print(f'Epoch {epoch}/{total}: {times[-1]:.2f} seconds', flush=True)

print('VGG13BN: 4 epochs, batch 128, SGD lr=0.1, crop/flip.', flush=True)
fu.train_from_scratch(net, ctx['train_imgs'], ctx['train_labs'],
    4, 0.1, 128, [60, 90], ctx['device'],
    augmenter=augmenter, progress_callback=progress)
assert len(times) == 4
print(f'Total: {sum(times):.2f} seconds ({sum(times)/60:.2f} minutes)', flush=True)
print(f'Mean per epoch: {sum(times)/4:.2f} seconds', flush=True)
