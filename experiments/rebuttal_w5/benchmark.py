"""Allocated-node W5 timing only; results remain outside production table data."""
import argparse
import json
import os
from pathlib import Path
import resource
import time

import numpy as np
import sklearn
import torch

import run as w5
from check import check


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--model', required=True, choices=['ConvNetBN', 'ResNet20BN'])
    args = parser.parse_args()
    if not os.environ.get('SLURM_JOB_ID'):
        raise RuntimeError('Benchmark inside a Slurm allocation.')
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    w5.install_stop_handlers()
    check()
    output = w5.ROOT/'rebuttal_w5_benchmark'/args.model
    output.mkdir(parents=True, exist_ok=True)
    settings = w5.plan()
    cell = next(c for c in settings['cells'] if c['model'] == args.model
                and c['selection'] == 'basis' and c['budget'] == .005)
    target = cell['targets'][0]
    before = time.monotonic()
    ctx = w5.replay.make_context(args.model, settings)
    timings = dict(model=args.model, device=torch.cuda.get_device_name(0),
                   torch_version=torch.__version__, sklearn_version=sklearn.__version__,
                   numpy_version=np.__version__,
                   job=os.environ['SLURM_JOB_ID'], cell=cell['id'], target=target,
                   setup_seconds=time.monotonic()-before, components={})
    for defense in ('deepknn', 'activation', 'advtrain'):
        before = time.monotonic()
        # Five AT epochs suffice to estimate 50; filtered victims finish all 50.
        w5.run_target(defense, cell, target, [0], ctx, output=output,
                      stop_after=5 if defense == 'advtrain' else None)
        directory = output/defense/cell['id']/f'target_{target}'/'victim_0'
        state = torch.load(directory/'training/checkpoint.pt', map_location='cpu',
                           weights_only=False)
        rows = state['history']
        stable = [r['seconds'] for r in rows[1:]]
        if not stable:
            stable = [r['seconds'] for r in rows]
        seconds = sum(stable)/len(stable)
        timings['components'][defense] = dict(wall_seconds=time.monotonic()-before,
            measured_epochs=state['epoch'], seconds_per_epoch=seconds,
            estimated_50_epoch_seconds=50*seconds,
            peak_host_gib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024**2,
            peak_gpu_gib=torch.cuda.max_memory_allocated()/1024**3)
        if defense != 'advtrain':
            filter_summary = output/'filters'/cell['id']/f'target_{target}'/defense/'summary.json'
            timings['components'][defense]['filter'] = json.loads(filter_summary.read_text())
        w5.atomic_json(output/'timings.json', timings)
        print('TIMING', defense, json.dumps(timings['components'][defense]), flush=True)
        del state
    print('BENCHMARK COMPLETE', json.dumps(timings), flush=True)


if __name__ == '__main__':
    try:
        main()
    except w5.Paused as error:
        print(str(error), flush=True)
        raise SystemExit(75)
