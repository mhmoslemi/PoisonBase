#!/usr/bin/env python3
"""Eight initial jobs: subset, three shared models, targets, three methods."""
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = '/home/mmoslem3/scratch/PoisonBase'
JOBS = [('prepare', '0')] + [('surrogate', str(i)) for i in range(3)] + \
       [('targets', '0')] + [('experiment', m) for m in ('random', 'minus-m', 'basis')]


def body(phase, identifier):
    gpu = '' if phase == 'prepare' else '#SBATCH --gres=gpu:l40s:1\n'
    cpus, memory = (2, '8G') if phase == 'prepare' else (8, '64G')
    return f'''#!/bin/bash
#SBATCH --account=aip-boyuwang
#SBATCH --job-name=in100_{phase}_{identifier}
#SBATCH --time=03:20:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task={cpus}
#SBATCH --mem={memory}
{gpu}#SBATCH --signal=B:USR1@600
#SBATCH --requeue
#SBATCH --chdir={ROOT}
#SBATCH --output={ROOT}/sbatch/logs/in100_{phase}_{identifier}-%j.out

export PHASE={phase}
export PHASE_ID={identifier}
source {ROOT}/sbatch/imagenet100_gm_8x6_20260921/_job_common.sh
'''


if __name__ == '__main__':
    for phase, identifier in JOBS:
        path = HERE / f'job_{phase}_{identifier}.sh'
        path.write_text(body(phase, identifier))
        path.chmod(0o755)
    print('Generated eight ImageNet jobs with shared-model dependencies.')
