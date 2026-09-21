#!/usr/bin/env python3
"""Generate exactly the 40 additional ConvNet scoring experiments."""
import csv
from pathlib import Path
from run_cell import cells

HERE = Path(__file__).resolve().parent
CLUSTER_ROOT = '/home/mmoslem3/scratch/PoisonBase'


def job_text(cell):
    tag = (f"score{cell['id']}_{cell['attack']}_"
           f"b{cell['budget'].replace('.', '')}_{cell['selector']}")
    return f'''#!/bin/bash
#SBATCH --account=aip-boyuwang
#SBATCH --job-name={tag}
#SBATCH --time=03:20:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=7G
#SBATCH --gres=gpu:l40s:1
#SBATCH --signal=B:USR1@300
#SBATCH --chdir={CLUSTER_ROOT}
#SBATCH --output={CLUSTER_ROOT}/sbatch/logs/{tag}-%j.out

# One experiment: ConvNetBN / dog-bird / {cell['attack']} / rho={cell['budget']}
# New selector: {cell['selector']}; exactly 8 targets x 6 victims.
export ROOT="${{ROOT:-{CLUSTER_ROOT}}}"
export ENV_ACTIVATE="${{ENV_ACTIVATE:-/home/mmoslem3/ENV/bin/activate}}"
export SCORE_JOB_ID={cell['id']}
source "$ROOT/sbatch/score_alternatives_8x6_20260921/_job_common.sh"
'''


def filename(cell):
    return f"job_{cell['id']}_{cell['attack']}_b{cell['budget'].replace('.', '')}_{cell['selector']}.sh"


def main():
    with (HERE / 'manifest.tsv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, delimiter='\t', lineterminator='\n',
                                fieldnames=['id', 'attack', 'budget', 'selector',
                                            'num_targets', 'num_victims', 'job_file'])
        writer.writeheader()
        for cell in cells():
            path = HERE / filename(cell)
            path.write_text(job_text(cell))
            path.chmod(0o755)
            writer.writerow(dict(cell, num_targets=8, num_victims=6, job_file=path.name))
    print('Generated 40 jobs: five new selectors x eight ConvNet settings.')


if __name__ == '__main__':
    main()
