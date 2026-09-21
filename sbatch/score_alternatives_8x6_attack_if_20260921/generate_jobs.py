#!/usr/bin/env python3
"""Generate the same 40 experiments using attack_if paths; never submit jobs."""
import csv
from pathlib import Path

HERE = Path(__file__).resolve().parent
SHARED = HERE.parent / 'score_alternatives_8x6_20260921'
CLUSTER_ROOT = '/home/mmoslem3/scratch/attack_if'


def source_rows():
    with (SHARED / 'manifest.tsv').open(newline='') as handle:
        rows = list(csv.DictReader(handle, delimiter='\t'))
    if len(rows) != 40:
        raise ValueError('expected 40 experiments in the original manifest')
    if any((r['num_targets'], r['num_victims']) != ('8', '6') for r in rows):
        raise ValueError('original manifest must use eight targets and six victims')
    return rows


def job_text(cell):
    tag = (f"ascore{cell['id']}_{cell['attack']}_"
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
export ROOT={CLUSTER_ROOT}
export ENV_ACTIVATE=/home/mmoslem3/ENV/bin/activate
export DATA_ROOT=/home/mmoslem3/scratch/data
export SCORE_JOB_ID={cell['id']}
source "$ROOT/sbatch/score_alternatives_8x6_attack_if_20260921/_job_common.sh"
'''


def main():
    rows = source_rows()
    for row in rows:
        path = HERE / row['job_file']
        path.write_text(job_text(row))
        path.chmod(0o755)
    with (HERE / 'manifest.tsv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), delimiter='\t',
                                lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)
    print('Generated 40 attack_if jobs: same experiments, aip-boyuwang, 03:20:00.')


if __name__ == '__main__':
    main()
