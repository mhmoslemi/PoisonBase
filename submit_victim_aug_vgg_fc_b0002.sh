#!/usr/bin/env bash
# Submit the six missing Table-3 victim-augmentation cells for
# VGG13 / BP (fc) / dog-bird / rho=0.002: RAND and GRAFT x three augs.

set -eu

ROOT=/home/mmoslem3/scratch/attack_if
JOB_DIR="$ROOT/sbatch/victim_aug_vgg_fc_b0002"
LOG_DIR="$ROOT/sbatch/logs"

set -- "$JOB_DIR"/job_*.sh
[ -e "$1" ] || {
    echo "ERROR: no victim-augmentation jobs found under $JOB_DIR" >&2
    exit 1
}
[ "$#" -eq 6 ] || {
    echo "ERROR: expected 6 victim-augmentation jobs; found $#" >&2
    exit 1
}

mkdir -p "$LOG_DIR"

for job in "$@"; do
    if [ "${DRY_RUN:-0}" = 1 ]; then
        echo "sbatch $job"
    else
        sbatch "$job"
    fi
done

echo 'Prepared six victim-only augmentation jobs (RAND/GRAFT x Crop+Flip/RandAugment/Cutout).'
