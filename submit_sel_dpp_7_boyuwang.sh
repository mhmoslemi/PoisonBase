#!/usr/bin/env bash
# Resubmit the seven dog-bird jobs that are not already running.
# Intentionally excluded:
#   job_08_vgg13_gradmatch.sh (Slurm job 5246590)
#   job_09_vgg13_sapa.sh      (Slurm job 5246591)

set -eu

PROJECT_ROOT=/home/mmoslem3/scratch/attack_if
JOB_DIR="$PROJECT_ROOT/sbatch/sel_dpp_9"
LOG_DIR="$PROJECT_ROOT/sbatch/logs"

set -- \
    "$JOB_DIR/job_01_convnet_fc.sh" \
    "$JOB_DIR/job_02_convnet_gradmatch.sh" \
    "$JOB_DIR/job_03_convnet_sapa.sh" \
    "$JOB_DIR/job_04_resnet20_fc.sh" \
    "$JOB_DIR/job_05_resnet20_gradmatch.sh" \
    "$JOB_DIR/job_06_resnet20_sapa.sh" \
    "$JOB_DIR/job_07_vgg13_fc.sh"

for job in "$@"; do
    [ -f "$job" ] || {
        echo "ERROR: missing job file: $job" >&2
        exit 1
    }
done

mkdir -p "$LOG_DIR"

for job in "$@"; do
    if [ "${DRY_RUN:-0}" = 1 ]; then
        echo "sbatch $job"
    else
        sbatch "$job"
    fi
done

echo "Prepared 7 dog-bird jobs on aip-boyuwang; jobs 5246590 and 5246591 were excluded."
