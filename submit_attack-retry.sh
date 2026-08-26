#!/bin/sh
# Retry unresolved attack jobs that previously terminated.
set -eu

ROOT="/home/mmoslem3/scratch/attack_if"
JOB="attack_003_convnet_gradmatch_dog_bird_b0_01_ours_std"
SCRIPT="$ROOT/sbatch/attack/$JOB.sh"
LOG_DIR="$ROOT/sbatch/logs"

mkdir -p "$LOG_DIR"

latest=$(
  find "$LOG_DIR" -maxdepth 1 -type f -name "$JOB-*.out" -print |
    sort |
    tail -n 1
)

if [ -n "$latest" ] && grep -q ' : ASR = ' "$latest"; then
  echo "skip: $JOB already completed in $latest"
  exit 0
fi

if command -v squeue >/dev/null 2>&1 &&
   squeue -h -n "$JOB" 2>/dev/null | grep -q .; then
  echo "skip: $JOB is already pending or running"
  exit 0
fi

echo "submit retry: $JOB (15 GB host memory)"
sbatch "$SCRIPT"
