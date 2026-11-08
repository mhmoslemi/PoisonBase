#!/bin/bash
set -Eeuo pipefail
ROOT="${1:?project root required}"
BATCH="${2:?snapshot required}"
TASK="${3:?task ID required}"
ENV_ACTIVATE="${4:?environment required}"
: "${SLURM_TMPDIR:?Run through submit_w1_resnet_bp_7x8_vulcan.sh}"
JOB_DIR="$ROOT/rebuttal_access_res_bp_7x8_result/jobs/$TASK"
mkdir -p "$JOB_DIR"
exec 8>"$JOB_DIR/worker.lock"
flock -n 8 || { echo 'Task already running.' >&2; exit 1; }
[ ! -f "$JOB_DIR/.done" ] || { echo 'Task already complete.'; exit 0; }
cd "$BATCH"
sha256sum --check --quiet code.sha256
module load StdEnv/2023 gcc/12.3 python/3.11.5 cuda/12.6 cudnn/9.10.0.56
source "$ENV_ACTIVATE"
WORK="$SLURM_TMPDIR/rebuttal_access_res_bp"
mkdir -p "$WORK/data" "$WORK/cache"
rsync -a "$ROOT/data/cifar-10-batches-py" "$WORK/data/"
export ACCESS_ROOT="$ROOT" ACCESS_BATCH="$BATCH" REVIEW_DATA="$WORK/data"
export PYTHONPATH="$BATCH/code" PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export HF_HOME="$WORK/cache/hf" TORCH_HOME="$WORK/cache/torch" XDG_CACHE_HOME="$WORK/cache"
export MPLCONFIGDIR="$WORK/matplotlib" PYTHONPYCACHEPREFIX="$WORK/pycache"
cd "$WORK"
srun --ntasks=1 --cpus-per-task=1 --unbuffered python -u "$BATCH/code/worker.py" "$TASK"
