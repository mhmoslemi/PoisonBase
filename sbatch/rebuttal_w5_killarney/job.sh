#!/bin/bash
set -Eeuo pipefail
ROOT="${1:?project root required}"
BATCH="${2:?snapshot directory required}"
TASK="${3:?task required}"
ENV_ACTIVATE="${4:?environment activation required}"
DATA_ROOT="${5:?data directory required}"
: "${SLURM_TMPDIR:?Run through the W5 submitter}"
export DEFENSE_DEADLINE=$(( $(date +%s) + 3*3600 - 300 ))
cd "$BATCH"
sha256sum --check --quiet code.sha256
module load StdEnv/2023 gcc/12.3 python/3.11.5 cuda/12.6 cudnn/9.10.0.56
source "$ENV_ACTIVATE"
WORK="$SLURM_TMPDIR/rebuttal_w5"
mkdir -p "$WORK/data" "$WORK/cache"
rsync -a "$DATA_ROOT/cifar-10-batches-py" "$WORK/data/"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
export PYTHONPYCACHEPREFIX="$WORK/pycache"
export XDG_CACHE_HOME="$WORK/cache" HF_HOME="$WORK/cache/hf" TORCH_HOME="$WORK/cache/torch"
export W5_ROOT="$ROOT" W5_PLAN="$BATCH/plan.json"
export DEFENSE_ROOT="$ROOT" DEFENSE_DATA="$WORK/data"
export PYTHONPATH="$BATCH/code"
cd "$ROOT"
srun --ntasks=1 --cpus-per-task=1 --unbuffered \
    python -u "$BATCH/code/experiments/rebuttal_w5/batch.py" --task "$TASK"
