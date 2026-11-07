#!/bin/bash
set -Eeuo pipefail

ROOT="${ACCESS_ROOT:-/home/mmoslem3/scratch/PoisonBase}"
cd "$ROOT"
module load StdEnv/2023 gcc/12.3 python/3.11.5 cuda/12.6 cudnn/9.10.0.56
source "${ENV_ACTIVATE:-/home/mmoslem3/ENV/bin/activate}"

export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
export ACCESS_ROOT="$ROOT"
export HF_HOME="$ROOT/rebuttal_access_conv_5x4_result/downloads/hf"
export TORCH_HOME="$ROOT/rebuttal_access_conv_5x4_result/downloads/torch"
export XDG_CACHE_HOME="$ROOT/rebuttal_access_conv_5x4_result/downloads/cache"
export MPLCONFIGDIR="${SLURM_TMPDIR:?}/matplotlib"

mkdir -p "$SLURM_TMPDIR/data"
rsync -a "$ROOT/data/cifar-10-batches-py/" "$SLURM_TMPDIR/data/cifar-10-batches-py/"
export REVIEW_DATA="$SLURM_TMPDIR/data"

srun --ntasks=1 --cpus-per-task=1 python \
    "$ROOT/experiments/rebuttal_access_conv/runner.py" "$@"
