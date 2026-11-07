#!/bin/bash
set -Eeuo pipefail

[[ -n "${SLURM_JOB_ID:-}" ]] || {
    echo "ERROR: run this inside the interactive Slurm GPU allocation" >&2
    exit 1
}

ROOT=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd -P)
REVIEW_ROOT="$ROOT/reviewer_20261106_fir"
REVIEW_CODE="$ROOT/sbatch/reviewer_20261106/code"
POOL="$REVIEW_ROOT/reviewer_20261106_result/surrogates/ResNet20BN_f0.5_r2_554b00a4d3bc23c9"

module load StdEnv/2023 gcc/12.3 python/3.11.5 cuda/12.6 cudnn/9.10.0.56
source "${ENV_ACTIVATE:-/home/mmoslem3/ENV/bin/activate}"

export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
export REVIEW_ROOT REVIEW_CODE
export RESUME_POOL="$POOL"
export PYTHONPATH="$REVIEW_CODE/experiments/reviewer_sweep:$REVIEW_CODE${PYTHONPATH:+:$PYTHONPATH}"
export HF_HOME="$REVIEW_ROOT/reviewer_20261106_result/downloads/hf"
export TORCH_HOME="$REVIEW_ROOT/reviewer_20261106_result/downloads/torch"
export XDG_CACHE_HOME="$REVIEW_ROOT/reviewer_20261106_result/downloads/cache"
export MPLCONFIGDIR="${SLURM_TMPDIR:?}/matplotlib"

mkdir -p "$SLURM_TMPDIR/data"
rsync -a "$REVIEW_ROOT/data/cifar-10-batches-py/" "$SLURM_TMPDIR/data/cifar-10-batches-py/"
export REVIEW_DATA="$SLURM_TMPDIR/data"

# net_0.pt through net_5.pt already exist. Parts 1--3 skip net_5 and train
# only the missing net_6.pt through net_19.pt.
for part in 1 2 3; do
    before=$(find "$POOL" -maxdepth 1 -type f -name 'net_*.pt' -printf '.' | wc -c)
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting part $part/3; $before/20 checkpoints complete"
    srun --ntasks=1 --cpus-per-task=1 python \
        "$ROOT/resume_resnet50_r2_progress.py" --part "$part"
    after=$(find "$POOL" -maxdepth 1 -type f -name 'net_*.pt' -printf '.' | wc -c)
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Finished part $part/3; $after/20 checkpoints complete"
done

count=$(find "$POOL" -maxdepth 1 -type f -name 'net_*.pt' -printf '.' | wc -c)
[[ "$count" -eq 20 ]] || {
    echo "ERROR: expected 20 checkpoints, found $count in $POOL" >&2
    exit 1
}
echo "Complete: ResNet20 50% r2 has all 20 checkpoints."
