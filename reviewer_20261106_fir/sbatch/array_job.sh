#!/bin/bash
#SBATCH --account=def-bghaddar
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=6G
#SBATCH --gpus-per-node=h100:1
#SBATCH --time=00:15:00
#SBATCH --output=/scratch/mmoslem3/PoisonBase/sbatch/reviewer_20261106/logs/%x-%A_%a.out
set -Eeuo pipefail
ROOT="${REVIEW_ROOT:-/scratch/mmoslem3/PoisonBase}"
cd "$ROOT"
module load StdEnv/2023 gcc/12.3 python/3.11.5 cuda/12.6 cudnn/9.10.0.56
source "${ENV_ACTIVATE:-/home/mmoslem3/ENV/bin/activate}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
export HF_HOME="$ROOT/reviewer_20261106_result/downloads/hf"
export TORCH_HOME="$ROOT/reviewer_20261106_result/downloads/torch"
export XDG_CACHE_HOME="$ROOT/reviewer_20261106_result/downloads/cache"
export MPLCONFIGDIR="$SLURM_TMPDIR/matplotlib"
export REVIEW_ROOT="$ROOT"
export REVIEW_CODE="${REVIEW_CODE:-$ROOT/sbatch/reviewer_20261106/code}"
export PYTHONPATH="$REVIEW_CODE/experiments/reviewer_sweep:$REVIEW_CODE${PYTHONPATH:+:$PYTHONPATH}"
mkdir -p "$SLURM_TMPDIR/data"
rsync -a "$ROOT/data/cifar-10-batches-py/" "$SLURM_TMPDIR/data/cifar-10-batches-py/"
export REVIEW_DATA="$SLURM_TMPDIR/data"
srun --ntasks=1 --cpus-per-task=1 python "$REVIEW_CODE/experiments/reviewer_sweep/run.py" "$@"
