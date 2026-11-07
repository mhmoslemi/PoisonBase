#!/bin/bash
# Run with srun --jobid=<the existing allocation> ... bash this-file <model>.
set -Eeuo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
MODEL="${1:?ConvNetBN or ResNet20BN required}"
: "${SLURM_JOB_ID:?Use the allocated GPU, not the login node}"
module load StdEnv/2023 gcc/12.3 python/3.11.5 cuda/12.6 cudnn/9.10.0.56
source /home/mmoslem3/ENV/bin/activate
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
export XDG_CACHE_HOME="$SLURM_TMPDIR/w5_cache"
export HF_HOME="$SLURM_TMPDIR/w5_cache/hf"
export PYTHONPATH="$ROOT"
cd "$ROOT"
python -u experiments/rebuttal_w5/benchmark.py --model "$MODEL"
