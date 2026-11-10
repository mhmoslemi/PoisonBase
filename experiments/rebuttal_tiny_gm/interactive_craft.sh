#!/bin/bash
set -Eeuo pipefail
: "${SLURM_JOB_ID:?Run this inside your interactive GPU allocation.}"
VARIANT="${1:?selection variant}"
shift
MODE=craft
[ "$#" -eq 0 ] || MODE=victim
ROOT="$(cd -- "$(dirname -- "$0")/../.." && pwd -P)"
BATCH="$ROOT/sbatch/rebuttal_tiny_gm_vgg13_5x5"
cd "$BATCH"
sha256sum --check --quiet code.sha256
module load StdEnv/2023 gcc/12.3 python/3.11.5 cuda/12.6 cudnn/9.10.0.56
source /home/mmoslem3/ENV/bin/activate
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
export XDG_CACHE_HOME="${SLURM_TMPDIR:?}/cache" HF_HOME="$SLURM_TMPDIR/cache/hf" TORCH_HOME="$SLURM_TMPDIR/cache/torch"
ALLOCATION_END="$(squeue -h -j "$SLURM_JOB_ID" -o '%e')"
export MR_DEADLINE=$(( $(date -d "$ALLOCATION_END" +%s) - 60 ))
cd "$ROOT"
mkdir -p rebuttal_tiny_gm_vgg13_5x5_result/logs
exec > >(tee -a "rebuttal_tiny_gm_vgg13_5x5_result/logs/interactive_${MODE}_${VARIANT}_${SLURM_JOB_ID}.log") 2>&1
echo "Starting $MODE $VARIANT on allocation $SLURM_JOB_ID $*"
srun --ntasks=1 --cpus-per-task=1 --unbuffered \
    python -u "$ROOT/experiments/rebuttal_tiny_gm/interactive_ablation.py" --root "$ROOT" --variant "$VARIANT" "$@"
