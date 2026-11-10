#!/bin/bash
set -Eeuo pipefail
ROOT="${1:?project root}"
BATCH="${2:?frozen snapshot}"
SECONDS_REQUESTED="${3:?time limit}"
TASK="${4:?original craft task}"
TARGET="${5:?target index}"
cd "$BATCH"
sha256sum --check --quiet code.sha256
export MR_DEADLINE=$(( $(date +%s) + SECONDS_REQUESTED - 60 ))
module load StdEnv/2023 gcc/12.3 python/3.11.5 cuda/12.6 cudnn/9.10.0.56
source /home/mmoslem3/ENV/bin/activate
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$BATCH/code"
export XDG_CACHE_HOME="$SLURM_TMPDIR/cache" HF_HOME="$SLURM_TMPDIR/cache/hf" TORCH_HOME="$SLURM_TMPDIR/cache/torch"
cd "$ROOT"
srun --ntasks=1 --cpus-per-task=1 --unbuffered \
    python -u "$ROOT/experiments/rebuttal_tiny_gm/check_craft_target.py"
exec srun --ntasks=1 --cpus-per-task=1 --unbuffered \
    python -u "$ROOT/experiments/rebuttal_tiny_gm/craft_target.py" \
    --root "$ROOT" --batch "$BATCH" --task "$TASK" --target "$TARGET"
