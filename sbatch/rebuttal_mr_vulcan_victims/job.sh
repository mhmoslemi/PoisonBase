#!/bin/bash
set -Eeuo pipefail
ROOT="${1:?project root required}"
BATCH="$ROOT/sbatch/rebuttal_mr_killarney"
WRAPPER="$ROOT/sbatch/rebuttal_mr_vulcan_victims"
TASK="${2:?task ID required}"
ENV_ACTIVATE="${3:?environment required}"
: "${SLURM_TMPDIR:?Use submit_mr_tiny_gm_victims_vulcan_ghaddar.sh}"
cd "$BATCH"
sha256sum --check --quiet code.sha256
(cd "$ROOT" && sha256sum --check --quiet "$WRAPPER/poisons.sha256")
WORK="$SLURM_TMPDIR/rebuttal_mr_victims"
mkdir -p "$WORK/data" "$WORK/cache"
module load StdEnv/2023 gcc/12.3 python/3.11.5 cuda/12.6 cudnn/9.10.0.56
source "$ENV_ACTIVATE"
rsync -a "$ROOT/data/tinyimagenet.pt" "$WORK/data/tinyimagenet.pt"
(cd "$WORK/data" && awk -F '\t' 'NR>1 && $3=="TinyImageNet"{print $4 "  " $1}' "$BATCH/data.tsv" | sha256sum --check --quiet)
export MR_DEADLINE=$(( $(date +%s) + 14400 - 180 ))
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$BATCH/code"
export XDG_CACHE_HOME="$WORK/cache" HF_HOME="$WORK/cache/hf" TORCH_HOME="$WORK/cache/torch"
export PYTHONPYCACHEPREFIX="$WORK/pycache"
cd "$ROOT"
srun --ntasks=1 --cpus-per-task=1 --unbuffered \
    python -u "$BATCH/code/experiments/rebuttal_mr/run.py" --task "$TASK" \
    --plan "$BATCH/plan.json" --root "$ROOT" --data "$WORK/data" --cache "$WORK/cache"
