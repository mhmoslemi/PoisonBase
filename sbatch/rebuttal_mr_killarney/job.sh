#!/bin/bash
set -Eeuo pipefail
ROOT="${1:?project root required}"
BATCH="${2:?snapshot required}"
TASK="${3:?task required}"
ENV_ACTIVATE="${4:?environment activation required}"
DATA_ROOT="${5:?data root required}"
: "${SLURM_TMPDIR:?Run through submit_mr_killarney_ghaddar.sh}"
cd "$BATCH"
sha256sum --check --quiet code.sha256
SECONDS_REQUESTED="$(jq -er --arg task "$TASK" '.tasks[] | select(.id==$task) | .seconds' plan.json)"
DATASET="$(jq -er --arg task "$TASK" '.tasks[] | select(.id==$task) | .dataset' plan.json)"
export MR_DEADLINE=$(( $(date +%s) + SECONDS_REQUESTED - 180 ))
module load StdEnv/2023 gcc/12.3 python/3.11.5 cuda/12.6 cudnn/9.10.0.56
source "$ENV_ACTIVATE"
WORK="$SLURM_TMPDIR/rebuttal_mr"
mkdir -p "$WORK/data" "$WORK/cache"
while IFS=$'\t' read -r relative bytes dataset hash; do
    if [ "$DATASET" = all ] || [ "$DATASET" = "$dataset" ]; then
        mkdir -p "$WORK/data/$(dirname "$relative")"
        rsync -a "$DATA_ROOT/$relative" "$WORK/data/$relative"
    fi
done < <(tail -n +2 "$BATCH/data.tsv")
while IFS=$'\t' read -r relative bytes dataset hash; do
    if [ "$DATASET" = all ] || [ "$DATASET" = "$dataset" ]; then
        source_relative="$relative"
        case "$dataset" in
            CIFAR100|TinyImageNet) source_relative="${relative%/*}_new/${relative##*/}" ;;
        esac
        relative="${relative#cache/surrogates/}"
        mkdir -p "$WORK/surrogates/$(dirname "$relative")"
        rsync -a "$ROOT/$source_relative" "$WORK/surrogates/$relative"
    fi
done < <(tail -n +2 "$BATCH/surrogates.tsv")
cd "$WORK/data"
awk -F '\t' -v dataset="$DATASET" 'NR>1 && (dataset=="all" || $3==dataset){print $4 "  " $1}' "$BATCH/data.tsv" | sha256sum --check --quiet
cd "$WORK/surrogates"
awk -F '\t' -v dataset="$DATASET" 'NR>1 && (dataset=="all" || $3==dataset){sub(/^cache\/surrogates\//,"",$1); print $4 "  " $1}' "$BATCH/surrogates.tsv" | sha256sum --check --quiet
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
export XDG_CACHE_HOME="$WORK/cache" HF_HOME="$WORK/cache/hf" TORCH_HOME="$WORK/cache/torch"
export PYTHONPYCACHEPREFIX="$WORK/pycache" PYTHONPATH="$BATCH/code"
cd "$ROOT"
srun --ntasks=1 --cpus-per-task=1 --unbuffered \
    python -u "$BATCH/code/experiments/rebuttal_mr/run.py" --task "$TASK" \
    --plan "$BATCH/plan.json" --root "$ROOT" --data "$WORK/data" --cache "$WORK/surrogates"
