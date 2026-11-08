#!/bin/bash
set -Eeuo pipefail
ROOT="${1:?project root required}"
TASK="${2:?task ID required}"
ENV_ACTIVATE="${3:?environment activation required}"
DATA_ROOT="${4:?data directory required}"
: "${SLURM_TMPDIR:?Submit with submit_w4_20_killarney_ghaddar.sh}"
BATCH="$ROOT/sbatch/rebuttal_w4"
PLAN="$BATCH/plan.json"
JOB_DIR="$ROOT/rebuttal_w4_result/jobs/$TASK"
WORK="$SLURM_TMPDIR/rebuttal_w4"
DEADLINE=$(( $(date +%s) + 5 * 3600 - 300 ))
mkdir -p "$JOB_DIR"
exec 8>"$JOB_DIR/.worker.lock"
flock -n 8 || { echo 'Task already running.' >&2; exit 1; }
if [ -f "$JOB_DIR/.done" ]; then echo 'Task already complete.'; exit 0; fi
cd "$BATCH"
sha256sum --check --quiet code.sha256
MODEL="$(jq -er --arg id "$TASK" '. as $plan | .tasks[] | select(.id==$id) | .cell as $cell | $plan.cells[] | select(.id==$cell) | .model' "$PLAN")"
module load StdEnv/2023 gcc/12.3 python/3.11.5 cuda/12.6 cudnn/9.10.0.56
source "$ENV_ACTIVATE"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
export PYTHONPYCACHEPREFIX="$WORK/pycache"
export PYTHONPATH="$BATCH/code"
export XDG_CACHE_HOME="$WORK/downloads" HF_HOME="$WORK/downloads/hf" TORCH_HOME="$WORK/downloads/torch"
surrogates="surrogates/${MODEL}_60ep_lr0.1_bs128_seed42"
mkdir -p "$WORK/data" "$WORK/cache/$surrogates"
rsync -a "$DATA_ROOT/cifar-10-batches-py" "$WORK/data/"
for ((index=0; index<5; index++)); do
    rsync -a "$ROOT/cache/$surrogates/net_$index.pt" "$WORK/cache/$surrogates/"
done
# All Python checks and scientific work run inside this allocation.
python -m py_compile "$BATCH/code/worker.py" "$BATCH/code/check.py"
python -u "$BATCH/code/check.py"
notify_stop() { scancel --signal=USR1 "${SLURM_JOB_ID}.0" 2>/dev/null || true; }
trap notify_stop USR1 TERM INT
cd "$WORK"
srun --ntasks=1 --cpus-per-task=1 --unbuffered \
    python "$BATCH/code/worker.py" --plan "$PLAN" --task "$TASK" \
    --root "$ROOT" --data "$WORK/data" --cache "$WORK/cache" --deadline "$DEADLINE" &
STEP_PID=$!
status=0
wait "$STEP_PID" || status=$?
if kill -0 "$STEP_PID" 2>/dev/null; then
    status=0
    wait "$STEP_PID" || status=$?
fi
if [ "$status" -eq 75 ]; then
    echo 'Saved progress; rerun submit_w4_20_killarney_ghaddar.sh to resume this task.'
fi
exit "$status"
