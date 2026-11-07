#!/bin/bash
set -Eeuo pipefail
ROOT="${1:?project root required}"
TASK="${2:?task ID required}"
ENV_ACTIVATE="${3:?environment required}"
: "${SLURM_TMPDIR:?Submit with submit_meta_influence_lu.sh}"
BATCH="$ROOT/sbatch/rebuttal_meta_influence"
PLAN="$BATCH/plan.json"
record="$(jq -ce --arg id "$TASK" '.tasks[] | select(.id==$id)' "$PLAN")"
MODEL="$(jq -r '.model' <<< "$record")"
STAGE="$(jq -r '.stage' <<< "$record")"
duration="$(jq -r '.time' <<< "$record")"
IFS=: read -r hours minutes seconds <<< "$duration"
allowance=$((10#$hours * 3600 + 10#$minutes * 60 + 10#$seconds - 240))
JOB_DIR="$ROOT/rebuttal_meta_influence_result/jobs/$TASK"
WORK="$SLURM_TMPDIR/rebuttal_meta_influence"
mkdir -p "$JOB_DIR"
exec 8>"$JOB_DIR/.worker.lock"
flock -n 8 || { echo 'This task is already running.' >&2; exit 1; }
if [ -f "$JOB_DIR/.done" ]; then echo 'Task already complete.'; exit 0; fi
cd "$BATCH"
sha256sum --check --quiet code.sha256
module load StdEnv/2023 gcc/12.3 python/3.11.5 cuda/12.6 cudnn/9.10.0.56
source "$ENV_ACTIVATE"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$BATCH/code"
export XDG_CACHE_HOME="$WORK/downloads" HF_HOME="$WORK/downloads/hf" TORCH_HOME="$WORK/downloads/torch"
surrogates="surrogates/${MODEL}_60ep_lr0.1_bs128_seed42"
victims="clean_victims/${MODEL}_50ep_lr0.1_bs125_wd0_seed42"
mkdir -p "$WORK/data" "$WORK/cache/$surrogates"
rsync -a "$ROOT/data/cifar-10-batches-py" "$WORK/data/"
for ((index=0; index<20; index++)); do
    rsync -a "$ROOT/cache/$surrogates/net_$index.pt" "$WORK/cache/$surrogates/"
done
if [ "$STAGE" = evaluate ]; then
    mkdir -p "$WORK/cache/$victims"
    for ((index=0; index<6; index++)); do
        rsync -a "$ROOT/cache/$victims/net_$index.pt" "$WORK/cache/$victims/"
    done
fi
STEP_PID=''
notify_stop() {
    # Send the signal to the Python task, which checkpoints before exiting.
    scancel --signal=USR1 "${SLURM_JOB_ID}.0" 2>/dev/null || true
}
trap notify_stop USR1 TERM INT
cd "$WORK"
srun --ntasks=1 --cpus-per-task=1 --unbuffered \
    python "$BATCH/code/experiments/rebuttal_selection/worker.py" \
    --plan "$PLAN" --task "$TASK" --root "$ROOT" \
    --data "$WORK/data" --cache "$WORK/cache" --seconds "$allowance" &
STEP_PID=$!
status=0
wait "$STEP_PID" || status=$?
# A trapped signal can interrupt bash's wait before the checkpointing task exits.
if kill -0 "$STEP_PID" 2>/dev/null; then
    status=0
    wait "$STEP_PID" || status=$?
fi
if [ "$status" -eq 75 ]; then
    echo 'Task paused with saved progress; rerun submit_meta_influence_lu.sh to resume.'
fi
exit "$status"
