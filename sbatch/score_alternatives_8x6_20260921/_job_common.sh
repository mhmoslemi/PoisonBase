#!/usr/bin/env bash
# One allocated GPU runs one selector/attack/budget cell (48 victim trials).
set -Eeuo pipefail

: "${SCORE_JOB_ID:?The individual job must set SCORE_JOB_ID}"
: "${SLURM_TMPDIR:?Submit with sbatch; SLURM_TMPDIR is required}"
ROOT="${ROOT:-/home/mmoslem3/scratch/PoisonBase}"
ENV_ACTIVATE="${ENV_ACTIVATE:-/home/mmoslem3/ENV/bin/activate}"
DATA_ROOT="${DATA_ROOT:-$ROOT/data}"
CACHE_ROOT="${CACHE_ROOT:-$ROOT/cache}"
BATCH_DIR="$ROOT/sbatch/score_alternatives_8x6_20260921"
RESULT_ROOT="${RESULT_ROOT:-$ROOT/score_alternatives_8x6_20260921_result}"
CELL_ROOT="$RESULT_ROOT/cell_$SCORE_JOB_ID"
RUN_ROOT="${RUN_ROOT:-$SLURM_TMPDIR/PoisonBase_score_$SCORE_JOB_ID}"
SURROGATES=surrogates/ConvNetBN_60ep_lr0.1_bs128_seed42
VICTIMS=clean_victims/ConvNetBN_50ep_lr0.1_bs125_wd0_seed42
STEP_PID=''
STAGED=0

die() { printf 'ERROR: %s\n' "$*" >&2; exit 1; }
sync_outputs() {
    if [[ "$STAGED" == 1 && -d "$RUN_ROOT/results" ]]; then
        mkdir -p "$CELL_ROOT"
        rsync -a --exclude='.lock' --exclude='*.tmp' \
            "$RUN_ROOT/results/" "$CELL_ROOT/"
    fi
}
finish() {
    local status=$?
    trap - EXIT
    sync_outputs || status=1
    exit "$status"
}
handle_signal() {
    trap '' USR1 TERM INT
    printf 'Stopping experiment %s and saving completed trials.\n' "$SCORE_JOB_ID"
    if [[ -n "$STEP_PID" ]]; then
        kill -TERM "$STEP_PID" 2>/dev/null || true
        wait "$STEP_PID" || true
    fi
    exit 143
}

if command -v module >/dev/null 2>&1; then
    module load python/3.11.5 cuda/12.6 cudnn
fi
source "$ENV_ACTIVATE"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export PYTHONUNBUFFERED=1
python -c 'import torch; assert torch.cuda.is_available(), "CUDA GPU required"; print(torch.cuda.get_device_name(0))'
[[ "$SCORE_JOB_ID" =~ ^[0-9]{3}$ ]] || die 'invalid experiment ID'
[[ -d "$DATA_ROOT/cifar-10-batches-py" ]] || die "missing CIFAR-10: $DATA_ROOT"

# Use the existing table's checkpoints. Fail early instead of silently spending
# the cell allocation training a different or incomplete shared model ensemble.
for ((i=0; i<20; i++)); do
    [[ -s "$CACHE_ROOT/$SURROGATES/net_$i.pt" ]] || \
        die "missing shared surrogate: $CACHE_ROOT/$SURROGATES/net_$i.pt"
done
for ((i=0; i<6; i++)); do
    [[ -s "$CACHE_ROOT/$VICTIMS/net_$i.pt" ]] || \
        die "missing clean victim: $CACHE_ROOT/$VICTIMS/net_$i.pt"
done

# Hold this persistent lock across staging, execution, and final sync so a
# duplicate submission cannot overwrite an active job's partial results.
mkdir -p "$RESULT_ROOT/.locks"
exec 9>"$RESULT_ROOT/.locks/cell_$SCORE_JOB_ID.lock"
flock -n 9 || die "cell $SCORE_JOB_ID is already running"
trap finish EXIT
trap handle_signal USR1 TERM INT

mkdir -p "$RUN_ROOT/data" "$RUN_ROOT/cache/$SURROGATES" \
    "$RUN_ROOT/cache/$VICTIMS" "$RUN_ROOT/results"
for file in final_update.py networks.py utils.py; do
    rsync -a "$ROOT/$file" "$RUN_ROOT/"
done
rsync -a "$BATCH_DIR/run_cell.py" "$RUN_ROOT/"
rsync -a "$DATA_ROOT/cifar-10-batches-py" "$RUN_ROOT/data/"
for ((i=0; i<20; i++)); do
    rsync -a "$CACHE_ROOT/$SURROGATES/net_$i.pt" "$RUN_ROOT/cache/$SURROGATES/"
done
for ((i=0; i<6; i++)); do
    rsync -a "$CACHE_ROOT/$VICTIMS/net_$i.pt" "$RUN_ROOT/cache/$VICTIMS/"
done
if [[ -d "$CELL_ROOT" ]]; then
    rsync -a --exclude='.lock' --exclude='*.tmp' "$CELL_ROOT/" "$RUN_ROOT/results/"
fi
STAGED=1
cd "$RUN_ROOT"
srun --ntasks=1 --unbuffered python "$RUN_ROOT/run_cell.py" \
    --id "$SCORE_JOB_ID" --work-root "$RUN_ROOT" &
STEP_PID=$!
set +e
wait "$STEP_PID"
status=$?
set -e
STEP_PID=''
exit "$status"
