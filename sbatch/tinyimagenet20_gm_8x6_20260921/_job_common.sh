#!/usr/bin/env bash
# Shared TinyImageNet runtime: stage only the fixed subset, checkpoint, and requeue.
set -Eeuo pipefail
: "${PHASE:?}"
: "${PHASE_ID:?}"
: "${SLURM_TMPDIR:?Submit with sbatch}"
ROOT=/home/mmoslem3/scratch/PoisonBase
OUTPUT="$ROOT/tinyimagenet20_gm_8x6_20260921_result"
case "${TINYIMAGENET_SLURM_ACCOUNT:-aip-boyuwang}" in
    aip-boyuwang) ;;
    aip-yiweilu) OUTPUT="${OUTPUT}_yiweilu" ;;
    *) echo 'Unsupported TinyImageNet Slurm account' >&2; exit 1 ;;
esac
WORK="$SLURM_TMPDIR/tinyimagenet20_${PHASE}_${PHASE_ID}"
LOCAL_OUTPUT="$WORK/results"
PID=''
REQUEUE=0
STAGED=0

sync_output() {
    [[ "$STAGED" == 1 ]] || return 0
    case "$PHASE" in
        surrogate)
            mkdir -p "$OUTPUT/models"
            [[ ! -f "$LOCAL_OUTPUT/models/surrogate_$PHASE_ID.pt" ]] || \
                rsync -a "$LOCAL_OUTPUT/models/surrogate_$PHASE_ID.pt" "$OUTPUT/models/"
            ;;
        targets)
            [[ ! -f "$LOCAL_OUTPUT/targets.json" ]] || rsync -a "$LOCAL_OUTPUT/targets.json" "$OUTPUT/"
            ;;
        experiment)
            mkdir -p "$OUTPUT/$PHASE_ID"
            [[ ! -d "$LOCAL_OUTPUT/$PHASE_ID" ]] || \
                rsync -a --exclude='*.tmp' "$LOCAL_OUTPUT/$PHASE_ID/" "$OUTPUT/$PHASE_ID/"
            ;;
    esac
}
finish() {
    local status=$?
    trap - EXIT
    sync_output || status=1
    exit "$status"
}
stop_step() {
    [[ "$1" != USR1 ]] || REQUEUE=1
    [[ "$1" == USR1 ]] || REQUEUE=0
    if [[ -n "$PID" ]]; then
        kill -USR1 "$PID" 2>/dev/null || true
    fi
}

if command -v module >/dev/null 2>&1; then
    module load python/3.11.5 cuda/12.6 cudnn
fi
source /home/mmoslem3/ENV/bin/activate
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONUNBUFFERED=1
mkdir -p "$OUTPUT/.locks" "$WORK/experiments" "$LOCAL_OUTPUT/models"
exec 9>"$OUTPUT/.locks/${PHASE}_${PHASE_ID}.lock"
flock -n 9 || { echo 'This experiment stage is already running.' >&2; exit 1; }
trap finish EXIT
trap 'stop_step USR1' USR1
trap 'stop_step TERM' TERM
trap 'stop_step INT' INT

if [[ "$PHASE" == prepare ]]; then
    python "$ROOT/experiments/tinyimagenet20_gm.py" prepare \
        --data-root "${TINYIMAGENET_FILE:-$ROOT/data/tinyimagenet.pt}" --output "$OUTPUT"
    exit 0
fi

for file in final_update.py networks.py utils.py; do
    rsync -a "$ROOT/$file" "$WORK/"
done
rsync -a "$ROOT/experiments/tinyimagenet20_gm.py" "$WORK/experiments/"
rsync -a "$ROOT/experiments/tinyimagenet20_data.py" "$WORK/experiments/"
rsync -a "$ROOT/experiments/imagenet100_gm.py" "$WORK/experiments/"
rsync -a "$ROOT/experiments/imagenet100_data.py" "$WORK/experiments/"
rsync -a "$OUTPUT/manifest.json" "$LOCAL_OUTPUT/"

# Preparation exports only 4,000 train images plus the selected validation
# images. The full tensor archive is never staged onto a GPU node.
python - "$OUTPUT/manifest.json" "$WORK/data" "$PHASE" <<'PY'
import json, pathlib, subprocess, sys
manifest = json.load(open(sys.argv[1]))
destination = pathlib.Path(sys.argv[2])
splits = ('val',) if sys.argv[3] == 'targets' else ('train', 'val')
for split in splits:
    (destination / split).mkdir(parents=True, exist_ok=True)
    sources = [str(pathlib.Path(manifest['data_root']) / split / name) for name in manifest['classes']]
    subprocess.run(['rsync', '-a', *sources, str(destination / split) + '/'], check=True)
PY

case "$PHASE" in
    surrogate)
        [[ ! -f "$OUTPUT/models/surrogate_$PHASE_ID.pt" ]] || \
            rsync -a "$OUTPUT/models/surrogate_$PHASE_ID.pt" "$LOCAL_OUTPUT/models/"
        ;;
    targets|experiment)
        for i in 0 1 2; do
            rsync -a "$OUTPUT/models/surrogate_$i.pt" "$LOCAL_OUTPUT/models/"
        done
        if [[ "$PHASE" == experiment ]]; then
            rsync -a "$OUTPUT/targets.json" "$LOCAL_OUTPUT/"
            mkdir -p "$LOCAL_OUTPUT/$PHASE_ID"
            [[ ! -d "$OUTPUT/$PHASE_ID" ]] || \
                rsync -a --exclude='*.tmp' "$OUTPUT/$PHASE_ID/" "$LOCAL_OUTPUT/$PHASE_ID/"
        fi
        ;;
    *) echo "Unknown phase $PHASE" >&2; exit 1 ;;
esac
STAGED=1
if [[ "$REQUEUE" == 1 ]]; then
    # A large initial staging transfer may use most of the first allocation.
    # No training progress has been started or discarded in this case.
    scontrol requeue "$SLURM_JOB_ID"
    exit 0
fi

python "$WORK/experiments/tinyimagenet20_gm.py" "$PHASE" --id "$PHASE_ID" \
    --data-root "$WORK/data" --output "$LOCAL_OUTPUT" \
    --workers 8 &
PID=$!
status=0
while :; do
    set +e
    wait "$PID"
    status=$?
    set -e
    kill -0 "$PID" 2>/dev/null || break
done
PID=''
sync_output
STAGED=0
if [[ "$status" == 75 && "$REQUEUE" == 1 ]]; then
    echo 'Checkpoint synced; requeueing this same job for the next allocation.'
    scontrol requeue "$SLURM_JOB_ID"
    exit 0
fi
exit "$status"
