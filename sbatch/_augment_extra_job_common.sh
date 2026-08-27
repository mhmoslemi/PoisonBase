#!/usr/bin/env bash
# Shared Vulcan runtime for one unfinished cell of augment-extra.tex.
# A cell is exactly one (configuration, selector, victim augmentation) run.

set -Eeuo pipefail

SOURCE_ROOT="${SOURCE_ROOT:-/home/mmoslem3/scratch/PoisonBase}"
PERSIST_DATA_ROOT="${PERSIST_DATA_ROOT:-$SOURCE_ROOT/data}"
PYTHON_ENV="${PYTHON_ENV:-/home/mmoslem3/ENV}"
RUN_ROOT="$SLURM_TMPDIR/augment_extra"
LOCAL_DATA_ROOT="$RUN_ROOT/data"
SYNCED=0
STEP_PID=""

say() { printf '%s\n' "$*"; }
die() { say "ERROR: $*" >&2; exit 1; }

for var in ROW_ID MODEL ATTACK BUDGET TARGET_SELECT SELECTION SEL_ALPHA \
           AUGMENT ATTACK_RUN_NAME RUN_RANDOM RUN_GREEDY RUN_DPP2 \
           RUN_DPP025 RUN_DPP01; do
    [ -n "${!var:-}" ] || die "$var is unset"
done

case "$AUGMENT" in
    none) DEF_TAG=none ;;
    standard|randaug|cutout) DEF_TAG="none+aug-$AUGMENT" ;;
    *) die "unsupported augmentation: $AUGMENT" ;;
esac
DEFENSE_RUN_NAME="${ATTACK_RUN_NAME}__def-${DEF_TAG}"
TARGET_FILE="aug_${MODEL}_${ATTACK}_dog-bird_b${BUDGET}.json"
CACHE_NAME="${MODEL}_${DEF_TAG}_50ep_lr0.1_bs125_wd0_seed42"

copy_dir_if_present() {
    local src="$1" dst="$2"
    if [ -d "$src" ]; then
        mkdir -p "$dst"
        rsync -a --exclude='.lock' --exclude='*.tmp' "$src/" "$dst/"
    fi
}

sync_outputs() {
    [ "$SYNCED" = 0 ] || return 0
    SYNCED=1
    say "sync: augment-extra cell -> $SOURCE_ROOT"
    if [ -d "$RUN_ROOT/augment_extra_result/$DEFENSE_RUN_NAME" ]; then
        mkdir -p "$SOURCE_ROOT/augment_extra_result/$DEFENSE_RUN_NAME"
        rsync -a --exclude='.lock' --exclude='*.tmp' \
            "$RUN_ROOT/augment_extra_result/$DEFENSE_RUN_NAME/" \
            "$SOURCE_ROOT/augment_extra_result/$DEFENSE_RUN_NAME/"
    fi
    if [ -d "$RUN_ROOT/cache/defended_victims/$CACHE_NAME" ]; then
        mkdir -p "$SOURCE_ROOT/cache/defended_victims/$CACHE_NAME"
        rsync -a --ignore-existing --exclude='*.tmp' \
            "$RUN_ROOT/cache/defended_victims/$CACHE_NAME/" \
            "$SOURCE_ROOT/cache/defended_victims/$CACHE_NAME/"
    fi
    if [ -f "$RUN_ROOT/target_sets/$TARGET_FILE" ]; then
        mkdir -p "$SOURCE_ROOT/target_sets"
        rsync -a "$RUN_ROOT/target_sets/$TARGET_FILE" \
            "$SOURCE_ROOT/target_sets/$TARGET_FILE"
    fi
}

handle_signal() {
    local signal="$1"
    say "signal: received $signal; stopping the step before final sync"
    if [ -n "$STEP_PID" ]; then
        kill -TERM "$STEP_PID" 2>/dev/null || true
        wait "$STEP_PID" 2>/dev/null || true
    fi
    sync_outputs
    trap - EXIT
    exit 143
}

[ -n "${SLURM_TMPDIR:-}" ] || die 'SLURM_TMPDIR is unset; submit with sbatch'
if command -v module >/dev/null 2>&1; then
    module load python/3.11.5 cuda/12.6 cudnn
else
    say "environment modules unavailable; using $PYTHON_ENV directly"
fi
source "$PYTHON_ENV/bin/activate"

trap 'handle_signal USR1' USR1
trap 'handle_signal TERM' TERM
trap 'handle_signal INT' INT
trap sync_outputs EXIT

mkdir -p "$RUN_ROOT" "$LOCAL_DATA_ROOT" "$RUN_ROOT/ours_result" \
         "$RUN_ROOT/augment_extra_result" "$RUN_ROOT/cache/defended_victims" \
         "$RUN_ROOT/target_sets"

for file in defense.py final_update.py networks.py utils.py victim_aug.py; do
    [ -f "$SOURCE_ROOT/$file" ] || die "required source file missing: $SOURCE_ROOT/$file"
    rsync -a "$SOURCE_ROOT/$file" "$RUN_ROOT/"
done
[ -d "$PERSIST_DATA_ROOT/cifar-10-batches-py" ] || \
    die "CIFAR-10 input missing: $PERSIST_DATA_ROOT/cifar-10-batches-py"
rsync -a "$PERSIST_DATA_ROOT/cifar-10-batches-py" "$LOCAL_DATA_ROOT/"

# Every selector in a row is evaluated on the same five targets.  Recompute the
# intersection from the actual saved poison caches, then publish it atomically.
python - "$SOURCE_ROOT" "$TARGET_FILE" "$MODEL" "$ATTACK" "$BUDGET" \
    "$RUN_RANDOM" "$RUN_GREEDY" "$RUN_DPP2" "$RUN_DPP025" "$RUN_DPP01" <<'PY'
import json, os, sys
root, filename, model, attack, budget, *runs = sys.argv[1:]
sys.path.insert(0, root)
import defense

sets, report = [], []
for run in runs:
    path = os.path.join(root, 'ours_result', run)
    targets = sorted(set(defense.cached_targets(path))) if os.path.isdir(path) else []
    report.append('%s=%d' % (run, len(targets)))
    sets.append(set(targets))
have = sorted(set.intersection(*sets)) if sets else []
if len(have) < 5:
    raise SystemExit('need 5 targets saved under all selectors; found %d\n%s'
                     % (len(have), '\n'.join(report)))
have = have[:5]
blob = {
    '_generated_by': 'augment-extra five-selector target intersection',
    '_combo': '%s / %s / dog-bird / b%s' % (model, attack, budget),
    '_pair_sels': 'random greedy dpp2 dpp0.25 dpp0.1',
    '_per_selection': report,
    'pairs': {'dog-bird': {'indices': have}},
}
out_dir = os.path.join(root, 'target_sets')
os.makedirs(out_dir, exist_ok=True)
path = os.path.join(out_dir, filename)
tmp = '%s.%d.tmp' % (path, os.getpid())
with open(tmp, 'w') as handle:
    json.dump(blob, handle, indent=1)
os.replace(tmp, path)
print('paired targets:', have)
PY

rsync -a "$SOURCE_ROOT/target_sets/$TARGET_FILE" "$RUN_ROOT/target_sets/"
[ -d "$SOURCE_ROOT/ours_result/$ATTACK_RUN_NAME" ] || \
    die "saved poison run missing: $SOURCE_ROOT/ours_result/$ATTACK_RUN_NAME"
copy_dir_if_present "$SOURCE_ROOT/ours_result/$ATTACK_RUN_NAME" \
                    "$RUN_ROOT/ours_result/$ATTACK_RUN_NAME"
copy_dir_if_present "$SOURCE_ROOT/augment_extra_result/$DEFENSE_RUN_NAME" \
                    "$RUN_ROOT/augment_extra_result/$DEFENSE_RUN_NAME"
copy_dir_if_present "$SOURCE_ROOT/cache/defended_victims/$CACHE_NAME" \
                    "$RUN_ROOT/cache/defended_victims/$CACHE_NAME"

SEL_FLAGS=(--base random)
case "$SELECTION" in
    random) SEL_FLAGS=(--base random) ;;
    greedy) SEL_FLAGS=(--base ours --base_dist cosine --lambda_margin 1.0) ;;
    dpp2|dpp025|dpp01)
        SEL_FLAGS=(--base ours --base_dist cosine --lambda_margin 1.0 \
                   --sel_dpp --sel_alpha "$SEL_ALPHA")
        ;;
    *) die "unsupported selection: $SELECTION" ;;
esac

say "job: $SLURM_JOB_ID $SLURM_JOB_NAME on $(hostname)"
say "cell: row=$ROW_ID selection=$SELECTION augmentation=$AUGMENT"
say "protocol: 5 paired targets x 4 victims"
python -c 'import torch; assert torch.cuda.is_available(); print("gpu:", torch.cuda.get_device_name(0))'

cd "$RUN_ROOT"
srun --ntasks=1 python defense.py \
    --dataset CIFAR10 --data_path "$LOCAL_DATA_ROOT" --seed 42 \
    --cache_dir "$RUN_ROOT/cache" --out_dir "$RUN_ROOT/ours_result" \
    --defense_out_dir "$RUN_ROOT/augment_extra_result" \
    --model "$MODEL" --attack "$ATTACK" "${SEL_FLAGS[@]}" \
    --class_pair dog-bird --pair_order poison-target \
    --budget "$BUDGET" --epsilon 0.0313725 \
    --craft_ensemble 5 --target_select "$TARGET_SELECT" \
    --target_idx_file "$RUN_ROOT/target_sets/$TARGET_FILE" \
    --defense none --victim_aug "$AUGMENT" \
    --num_targets 5 --num_victims 4 --victim_epochs 50 \
    --victim_lr 0.1 --victim_bs 125 --victim_decay 40 --victim_wd 0.0 \
    --clean_baseline &
STEP_PID=$!
set +e
wait "$STEP_PID"
status=$?
set -e
STEP_PID=""
sync_outputs
trap - EXIT
exit "$status"
