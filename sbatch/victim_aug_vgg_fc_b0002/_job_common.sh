#!/usr/bin/env bash
# Victim-only replay for the missing VGG13 / BP / rho=0.002 block in Table 3.
# The selected bases and optimized perturbations must already exist in
# ours_result; this launcher never invokes the crafting pipeline.

set -Eeuo pipefail

: "${SELECTION:?SELECTION must be set by the sbatch wrapper}"
: "${AUGMENT:?AUGMENT must be set by the sbatch wrapper}"

case "$SELECTION" in
    random|greedy) ;;
    *) echo "ERROR: unsupported SELECTION=$SELECTION" >&2; exit 1 ;;
esac
case "$AUGMENT" in
    standard|randaug|cutout) ;;
    *) echo "ERROR: unsupported AUGMENT=$AUGMENT" >&2; exit 1 ;;
esac

SOURCE_ROOT=/home/mmoslem3/scratch/attack_if
PERSIST_DATA_ROOT=/home/mmoslem3/scratch/attack_if/data
PYTHON_ENV=/home/mmoslem3/ENV
RUN_ROOT="$SLURM_TMPDIR/victim_aug_vgg_fc_b0002"
LOCAL_DATA_ROOT="$RUN_ROOT/data"

MODEL=VGG13BN
ATTACK=fc
CLASS_PAIR=dog-bird
BUDGET=0.002
TARGET_SELECT=3
NUM_TARGETS=5
NUM_VICTIMS=4

RUN_RANDOM='CIFAR10_VGG13BN_fc_random_dog-bird_b0.002_eps8_seed42_ce5_tgt3'
RUN_GRAFT='CIFAR10_VGG13BN_fc_ours_dog-bird_b0.002_eps8_seed42_lam1_cosine_ce5_tgt3'
case "$SELECTION" in
    random)
        ATTACK_RUN_NAME="$RUN_RANDOM"
        SEL_FLAGS=(--base random)
        ;;
    greedy)
        ATTACK_RUN_NAME="$RUN_GRAFT"
        SEL_FLAGS=(--base ours --base_dist cosine --lambda_margin 1.0)
        ;;
esac

DEF_TAG="none+aug-$AUGMENT"
DEFENSE_RUN_NAME="${ATTACK_RUN_NAME}__def-${DEF_TAG}"
TARGET_FILE="aug_${MODEL}_${ATTACK}_${CLASS_PAIR}_b${BUDGET}.json"
CACHE_NAME="${MODEL}_${DEF_TAG}_50ep_lr0.1_bs125_wd0_seed42"
SYNCED=0
STEP_PID=''

say() { printf '%s\n' "$*"; }
die() { say "ERROR: $*" >&2; exit 1; }

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
    say "sync: $DEFENSE_RUN_NAME -> $SOURCE_ROOT"
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
    say "signal: received $signal; stopping victim training before final sync"
    if [ -n "$STEP_PID" ]; then
        kill -TERM "$STEP_PID" 2>/dev/null || true
        wait "$STEP_PID" 2>/dev/null || true
    fi
    sync_outputs
    trap - EXIT
    exit 143
}

[ -n "${SLURM_TMPDIR:-}" ] || die 'SLURM_TMPDIR is unset; submit with sbatch'
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

# Pair RAND and GRAFT from their actual saved poison caches. Taking the same
# sorted first five targets in every job makes all six table cells comparable.
python - "$SOURCE_ROOT" "$RUN_ROOT/target_sets/$TARGET_FILE" \
    "$RUN_RANDOM" "$RUN_GRAFT" <<'PY'
import json, os, sys

root, output, random_run, graft_run = sys.argv[1:]
sys.path.insert(0, root)
import defense

report = {}
sets = []
for label, run in [('RAND', random_run), ('GRAFT', graft_run)]:
    path = os.path.join(root, 'ours_result', run)
    if not os.path.isdir(path):
        raise SystemExit('saved %s attack run is missing: %s' % (label, path))
    targets = set(defense.cached_targets(path))
    report[label] = len(targets)
    sets.append(targets)

paired = sorted(set.intersection(*sets))
if len(paired) < 5:
    raise SystemExit(
        'need at least 5 targets with saved RAND and GRAFT poisons; '
        'found %d (RAND=%d, GRAFT=%d). No poisons were recrafted.'
        % (len(paired), report['RAND'], report['GRAFT']))
paired = paired[:5]
blob = {
    '_generated_by': 'victim-only RAND/GRAFT cache intersection',
    '_combo': 'VGG13BN / fc / dog-bird / b0.002',
    '_pair_sels': 'random greedy',
    '_per_selection': 'RAND=%d GRAFT=%d' % (report['RAND'], report['GRAFT']),
    'pairs': {'dog-bird': {'indices': paired}},
}
os.makedirs(os.path.dirname(output), exist_ok=True)
with open(output, 'w') as handle:
    json.dump(blob, handle, indent=1)
print('paired targets:', paired)
PY

[ -d "$SOURCE_ROOT/ours_result/$ATTACK_RUN_NAME" ] || \
    die "saved poison run missing: $SOURCE_ROOT/ours_result/$ATTACK_RUN_NAME"
copy_dir_if_present "$SOURCE_ROOT/ours_result/$ATTACK_RUN_NAME" \
    "$RUN_ROOT/ours_result/$ATTACK_RUN_NAME"
copy_dir_if_present "$SOURCE_ROOT/augment_extra_result/$DEFENSE_RUN_NAME" \
    "$RUN_ROOT/augment_extra_result/$DEFENSE_RUN_NAME"
copy_dir_if_present "$SOURCE_ROOT/cache/defended_victims/$CACHE_NAME" \
    "$RUN_ROOT/cache/defended_victims/$CACHE_NAME"

say "job: $SLURM_JOB_ID $SLURM_JOB_NAME on $(hostname)"
say "cell: VGG13 / BP / rho=0.002 | $SELECTION | $AUGMENT"
say "protocol: victim-only replay on $NUM_TARGETS paired targets x $NUM_VICTIMS victims"
python -c 'import torch; assert torch.cuda.is_available(); print("gpu:", torch.cuda.get_device_name(0))'

cd "$RUN_ROOT"
srun --ntasks=1 python defense.py \
    --dataset CIFAR10 --data_path "$LOCAL_DATA_ROOT" --seed 42 \
    --cache_dir "$RUN_ROOT/cache" --out_dir "$RUN_ROOT/ours_result" \
    --defense_out_dir "$RUN_ROOT/augment_extra_result" \
    --model "$MODEL" --attack "$ATTACK" "${SEL_FLAGS[@]}" \
    --class_pair "$CLASS_PAIR" --pair_order poison-target \
    --budget "$BUDGET" --epsilon 0.0313725 \
    --craft_ensemble 5 --target_select "$TARGET_SELECT" \
    --target_idx_file "$RUN_ROOT/target_sets/$TARGET_FILE" \
    --defense none --victim_aug "$AUGMENT" \
    --num_targets "$NUM_TARGETS" --num_victims "$NUM_VICTIMS" \
    --victim_epochs 50 --victim_lr 0.1 --victim_bs 125 \
    --victim_decay 40 --victim_wd 0.0 --clean_baseline &
STEP_PID=$!
set +e
wait "$STEP_PID"
status=$?
set -e
STEP_PID=''
sync_outputs
trap - EXIT
exit "$status"
