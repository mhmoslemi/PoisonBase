#!/usr/bin/env bash
# Runtime for one fresh matched-architecture main-table BASIS rerun.
# Outputs go to a dedicated result root, so incomplete historical runs are never
# interpreted as completed trials for these jobs.

set -Eeuo pipefail

ROOT="${ROOT:-/home/mmoslem3/scratch/PoisonBase}"
ENV_ACTIVATE="${ENV_ACTIVATE:-/home/mmoslem3/ENV/bin/activate}"
DATA_ROOT="${DATA_ROOT:-$ROOT/data}"
CACHE_ROOT="${CACHE_ROOT:-$ROOT/cache}"
RESULT_ROOT="${RESULT_ROOT:-$ROOT/basis_k_rand_rerun_20260919_result}"

say() { printf '%s\n' "$*"; }
die() { say "ERROR: $*" >&2; exit 1; }

for required in MAIN_INDEX MAIN_MODEL MAIN_ATTACK MAIN_CLASS_PAIR MAIN_BUDGET \
                MAIN_TARGET_DEGREE MAIN_NUM_TARGETS MAIN_NUM_VICTIMS \
                ORIGINAL_COMMAND; do
    [ -n "${!required:-}" ] || die "$required is unset"
done

[ -n "${SLURM_TMPDIR:-}" ] || die "SLURM_TMPDIR is unset; submit this file with sbatch"
[ "$MAIN_MODEL" = ResNet20BN ] || die "unexpected model: $MAIN_MODEL"
[ "$MAIN_ATTACK" = fc ] || die "unexpected attack: $MAIN_ATTACK"
case "$MAIN_CLASS_PAIR" in dog-bird|frog-airplane) ;; *) die "bad class pair: $MAIN_CLASS_PAIR" ;; esac
case "$MAIN_NUM_TARGETS" in 8|10) ;; *) die "targets must be 8 or 10" ;; esac
[ "$MAIN_NUM_VICTIMS" = 6 ] || die "victims must be 6"

if command -v module >/dev/null 2>&1; then
    module load python/3.11.5 cuda/12.6 cudnn
fi
[ -f "$ENV_ACTIVATE" ] || die "environment activation missing: $ENV_ACTIVATE"
# shellcheck disable=SC1090
source "$ENV_ACTIVATE"
python -c 'import torch; assert torch.cuda.is_available(); print("gpu:", torch.cuda.get_device_name(0))'

[ -f "$ROOT/final_update.py" ] || die "missing $ROOT/final_update.py"
[ -d "$DATA_ROOT/cifar-10-batches-py" ] || die "missing CIFAR-10 data under $DATA_ROOT"
TARGET_FILE="$ROOT/target_sets/${MAIN_MODEL}_${MAIN_ATTACK}_${MAIN_CLASS_PAIR}.json"
[ -s "$TARGET_FILE" ] || die "missing pinned target set: $TARGET_FILE"

python - "$TARGET_FILE" "$MAIN_CLASS_PAIR" "$MAIN_NUM_TARGETS" <<'PY'
import json
import sys

path, pair, count = sys.argv[1], sys.argv[2], int(sys.argv[3])
with open(path) as handle:
    payload = json.load(handle)
indices = list(map(int, payload['pairs'][pair]['indices']))[:count]
if len(indices) != count or len(set(indices)) != count:
    raise SystemExit('%s must contain at least %d unique %s targets' %
                     (path, count, pair))
print('targets: %d unique pinned indices:' % count, ' '.join(map(str, indices)))
PY

cache_has_nets() {
    local directory="$1" count="$2" index
    for ((index = 0; index < count; index++)); do
        [ -s "$directory/net_${index}.pt" ] || return 1
    done
}

ensure_caches() {
    local surrogate_dir victim_dir lock_fd
    surrogate_dir="$CACHE_ROOT/surrogates/${MAIN_MODEL}_60ep_lr0.1_bs128_seed42"
    victim_dir="$CACHE_ROOT/clean_victims/${MAIN_MODEL}_50ep_lr0.1_bs125_wd0_seed42"
    mkdir -p "$CACHE_ROOT/.basis_k_rand_locks" "$RESULT_ROOT"
    exec {lock_fd}>"$CACHE_ROOT/.basis_k_rand_locks/${MAIN_MODEL}.lock"
    flock -x "$lock_fd"
    if ! cache_has_nets "$surrogate_dir" 20 || ! cache_has_nets "$victim_dir" 6; then
        say "cache: completing shared $MAIN_MODEL surrogate/victim checkpoints"
        python "$ROOT/final_update.py" \
            --dataset CIFAR10 --data_path "$DATA_ROOT" --seed 42 --gpus all \
            --cache_dir "$CACHE_ROOT" --out_dir "$RESULT_ROOT" \
            --model "$MAIN_MODEL" \
            --num_surrogates 20 --surrogate_epochs 60 --surrogate_lr 0.1 \
            --surrogate_bs 128 --surrogate_decay 35 45 --surrogate_wd 0 \
            --num_victims 6 --victim_epochs 50 --victim_lr 0.1 \
            --victim_bs 125 --victim_decay 40 --victim_wd 0 \
            --precompute_only
    fi
    cache_has_nets "$surrogate_dir" 20 || die "surrogate cache remains incomplete"
    cache_has_nets "$victim_dir" 6 || die "victim cache remains incomplete"
    flock -u "$lock_fd"
    exec {lock_fd}>&-
}

ensure_caches

MAIN_RUN_NAME="CIFAR10_${MAIN_MODEL}_${MAIN_ATTACK}_ours_${MAIN_CLASS_PAIR}_b${MAIN_BUDGET}_eps8_seed42_lam1_cosine_ce5_tgt${MAIN_TARGET_DEGREE}"
mkdir -p "$RESULT_ROOT/.submission_locks"
exec {RUN_LOCK_FD}>"$RESULT_ROOT/.submission_locks/${MAIN_RUN_NAME}.lock"
say "run: waiting for duplicate-submission lock"
flock -x "$RUN_LOCK_FD"

say "job: ${SLURM_JOB_ID:-unknown} ${SLURM_JOB_NAME:-unknown} on $(hostname)"
say "config: $ORIGINAL_COMMAND"
say "protocol: method=ours Jacobian=off targets=$MAIN_NUM_TARGETS victims=$MAIN_NUM_VICTIMS selector_K=20"
say "output: $RESULT_ROOT/$MAIN_RUN_NAME"

srun --ntasks=1 python "$ROOT/final_update.py" \
    --dataset CIFAR10 --data_path "$DATA_ROOT" --seed 42 --gpus all \
    --cache_dir "$CACHE_ROOT" --out_dir "$RESULT_ROOT" \
    --model "$MAIN_MODEL" --attack "$MAIN_ATTACK" \
    --base ours --base_dist cosine --lambda_margin 1 \
    --class_pair "$MAIN_CLASS_PAIR" --pair_order poison-target \
    --budget "$MAIN_BUDGET" --epsilon 0.0313725 \
    --craft_steps 250 --craft_alpha 0.0039216 --restarts 8 --fc_restarts 1 \
    --craft_ensemble 5 --craft_aug \
    --num_surrogates 20 --surrogate_epochs 60 --surrogate_lr 0.1 \
    --surrogate_bs 128 --surrogate_decay 35 45 --surrogate_wd 0 \
    --num_targets "$MAIN_NUM_TARGETS" --target_select "$MAIN_TARGET_DEGREE" \
    --target_idx_file "$TARGET_FILE" --rank_on_victims \
    --num_victims "$MAIN_NUM_VICTIMS" --victim_epochs 50 --victim_lr 0.1 \
    --victim_bs 125 --victim_decay 40 --victim_wd 0 --clean_baseline

python - "$RESULT_ROOT/$MAIN_RUN_NAME/results.csv" "$TARGET_FILE" \
    "$MAIN_CLASS_PAIR" "$MAIN_NUM_TARGETS" "$MAIN_NUM_VICTIMS" <<'PY'
import csv
import json
import sys
from collections import Counter

csv_path, target_path, pair = sys.argv[1:4]
num_targets, num_victims = map(int, sys.argv[4:])
with open(target_path) as handle:
    expected = list(map(int, json.load(handle)['pairs'][pair]['indices']))[:num_targets]
with open(csv_path, newline='') as handle:
    rows = list(csv.DictReader(handle))

pairs = [(int(row['target_idx']), int(row['victim_id'])) for row in rows]
counts = Counter(target for target, _ in pairs)
expected_rows = num_targets * num_victims
problems = []
if len(rows) != expected_rows:
    problems.append('expected %d rows, found %d' % (expected_rows, len(rows)))
if len(set(pairs)) != expected_rows:
    problems.append('expected %d unique target/victim pairs, found %d' %
                    (expected_rows, len(set(pairs))))
if set(counts) != set(expected):
    problems.append('target IDs differ from the pinned set')
for target in expected:
    victims = sorted(victim for row_target, victim in pairs if row_target == target)
    if victims != list(range(num_victims)):
        problems.append('target %d victim IDs are %s' % (target, victims))
if problems:
    raise SystemExit('incomplete result: ' + '; '.join(problems))
print('verified: %d targets x %d victims = %d unique evaluations' %
      (num_targets, num_victims, expected_rows))
PY
