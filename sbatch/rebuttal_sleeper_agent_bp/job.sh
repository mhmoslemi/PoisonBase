#!/bin/bash
# One allocated GPU handles one missing gradient-norm/BP table cell.
set -Eeuo pipefail
ROOT="${1:?project root required}"
MODEL="${2:?model required}"
BUDGET="${3:?budget required}"
ENV_ACTIVATE="${4:?environment required}"
: "${SLURM_JOB_ID:?Submit through submit_sleeper_agent_bp.sh}"
: "${SLURM_TMPDIR:?Slurm local storage is required}"
case "$MODEL" in ConvNetBN) TARGET_DEGREE=50;; ResNet20BN) TARGET_DEGREE=10;; *) exit 1;; esac
case "$BUDGET" in 0.002|0.005) ;; *) exit 1;; esac
CELL="$ROOT/rebuttal_sleeper_agent_bp_result/${MODEL}_b${BUDGET}"
WORK="$SLURM_TMPDIR/rebuttal_sleeper_agent_bp"
SURROGATES="surrogates/${MODEL}_60ep_lr0.1_bs128_seed42"
VICTIMS="clean_victims/${MODEL}_50ep_lr0.1_bs125_wd0_seed42"
METRICS="selector_metrics/CIFAR10_${MODEL}_60ep_select10_lr0.1_bs128_decay35-45_wd0_seed42/class5"
RUN_NAME="CIFAR10_${MODEL}_fc_ours_dog-bird_b${BUDGET}_eps8_seed42_lam1_cosine_selgao-gradnorm_ep10_K20_ce5_tgt${TARGET_DEGREE}"

mkdir -p "$CELL"
exec 8>"$CELL/.cell.lock"
flock -n 8 || { echo 'This cell is already running.' >&2; exit 1; }
if [ -f "$CELL/.done" ]; then echo 'Cell already complete.'; exit 0; fi
# The exclusive cell lock proves that no other worker owns this dedicated run.
# Remove the Python marker a terminated attempt can leave, allowing an immediate resume.
rm -f -- "$CELL/$RUN_NAME/.lock"

module load StdEnv/2023 gcc/12.3 python/3.11.5 cuda/12.6 cudnn/9.10.0.56
source "$ENV_ACTIVATE"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
export XDG_CACHE_HOME="$WORK/downloads" HF_HOME="$WORK/downloads/hf" TORCH_HOME="$WORK/downloads/torch"
mkdir -p "$WORK/data" "$WORK/cache/$SURROGATES" "$WORK/cache/$VICTIMS" "$WORK/cache/$METRICS"
for source_file in final_update.py networks.py utils.py; do
    rsync -a "$ROOT/$source_file" "$WORK/"
done
rsync -a "$ROOT/data/cifar-10-batches-py" "$WORK/data/"
for ((index=0; index<20; index++)); do
    rsync -a "$ROOT/cache/$SURROGATES/net_$index.pt" "$WORK/cache/$SURROGATES/"
    rsync -a "$ROOT/cache/$METRICS/net_$index.npz" "$WORK/cache/$METRICS/"
done
for ((index=0; index<6; index++)); do
    rsync -a "$ROOT/cache/$VICTIMS/net_$index.pt" "$WORK/cache/$VICTIMS/"
done
# Reuse the first eight BP targets from the original model-specific target set.
python - "$ROOT/target_sets/${MODEL}_fc_dog-bird.json" "$WORK/targets.json" <<'PY'
import json
import sys
with open(sys.argv[1]) as handle:
    targets = json.load(handle)['pairs']['dog-bird']['indices'][:8]
if len(targets) != 8 or len(set(targets)) != 8:
    raise SystemExit('Expected eight distinct BP targets')
with open(sys.argv[2], 'w') as handle:
    json.dump({'pairs': {'dog-bird': {'indices': targets}}}, handle)
print('Pinned BP targets:', targets, flush=True)
PY

STEP_PID=''
stop_job() {
    trap '' USR1 TERM INT
    echo 'Stopping; completed victim trials and poisons are already saved on scratch.'
    if [ -n "$STEP_PID" ]; then
        kill -TERM "$STEP_PID" 2>/dev/null || true
        wait "$STEP_PID" || true
    fi
    exit 143
}
trap stop_job USR1 TERM INT
cd "$WORK"
printf 'Cell: %s BP rho=%s; 8 targets x 6 victims; victims: 50 epochs\nResults: %s\n' "$MODEL" "$BUDGET" "$CELL"
# Output goes directly to persistent scratch. The existing runner skips completed
# target/victim pairs and reuses saved base selections and perturbations.
srun --ntasks=1 --cpus-per-task=1 --unbuffered python "$WORK/final_update.py" \
    --dataset CIFAR10 --data_path "$WORK/data" --seed 42 --gpus all \
    --cache_dir "$WORK/cache" --out_dir "$CELL" \
    --model "$MODEL" --attack fc --fc_mode sample --base ours \
    --base_dist cosine --lambda_margin 1 --sel_criterion gao-gradnorm \
    --selector_metric_dir "$WORK/cache/selector_metrics" --sel_metric_epoch 10 --sel_K 20 \
    --class_pair dog-bird --pair_order poison-target --budget "$BUDGET" --epsilon 0.0313725 \
    --craft_steps 250 --craft_alpha 0.0039216 --restarts 8 --fc_restarts 1 \
    --craft_ensemble 5 --craft_aug --dsa_strategy color_crop_cutout_flip_scale_rotate \
    --num_surrogates 20 --surrogate_epochs 60 --surrogate_lr 0.1 \
    --surrogate_bs 128 --surrogate_decay 35 45 --surrogate_wd 0 \
    --num_targets 8 --target_select "$TARGET_DEGREE" --target_idx_file "$WORK/targets.json" \
    --keep_pinned_targets --num_victims 6 --victim_epochs 50 --victim_lr 0.1 \
    --victim_bs 125 --victim_decay 40 --victim_wd 0 --clean_baseline &
STEP_PID=$!
status=0
wait "$STEP_PID" || status=$?
STEP_PID=''
[ "$status" -eq 0 ] || exit "$status"

python - "$CELL/$RUN_NAME/results.csv" "$WORK/targets.json" "$MODEL" "$BUDGET" <<'PY'
import csv
import json
import sys
with open(sys.argv[2]) as handle:
    targets = json.load(handle)['pairs']['dog-bird']['indices']
with open(sys.argv[1], newline='') as handle:
    rows = list(csv.DictReader(handle))
pairs = [(int(row['target_idx']), int(row['victim_id'])) for row in rows]
expected = {(target, victim) for target in targets for victim in range(6)}
if len(rows) != 48 or set(pairs) != expected:
    raise SystemExit('Incomplete cell: expected 48 distinct target/victim pairs')
if any(row['model'] != sys.argv[3] or row['attack'] != 'fc'
       or abs(float(row['budget']) - float(sys.argv[4])) > 1e-12
       or row['success'] not in ('0', '1') for row in rows):
    raise SystemExit('Cell metadata does not match the requested experiment')
print('Complete: eight targets x six victims = 48 evaluations.', flush=True)
PY
touch "$CELL/.done"
