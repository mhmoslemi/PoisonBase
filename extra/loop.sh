#!/usr/bin/env bash

# ---- GPUs visible to this script (edit this list) ----
# GPU_IDS="0 1 2 3"
# export CUDA_VISIBLE_DEVICES="$(echo $GPU_IDS | tr ' ' ',')"

#
# Full sweep of the diversity-aware base selection, mode: dpp
#   greedy log-det (DPP MAP) with quality q_i = exp(-SEL_ALPHA * score_i).
#   Small alpha = more diversity; large alpha reproduces plain --base ours.
#
# Loops over 3 models x 2 class pairs x every budget = 36 runs, in that order
# (all budgets of one combo finish before the next combo starts, so the target
# set and the cached surrogates for that combo are only paid for once).
#
# Runs final_update.py, which spreads each run over every gpu in GPU_IDS: one
# target per gpu, and when fewer targets are left than gpus the idle ones help
# with the remaining victims of the targets still running.
#
# The ONLY difference from a plain --base ours run is how the N_p bases are
# picked from the identical per-candidate score; every other hyperparameter
# matches the random-base sweep.
#
# Targets: pinned per combo with --target_idx_file to
# target_sets/<MODEL>_<ATTACK>_<PAIR>.json, i.e. the exact 10 test images the
# random-base run for that combo attacked. The difficulty label and the
# craft-memory flags are read from sweep_config.json. Nothing is guessed here.
# A combo whose target file is missing is skipped and reported at the end
# rather than killing the whole sweep.
#
# The run name gets a _seldpp<alpha> suffix, so these land in their own
# directories and cannot collide with the existing --base ours results.
#
# Everything honours the environment, so you can run any slice of the sweep:
#     MODELS=VGG13BN sh loop.sh
#     PAIRS=dog-bird BUDGETS="0.02 0.04" sh loop.sh
#     ATTACK=gradmatch sh loop.sh
#     FORCE=0 sh loop.sh          # resume instead of redoing finished trials

# MODELS="${MODELS:-ConvNetBN ResNet20BN VGG13BN}"
MODELS="${MODELS:-ConvNetBN}"
PAIRS="${PAIRS:-dog-bird frog-airplane}"
# BUDGETS="${BUDGETS:-0.001 0.002 0.005 0.01 0.02 0.04}"
BUDGETS="${BUDGETS:-0.001}"
ATTACK="${ATTACK:-gradmatch}"

SEL_ALPHA="${SEL_ALPHA:-2}"

# FORCE=1 redoes every run from scratch: results.csv is dropped and every delta
# is re-crafted. The cached surrogates / clean victims are reused either way.
# Set FORCE=0 to pick a killed sweep up where it stopped instead.
FORCE="${FORCE:-1}"

DATASET=CIFAR10
DATA_PATH=/home/ubuntu/mohammad/data
OUT_DIR=ours_result
CACHE_DIR=./cache
SEED=42

source ../../.venv/bin/activate
cd /home/ubuntu/mohammad/base/attack_if

if [ "$FORCE" = "1" ]; then
    FORCE_FLAG="--FORCE"
else
    FORCE_FLAG=""
fi

N_BUDGETS=$(echo $BUDGETS | wc -w)
TOTAL=$(( $(echo $MODELS | wc -w) * $(echo $PAIRS | wc -w) * N_BUDGETS ))
DONE=0
FAILED=""
SKIPPED=""
SWEEP_T0=$(date +%s)

echo "=========================================================================="
echo " dpp sweep | attack=$ATTACK  alpha=$SEL_ALPHA  force=$FORCE  gpus=$GPU_IDS"
echo " models : $MODELS"
echo " pairs  : $PAIRS"
echo " budgets: $BUDGETS"
echo " $TOTAL runs total"
echo "=========================================================================="
echo

for MODEL in $MODELS; do
for CLASS_PAIR in $PAIRS; do

    # --- difficulty label + craft-memory flags, straight from sweep_config.json ---
    CFG="$(python - "$MODEL" "$ATTACK" "$CLASS_PAIR" <<'PY'
import json, sys
model, attack, pair = sys.argv[1:4]
cfg = json.load(open('sweep_config.json'))
try:
    tgt = cfg['difficulty'][model][attack][pair]
except KeyError:
    sys.exit('sweep_config.json has no difficulty for %s / %s / %s' % (model, attack, pair))
mem = cfg['memory'].get(model, {}).get(attack, cfg['memory_default'])
print('CFG_TGT=%s' % tgt)
print("CFG_MEM='%s'" % ('--craft_lowmem --craft_batch 256 --fast_gradmatch'
                        if mem['craft_lowmem'] else ''))
PY
)"
    if [ $? -ne 0 ]; then
        echo "!! skipping $MODEL / $ATTACK / $CLASS_PAIR: not in sweep_config.json"
        SKIPPED="$SKIPPED  $MODEL/$ATTACK/$CLASS_PAIR (no sweep_config entry)
"
        DONE=$(( DONE + N_BUDGETS ))
        continue
    fi
    eval "$CFG"

    IDX="target_sets/${MODEL}_${ATTACK}_${CLASS_PAIR}.json"
    if [ ! -s "$IDX" ]; then
        echo "!! skipping $MODEL / $ATTACK / $CLASS_PAIR: $IDX missing"
        echo "   (run ours.sh once for this combo to regenerate it)"
        SKIPPED="$SKIPPED  $MODEL/$ATTACK/$CLASS_PAIR (no $IDX)
"
        DONE=$(( DONE + N_BUDGETS ))
        continue
    fi

    echo "=== dpp selection | $MODEL / $ATTACK / $CLASS_PAIR | SEL_ALPHA=$SEL_ALPHA ==="
    echo "    targets pinned from $IDX (same 10 the random-base run attacked)"
    echo "    difficulty label tgt$CFG_TGT   craft flags: ${CFG_MEM:-none}"
    echo

    for bug in $BUDGETS; do
        DONE=$(( DONE + 1 ))
        echo "--- [$DONE/$TOTAL] $MODEL / $CLASS_PAIR / budget $bug ---"
        RUN_T0=$(date +%s)

        python final_update.py \
            --dataset "$DATASET" --data_path "$DATA_PATH" --seed "$SEED" \
            --cache_dir "$CACHE_DIR" --out_dir "$OUT_DIR" \
            --model "$MODEL" --attack "$ATTACK" --base ours \
            --class_pair "$CLASS_PAIR" --pair_order poison-target \
            --budget "$bug" --epsilon 0.0313725 \
            --craft_steps 250 --craft_alpha 0.0039216 \
            --restarts 8 --craft_ensemble 5 $CFG_MEM \
            --base_dist cosine --lambda_margin 1.0 \
            --sel_dpp --sel_alpha "$SEL_ALPHA" \
            --num_surrogates 20 --surrogate_epochs 60 --surrogate_decay 35 45 \
            --num_targets 10 --target_select "$CFG_TGT" \
            --target_idx_file "$IDX" \
            --num_victims 6 --victim_epochs 50 --victim_lr 0.1 --victim_bs 125 \
            --victim_decay 40 --victim_wd 0.0 \
            --clean_baseline $FORCE_FLAG

        RC=$?
        MIN=$(( ($(date +%s) - RUN_T0) / 60 ))
        if [ $RC -ne 0 ]; then
            echo "!! FAILED (exit $RC) after ${MIN}m: $MODEL / $CLASS_PAIR / $bug"
            FAILED="$FAILED  $MODEL/$ATTACK/$CLASS_PAIR budget=$bug (exit $RC)
"
        else
            echo "    done in ${MIN}m"
        fi
        echo
    done
done
done

# --- summary ---------------------------------------------------------------
echo "=========================================================================="
echo " sweep finished in $(( ($(date +%s) - SWEEP_T0) / 60 )) min"
if [ -n "$SKIPPED" ]; then
    echo " skipped combos:"
    printf "%s" "$SKIPPED"
fi
if [ -n "$FAILED" ]; then
    echo " FAILED runs:"
    printf "%s" "$FAILED"
    echo " rerun the sweep with FORCE=0 to redo only what is missing"
    echo "=========================================================================="
    exit 1
fi
echo " all $TOTAL runs ok -- summary rows appended to $OUT_DIR/summary_all.csv"
echo "=========================================================================="
