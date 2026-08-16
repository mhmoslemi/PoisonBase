#!/usr/bin/env bash
#
# Diversity-aware base selection, mode: dpp
#   greedy log-det (DPP MAP) with quality q_i = exp(-SEL_ALPHA * score_i).
#   Small alpha = more diversity; large alpha reproduces plain --base ours.
#
# Runs final_update.py (final.py is untouched). The ONLY difference from a plain
# --base ours run is how the N_p bases are picked from the identical per-candidate
# score; every other hyperparameter matches the random-base sweep.
#
# Targets: pinned with --target_idx_file to target_sets/<MODEL>_<ATTACK>_<PAIR>.json,
# i.e. the exact 10 test images the random-base run for this combo attacked. The
# difficulty label and the craft-memory flags are read from sweep_config.json, so
# this combo is set up the same way ours.sh sets it up. Nothing is guessed here.
#
# The run name gets a _sel_alpha<value> suffix, so these land in their own
# directories and cannot collide with the existing --base ours results.
#
# Edit the four knobs below (they honour the environment, so you can sweep:
#     MODELS="VGG13BN ResNet20BN" CLASS_PAIRS="dog-bird frog-airplane" sh sel_dpp_grad1_.sh )
#
# MODELS and CLASS_PAIRS are space-separated LISTS, swept as model x pair.  The
# difficulty label, the target file and the craft flags are all resolved inside
# the loops, per (model, pair), because sweep_config.json is keyed by a single
# model and a single pair name -- passing a whole list as one key is what makes
# the lookup fail.  (MODEL / CLASS_PAIR, singular, are still honoured so the old
# one-at-a-time invocations keep working.)

# MODELS=(ConvNetBN VGG13BN ResNet20BN)
# ATTACKS=(fc gradmatch)
# BASES=(random ours)
# CLASS_PAIRS=(dog-bird frog-airplane)

MODELS="${MODELS:-${MODEL:-VGG13BN ResNet20BN}}"
ATTACK="${ATTACK:-gradmatch}"
CLASS_PAIRS="${CLASS_PAIRS:-${CLASS_PAIR:-frog-airplane dog-bird}}"
# BUDGETS="${BUDGETS:-0.001 0.002 0.005 0.01 0.02 0.04}"
BUDGETS="${BUDGETS:-0.002}"

SEL_ALPHA="${SEL_ALPHA:-2.0}"

DATASET=CIFAR10
DATA_PATH=/home/mmoslem3/scratch/data
OUT_DIR=ours_result
CACHE_DIR=./cache
SEED=42

source /home/mmoslem3/ENV/bin/activate
cd /home/mmoslem3/scratch/attack_if

echo "=== dpp selection | $ATTACK | models: $MODELS | pairs: $CLASS_PAIRS | SEL_ALPHA=$SEL_ALPHA ==="
echo "    budgets: $BUDGETS"
echo

rc=0

for model in $MODELS; do
for pair in $CLASS_PAIRS; do

    # --- difficulty label + craft-memory flags, straight from sweep_config.json.
    # Resolved per (model, pair): the difficulty label is keyed by both and the
    # craft flags by model, so this cannot be hoisted out of the loops.
    CFG="$(python - "$model" "$ATTACK" "$pair" <<'PY'
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
)" || { echo "!! skipping $model / $pair (no sweep_config.json entry)"; echo; rc=1; continue; }
    eval "$CFG"

    IDX="target_sets/${model}_${ATTACK}_${pair}.json"
    [ -s "$IDX" ] || {
        echo "!! target file $IDX missing -- run ours.sh once to regenerate it"
        echo "!! skipping $model / $pair"; echo; rc=1; continue; }

    echo "=== $model / $ATTACK / $pair | SEL_ALPHA=$SEL_ALPHA ==="
    echo "    targets pinned from $IDX (same 10 the random-base run attacked)"
    echo "    difficulty label tgt$CFG_TGT   craft flags: ${CFG_MEM:-none}"
    echo

    for bug in $BUDGETS; do
        echo "--- $model | pair $pair | budget $bug ---"
        python final_update.py \
            --dataset "$DATASET" --data_path "$DATA_PATH" --seed "$SEED" \
            --cache_dir "$CACHE_DIR" --out_dir "$OUT_DIR" \
            --model "$model" --attack "$ATTACK" --base ours \
            --class_pair "$pair" --pair_order poison-target \
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
            --clean_baseline
    done
done
done

# --no_resume / --recompute_deltas are deliberately not passed, so a shard that
# dies can simply be rerun and will pick up where it stopped.

# non-zero if any (model, pair) was skipped, so a bad config is visible in $?
exit $rc
