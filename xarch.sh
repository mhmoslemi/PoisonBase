#!/usr/bin/env bash
#
# One off-diagonal block of the cross-architecture table (tab:cross-architecture).
#
# S = the architecture whose surrogates PICK THE BASES, A = V = the architecture
# the poisons are crafted on and the victims are trained with. final_update.py
# grew a --sel_model flag for exactly this: --model is A, --sel_model is S. When
# they differ the run loads a second surrogate ensemble (S's, straight from
# cache/surrogates -- same seeds, so they are literally the nets S's own run
# selected with), uses it for select_base_ours_div, and crafts/trains with A.
#
# Only the DPP column needs runs. --base random draws from the poison class with
# a per-target rng and never touches a net, so an S != A random run would be
# bit-identical to the S = A one; final_update.py rejects that combination
# instead of writing a duplicate under a new name.
#
# Protocol matches the diagonal cells already in the table: the 5 targets that
# cell was scored on (first 5 of the sorted Random-DPP intersection, pinned in
# target_sets/xarch_<A>_<attack>_dog-bird_b0.005.json) x victims 0-3 = 20 trials.
# Because build_network seeds from seed*100000 + tidx*100 + victim_id, victims
# 0-3 here are the same nets the diagonal cell used -- the column is paired
# trial-for-trial, only the base selection differs.
#
# Every other flag is copied verbatim from the diagonal run's own "args:" line,
# so the only difference between this run and the S = A one is --sel_model.
#
# Not called directly -- xa1.sh .. xa6.sh set MODEL/ATTACK and source it.
#
#   MODEL=ResNet20BN ATTACK=fc SEL_MODELS="ConvNetBN VGG13BN" sh ./xarch.sh
#   DRY_RUN=1 ... sh ./xarch.sh     # print the commands and stop

set -u

MODEL="${MODEL:?set MODEL (the A = V architecture)}"
ATTACK="${ATTACK:?set ATTACK (fc | gradmatch)}"
SEL_MODELS="${SEL_MODELS:?set SEL_MODELS (the S architectures)}"
DRY_RUN="${DRY_RUN:-}"

DATASET=CIFAR10
DATA_PATH=/home/mmoslem3/scratch/data
OUT_DIR=ours_result
CACHE_DIR=./cache
SEED=42
PAIR=dog-bird
BUDGET=0.005

source /home/mmoslem3/ENV/bin/activate
cd /home/mmoslem3/scratch/attack_if

if [ -z "$DRY_RUN" ]; then
    python -c 'import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)' || {
        echo "xarch.sh: no CUDA device visible -- get a GPU allocation first"; exit 1; }
fi

# difficulty label (only names the run dir -- the targets are pinned) and the
# craft-memory flags, both read from sweep_config.json the way sel_dpp.sh does,
# so VGG13BN/gradmatch keeps its --craft_lowmem --fast_gradmatch
CFG="$(python - "$MODEL" "$ATTACK" "$PAIR" <<'PY'
import json, sys
model, attack, pair = sys.argv[1:4]
cfg = json.load(open('sweep_config.json'))
print('CFG_TGT=%s' % cfg['difficulty'][model][attack][pair])
mem = cfg['memory'].get(model, {}).get(attack, cfg['memory_default'])
print("CFG_MEM='%s'" % ('--craft_lowmem --craft_batch 256 --fast_gradmatch'
                        if mem['craft_lowmem'] else ''))
PY
)" || exit 1
eval "$CFG"

IDX="target_sets/xarch_${MODEL}_${ATTACK}_${PAIR}_b${BUDGET}.json"
[ -s "$IDX" ] || { echo "xarch.sh: missing pinned target file $IDX"; exit 1; }

for S in $SEL_MODELS; do

    [ "$S" = "$MODEL" ] && { echo "xarch.sh: S = A = $S is the diagonal cell, skipping"; continue; }

    TAG="${DATASET}_${MODEL}_${ATTACK}_ours_${PAIR}_b${BUDGET}_eps8_seed${SEED}"
    TAG="${TAG}_lam1_cosine_seldpp2_selarch${S}_ce5_tgt${CFG_TGT}"

    echo "=== cross-arch | S=$S  ->  A=V=$MODEL | $ATTACK / $PAIR / budget $BUDGET ==="
    echo "    targets: pinned from $IDX (the 5 the diagonal cell used)"
    echo "    craft flags: ${CFG_MEM:-none}   run dir: $TAG"

    if grep -q "==== $TAG : ASR = " "$OUT_DIR/$TAG/log.txt" 2>/dev/null; then
        echo "    already complete: $(grep -o 'ASR = .*====' "$OUT_DIR/$TAG/log.txt" | tail -1)"
        echo
        continue
    fi

    CMD="python final_update.py \
        --dataset $DATASET --data_path $DATA_PATH --seed $SEED \
        --cache_dir $CACHE_DIR --out_dir $OUT_DIR \
        --model $MODEL --sel_model $S --attack $ATTACK --base ours \
        --class_pair $PAIR --pair_order poison-target \
        --budget $BUDGET --epsilon 0.0313725 \
        --craft_steps 250 --craft_alpha 0.0039216 \
        --restarts 8 --craft_ensemble 5 $CFG_MEM \
        --base_dist cosine --lambda_margin 1.0 \
        --sel_dpp --sel_alpha 2.0 \
        --num_surrogates 20 --surrogate_epochs 60 --surrogate_decay 35 45 \
        --num_targets 5 --target_select $CFG_TGT --target_idx_file $IDX \
        --num_victims 4 --victim_epochs 50 --victim_lr 0.1 --victim_bs 125 \
        --victim_decay 40 --victim_wd 0.0 \
        --clean_baseline"

    if [ -n "$DRY_RUN" ]; then
        echo "$CMD" | tr -s ' '
    else
        # no --no_resume / --recompute_deltas: a killed run picks up at the first
        # missing (target, victim) trial and re-reads its own poison_cache
        eval "$CMD" || exit 1
    fi
    echo

done
