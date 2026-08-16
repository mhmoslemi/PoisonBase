#!/usr/bin/env bash

DATASET=CIFAR10
DATA_PATH=/home/mmoslem3/scratch/data
OUT_DIR=ours_result
CACHE_DIR=./cache
SEED=42

# MODELS=(ConvNetBN VGG13BN ResNet20BN)
# ATTACKS=(fc gradmatch)
# BASES=(random ours)
# CLASS_PAIRS=(dog-bird frog-airplane)
# Edit these three -- everything difficulty/memory related follows from them.
# They also honour the environment, so you can sweep without editing:
#     for p in dog-bird frog-airplane; do MODEL=VGG13BN ATTACK=fc CLASS_PAIR=$p sh ours.sh; done
MODEL="${MODEL:-ConvNetBN}"
ATTACK="${ATTACK:-gradmatch}"
BASE="${BASE:-ours}"
CLASS_PAIR="${CLASS_PAIR:-frog-airplane}"

PAIR_ORDER=poison-target

EPSILON=0.0313725         # 16/255. use 0.0313725 for 8/255
CRAFT_STEPS=250
CRAFT_ALPHA=0.0039216   # 1/255. fc: PGD sign step. gradmatch: signed-Adam lr
RESTARTS=8
CRAFT_ENSEMBLE=5        # 0 = use all surrogates

# Targets are NOT chosen by difficulty. sweep_config.json stores the exact 10
# test-set indices the random-base run for this MODEL + ATTACK + CLASS_PAIR
# attacked, and they are handed to final.py with --target_idx_file, which beats
# the degree inside select_targets(). Same images, paired comparison, no drift.
#
# (A degree cannot promise that: it re-ranks the eligible pool by clean-ensemble
# p_adv at run time, and the clean ensemble is not bit-identical between runs --
# the ResNet20BN/fc/dog-bird tgt10 runs logged eligible 972 on some invocations
# and 974 on others, which slides the window onto different images.)
#
# TARGET_SELECT survives only as the _tgt<N> run-name suffix so the run still
# lines up with its table.tex row. Override either for one run:
#     TARGET_SELECT=20 sh ours.sh
#     TARGET_IDX_FILE=my_targets.json sh ours.sh
TARGET_SELECT="${TARGET_SELECT:-}"
NUM_TARGETS=10
# NUM_TARGETS=4

BASE_DIST=cosine    #l2
LAMBDA=1.0

NUM_SURROGATES=5
SURROGATE_EPOCHS=60
SURROGATE_DECAY="35 45"


NUM_VICTIMS=6
# NUM_VICTIMS=3
# NUM_VICTIMS=5
VICTIM_EPOCHS=50
VICTIM_LR=0.1
VICTIM_BS=125
VICTIM_DECAY="40"

# FAST / LOWMEM are looked up too, for the same reason: the random-base sweep
# used --craft_lowmem --fast_gradmatch for VGG13BN+gradmatch at EVERY budget and
# for nothing else, so matching it is what keeps the crafting path identical.
# --craft_lowmem does one surrogate and one CRAFT_BATCH slice of poisons at a
# time; same exact objective, ~1.5-2x crafting time, and it takes precedence over
# --fast_gradmatch. Both are ignored when ATTACK=fc.
# Override for one run:  LOWMEM="--craft_lowmem" FAST="" sh ours.sh
FAST="${FAST-__auto__}"
LOWMEM="${LOWMEM-__auto__}"
CRAFT_BATCH=256

source /home/mmoslem3/ENV/bin/activate
cd /home/mmoslem3/scratch/attack_if

# ---------------------------------------------------------------------------
# Auto-configure from sweep_config.json: whatever the random-base sweep used for
# this MODEL / ATTACK / CLASS_PAIR. Dies rather than guessing on an unknown combo.
# ---------------------------------------------------------------------------
CFG="$(python - "$MODEL" "$ATTACK" "$CLASS_PAIR" <<'PY'
import json, os, sys
model, attack, pair = sys.argv[1:4]
cfg = json.load(open('sweep_config.json'))

def pick(section, what):
    try:
        return cfg[section][model][attack][pair]
    except KeyError:
        sys.exit('sweep_config.json has no %s for %s / %s / %s -- add it there '
                 '(and say where it came from) rather than guessing here.'
                 % (what, model, attack, pair))

targets = pick('targets', 'target list')
tgt = pick('difficulty', 'difficulty')

# The exact 10 images the random-base run attacked, in final.py's
# --target_idx_file format. Regenerated every run, so sweep_config.json stays
# the single source of truth and this file is only a view of it.
os.makedirs('target_sets', exist_ok=True)
path = 'target_sets/%s_%s_%s.json' % (model, attack, pair)
with open(path, 'w') as f:
    json.dump({'_generated_by': 'ours.sh from sweep_config.json -- do not hand-edit',
               '_combo': '%s / %s / %s' % (model, attack, pair),
               'pairs': {pair: {'indices': targets}}}, f, indent=1)

mem = cfg['memory'].get(model, {}).get(attack, cfg['memory_default'])
print('CFG_TARGET_SELECT=%s' % tgt)
print('CFG_TARGET_IDX_FILE=%s' % path)
print('CFG_TARGETS="%s"' % ' '.join(str(t) for t in targets))
print('CFG_LOWMEM=%s' % ('--craft_lowmem'   if mem['craft_lowmem']   else "''"))
print('CFG_FAST=%s'   % ('--fast_gradmatch' if mem['fast_gradmatch'] else "''"))
PY
)" || exit 1
eval "$CFG"

TARGET_IDX_FILE="${TARGET_IDX_FILE:-$CFG_TARGET_IDX_FILE}"
[ -s "$TARGET_IDX_FILE" ] || { echo "target idx file $TARGET_IDX_FILE missing/empty"; exit 1; }

[ -z "$TARGET_SELECT" ] && TARGET_SELECT="$CFG_TARGET_SELECT"

MEM_NOTE="matches the random-base sweep"
if [ "$LOWMEM" = "__auto__" ]; then LOWMEM="$CFG_LOWMEM"; else MEM_NOTE="OVERRIDDEN"; fi
if [ "$FAST"   = "__auto__" ]; then FAST="$CFG_FAST";     else MEM_NOTE="OVERRIDDEN"; fi

echo "=== $MODEL / $ATTACK / $CLASS_PAIR / base=$BASE ==="
echo "    targets       = $CFG_TARGETS"
echo "                    pinned from $TARGET_IDX_FILE -- the exact 10 the random-base run attacked"
echo "    target_select = $TARGET_SELECT   (label only: it sets the _tgt$TARGET_SELECT run-name suffix;"
echo "                    --target_idx_file wins over the degree inside final.py)"
echo "    craft flags   = ${LOWMEM:-none} ${FAST}${LOWMEM:+  (craft_batch $CRAFT_BATCH)}   ($MEM_NOTE)"
echo

# 0.04 0.02 0.01 0.005 0.002 0.001
# for bug in 0.04 0.02 0.01; do
# for bug in 0.04 0.005 0.002; do

# for bug in 0.02; do
for bug in 0.005 0.002 0.001; do
# for bug in 0.01; do
# for bug in 0.001; do

# for bug in 0.005 0.002 0.001; do

${DRY_RUN:+echo} python final.py \
    --dataset "$DATASET" --data_path "$DATA_PATH" --seed "$SEED" \
    --model "$MODEL" --attack "$ATTACK" --base "$BASE" \
    --class_pair "$CLASS_PAIR" --pair_order "$PAIR_ORDER" \
    --budget "$bug" --epsilon "$EPSILON" \
    --craft_steps "$CRAFT_STEPS" --craft_alpha "$CRAFT_ALPHA" \
    --restarts "$RESTARTS" --craft_ensemble "$CRAFT_ENSEMBLE" $FAST \
    $LOWMEM --craft_batch "$CRAFT_BATCH" \
    --base_dist "$BASE_DIST" --lambda_margin "$LAMBDA" \
    --num_surrogates "$NUM_SURROGATES" --surrogate_epochs "$SURROGATE_EPOCHS" \
    --surrogate_decay $SURROGATE_DECAY \
    --num_targets "$NUM_TARGETS" --target_select "$TARGET_SELECT" \
    --target_idx_file "$TARGET_IDX_FILE" \
    --num_victims "$NUM_VICTIMS" --victim_epochs "$VICTIM_EPOCHS" \
    --victim_lr "$VICTIM_LR" --victim_bs "$VICTIM_BS" --victim_decay $VICTIM_DECAY \
    --victim_wd 0.0 --clean_baseline \
    --cache_dir "$CACHE_DIR" --out_dir "$OUT_DIR" \
    --no_resume --recompute_deltas

done



# 40%(6/15) asr, FC, dog-bird, easyineess 10, random select, budget 0.005, resnet
# 68.8% asr, FC, dog-bird, easyineess 10, random select, budget 0.05, resnet

# 60%(9/15) asr, FC, dog-bird, easyineess 10, ours select, budget 0.005, resnet
# 50% (8/16) asr, FC, dog-bird, easyineess 10, ours select, budget 0.05, resnet

