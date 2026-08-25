#!/usr/bin/env bash
#
# Defense sweep: replay the poisons that are ALREADY on disk through a defended
# victim training, once per base selection, so the effect of the base selection
# can be read off under a defense instead of against an undefended victim.
#
# Nothing is crafted here. For each (MODEL, ATTACK, CLASS_PAIR, budget) cell this
# script
#
#   1. asks defense.py which targets each base selection has saved poisons for,
#   2. intersects those lists and pins the intersection with --target_idx_file,
#      so every selection is judged on exactly the same target images, and
#   3. runs each DEFENSE on each SELECTION over that pinned set.
#
# The victim hyperparameters below are the ones the attack runs used, so
# `DEFENSES=none` reproduces the undefended numbers (build_network reseeds from
# seed*100000 + tidx*100 + victim_id, so the same trial is the same trial).
#
# Sweep by editing the knobs or from the environment:
#   for p in dog-bird frog-airplane; do MODEL=VGG13BN CLASS_PAIR=$p sh defense.sh; done
#   DEFENSES="none epic" SELS="random dpp" BUDGETS=0.005 sh defense.sh
#
# By default this replays the historical non-Jacobian poison caches. To replay
# Greedy_J / DPP_J caches, enable the same controls used while crafting them:
#   USE_JACOBIAN_SCORE=1 JACOBIAN_WEIGHT=1.0 SELS="ours dpp" sh defense.sh
# JACOBIAN_BATCH_SIZE is metadata only here (defense.py does not recompute the
# score), but passing it keeps the replay summary faithful to the attack setup.

MODEL="${MODEL:-ResNet20BN}"
ATTACK="${ATTACK:-fc}"
CLASS_PAIR="${CLASS_PAIR:-dog-bird}"
BUDGETS="${BUDGETS:-0.002 0.01}"

# Base selections to compare. random | ours | dpp  (dpp = ours + --sel_dpp)
# SELS="${SELS:-random ours dpp}"
SELS="${SELS:-ours}"
SEL_ALPHA="${SEL_ALPHA:-2.0}"

# Defenses to run. Anything defense.py accepts, including composed ones:
#   none | epic | friends | noise | advtrain | epic+friends
# DEFENSES="${DEFENSES:-none epic friends}"
DEFENSES="${DEFENSES:-epic friends}"

DATASET=CIFAR10
DATA_PATH=/home/mmoslem3/scratch/data
OUT_DIR=ours_result           # where the attack runs (and their poisons) live
DEF_OUT_DIR=defense_result    # where this writes
CACHE_DIR=./cache
SEED=42
EPSILON=0.0313725             # 8/255, matches the attack runs

USE_JACOBIAN_SCORE="${USE_JACOBIAN_SCORE:-0}"
JACOBIAN_WEIGHT="${JACOBIAN_WEIGHT:-1.0}"
JACOBIAN_BATCH_SIZE="${JACOBIAN_BATCH_SIZE:-64}"
case "$USE_JACOBIAN_SCORE" in
    0|1) ;;
    *) echo "USE_JACOBIAN_SCORE=$USE_JACOBIAN_SCORE (expected: 0 or 1)"; exit 1 ;;
esac
python - "$JACOBIAN_WEIGHT" "$JACOBIAN_BATCH_SIZE" <<'PY' || exit 1
import sys
try:
    weight = float(sys.argv[1])
    batch_size = int(sys.argv[2])
except ValueError:
    sys.exit('JACOBIAN_WEIGHT must be a number and JACOBIAN_BATCH_SIZE an integer')
if weight < 0:
    sys.exit('JACOBIAN_WEIGHT must be nonnegative')
if batch_size <= 0:
    sys.exit('JACOBIAN_BATCH_SIZE must be positive')
PY


NUM_VICTIMS="${NUM_VICTIMS:-5}"
# tab:utility-matched-defense retunes each defense so its CLEAN accuracy stays
# within two points of the undefended model, so the strength has to be settable
# from outside. Empty means defense.py's own default (EPIC 0.1, FRIENDS 8).
EPIC_SUBSET="${EPIC_SUBSET:-}"
NOISE_EPS="${NOISE_EPS:-}"
# FRIENDS has two independent magnitudes. --noise_eps scales only the RANDOM
# (bernoulli) component; the friendly noise itself is bounded by --friendly_clamp,
# 16/255 by default, and is what the defense is actually named after. Sweeping
# noise_eps alone leaves the dominant term untouched, which is why the clean-utility
# curve is flat in noise_eps.
FRIENDLY_CLAMP="${FRIENDLY_CLAMP:-}"
NUM_TARGETS="${NUM_TARGETS:-5}"
VICTIM_EPOCHS=50
VICTIM_LR=0.1
VICTIM_BS=125
VICTIM_DECAY=40

source /home/mmoslem3/ENV/bin/activate
cd /home/mmoslem3/scratch/attack_if

# The difficulty label only has to match the attack run so the _tgt<N> suffix of
# the run name lines up. Straight from sweep_config.json, same as sel_dpp.sh.
# TARGET_SELECT overrides the sweep_config.json lookup. Needed wherever the crafts
# being replayed were not produced by the main sweep: sweep_config.json only
# records a difficulty for the (model, attack, pair) combos table.tex reports, so
# a combo it never covered -- ConvNetBN/sapa, for instance -- has no entry and is
# refused here rather than guessed. Pass the degree the crafts actually used.
TARGET_SELECT="${TARGET_SELECT:-}"
if [ -n "$TARGET_SELECT" ]; then
    CFG_TGT="$TARGET_SELECT"
else
CFG="$(python - "$MODEL" "$ATTACK" "$CLASS_PAIR" <<'PY'
import json, sys
model, attack, pair = sys.argv[1:4]
cfg = json.load(open('sweep_config.json'))
try:
    print('CFG_TGT=%s' % cfg['difficulty'][model][attack][pair])
except KeyError:
    sys.exit('sweep_config.json has no difficulty for %s / %s / %s'
             % (model, attack, pair))
PY
)" || exit 1
eval "$CFG"
fi

sel_flags() {
    case "$1" in
        random) echo "--base random" ;;
        ours)   echo "--base ours --base_dist cosine --lambda_margin 1.0" ;;
        dpp)    echo "--base ours --base_dist cosine --lambda_margin 1.0 --sel_dpp --sel_alpha $SEL_ALPHA" ;;
        *)      echo "UNKNOWN_SEL" ;;
    esac
}

jacobian_flags() {
    if [ "$USE_JACOBIAN_SCORE" = "1" ] && [ "$1" != "random" ]; then
        echo "--use_jacobian_score --jacobian_weight $JACOBIAN_WEIGHT --jacobian_batch_size $JACOBIAN_BATCH_SIZE"
    else
        echo ""
    fi
}

mkdir -p target_sets "$DEF_OUT_DIR"

# --- pair the selections: one pinned target set per budget --------------------
# build_run_name and the cache reader come from the real modules, so this can
# never drift from what defense.py itself will look for.
PLAN="$(python - "$MODEL" "$ATTACK" "$CLASS_PAIR" "$CFG_TGT" "$SEED" "$EPSILON" \
                 "$OUT_DIR" "$SEL_ALPHA" "$SELS" "$BUDGETS" \
                 "$USE_JACOBIAN_SCORE" "$JACOBIAN_WEIGHT" \
                 "$JACOBIAN_BATCH_SIZE" <<'PY'
import argparse, json, os, sys
import final_update as FU
import defense as DEF

(model, attack, pair, tgt, seed, eps, out_dir, sel_alpha,
 sels, budgets, use_jacobian, jacobian_weight,
 jacobian_batch_size) = sys.argv[1:14]
seed, eps, sel_alpha = int(seed), float(eps), float(sel_alpha)
use_jacobian = bool(int(use_jacobian))
jacobian_weight = float(jacobian_weight)
jacobian_batch_size = int(jacobian_batch_size)
sels, budgets = sels.split(), [float(b) for b in budgets.split()]

def ns(base, budget, dpp):
    return argparse.Namespace(
        dataset='CIFAR10', model=model, attack=attack, base=base,
        class_pair=pair, budget=budget, epsilon=eps, seed=seed,
        lambda_margin=1.0, base_dist='cosine',
        sel_filter=False, sel_pca=False, sel_mmr=False, sel_dpp=dpp,
        sel_pool=3.0, sel_mu=1.0, sel_alpha=sel_alpha,
        fc_mode='sample', sharp_mode='worst', sharp_sigma=0.05,
        sharp_samples=20, craft_ensemble=5, target_select=int(tgt),
        use_jacobian_score=(use_jacobian and base == 'ours'),
        jacobian_weight=jacobian_weight,
        jacobian_batch_size=jacobian_batch_size)

SPEC = {'random': ('random', False), 'ours': ('ours', False), 'dpp': ('ours', True)}
ok = []
for b in budgets:
    have, report = None, []
    for s in sels:
        base, dpp = SPEC[s]
        d = os.path.join(out_dir, FU.build_run_name(ns(base, b, dpp)))
        ts = set(DEF.cached_targets(d)) if os.path.isdir(d) else set()
        report.append('%s=%d' % (s, len(ts)))
        have = ts if have is None else (have & ts)
    have = sorted(have or [])
    jac_tag = ('_jacw%g' % jacobian_weight
               if use_jacobian and any(s != 'random' for s in sels) else '')
    path = 'target_sets/def_%s_%s_%s_b%g%s.json' % (
        model, attack, pair, b, jac_tag)
    with open(path, 'w') as f:
        json.dump({'_generated_by': 'defense.sh -- intersection of the targets '
                                    'every base selection has saved poisons for',
                   '_combo': '%s / %s / %s / b%g' % (model, attack, pair, b),
                   '_per_selection': ' '.join(report),
                   'pairs': {pair: {'indices': have}}}, f, indent=1)
    print('#  b%-6g %-28s -> %d paired target(s)' % (b, ' '.join(report), len(have)),
          file=sys.stderr)
    if have:
        ok.append('%g:%s' % (b, path))
    else:
        print('#  b%g skipped: no target has poisons under every selection' % b,
              file=sys.stderr)
print(' '.join(ok))
PY
)" || exit 1

echo "=== defense sweep | $MODEL / $ATTACK / $CLASS_PAIR (tgt$CFG_TGT) ==="
echo "    selections : $SELS   (dpp alpha $SEL_ALPHA)"
if [ "$USE_JACOBIAN_SCORE" = "1" ]; then
    echo "    selector   : Jacobian enabled for ours/dpp (weight $JACOBIAN_WEIGHT, batch $JACOBIAN_BATCH_SIZE); random unchanged"
else
    echo "    selector   : Jacobian disabled (historical attack-run names)"
fi
echo "    defenses   : $DEFENSES"
echo "    poisons    : replayed from $OUT_DIR, results into $DEF_OUT_DIR"
echo

[ -n "$PLAN" ] || { echo "nothing to run: no budget has poisons under every selection"; exit 1; }

for entry in $PLAN; do
    bug="${entry%%:*}"
    idx="${entry#*:}"
    for def in $DEFENSES; do
        for sel in $SELS; do
            flags="$(sel_flags "$sel")"
            jac_flags="$(jacobian_flags "$sel")"
            [ "$flags" = "UNKNOWN_SEL" ] && { echo "unknown SEL '$sel'"; exit 1; }
            echo "--- budget $bug | defense $def | selection $sel ---"
            ${DRY_RUN:+echo} python defense.py \
                --dataset "$DATASET" --data_path "$DATA_PATH" --seed "$SEED" \
                --cache_dir "$CACHE_DIR" --out_dir "$OUT_DIR" \
                --defense_out_dir "$DEF_OUT_DIR" \
                --model "$MODEL" --attack "$ATTACK" $flags \
                --class_pair "$CLASS_PAIR" --pair_order poison-target \
                --budget "$bug" --epsilon "$EPSILON" \
                --craft_ensemble 5 --target_select "$CFG_TGT" \
                --target_idx_file "$idx" \
                --defense "$def" \
                $jac_flags \
                ${EPIC_SUBSET:+--epic_subset_size "$EPIC_SUBSET"} \
                ${NOISE_EPS:+--noise_eps "$NOISE_EPS"} \
                ${FRIENDLY_CLAMP:+--friendly_clamp "$FRIENDLY_CLAMP"} \
                ${NUM_TARGETS:+--num_targets "$NUM_TARGETS"} \
                --num_victims "$NUM_VICTIMS" --victim_epochs "$VICTIM_EPOCHS" \
                --victim_lr "$VICTIM_LR" --victim_bs "$VICTIM_BS" \
                --victim_decay "$VICTIM_DECAY" --victim_wd 0.0 \
                --clean_baseline
        done
    done
done

# --no_resume is deliberately not passed: a shard that dies can just be rerun and
# picks up from results.csv where it stopped.
