#!/usr/bin/env bash
#
# Augmentation sweep: replay the poisons that are ALREADY on disk through a
# victim that AUGMENTS during training, once per base selection, so the effect of
# the base selection can be read off against an augmenting victim.
#
# Nothing is crafted here. The saved perturbation is written into the training
# image once, before training starts, and the augmentation is resampled on top of
# it every epoch (defense.py -> train_victim_defended -> victim_aug.py). The
# victim augments a poisoned dataset; it never sees the perturbation.
#
# MODEL / ATTACK / CLASS_PAIR are LISTS: the sweep is their full cross product,
# and each combo is configured from sweep_config.json on its own (the difficulty
# label differs per combo). A combo that is not in sweep_config.json, or whose
# budgets have no paired poisons, is skipped and reported at the end rather than
# killing the sweep.
#
# Within a combo, for each budget this script
#
#   1. asks which targets each base selection has saved poisons for,
#   2. intersects those lists and pins the intersection with --target_idx_file,
#      so every selection is judged on exactly the same target images, and
#   3. runs each AUG on each SELECTION over that pinned set.
#
# AUGS=none is the control: it is the same code path as the undefended attack
# runs, so it reproduces their numbers and gives the row every augmented row is
# read against. defense.sh with DEFENSES=none writes that same row, so if you
# have already run that, you do not need to pay for it again here.
#
#   AUGS="none standard" SELS="random ours" BUDGETS=0.005 sh aug.sh
#   MODEL=VGG13BN ATTACK=gradmatch sh aug.sh
#   DRY_RUN=1 sh aug.sh          # print the commands instead of running them

MODEL="${MODEL:-ResNet20BN}"
ATTACK="${ATTACK:-fc gradmatch}"
CLASS_PAIR="${CLASS_PAIR:-dog-bird frog-airplane}"
BUDGETS="${BUDGETS:-0.002 0.005 0.02}"

# Base selections to compare. random | ours | dpp  (dpp = ours + --sel_dpp)
SELS="${SELS:-random dpp}"
SEL_ALPHA="${SEL_ALPHA:-2.0}"

# Victim-training augmentations. Anything victim_aug.py accepts:
#   none      no augmentation (the protocol the attacks were crafted against)
#   standard  RandomCrop(32, padding=4) + RandomHorizontalFlip(0.5)
#   randaug   standard, then RandAugment(num_ops=2, magnitude=9)
#   cutout    standard, then one random 16x16 Cutout region
#   dsa       the DiffAugment strategy that was already wired up
AUGS="${AUGS:-standard randaug cutout}"

# Defense to run underneath. 'none' isolates the augmentation; set it to epic /
# friends / ... to ask what augmentation adds on top of a real defense.
DEFENSE="${DEFENSE:-none}"

DATASET=CIFAR10
DATA_PATH=/home/mmoslem3/scratch/data
OUT_DIR=ours_result           # where the attack runs (and their poisons) live
DEF_OUT_DIR=defense_result    # where this writes
CACHE_DIR=./cache
SEED=42
EPSILON=0.0313725             # 8/255, matches the attack runs

NUM_VICTIMS="${NUM_VICTIMS:-6}"
VICTIM_EPOCHS="${VICTIM_EPOCHS:-50}"
VICTIM_LR=0.1
VICTIM_BS=125
VICTIM_DECAY=40

source /home/mmoslem3/ENV/bin/activate
cd /home/mmoslem3/scratch/attack_if

mkdir -p target_sets "$DEF_OUT_DIR"

sel_flags() {
    case "$1" in
        random) echo "--base random" ;;
        ours)   echo "--base ours --base_dist cosine --lambda_margin 1.0" ;;
        dpp)    echo "--base ours --base_dist cosine --lambda_margin 1.0 --sel_dpp --sel_alpha $SEL_ALPHA" ;;
        *)      echo "UNKNOWN_SEL" ;;
    esac
}

echo "=== augmentation sweep ==="
echo "    models     : $MODEL"
echo "    attacks    : $ATTACK"
echo "    pairs      : $CLASS_PAIR"
echo "    budgets    : $BUDGETS"
echo "    selections : $SELS   (dpp alpha $SEL_ALPHA)"
echo "    augs       : $AUGS   (under defense $DEFENSE)"
echo "    poisons    : replayed from $OUT_DIR, results into $DEF_OUT_DIR"
echo

SKIPPED=""
NRUN=0

for model in $MODEL; do
for attack in $ATTACK; do
for pair in $CLASS_PAIR; do

    # ---- this combo's difficulty label ------------------------------------
    # It only has to match the attack run so the _tgt<N> suffix of the run name
    # lines up. Straight from sweep_config.json, same as defense.sh.
    CFG="$(python - "$model" "$attack" "$pair" <<'PY'
import json, sys
model, attack, pair = sys.argv[1:4]
cfg = json.load(open('sweep_config.json'))
try:
    print('CFG_TGT=%s' % cfg['difficulty'][model][attack][pair])
except KeyError:
    sys.exit('sweep_config.json has no difficulty for %s / %s / %s -- add it '
             'there (and say where it came from) rather than guessing here.'
             % (model, attack, pair))
PY
)" || { SKIPPED="$SKIPPED
    $model / $attack / $pair : not in sweep_config.json"; continue; }
    eval "$CFG"

    # ---- pair the selections: one pinned target set per budget -------------
    # build_run_name and the cache reader come from the real modules, so this
    # can never drift from what defense.py itself will look for.
    PLAN="$(python - "$model" "$attack" "$pair" "$CFG_TGT" "$SEED" "$EPSILON" \
                     "$OUT_DIR" "$SEL_ALPHA" "$SELS" "$BUDGETS" <<'PY'
import argparse, json, os, sys
import final_update as FU
import defense as DEF

(model, attack, pair, tgt, seed, eps, out_dir, sel_alpha,
 sels, budgets) = sys.argv[1:11]
seed, eps, sel_alpha = int(seed), float(eps), float(sel_alpha)
sels, budgets = sels.split(), [float(b) for b in budgets.split()]

def ns(base, budget, dpp):
    return argparse.Namespace(
        dataset='CIFAR10', model=model, attack=attack, base=base,
        class_pair=pair, budget=budget, epsilon=eps, seed=seed,
        lambda_margin=1.0, base_dist='cosine',
        sel_filter=False, sel_pca=False, sel_mmr=False, sel_dpp=dpp,
        sel_pool=3.0, sel_mu=1.0, sel_alpha=sel_alpha,
        fc_mode='sample', sharp_mode='worst', sharp_sigma=0.05,
        sharp_samples=20, craft_ensemble=5, target_select=int(tgt))

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
    path = 'target_sets/aug_%s_%s_%s_b%g.json' % (model, attack, pair, b)
    with open(path, 'w') as f:
        json.dump({'_generated_by': 'aug.sh -- intersection of the targets '
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
)" || { SKIPPED="$SKIPPED
    $model / $attack / $pair : planning failed"; continue; }

    echo "=== $model / $attack / $pair (tgt$CFG_TGT) ==="
    if [ -z "$PLAN" ]; then
        echo "    no budget has poisons under every selection -- skipped"
        echo
        SKIPPED="$SKIPPED
    $model / $attack / $pair : no paired poisons for any budget"
        continue
    fi

    for entry in $PLAN; do
        bug="${entry%%:*}"
        idx="${entry#*:}"
        for aug in $AUGS; do
            for sel in $SELS; do
                flags="$(sel_flags "$sel")"
                [ "$flags" = "UNKNOWN_SEL" ] && { echo "unknown SEL '$sel'"; exit 1; }
                echo "--- $model/$attack/$pair | budget $bug | aug $aug | selection $sel ---"
                NRUN=$((NRUN + 1))
                ${DRY_RUN:+echo} python defense.py \
                    --dataset "$DATASET" --data_path "$DATA_PATH" --seed "$SEED" \
                    --cache_dir "$CACHE_DIR" --out_dir "$OUT_DIR" \
                    --defense_out_dir "$DEF_OUT_DIR" \
                    --model "$model" --attack "$attack" $flags \
                    --class_pair "$pair" --pair_order poison-target \
                    --budget "$bug" --epsilon "$EPSILON" \
                    --craft_ensemble 5 --target_select "$CFG_TGT" \
                    --target_idx_file "$idx" \
                    --defense "$DEFENSE" --victim_aug "$aug" \
                    --num_victims "$NUM_VICTIMS" --victim_epochs "$VICTIM_EPOCHS" \
                    --victim_lr "$VICTIM_LR" --victim_bs "$VICTIM_BS" \
                    --victim_decay "$VICTIM_DECAY" --victim_wd 0.0 \
                    --clean_baseline
            done
        done
    done
    echo

done
done
done

echo "=== sweep finished: $NRUN run(s) ==="
if [ -n "$SKIPPED" ]; then
    echo "skipped combos:$SKIPPED"
fi

# --no_resume is deliberately not passed: a shard that dies can just be rerun and
# picks up from results.csv where it stopped.
