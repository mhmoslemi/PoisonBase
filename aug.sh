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

MODEL="${MODEL:-ConvNetBN}"
ATTACK="${ATTACK:-fc gradmatch}"
CLASS_PAIR="${CLASS_PAIR:-dog-bird frog-airplane}"
BUDGETS="${BUDGETS:-0.002 0.005 0.02}"

# Base selections to RUN. random | ours | dpp  (dpp = ours + --sel_dpp)
SELS="${SELS:-random dpp}"
SEL_ALPHA="${SEL_ALPHA:-2.0}"

# Base selections to PIN THE TARGETS OVER. The targets are the intersection of
# what these selections have poisons for, so every row of the table is scored on
# the same images. This is deliberately separate from SELS: the per-row shards
# (aug01.sh ...) run ONE selection each, but must still be pinned to the
# intersection over ALL of them, or the Random row and the DPP row would each
# pick their own targets and stop being comparable.
PAIR_SELS="${PAIR_SELS:-$SELS}"

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
# which craft-time augmentation the poisons being replayed were built with. Empty
# is the default DSA strategy, i.e. the main-sweep crafts.
CRAFT_AUG="${CRAFT_AUG:-}"
DSA_STRATEGY="${DSA_STRATEGY:-}"

# Which target-selection protocol the poisons being replayed were crafted under.
# Empty is the main sweep: the difficulty degree comes from sweep_config.json and
# the run name carries a _tgt<N> suffix. The appendix crafts sample targets
# uniformly instead (--target_select random, no suffix), so set TARGET_SELECT=random
# to replay those -- otherwise build_run_name below looks for a _tgt<N> directory
# that was never created and the combo is skipped as "no paired poisons".
TARGET_SELECT="${TARGET_SELECT:-}"

# How many of the pinned targets to actually run. 0 = all of them.
# defense.py takes the FIRST n of the pinned list, and that list is the sorted
# intersection written to target_sets/aug_<combo>_b<budget>.json, so the same n
# targets are used by every selection and every augmentation -- and the No Aug.
# column can be recomputed from the existing attack logs by filtering them to
# those same target indices and to victim_id < NUM_VICTIMS.
NUM_TARGETS="${NUM_TARGETS:-0}"
VICTIM_EPOCHS="${VICTIM_EPOCHS:-50}"
VICTIM_LR=0.1
VICTIM_BS=125
VICTIM_DECAY=40

source /home/mmoslem3/ENV/bin/activate
cd /home/mmoslem3/scratch/attack_if

# ---- preflight: fail fast if this shell cannot actually see a gpu -----------
# defense.py's resolve_gpus() falls back to the cpu when it finds no gpu, and a
# cpu victim is ~100x slower, so a login-node launch would sit there looking
# like it is working for days. Every shard execs this file, so checking once
# here covers all 16. Also catches a gpu that is visible but unusable (someone
# else's process holding all the memory, a broken NVML, a stale allocation).
python - <<'PY' || exit 1
import sys
try:
    import torch
except Exception as e:
    sys.exit('preflight FAILED: cannot import torch: %s' % e)
if not torch.cuda.is_available():
    sys.exit('preflight FAILED: CUDA is not available. This is almost certainly '
             'a login node -- get a gpu allocation (salloc / srun --jobid=... '
             '--overlap) and rerun. Refusing to fall back to the cpu.')
n = torch.cuda.device_count()
if n < 1:
    sys.exit('preflight FAILED: torch.cuda.is_available() is True but '
             'device_count() is 0')
try:
    x = torch.zeros(2048, 2048, device='cuda:0')
    float((x + 1).sum().item())
    del x
    free, total = torch.cuda.mem_get_info(0)
except Exception as e:
    sys.exit('preflight FAILED: gpu 0 is visible but unusable: %s' % e)
if free < 3 * 2 ** 30:
    sys.exit('preflight FAILED: only %.1f GiB free on gpu 0; a victim needs a '
             'few GiB. Something else is using this gpu.' % (free / 2 ** 30))
print('preflight ok: %d gpu(s), %s, %.1f/%.1f GiB free'
      % (n, torch.cuda.get_device_name(0), free / 2 ** 30, total / 2 ** 30))
PY

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
echo "    targets    : intersection over '$PAIR_SELS'${NUM_TARGETS:+, first $NUM_TARGETS}"
echo "    victims    : $NUM_VICTIMS"
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
    if [ -n "$TARGET_SELECT" ]; then
        CFG_TGT="$TARGET_SELECT"
    else
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
    fi

    # ---- pair the selections: one pinned target set per budget -------------
    # build_run_name and the cache reader come from the real modules, so this
    # can never drift from what defense.py itself will look for.
    PLAN="$(python - "$model" "$attack" "$pair" "$CFG_TGT" "$SEED" "$EPSILON" \
                     "$OUT_DIR" "$SEL_ALPHA" "$PAIR_SELS" "$BUDGETS" <<'PY'
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
        sharp_samples=20, craft_ensemble=5,
        target_select=(int(tgt) if tgt.lstrip('-').isdigit() else tgt),
        craft_aug=(os.environ.get('CRAFT_AUG', '') != '--no_craft_aug'),
        dsa_strategy=(os.environ.get('DSA_STRATEGY') or FU.DSA_DEFAULT))

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
    # keyed on the combo, not on the selection, so the per-row shards all pin the
    # SAME file -- write it atomically, since several of them run at once
    blob = {'_generated_by': 'aug.sh -- intersection of the targets every base '
                             'selection in PAIR_SELS has saved poisons for',
            '_combo': '%s / %s / %s / b%g' % (model, attack, pair, b),
            '_pair_sels': ' '.join(sels),
            '_per_selection': ' '.join(report),
            'pairs': {pair: {'indices': have}}}
    tmp = '%s.%d.tmp' % (path, os.getpid())
    with open(tmp, 'w') as f:
        json.dump(blob, f, indent=1)
    os.replace(tmp, path)
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
                    ${CRAFT_AUG:+$CRAFT_AUG} \
                    ${DSA_STRATEGY:+--dsa_strategy "$DSA_STRATEGY"} \
                    --defense "$DEFENSE" --victim_aug "$aug" \
                    --num_targets "$NUM_TARGETS" \
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
