#!/usr/bin/env bash
#
# Base-selection TIMING driver -- companion to sel_dpp.sh.
#
# sel_dpp.sh measures ASR/CTA end to end (crafting + victim training included).
# This script measures ONLY the base-selection step: how long it takes to
# compute the per-candidate quantities (feature distance d, margin M, the
# Jacobian backbone-gradient interaction A, the raw representation inner
# product R -- whichever SELECT actually needs) and how long it takes to sort
# / rank the candidate pool from them, plus the peak GPU memory that step
# uses. No poison is ever crafted and no victim is ever trained.
#
# SELECT is the same selector menu as sel_dpp.sh (see there for exact
# per-selector formulas). For MODEL/ATTACK/CLASS_PAIR/BUDGETS this script
# picks the SAME --num_targets real targets a normal run would (pinned file if
# target_sets/ already has one, else the combo's difficulty degree from
# sweep_config.json, exactly like sel_dpp.sh), then times each selector
# TIME_REPEATS times per target. With the defaults (10 targets x 10 repeats)
# that is 100 timed selections per (model, attack, pair, select, budget) combo.
#
# Runs final_update_time.py --time_mode (final_update.py and final.py are both
# untouched). final_update_time.py's --time_mode path still trains/loads the
# real surrogate ensemble -- selection needs real nets -- it just never crafts
# a poison or trains a victim, so a combo that already has cached surrogates in
# CACHE_DIR times in seconds instead of the hours a full sel_dpp.sh run takes.
#
# Per (model, attack, pair, select, budget) combo this prints, and saves to
# <OUT_DIR>/TIME_<run name>/timing.json:
#   compute time  (the expensive per-candidate forward/backward passes)
#   sort time     (combining the standardized components + topk / greedy DPP)
#   end-to-end    (compute + sort)
#   peak GPU mem
# each as mean +/- std over the TIME_REPEATS x --num_targets measured calls.
#
# Usage, same env-var sweep style as sel_dpp.sh:
#
#     sh sel_dpp_time.sh
#     SELECT="ours dpp exact a-mr" MODEL=VGG13BN sh sel_dpp_time.sh
#     SELECT="minus-m r a a-minus-m a-plus-r minus-m-times-r" sh sel_dpp_time.sh
#     NUM_TARGETS=10 TIME_REPEATS=20 SELECT=dpp sh sel_dpp_time.sh
#     BUDGETS="0.001 0.01 0.04" SELECT=dpp sh sel_dpp_time.sh   # N_p scaling
#

MODEL="${MODEL:-VGG13BN}"
ATTACK="${ATTACK:-fc}"
CLASS_PAIR="${CLASS_PAIR:-frog-airplane}"
BUDGETS="${BUDGETS:-0.001}"
SELECT="${SELECT:-dpp}"

SEL_ALPHA="${SEL_ALPHA:-2.0}"        # SELECT=dpp only
USE_JACOBIAN_SCORE="${USE_JACOBIAN_SCORE:-1}"
JACOBIAN_WEIGHT="${JACOBIAN_WEIGHT:-1.0}"
JACOBIAN_BATCH_SIZE="${JACOBIAN_BATCH_SIZE:-64}"
case "$USE_JACOBIAN_SCORE" in
    0|1) ;;
    *) echo "USE_JACOBIAN_SCORE=$USE_JACOBIAN_SCORE (expected: 0 or 1)"; exit 1 ;;
esac

DATASET="${DATASET:-CIFAR10}"
DATA_PATH="${DATA_PATH:-/home/mmoslem3/scratch/data}"
OUT_DIR="${OUT_DIR:-ours_result}"
CACHE_DIR="${CACHE_DIR:-./cache}"
SEED="${SEED:-42}"
PROJECT_ROOT="${PROJECT_ROOT:-/home/mmoslem3/scratch/attack_if}"
PYTHON_ENV="${PYTHON_ENV:-/home/mmoslem3/ENV}"

# how many real targets to select and how many timed repeats per target --
# defaults give 10 x 10 = 100 measured selections per combo.
NUM_TARGETS="${NUM_TARGETS:-10}"
TIME_REPEATS="${TIME_REPEATS:-10}"

# Difficulty degree to select targets with the FIRST time a combo is run, i.e.
# when target_sets/<MODEL>_<ATTACK>_<PAIR>.json does not exist yet. 0..100
# (0 = easiest, 100 = hardest) or easiest | hardest | random | first. Once that
# file exists the pinned targets win and this is ignored -- so a timing run
# always measures selection over the SAME images a real sel_dpp.sh run would.
# Empty -> fall back to the combo's difficulty label in sweep_config.json.
# TARGET_SELECT="${TARGET_SELECT:-70}"

SHARP_MODE="${SHARP_MODE:-worst}"    # ATTACK=sapa only: worst | avg
SHARP_SIGMA="${SHARP_SIGMA:-0.05}"   # ATTACK=sapa only, same meaning as sel_dpp.sh

source "$PYTHON_ENV/bin/activate"
cd "$PROJECT_ROOT"

# --- refuse to start on a node with no GPU ------------------------------------
# klogin* has no CUDA driver. resolve_gpus() returns [] when torch.cuda.is_available()
# is False, so final_update_time.py SILENTLY falls back to cpu -- a "timing" that
# ran on cpu is not the number anyone wants. Fail in 2 s instead of a slow run.
if [ -z "$ALLOW_CPU" ]; then
    python -c "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)" || {
        echo "!! no CUDA device visible on $(hostname) -- refusing to start."
        echo "   final_update_time.py would fall back to cpu and the timing would be"
        echo "   meaningless. run this inside an allocation, e.g.:"
        echo "     salloc --account=aip-boyuwang --gres=gpu:l40s:1 --cpus-per-task=4 \\"
        echo "            --mem=32G --time=02:00:00"
        echo "   or attach to one you already hold:   srun --jobid=<id> --pty bash"
        echo "   (ALLOW_CPU=1 bypasses this check, for a deliberate cpu timing)"
        exit 1
    }
fi

# reject a bad SELECT once, up front, instead of after minutes of surrogate training
for sel in $SELECT; do
    case "$sel" in
        random|ours|dpp|exact|a-mr|minus-m|r|a|a-minus-m|a-plus-r|minus-m-times-r) ;;
        *) echo "unknown SELECT=$sel (expected: random | ours | dpp | exact | a-mr | minus-m | r | a | a-minus-m | a-plus-r | minus-m-times-r)"; exit 1 ;;
    esac
done

# same for TARGET_SELECT -- final_update_time.py takes 0..100 or one of the four words
case "$TARGET_SELECT" in
    ''|easiest|hardest|random|first) ;;
    *[!0-9]*) echo "unknown TARGET_SELECT=$TARGET_SELECT (expected: 0..100 |" \
                   "easiest | hardest | random | first)"; exit 1 ;;
    *) [ "$TARGET_SELECT" -le 100 ] || { echo "TARGET_SELECT=$TARGET_SELECT out of range 0..100"; exit 1; } ;;
esac

for model in $MODEL; do
for attack in $ATTACK; do
for pair in $CLASS_PAIR; do

    # --- difficulty label, straight from sweep_config.json (same source as
    # sel_dpp.sh, so a timing run's targets line up with a real run's) --------
    CFG="$(python - "$model" "$attack" "$pair" <<'PY'
import json, sys
model, attack, pair = sys.argv[1:4]
cfg = json.load(open('sweep_config.json'))
# sapa is gradmatch + a sharpness-aware target gradient: same difficulty label,
# so it reads the gradmatch entry rather than needing its own.
key = 'gradmatch' if attack == 'sapa' else attack
try:
    tgt = cfg['difficulty'][model][key][pair]
except KeyError:
    sys.exit('sweep_config.json has no difficulty for %s / %s / %s' % (model, key, pair))
print('CFG_TGT=%s' % tgt)
print('CFG_KEY=%s' % key)
PY
    )" || { echo "!! skipping $model / $attack / $pair"; echo; continue; }
    eval "$CFG"

    # --- targets: pinned if the file is there, difficulty degree otherwise -----
    # sapa falls back to the gradmatch target set, so the two attacks are timed
    # on the identical images
    IDX="target_sets/${model}_${attack}_${pair}.json"
    if [ ! -s "$IDX" ] && [ "$attack" != "$CFG_KEY" ]; then
        IDX="target_sets/${model}_${CFG_KEY}_${pair}.json"
    fi
    if [ -s "$IDX" ]; then
        TGT_FLAGS="--target_idx_file $IDX"
        TGT_DEG="$CFG_TGT"
        TGT_NOTE="pinned from $IDX (same images sel_dpp.sh would attack)"
        [ -n "$TARGET_SELECT" ] && \
            TGT_NOTE="$TGT_NOTE; TARGET_SELECT=$TARGET_SELECT ignored -- combo already pinned"
    elif [ -n "$TARGET_SELECT" ]; then
        TGT_FLAGS=""
        TGT_DEG="$TARGET_SELECT"
        TGT_NOTE="no pinned set -- selecting by TARGET_SELECT=$TARGET_SELECT"
    else
        TGT_FLAGS=""
        TGT_DEG="$CFG_TGT"
        TGT_NOTE="no pinned set found -- selecting by difficulty degree tgt$CFG_TGT"
    fi

for sel in $SELECT; do

    # --- selector-specific flags, identical mapping to sel_dpp.sh --------------
    case "$sel" in
        random) BASE=random; SEL_FLAGS="";                              SEL_NOTE="random" ;;
        ours)   BASE=ours;   SEL_FLAGS="";                              SEL_NOTE="ours (plain greedy top-N_p by score)" ;;
        dpp)    BASE=ours;   SEL_FLAGS="--sel_dpp --sel_alpha $SEL_ALPHA"; SEL_NOTE="dpp (alpha=$SEL_ALPHA)" ;;
        exact)  BASE=ours;   SEL_FLAGS="--sel_exact_alignment --jacobian_batch_size $JACOBIAN_BATCH_SIZE"; SEL_NOTE="exact full-parameter g_i^T g_t (per-surrogate standardized)" ;;
        a-mr)   BASE=ours;   SEL_FLAGS="--sel_a_minus_mr --jacobian_batch_size $JACOBIAN_BATCH_SIZE"; SEL_NOTE="A - MR: standardized A + (-M)*R" ;;
        minus-m) BASE=ours; SEL_FLAGS="--sel_component minus-m --jacobian_batch_size $JACOBIAN_BATCH_SIZE"; SEL_NOTE="component ablation: -M" ;;
        r) BASE=ours; SEL_FLAGS="--sel_component r --jacobian_batch_size $JACOBIAN_BATCH_SIZE"; SEL_NOTE="component ablation: R" ;;
        a) BASE=ours; SEL_FLAGS="--sel_component a --jacobian_batch_size $JACOBIAN_BATCH_SIZE"; SEL_NOTE="component ablation: A" ;;
        a-minus-m) BASE=ours; SEL_FLAGS="--sel_component a-minus-m --jacobian_batch_size $JACOBIAN_BATCH_SIZE"; SEL_NOTE="component ablation: A-M" ;;
        a-plus-r) BASE=ours; SEL_FLAGS="--sel_component a-plus-r --jacobian_batch_size $JACOBIAN_BATCH_SIZE"; SEL_NOTE="component ablation: A+R" ;;
        minus-m-times-r) BASE=ours; SEL_FLAGS="--sel_component minus-m-times-r --jacobian_batch_size $JACOBIAN_BATCH_SIZE"; SEL_NOTE="component ablation: (-M)*R" ;;
    esac
    JACOBIAN_FLAGS=""
    JACOBIAN_NOTE="Jacobian score disabled"
    if [ "$sel" = "exact" ]; then
        JACOBIAN_NOTE="exact selector uses full-parameter gi^T gt (batch=$JACOBIAN_BATCH_SIZE)"
    elif [ "$sel" = "a-mr" ]; then
        JACOBIAN_NOTE="A - MR uses paper A/M/R (batch=$JACOBIAN_BATCH_SIZE); average then standardize, then A + (-M)*R"
    elif [ "$sel" = "minus-m" ] || [ "$sel" = "r" ] || \
         [ "$sel" = "a" ] || [ "$sel" = "a-minus-m" ] || \
         [ "$sel" = "a-plus-r" ] || [ "$sel" = "minus-m-times-r" ]; then
        JACOBIAN_NOTE="$SEL_NOTE; needed raw components are averaged over surrogates, then standardized (batch=$JACOBIAN_BATCH_SIZE)"
    elif [ "$USE_JACOBIAN_SCORE" = "1" ]; then
        case "$sel" in
            random)
                JACOBIAN_NOTE="Jacobian score not applicable to random; run unchanged"
                echo "    Jacobian score not applicable to SELECT=random; leaving random run unchanged"
                ;;
            *)
                JACOBIAN_FLAGS="--use_jacobian_score --jacobian_weight $JACOBIAN_WEIGHT --jacobian_batch_size $JACOBIAN_BATCH_SIZE"
                JACOBIAN_NOTE="Jacobian score enabled (weight=$JACOBIAN_WEIGHT, batch=$JACOBIAN_BATCH_SIZE)"
                ;;
        esac
    fi

    # sigma is a real loop for sapa and a single no-op pass for everything else,
    # so setting SHARP_SIGMA can never duplicate an fc / gradmatch timing
    if [ "$attack" = "sapa" ]; then SIGMAS="$SHARP_SIGMA"; else SIGMAS="-"; fi

for sig in $SIGMAS; do

    # sapa's sharpness knobs only change the CRAFTING target gradient, which
    # this script never runs -- kept only so the printed header matches
    # sel_dpp.sh's, not passed to final_update_time.py.
    if [ "$attack" = "sapa" ]; then
        SHARP_NOTE=" | sharp $SHARP_MODE sigma=$sig"
    else
        SHARP_NOTE=""
    fi

    echo "=== TIMING $SEL_NOTE | $model / $attack / $pair$SHARP_NOTE ==="
    echo "    targets: $TGT_NOTE ($NUM_TARGETS targets x $TIME_REPEATS repeats)"
    echo "    difficulty label tgt$TGT_DEG"
    echo "    $JACOBIAN_NOTE"
    echo "    budgets: $BUDGETS"
    echo

    for bug in $BUDGETS; do
        echo "--- $sel | $model / $attack / $pair$SHARP_NOTE | budget $bug ---"
        python final_update_time.py \
            --dataset "$DATASET" --data_path "$DATA_PATH" --seed "$SEED" \
            --cache_dir "$CACHE_DIR" --out_dir "$OUT_DIR" \
            --model "$model" --attack "$attack" --base "$BASE" \
            --class_pair "$pair" --pair_order poison-target \
            --budget "$bug" \
            --base_dist cosine --lambda_margin 1.0 \
            $SEL_FLAGS $JACOBIAN_FLAGS \
            --num_surrogates 20 --surrogate_epochs 60 --surrogate_decay 35 45 \
            --num_targets "$NUM_TARGETS" --target_select "$TGT_DEG" \
            $TGT_FLAGS \
            --time_mode --time_repeats "$TIME_REPEATS"
    done
    echo

done
done
done
done
done
