#!/usr/bin/env bash
#
# Base-selection sweep driver.
#
# SELECT picks how the N_p bases are chosen; everything else (crafting, victims,
# surrogates) is identical across the five, so the comparison is paired:
#
#   random   uniform over the poison class          -> --base random
#   ours     lowest standardized d(x) + lam*M(x),   -> --base ours   (no --sel_*)
#            the plain greedy top-N_p by score
#   dpp      greedy log-det (DPP MAP) with quality  -> --base ours --sel_dpp
#            q_i = exp(-SEL_ALPHA * score_i). Small alpha = more diversity,
#            large alpha reproduces plain 'ours'.
#   exact    exact full-parameter g_i^T g_t         -> --base ours
#                                                    --sel_exact_alignment
#   a-mr     A_i + (-M_i) * R_i, using the paper's  -> --base ours
#            A, M, and R definitions                  --sel_a_minus_mr
# The exact mode standardizes each surrogate before averaging. A-MR follows the
# paper: average each component over surrogates, standardize A/M/R across the
# candidate pool, and then form A + (-M)*R.
#
# Set USE_JACOBIAN_SCORE=1 to add the exact standardized backbone-gradient
# interaction to the pointwise quality cost for ours or dpp. JACOBIAN_WEIGHT is
# beta and JACOBIAN_BATCH_SIZE controls memory without changing run identity:
#
#     USE_JACOBIAN_SCORE=1 JACOBIAN_WEIGHT=1.0 SELECT=dpp sh sel_dpp.sh
#
# SELECT is the authority: SEL_ALPHA is read ONLY when SELECT=dpp. Under
# random / ours / exact / a-mr it is ignored and does not appear in the run
# name, so setting it cannot silently change a non-dpp run.
#
# Runs final_update.py (final.py is untouched).
#
# Targets: if target_sets/<MODEL>_<ATTACK>_<PAIR>.json exists it is pinned with
# --target_idx_file, i.e. the exact 10 test images the random-base run for this
# combo attacked, and TARGET_SELECT is ignored -- a combo can never silently
# change targets once it has been pinned. If the file does not exist this is the
# first run for the combo, and the targets are selected by difficulty degree:
# TARGET_SELECT if you set it, otherwise the combo's label in sweep_config.json.
# Either way the degree also names the run dir (_tgt<N>), so with a pinned file
# the combo's own label is used there and not whatever TARGET_SELECT holds.
# The craft-memory flags come from sweep_config.json, so each combo is set up
# exactly the way ours.sh sets it up. Nothing is guessed.
#
# ATTACK=sapa is gradmatch plus a sharpness-aware target gradient, so it has the
# same difficulty label, the same memory profile and the same target set. Both
# lookups therefore fall back to the gradmatch entry (reported on the run header),
# which is also what makes sapa vs gradmatch a paired comparison on identical
# target images. SHARP_MODE / SHARP_SIGMA are read ONLY when ATTACK=sapa.
#
# MODEL, ATTACK, CLASS_PAIR, BUDGETS, SELECT and SHARP_SIGMA each take ONE OR
# MORE whitespace-separated values and are swept as nested loops, so a single
# call can cover the whole grid:
#
#     sh sel_dpp.sh
#     MODEL=VGG13BN CLASS_PAIR="dog-bird frog-airplane" sh sel_dpp.sh
#     SELECT="random ours dpp" MODEL="ResNet20BN VGG13BN" ATTACK="fc gradmatch" \
#         BUDGETS="0.001 0.002 0.005" sh sel_dpp.sh
#     SELECT=exact JACOBIAN_BATCH_SIZE=64 sh sel_dpp.sh
#     SELECT=a-mr JACOBIAN_BATCH_SIZE=64 sh sel_dpp.sh
#     ATTACK=sapa SHARP_SIGMA="0.01 0.05 0.1" SELECT=dpp sh sel_dpp.sh
#     TARGET_SELECT=30 MODEL=VGG13BN ATTACK=sapa sh sel_dpp.sh   # first run of a
#         combo: pick its 10 targets at difficulty 30. Once target_sets/ has the
#         file, the same call reuses those 10 and TARGET_SELECT does nothing.
#
# A combo missing from sweep_config.json is reported and skipped, so one hole
# does not kill a long sweep.


# MODELS=(ConvNetBN VGG13BN ResNet20BN)
# ATTACKS=(fc gradmatch sapa)
# BASES=(random ours)
# CLASS_PAIRS=(dog-bird frog-airplane)

MODEL="${MODEL:-VGG13BN}"
ATTACK="${ATTACK:-fc}"
CLASS_PAIR="${CLASS_PAIR:-frog-airplane}"
# BUDGETS="${BUDGETS:-0.002 0.005 0.02 0.001 0.01 0.04}"
BUDGETS="${BUDGETS:-0.001}"
SELECT="${SELECT:-dpp}"

# BUDGETS="${BUDGETS:-0.001 0.002 0.005 0.01 0.02 0.04}"

SEL_ALPHA="${SEL_ALPHA:-2.0}"        # SELECT=dpp only
USE_JACOBIAN_SCORE="${USE_JACOBIAN_SCORE:-1}"
JACOBIAN_WEIGHT="${JACOBIAN_WEIGHT:-1.0}"
JACOBIAN_BATCH_SIZE="${JACOBIAN_BATCH_SIZE:-64}"
case "$USE_JACOBIAN_SCORE" in
    0|1) ;;
    *) echo "USE_JACOBIAN_SCORE=$USE_JACOBIAN_SCORE (expected: 0 or 1)"; exit 1 ;;
esac

# Difficulty degree to select targets with the FIRST time a combo is run, i.e.
# when target_sets/<MODEL>_<ATTACK>_<PAIR>.json does not exist yet. 0..100
# (0 = easiest, 100 = hardest) or easiest | hardest | random | first. Once that
# file exists the pinned 10 images win and this is ignored, so re-running a combo
# with a different TARGET_SELECT can never silently swap its targets.
# Empty -> fall back to the combo's difficulty label in sweep_config.json.
# TARGET_SELECT="${TARGET_SELECT:-70}"

SHARP_MODE="${SHARP_MODE:-worst}"    # ATTACK=sapa only: worst | avg
SHARP_SIGMA="${SHARP_SIGMA:-0.05}"   # ATTACK=sapa only. worst: l2 radius (SAM rho).
                                     # avg: PER-ELEMENT std, use ~1e-3 there.

DATASET="${DATASET:-CIFAR10}"
DATA_PATH="${DATA_PATH:-/home/mmoslem3/scratch/data}"
OUT_DIR="${OUT_DIR:-ours_result}"
CACHE_DIR="${CACHE_DIR:-./cache}"
SEED="${SEED:-42}"
PROJECT_ROOT="${PROJECT_ROOT:-/home/mmoslem3/scratch/attack_if}"
PYTHON_ENV="${PYTHON_ENV:-/home/mmoslem3/ENV}"
NUM_TARGETS="${NUM_TARGETS:-8}"
NUM_VICTIMS="${NUM_VICTIMS:-5}"
RECOMPUTE_DELTAS="${RECOMPUTE_DELTAS:-0}"
case "$RECOMPUTE_DELTAS" in
    0) RECOMPUTE_FLAGS="" ;;
    1) RECOMPUTE_FLAGS="--recompute_deltas" ;;
    *) echo "RECOMPUTE_DELTAS=$RECOMPUTE_DELTAS (expected: 0 or 1)"; exit 1 ;;
esac

source "$PYTHON_ENV/bin/activate"
cd "$PROJECT_ROOT"

# --- refuse to start on a node with no GPU ------------------------------------
# klogin* has no CUDA driver. resolve_gpus() returns [] when torch.cuda.is_available()
# is False, so final_update.py SILENTLY falls back to cpu and then gets OOM-killed
# on any real budget instead of telling you why. Fail in 2 s instead of 20 min.
if [ -z "$ALLOW_CPU" ]; then
    python -c "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)" || {
        echo "!! no CUDA device visible on $(hostname) -- refusing to start."
        echo "   final_update.py would fall back to cpu and be OOM-killed on big budgets."
        echo "   run this inside an allocation, e.g.:"
        echo "     salloc --account=aip-boyuwang --gres=gpu:l40s:1 --cpus-per-task=4 \\"
        echo "            --mem=32G --time=12:00:00"
        echo "   or attach to one you already hold:   srun --jobid=<id> --pty bash"
        echo "   (ALLOW_CPU=1 bypasses this check)"
        exit 1
    }
fi

# reject a bad SELECT once, up front, instead of after hours of surrogate training
for sel in $SELECT; do
    case "$sel" in
        random|ours|dpp|exact|a-mr) ;;
        *) echo "unknown SELECT=$sel (expected: random | ours | dpp | exact | a-mr)"; exit 1 ;;
    esac
done

# same for TARGET_SELECT -- final_update.py takes 0..100 or one of the four words
case "$TARGET_SELECT" in
    ''|easiest|hardest|random|first) ;;
    *[!0-9]*) echo "unknown TARGET_SELECT=$TARGET_SELECT (expected: 0..100 |" \
                   "easiest | hardest | random | first)"; exit 1 ;;
    *) [ "$TARGET_SELECT" -le 100 ] || { echo "TARGET_SELECT=$TARGET_SELECT out of range 0..100"; exit 1; } ;;
esac

for model in $MODEL; do
for attack in $ATTACK; do
for pair in $CLASS_PAIR; do

    # --- difficulty label + craft-memory flags, straight from sweep_config.json -
    CFG="$(python - "$model" "$attack" "$pair" <<'PY'
import json, sys
model, attack, pair = sys.argv[1:4]
cfg = json.load(open('sweep_config.json'))
# sapa is gradmatch + a sharpness-aware target gradient: same crafting cost, same
# difficulty, so it reads the gradmatch entry rather than needing its own.
key = 'gradmatch' if attack == 'sapa' else attack
try:
    tgt = cfg['difficulty'][model][key][pair]
except KeyError:
    sys.exit('sweep_config.json has no difficulty for %s / %s / %s' % (model, key, pair))
mem = cfg['memory'].get(model, {}).get(key, cfg['memory_default'])
print('CFG_TGT=%s' % tgt)
print('CFG_KEY=%s' % key)
print("CFG_MEM='%s'" % ('--craft_lowmem --craft_batch 256 --fast_gradmatch'
                        if mem['craft_lowmem'] else ''))
PY
    )" || { echo "!! skipping $model / $attack / $pair"; echo; continue; }
    eval "$CFG"

    # --- targets: pinned if the file is there, difficulty degree otherwise -----
    # sapa falls back to the gradmatch target set, so the two attacks are compared
    # on the identical 10 images
    IDX="target_sets/${model}_${attack}_${pair}.json"
    if [ ! -s "$IDX" ] && [ "$attack" != "$CFG_KEY" ]; then
        IDX="target_sets/${model}_${CFG_KEY}_${pair}.json"
    fi
    # TGT_DEG is what --target_select gets. With a pinned file the selector never
    # runs, but the degree still names the run dir (_tgt<N>), so keep the combo's
    # own label there rather than whatever TARGET_SELECT happens to be set to.
    if [ -s "$IDX" ]; then
        TGT_FLAGS="--target_idx_file $IDX"
        TGT_DEG="$CFG_TGT"
        TGT_NOTE="pinned from $IDX (same 10 the random-base run attacked)"
        [ -n "$TARGET_SELECT" ] && \
            TGT_NOTE="$TGT_NOTE; TARGET_SELECT=$TARGET_SELECT ignored -- combo already pinned"
    elif [ -n "$TARGET_SELECT" ]; then
        TGT_FLAGS=""
        TGT_DEG="$TARGET_SELECT"
        TGT_NOTE="no pinned set -- first run for this combo, selecting by TARGET_SELECT=$TARGET_SELECT"
    else
        TGT_FLAGS=""
        TGT_DEG="$CFG_TGT"
        TGT_NOTE="no pinned set found -- selecting by difficulty degree tgt$CFG_TGT"
    fi

for sel in $SELECT; do

    # --- the one knob that differs between the five selections ----------------
    case "$sel" in
        random) BASE=random; SEL_FLAGS="";                              SEL_NOTE="random" ;;
        ours)   BASE=ours;   SEL_FLAGS="";                              SEL_NOTE="ours (plain greedy top-N_p by score)" ;;
        dpp)    BASE=ours;   SEL_FLAGS="--sel_dpp --sel_alpha $SEL_ALPHA"; SEL_NOTE="dpp (alpha=$SEL_ALPHA)" ;;
        exact)  BASE=ours;   SEL_FLAGS="--sel_exact_alignment --jacobian_batch_size $JACOBIAN_BATCH_SIZE"; SEL_NOTE="exact full-parameter g_i^T g_t (per-surrogate standardized)" ;;
        a-mr)   BASE=ours;   SEL_FLAGS="--sel_a_minus_mr --jacobian_batch_size $JACOBIAN_BATCH_SIZE"; SEL_NOTE="A - MR: standardized A + (-M)*R" ;;
    esac
    JACOBIAN_FLAGS=""
    JACOBIAN_NOTE="Jacobian score disabled"
    if [ "$sel" = "exact" ]; then
        JACOBIAN_NOTE="exact selector uses full-parameter gi^T gt (batch=$JACOBIAN_BATCH_SIZE)"
    elif [ "$sel" = "a-mr" ]; then
        JACOBIAN_NOTE="A - MR uses paper A/M/R (batch=$JACOBIAN_BATCH_SIZE); average then standardize, then A + (-M)*R"
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
    # so setting SHARP_SIGMA can never duplicate an fc / gradmatch run
    if [ "$attack" = "sapa" ]; then SIGMAS="$SHARP_SIGMA"; else SIGMAS="-"; fi

for sig in $SIGMAS; do

    if [ "$attack" = "sapa" ]; then
        SHARP_FLAGS="--sharp_mode $SHARP_MODE --sharp_sigma $sig"
        SHARP_NOTE=" | sharp $SHARP_MODE sigma=$sig"
    else
        SHARP_FLAGS=""
        SHARP_NOTE=""
    fi

    echo "=== $SEL_NOTE | $model / $attack / $pair$SHARP_NOTE ==="
    echo "    targets: $TGT_NOTE"
    echo "    difficulty label tgt$TGT_DEG   craft flags: ${CFG_MEM:-none}"
    echo "    $JACOBIAN_NOTE"
    echo "    budgets: $BUDGETS"
    echo

    for bug in $BUDGETS; do
        echo "--- $sel | $model / $attack / $pair$SHARP_NOTE | budget $bug ---"
        python final_update.py \
            --dataset "$DATASET" --data_path "$DATA_PATH" --seed "$SEED" \
            --cache_dir "$CACHE_DIR" --out_dir "$OUT_DIR" \
            --model "$model" --attack "$attack" --base "$BASE" \
            --class_pair "$pair" --pair_order poison-target \
            --budget "$bug" --epsilon 0.0313725 \
            --craft_steps 250 --craft_alpha 0.0039216 \
            --restarts 8 --craft_ensemble 5 $CFG_MEM \
            --base_dist cosine --lambda_margin 1.0 \
            $SEL_FLAGS $JACOBIAN_FLAGS $SHARP_FLAGS \
            --num_surrogates 20 --surrogate_epochs 60 --surrogate_decay 35 45 \
            --num_targets "$NUM_TARGETS" --target_select "$TGT_DEG" \
            $TGT_FLAGS \
            $RECOMPUTE_FLAGS \
            --num_victims "$NUM_VICTIMS" --victim_epochs 50 --victim_lr 0.1 --victim_bs 125 \
            --victim_decay 40 --victim_wd 0.0 \
            --clean_baseline
    done
    echo

done
done
done
done
done

# By default neither --no_resume nor --recompute_deltas is passed, so an
# interrupted shard resumes. Repair jobs may set RECOMPUTE_DELTAS=1 when the
# result CSV survived but its poison_cache did not.
