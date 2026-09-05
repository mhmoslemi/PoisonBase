#!/usr/bin/env bash
#
# Score-component SCATTER driver -- companion to sel_dpp.sh.
#
# sel_dpp.sh selects bases, crafts poisons and trains victims. This script runs
# the SAME config (same surrogates, same pinned targets, same run name) but
# stops right after the per-candidate score components are computed and plots
# them instead. For the proposed selector (SELECT=ours or SELECT=dpp) the
# pointwise score of candidate i is, per surrogate and then averaged,
#
#     std(d_i) + lambda * std(M_i)  -  beta * std(A_i)
#
# d = feature distance to the target (cosine here), M = logit margin toward
# y_adv (the boundary term), A = exact backbone-gradient interaction g_i^T g_t
# (the Jacobian term, beta = JACOBIAN_WEIGHT). The plot is, for a seeded random
# subset of SCATTER_POINTS candidates of the poison class,
#
#     x = std(d_i) + lambda * std(M_i)      (the score with A switched off)
#     y = std(A_i)
#
# one panel per target (scatter_target<t>), a grid of all targets
# (scatter_grid), and one pooled panel (scatter_pooled), plus the plotted
# points as CSV, the full-pool components as components.npz and a
# summary.json with Pearson / Spearman correlations. A is always computed here,
# whatever USE_JACOBIAN_SCORE says -- it is the y axis; USE_JACOBIAN_SCORE /
# JACOBIAN_WEIGHT only change the run name and the "score = x - beta y" note
# printed on the figure, so set them to the config you actually want to pair
# this plot with.
#
# Runs final_update_scatter.py --scatter_mode (final_update.py is untouched).
# Output goes to <OUT_DIR>/SCATTER_<run name>/ where <run name> is exactly the
# directory sel_dpp.sh would use for the same flags.
#
# Only SELECT=ours and SELECT=dpp are accepted: the other selectors (exact,
# a-mr, the component ablations) do not rank by d + lambda*M - beta*A, so the
# x axis would not be their score. Both share the same pointwise components;
# they differ only in the ranking step this script never runs.
#
# Targets are resolved exactly as in sel_dpp.sh: pinned from
# target_sets/<MODEL>_<ATTACK>_<PAIR>.json when it exists, else by difficulty
# degree (TARGET_SELECT, or the combo's label in sweep_config.json). The
# clean-victim pool is loaded from the cache to rank targets when there is no
# pinned file, same as a real run -- it is never used for anything else.
#
# Usage, same env-var sweep style as sel_dpp.sh:
#
#     sh sel_dpp_scatter.sh
#     SELECT=ours MODEL=VGG13BN CLASS_PAIR=dog-bird sh sel_dpp_scatter.sh
#     SCATTER_POINTS=2000 BUDGETS="0.001 0.01" sh sel_dpp_scatter.sh
#     USE_JACOBIAN_SCORE=0 SELECT=ours sh sel_dpp_scatter.sh   # pair with a no-A run
#

# MODELS=(ConvNetBN VGG13BN ResNet20BN)
# ATTACKS=(fc gradmatch sapa)
# BASES=(random ours)
# CLASS_PAIRS=(dog-bird frog-airplane)

MODEL="${MODEL:-ConvNetBN VGG13BN ResNet20BN}"
ATTACK="${ATTACK:-fc}"
# CLASS_PAIR: leave empty to get the per-DATASET default chosen below DATASET
# (dog-bird / frog-airplane only exist on CIFAR10). Set it explicitly to
# override, e.g. CLASS_PAIR="1-7 3-8" for SVHN or CLASS_PAIR="3-7" for CIFAR100.
CLASS_PAIR="${CLASS_PAIR:-}"
# BUDGETS="${BUDGETS:-0.002 0.005 0.02 0.001 0.01 0.04}"
BUDGETS="${BUDGETS:-0.02}"
SELECT="${SELECT:-dpp}"

# BUDGETS="${BUDGETS:-0.001 0.002 0.005 0.01 0.02 0.04}"

SEL_ALPHA="${SEL_ALPHA:-2.0}"        # SELECT=dpp only
USE_JACOBIAN_SCORE="${USE_JACOBIAN_SCORE:-1}"
JACOBIAN_WEIGHT="${JACOBIAN_WEIGHT:-1.0}"
JACOBIAN_BATCH_SIZE="${JACOBIAN_BATCH_SIZE:-128}"
# Crafting defaults are unchanged when these variables are not supplied.  They
# are environment knobs so a Slurm job can keep its FC settings in a separate,
# editable file instead of modifying this sweep driver.
CRAFT_STEPS="${CRAFT_STEPS:-250}"
CRAFT_ALPHA="${CRAFT_ALPHA:-0.0039216}"
FC_RESTARTS="${FC_RESTARTS:-1}"
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

DATASET="${DATASET:-CIFAR100}"
DATA_PATH="${DATA_PATH:-/home/mmoslem3/scratch/data}"

# CLASS_PAIR default depends on DATASET: CIFAR10's named pairs (dog-bird,
# frog-airplane) exist nowhere else. SVHN class names are the digit strings
# '0'..'9', so '1-7' works by name; CIFAR100 uses plain class INDICES
# ('3-7'), which final_update_scatter.py's parse_pair accepts whenever a token
# is not a literal class name. Which two classes does not matter for the
# component scatter, only that they exist.
if [ -z "$CLASS_PAIR" ]; then
    case "$DATASET" in
        CIFAR10) CLASS_PAIR="frog-airplane" ;;
        SVHN) CLASS_PAIR="1-7" ;;
        CIFAR100) CLASS_PAIR="3-7" ;;
        TinyImageNet) CLASS_PAIR="3-7" ;;
        *) CLASS_PAIR="0-1" ;;
    esac
fi
# Difficulty degree used when sweep_config.json has no entry for the combo
# (every non-CIFAR10 dataset). TARGET_SELECT, when set, still wins over this.
DEFAULT_TGT_DEG="${DEFAULT_TGT_DEG:-50}"
OUT_DIR="${OUT_DIR:-scatter_result}"
CACHE_DIR="${CACHE_DIR:-./cache}"
SEED="${SEED:-42}"
PROJECT_ROOT="${PROJECT_ROOT:-/home/mmoslem3/scratch/attack_if}"
PYTHON_ENV="${PYTHON_ENV:-/home/mmoslem3/ENV}"
NUM_TARGETS="${NUM_TARGETS:-10}"
SCATTER_POINTS="${SCATTER_POINTS:-2500}"   # random candidates plotted per target
SCATTER_FORMATS="${SCATTER_FORMATS:-pdf}"
NUM_VICTIMS="${NUM_VICTIMS:-5}"
RECOMPUTE_DELTAS="${RECOMPUTE_DELTAS:-0}"
case "$RECOMPUTE_DELTAS" in
    0) RECOMPUTE_FLAGS="" ;;
    1) RECOMPUTE_FLAGS="--recompute_deltas" ;;
    *) echo "RECOMPUTE_DELTAS=$RECOMPUTE_DELTAS (expected: 0 or 1)"; exit 1 ;;
esac
FORCE="${FORCE:-0}"
case "$FORCE" in
    0) FORCE_FLAGS="" ;;
    1) FORCE_FLAGS="--FORCE" ;;
    *) echo "FORCE=$FORCE (expected: 0 or 1)"; exit 1 ;;
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
        echo "   final_update_scatter.py would fall back to cpu and the Jacobian pass"
        echo "   over the whole candidate pool would take forever / be OOM-killed."
        echo "   run this inside an allocation, e.g.:"
        echo "     salloc --account=aip-boyuwang --gres=gpu:l40s:1 --cpus-per-task=4 \\"
        echo "            --mem=32G --time=12:00:00"
        echo "   or attach to one you already hold:   srun --jobid=<id> --pty bash"
        echo "   (ALLOW_CPU=1 bypasses this check)"
        exit 1
    }
fi

# reject a bad SELECT once, up front. Only ours / dpp rank by d + lam*M - beta*A,
# which is what the x axis is, so nothing else is accepted here.
for sel in $SELECT; do
    case "$sel" in
        ours|dpp) ;;
        *) echo "unknown SELECT=$sel for the scatter driver (expected: ours | dpp)"; exit 1 ;;
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
    # sweep_config.json is CIFAR10-only in practice. A dataset / model / attack /
    # pair combo it has never seen (SVHN, CIFAR100, or any pair besides the
    # CIFAR10 ones) still plots fine, it just has no paired-difficulty label, so
    # it falls back to degree DEFAULT_TGT_DEG instead of skipping the combo.
    CFG="$(python - "$model" "$attack" "$pair" "$DATASET" "$DEFAULT_TGT_DEG" <<'PY'
import json, sys
model, attack, pair, dataset, default_tgt = sys.argv[1:6]
cfg = json.load(open('sweep_config.json'))
# sapa is gradmatch + a sharpness-aware target gradient: same crafting cost, same
# difficulty, so it reads the gradmatch entry rather than needing its own.
key = 'gradmatch' if attack == 'sapa' else attack
try:
    if dataset != 'CIFAR10':
        raise KeyError(dataset)
    tgt = cfg['difficulty'][model][key][pair]
except KeyError:
    tgt = default_tgt
    sys.stderr.write('   note: sweep_config.json has no difficulty for %s / %s / %s / %s'
                     ' -- defaulting to degree %s\n' % (dataset, model, key, pair, tgt))
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
    # on the identical 10 images. CIFAR10 keeps the historical file names; every
    # other dataset gets its own prefix so '1-7' on SVHN and '1-7' on CIFAR100 can
    # never share a pinned set.
    if [ "$DATASET" = "CIFAR10" ]; then IDX_PREFIX=""; else IDX_PREFIX="${DATASET}_"; fi
    IDX="target_sets/${IDX_PREFIX}${model}_${attack}_${pair}.json"
    if [ ! -s "$IDX" ] && [ "$attack" != "$CFG_KEY" ]; then
        IDX="target_sets/${IDX_PREFIX}${model}_${CFG_KEY}_${pair}.json"
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

    # --- selector-specific flags -----------------------------------------------
    case "$sel" in
        ours)   BASE=ours;   SEL_FLAGS="";                              SEL_NOTE="ours (plain greedy top-N_p by score)" ;;
        dpp)    BASE=ours;   SEL_FLAGS="--sel_dpp --sel_alpha $SEL_ALPHA"; SEL_NOTE="dpp (alpha=$SEL_ALPHA)" ;;
    esac
    # The A term is computed regardless (it is the y axis); these flags only fix
    # the run name and the beta printed on the figure, so the plot pairs with
    # the sel_dpp.sh run that used the same USE_JACOBIAN_SCORE / JACOBIAN_WEIGHT.
    JACOBIAN_FLAGS="--jacobian_batch_size $JACOBIAN_BATCH_SIZE"
    JACOBIAN_NOTE="config has the Jacobian score disabled (ranking = x alone); A still computed for the y axis (batch=$JACOBIAN_BATCH_SIZE)"
    if [ "$USE_JACOBIAN_SCORE" = "1" ]; then
        JACOBIAN_FLAGS="--use_jacobian_score --jacobian_weight $JACOBIAN_WEIGHT --jacobian_batch_size $JACOBIAN_BATCH_SIZE"
        JACOBIAN_NOTE="config has the Jacobian score enabled (ranking = x - $JACOBIAN_WEIGHT y, batch=$JACOBIAN_BATCH_SIZE)"
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

    echo "=== SCATTER $SEL_NOTE | $model / $attack / $pair$SHARP_NOTE ==="
    echo "    x = std(d) + lam*std(M), y = std(A); $SCATTER_POINTS random candidates per target"
    echo "    targets: $TGT_NOTE"
    echo "    difficulty label tgt$TGT_DEG   craft flags: ${CFG_MEM:-none}"
    echo "    $JACOBIAN_NOTE"
    echo "    budgets: $BUDGETS"
    echo

    for bug in $BUDGETS; do
        echo "--- $sel | $model / $attack / $pair$SHARP_NOTE | budget $bug ---"
        python final_update_scatter.py \
            --dataset "$DATASET" --data_path "$DATA_PATH" --seed "$SEED" \
            --cache_dir "$CACHE_DIR" --out_dir "$OUT_DIR" \
            --model "$model" --attack "$attack" --base "$BASE" \
            --class_pair "$pair" --pair_order poison-target \
            --budget "$bug" --epsilon 0.0313725 \
            --craft_steps "$CRAFT_STEPS" --craft_alpha "$CRAFT_ALPHA" \
            --restarts 8 --fc_restarts "$FC_RESTARTS" --craft_ensemble 5 $CFG_MEM \
            --base_dist cosine --lambda_margin 1.0 \
            $SEL_FLAGS $JACOBIAN_FLAGS $SHARP_FLAGS \
            --num_surrogates 20 --surrogate_epochs 60 --surrogate_decay 35 45 \
            --num_targets "$NUM_TARGETS" --target_select "$TGT_DEG" \
            $TGT_FLAGS \
            $RECOMPUTE_FLAGS \
            $FORCE_FLAGS \
            --num_victims "$NUM_VICTIMS" --victim_epochs 50 --victim_lr 0.1 --victim_bs 125 \
            --victim_decay 40 --victim_wd 0.0 \
            --clean_baseline \
            --scatter_mode --scatter_points "$SCATTER_POINTS" \
            --scatter_formats "$SCATTER_FORMATS"
    done
    echo

done
done
done
done
done

# RECOMPUTE_DELTAS / FORCE are accepted for flag-compatibility with sel_dpp.sh
# but nothing here is resumed or cached: every call recomputes the components
# and overwrites the figures in SCATTER_<run name>/.
