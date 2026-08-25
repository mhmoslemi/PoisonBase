#!/usr/bin/env bash
#
# Cross-architecture dog--bird table launcher.
#
# S (SELECTOR_MODELS) chooses bases. A=V (MODELS) crafts poisons and trains
# victims. The paper's BP rows use --attack fc; GM uses --attack gradmatch.
# Every cell uses the xa* protocol: 20 surrogates, five pinned targets, four
# victims per target, budget 0.005, and DPP alpha 2.
#
# Examples:
#   sh cross_arch.sh
#   USE_JACOBIAN_SCORE=1 JACOBIAN_WEIGHT=1.0 sh cross_arch.sh
#   MODELS=ResNet20BN ATTACKS=gradmatch DRY_RUN=1 sh cross_arch.sh
#
# Interrupted runs are resumable: final_update.py reuses their run directory,
# poison cache, and completed (target, victim) trials.

set -u

# =========================== EDIT SETTINGS HERE ============================ #

# Whitespace-separated sweep axes. Supported names come from final_update.py.
MODELS="${MODELS:-ConvNetBN ResNet20BN VGG13BN}"       # attack/victim A=V
SELECTOR_MODELS="${SELECTOR_MODELS:-ConvNetBN ResNet20BN VGG13BN}"  # S
ATTACKS="${ATTACKS:-fc gradmatch}"                    # fc = BP in the table
SELECTIONS="${SELECTIONS:-random dpp}"

DATASET="${DATASET:-CIFAR10}"
CLASS_PAIR="${CLASS_PAIR:-dog-bird}"
BUDGET="${BUDGET:-0.005}"
EPSILON="${EPSILON:-0.0313725}"                       # 8/255 pixel budget
SEED="${SEED:-42}"

# These reproduce xa1.sh ... xa6.sh and xr1.sh ... xr3.sh.
NUM_SURROGATES="${NUM_SURROGATES:-20}"
CRAFT_ENSEMBLE="${CRAFT_ENSEMBLE:-5}"
NUM_TARGETS="${NUM_TARGETS:-5}"
NUM_VICTIMS="${NUM_VICTIMS:-4}"
SURROGATE_EPOCHS="${SURROGATE_EPOCHS:-60}"
VICTIM_EPOCHS="${VICTIM_EPOCHS:-50}"

BASE_DIST="${BASE_DIST:-cosine}"
LAMBDA_MARGIN="${LAMBDA_MARGIN:-1.0}"
SEL_ALPHA="${SEL_ALPHA:-2.0}"

# Exact Jacobian-aware pointwise score. It applies to DPP only; Random is
# deliberately unchanged. Batch size affects speed/memory, not run identity.
USE_JACOBIAN_SCORE="${USE_JACOBIAN_SCORE:-0}"         # must be 0 or 1
JACOBIAN_WEIGHT="${JACOBIAN_WEIGHT:-1.0}"             # beta, must be >= 0
JACOBIAN_BATCH_SIZE="${JACOBIAN_BATCH_SIZE:-64}"      # must be > 0

# The historical xa* launchers cover off-diagonal S != A cells. "auto" also
# runs matched DPP cells when Jacobian is enabled, since their _jacw* run names
# are isolated from historical caches. Set to 1 to request matched DPP cells in
# a fresh baseline tree, or 0 to always leave them to sel_dpp.sh.
RUN_MATCHED_DPP="${RUN_MATCHED_DPP:-auto}"             # auto | 0 | 1

PROJECT_DIR="${PROJECT_DIR:-$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)}"
DATA_PATH="${DATA_PATH:-/home/mmoslem3/scratch/data}"
OUT_DIR="${OUT_DIR:-ours_result}"
CACHE_DIR="${CACHE_DIR:-./cache}"
VENV_ACTIVATE="${VENV_ACTIVATE:-/home/mmoslem3/ENV/bin/activate}"

CRAFT_STEPS="${CRAFT_STEPS:-250}"
CRAFT_ALPHA="${CRAFT_ALPHA:-0.0039216}"
RESTARTS="${RESTARTS:-8}"
CRAFT_BATCH="${CRAFT_BATCH:-256}"
VICTIM_LR="${VICTIM_LR:-0.1}"
VICTIM_BS="${VICTIM_BS:-125}"
VICTIM_WD="${VICTIM_WD:-0.0}"

DRY_RUN="${DRY_RUN:-0}"                               # 1 prints commands
ALLOW_CPU="${ALLOW_CPU:-0}"                           # 1 skips CUDA check

# ========================================================================== #

die() {
    echo "cross_arch.sh: $*" >&2
    exit 1
}

case "$USE_JACOBIAN_SCORE" in
    0|1) ;;
    *) die "USE_JACOBIAN_SCORE=$USE_JACOBIAN_SCORE (expected 0 or 1)" ;;
esac
case "$RUN_MATCHED_DPP" in
    auto|0|1) ;;
    *) die "RUN_MATCHED_DPP=$RUN_MATCHED_DPP (expected auto, 0, or 1)" ;;
esac
case "$DRY_RUN" in
    0|1) ;;
    *) die "DRY_RUN=$DRY_RUN (expected 0 or 1)" ;;
esac

# Use Python for numeric validation so decimal and scientific notation work.
python - "$JACOBIAN_WEIGHT" "$JACOBIAN_BATCH_SIZE" <<'PY' || exit 1
import sys
try:
    weight = float(sys.argv[1])
    batch_size = int(sys.argv[2])
except ValueError as exc:
    raise SystemExit('cross_arch.sh: invalid Jacobian setting: %s' % exc)
if weight < 0:
    raise SystemExit('cross_arch.sh: JACOBIAN_WEIGHT must be nonnegative')
if batch_size <= 0:
    raise SystemExit('cross_arch.sh: JACOBIAN_BATCH_SIZE must be positive')
PY

for selection in $SELECTIONS; do
    case "$selection" in
        random|dpp) ;;
        *) die "unknown selection '$selection' (expected random or dpp)" ;;
    esac
done
for attack in $ATTACKS; do
    case "$attack" in
        fc|gradmatch) ;;
        *) die "unknown attack '$attack' (expected fc or gradmatch)" ;;
    esac
done

cd "$PROJECT_DIR" || exit 1
[ -f final_update.py ] || die "final_update.py not found in PROJECT_DIR=$PROJECT_DIR"

if [ "$DRY_RUN" = 0 ]; then
    if [ -n "$VENV_ACTIVATE" ]; then
        [ -f "$VENV_ACTIVATE" ] || die "venv activation file not found: $VENV_ACTIVATE"
        # shellcheck disable=SC1090
        . "$VENV_ACTIVATE"
    fi
    if [ "$ALLOW_CPU" != 1 ]; then
        python -c 'import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)' ||
            die "no CUDA device visible (set ALLOW_CPU=1 only for a deliberate CPU run)"
    fi
fi

# Architecture/attack difficulty labels copied from the xa/xr launchers. These
# label run directories; the actual target indices always come from IDX below.
target_degree() {
    case "$1:$2" in
        ConvNetBN:fc)        echo 50 ;;
        ConvNetBN:gradmatch) echo 70 ;;
        ResNet20BN:fc)       echo 10 ;;
        ResNet20BN:gradmatch) echo 14 ;;
        VGG13BN:fc)          echo 3 ;;
        VGG13BN:gradmatch)   echo 50 ;;
        *) die "no xa target-degree setting for model=$1 attack=$2" ;;
    esac
}

print_command() {
    printf '  '
    for arg in "$@"; do
        case "$arg" in
            *[!A-Za-z0-9_./:=+-]*) printf "'%s' " "$(printf '%s' "$arg" | sed "s/'/'\\\\''/g")" ;;
            *) printf '%s ' "$arg" ;;
        esac
    done
    printf '\n'
}

run_cell() {
    model="$1"
    attack="$2"
    selector_model="$3"
    selection="$4"
    degree="$(target_degree "$model" "$attack")"
    idx="target_sets/xarch_${model}_${attack}_${CLASS_PAIR}_b${BUDGET}.json"
    [ -s "$idx" ] || die "missing pinned target file: $idx"

    set -- python final_update.py \
        --dataset "$DATASET" --data_path "$DATA_PATH" --seed "$SEED" \
        --cache_dir "$CACHE_DIR" --out_dir "$OUT_DIR" \
        --model "$model" --sel_model "$selector_model" --attack "$attack" \
        --class_pair "$CLASS_PAIR" --pair_order poison-target \
        --budget "$BUDGET" --epsilon "$EPSILON" \
        --craft_steps "$CRAFT_STEPS" --craft_alpha "$CRAFT_ALPHA" \
        --restarts "$RESTARTS" --craft_ensemble "$CRAFT_ENSEMBLE" \
        --num_surrogates "$NUM_SURROGATES" \
        --surrogate_epochs "$SURROGATE_EPOCHS" --surrogate_decay 35 45 \
        --num_targets "$NUM_TARGETS" --target_select "$degree" \
        --target_idx_file "$idx" \
        --num_victims "$NUM_VICTIMS" --victim_epochs "$VICTIM_EPOCHS" \
        --victim_lr "$VICTIM_LR" --victim_bs "$VICTIM_BS" \
        --victim_decay 40 --victim_wd "$VICTIM_WD" \
        --clean_baseline

    # Match the one low-memory combination in xa6/xr3.
    if [ "$model" = VGG13BN ] && [ "$attack" = gradmatch ]; then
        set -- "$@" --craft_lowmem --craft_batch "$CRAFT_BATCH" --fast_gradmatch
    fi

    case "$selection" in
        random)
            set -- "$@" --base random
            ;;
        dpp)
            set -- "$@" --base ours --base_dist "$BASE_DIST" \
                --lambda_margin "$LAMBDA_MARGIN" --sel_dpp --sel_alpha "$SEL_ALPHA"
            if [ "$USE_JACOBIAN_SCORE" = 1 ]; then
                set -- "$@" --use_jacobian_score \
                    --jacobian_weight "$JACOBIAN_WEIGHT" \
                    --jacobian_batch_size "$JACOBIAN_BATCH_SIZE"
            fi
            ;;
    esac

    echo "=== S=$selector_model -> A=V=$model | $attack | $selection ==="
    echo "    targets=$idx ($NUM_TARGETS), victims=$NUM_VICTIMS, surrogates=$NUM_SURROGATES"
    if [ "$selection" = dpp ]; then
        echo "    Jacobian: enabled=$USE_JACOBIAN_SCORE weight=$JACOBIAN_WEIGHT batch=$JACOBIAN_BATCH_SIZE"
    else
        echo "    Jacobian: not applicable to Random; run is unchanged"
    fi

    if [ "$DRY_RUN" = 1 ]; then
        print_command "$@"
    else
        "$@" || exit 1
    fi
    echo
}

echo "=== cross-architecture sweep ==="
echo "    A=V models : $MODELS"
echo "    S models   : $SELECTOR_MODELS"
echo "    attacks    : $ATTACKS (fc is BP)"
echo "    selections : $SELECTIONS"
echo "    protocol   : $NUM_TARGETS targets x $NUM_VICTIMS victims, $NUM_SURROGATES surrogates"
echo "    Jacobian   : enabled=$USE_JACOBIAN_SCORE weight=$JACOBIAN_WEIGHT batch=$JACOBIAN_BATCH_SIZE"
echo

for model in $MODELS; do
for attack in $ATTACKS; do
for selector_model in $SELECTOR_MODELS; do
for selection in $SELECTIONS; do
    if [ "$selector_model" = "$model" ]; then
        run_matched="$RUN_MATCHED_DPP"
        if [ "$run_matched" = auto ]; then
            run_matched="$USE_JACOBIAN_SCORE"
        fi
        if [ "$selection" != dpp ] || [ "$run_matched" != 1 ]; then
            echo "=== S=A=V=$model | $attack | $selection: matched cell; reused/skipped ==="
            echo
            continue
        fi
    fi
    run_cell "$model" "$attack" "$selector_model" "$selection"
done
done
done
done

echo "=== cross-architecture sweep finished ==="
