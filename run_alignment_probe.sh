#!/usr/bin/env bash
#
# Driver for alignment_probe.py -- the exact <g_i, g_t> / Eq.(3) / Algorithm 1
# audit requested by the reviewer note on Section 3.2.
#
# For every (dataset, architecture, class pair, pinned target, cached surrogate)
# it writes, for EVERY candidate base:
#
#   g_full   exact <g_i, g_t> over all parameters
#   head_W   (r_i.r_t)(h_i.h_t)      Eq.(3) classifier-weight term  [closed form]
#   head_b   (r_i.r_t)               classifier-bias term           [closed form]
#   A        <grad_phi l_i, grad_phi L_t>   Algorithm 1's A_i       [exact JVP]
#   hTh      <h_i, h_t>              Algorithm 1's R_i
#   d_cos    1 - cos(h_i, h_t)       the feature term the CODE uses instead
#   margin   Algorithm 1's M_i, plus ||r_i||, <r_i,r_t>, p_adv plus
#   jker     u_t^T J_i J_t^T u_t     Jacobian interaction with the candidate
#                                    residual removed
#   ntk_tr   Hutchinson tr(J_i J_t^T), the scalar comparable to <h_i, h_t>
#
# and checks  g_full == head_W + head_b + A  against brute-force full-parameter
# gradients on VERIFY candidates per (target, surrogate).
#
# Nothing is retrained: the surrogates come out of $CACHE_DIR/surrogates/, the
# same pool the attack runs used. Set TRAIN_MISSING=1 to fill a hole.
#
# Usage
#   sh run_alignment_probe.sh                      # the whole grid, here and now
#   SUBMIT=1 sh run_alignment_probe.sh             # one SLURM job per combination
#   DATASETS=CIFAR10 MODELS=ConvNetBN sh run_alignment_probe.sh
#   DATASETS=SVHN sh run_alignment_probe.sh
#   NTK_PROBES=0 VERIFY=0 sh run_alignment_probe.sh    # fastest pass
#   MAX_CANDIDATES=500 sh run_alignment_probe.sh       # quick look at everything
#   DRY_RUN=1 sh run_alignment_probe.sh            # print the commands only
#
# DATASETS, MODELS, PAIRS, ATTACKS each take one or more whitespace-separated
# values and are swept as nested loops. Leaving MODELS or PAIRS empty uses the
# per-dataset defaults below, which are the combinations the paper reports.
#
# Cost. Per (surrogate, target) the candidate pool is swept 2 + NTK_PROBES times
# in forward-mode AD. Measured on one H100 with the defaults: 1.5 s per
# (surrogate, target) over CIFAR-10's full 5000-image pool with ConvNetBN, so
# about 5 minutes for a whole 20-surrogate x 10-target combo. VGG13 and
# ResNet18BN are several times heavier. NTK_PROBES=0 removes two thirds of the
# work and keeps everything except the tr(J_i J_t^T) column; MAX_CANDIDATES caps
# the pool for a first look.
#
# There is no GPU on a login node, so either run this inside an allocation
#   salloc --account=aip-boyuwang --gres=gpu:l40s:1 --cpus-per-task=4 \
#          --mem=32G --time=8:00:00
# or use SUBMIT=1, which writes and sbatches one job per combination.

set -u

DATASETS="${DATASETS:-CIFAR10 CIFAR100 SVHN TinyImageNet}"
MODELS="${MODELS:-}"                 # empty -> per-dataset default
PAIRS="${PAIRS:-}"                   # empty -> per-dataset default
ATTACKS="${ATTACKS:-fc gradmatch}"   # only picks the CIFAR-10 pinned target set

NUM_SURROGATES="${NUM_SURROGATES:-20}"
NUM_TARGETS="${NUM_TARGETS:-0}"      # 0 = every target in the pinned file
BATCH_SIZE="${BATCH_SIZE:-64}"
NTK_PROBES="${NTK_PROBES:-8}"        # 0 skips the tr(J_i J_t^T) estimate
VERIFY="${VERIFY:-8}"                # 0 skips the brute-force identity check
MAX_CANDIDATES="${MAX_CANDIDATES:-0}"   # 0 = the whole class pool
LAM="${LAM:-1.0}"
BETA="${BETA:-1.0}"
BUDGET="${BUDGET:-0.005}"            # only sets top-m for the overlap columns
WRITE_CSV="${WRITE_CSV:-0}"          # 1 also writes long-format csv.gz
TRAIN_MISSING="${TRAIN_MISSING:-0}"

DATA_PATH="${DATA_PATH:-/home/mmoslem3/scratch/data}"
CACHE_DIR="${CACHE_DIR:-./cache}"
TARGET_SETS="${TARGET_SETS:-./target_sets}"
OUT_DIR="${OUT_DIR:-alignment_probe_result}"
SEED="${SEED:-42}"
PROJECT_ROOT="${PROJECT_ROOT:-/home/mmoslem3/scratch/attack_if}"
PYTHON_ENV="${PYTHON_ENV:-/home/mmoslem3/ENV}"
DRY_RUN="${DRY_RUN:-0}"

# SUBMIT=1 writes one sbatch script per combination under $SB_DIR and submits it,
# instead of running here. Same accounting and log layout as sbatch/attack/*.
SUBMIT="${SUBMIT:-0}"
SB_DIR="${SB_DIR:-$PROJECT_ROOT/sbatch/alignment}"
SB_LOGS="${SB_LOGS:-$PROJECT_ROOT/sbatch/logs-alignment}"
SB_ACCOUNT="${SB_ACCOUNT:-aip-boyuwang}"
SB_TIME="${SB_TIME:-0-06:00:00}"
SB_GRES="${SB_GRES:-gpu:l40s:1}"
SB_MEM="${SB_MEM:-32G}"
SB_CPUS="${SB_CPUS:-4}"

# --- per-dataset defaults, matching the paper's reported combinations --------
default_models() {
    case "$1" in
        CIFAR10)      echo "ConvNetBN ResNet20BN VGG13BN" ;;
        CIFAR100)     echo "ResNet18BN" ;;
        SVHN)         echo "ConvNetBN" ;;
        TinyImageNet) echo "ResNet18BN" ;;
        *) echo "" ;;
    esac
}

# poison-target order, i.e. '<adversarial>-<target>', the same convention
# --class_pair uses everywhere else in this repo.
default_pairs() {
    case "$1" in
        CIFAR10)      echo "dog-bird frog-airplane" ;;
        CIFAR100)     echo "sea-willow_tree plain-bicycle wardrobe-lawn_mower bottle-road sunflower-cattle" ;;
        SVHN)         echo "9-2 6-1 8-3 7-5 0-4" ;;
        TinyImageNet) echo "n01443537-n01629819" ;;
        *) echo "" ;;
    esac
}

# Only CIFAR-10 has two distinct pinned target sets (fc vs gradmatch; sapa reuses
# gradmatch's). Everywhere else there is one file per pair, so looping ATTACKS
# there would just redo identical work.
attacks_for() {
    case "$1" in
        CIFAR10) echo "$ATTACKS" ;;
        *) echo "-" ;;
    esac
}

source "$PYTHON_ENV/bin/activate"
cd "$PROJECT_ROOT" || exit 1

# --- refuse to start on a node with no GPU -----------------------------------
# The JVP passes are ~10 forward-equivalents over the whole class pool per
# (surrogate, target). On CPU that is days, not minutes.
if [ -z "${ALLOW_CPU:-}" ] && [ "$DRY_RUN" != "1" ] && [ "$SUBMIT" != "1" ]; then
    python -c "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)" || {
        echo "!! no CUDA device visible on $(hostname) -- refusing to start."
        echo "   run this inside an allocation, e.g.:"
        echo "     salloc --account=aip-boyuwang --gres=gpu:l40s:1 --cpus-per-task=4 \\"
        echo "            --mem=32G --time=8:00:00"
        echo "   (ALLOW_CPU=1 bypasses this check and passes --allow_cpu)"
        exit 1
    }
fi

EXTRA=""
[ "$WRITE_CSV" = "1" ]     && EXTRA="$EXTRA --csv"
[ "$TRAIN_MISSING" = "1" ] && EXTRA="$EXTRA --train_missing"
[ -n "${ALLOW_CPU:-}" ]    && EXTRA="$EXTRA --allow_cpu"

mkdir -p "$OUT_DIR"
FAILED=""
TOTAL=0
OK=0

for ds in $DATASETS; do

    models="$MODELS"
    [ -z "$models" ] && models="$(default_models "$ds")"
    pairs="$PAIRS"
    [ -z "$pairs" ] && pairs="$(default_pairs "$ds")"
    if [ -z "$models" ] || [ -z "$pairs" ]; then
        echo "!! no default models/pairs for dataset $ds -- set MODELS and PAIRS"
        echo
        continue
    fi

for model in $models; do
for pair in $pairs; do
for atk in $(attacks_for "$ds"); do

    # target-set lookup mirrors alignment_probe.resolve_target_file, and is
    # repeated here only so a missing file is reported before the GPU is touched
    if [ "$atk" = "-" ]; then
        atk_flag=""
        cand_files="$TARGET_SETS/xdata_${ds}_${model}_${pair}.json
                    $TARGET_SETS/appx_tiny_${model}_${pair}.json
                    $TARGET_SETS/appx_broad_${model}_${pair}.json"
        tag="$ds/$model/$pair"
    else
        atk_flag="--attack $atk"
        cand_files="$TARGET_SETS/${model}_${atk}_${pair}.json"
        tag="$ds/$model/$pair (target set: $atk)"
    fi
    found=""
    for f in $cand_files; do
        [ -s "$f" ] && { found="$f"; break; }
    done
    if [ -z "$found" ]; then
        echo "!! no pinned target set for $tag -- looked for:"
        for f in $cand_files; do echo "     $f"; done
        echo "   skipping"
        echo
        continue
    fi

    echo "=== $tag ==="
    echo "    targets:    $found"
    echo "    surrogates: $NUM_SURROGATES from $CACHE_DIR/surrogates"
    echo "    probes:     ntk=$NTK_PROBES verify=$VERIFY batch=$BATCH_SIZE"

    CMD="python alignment_probe.py \
        --dataset $ds --data_path $DATA_PATH --model $model \
        --class_pair $pair --pair_order poison-target $atk_flag \
        --cache_dir $CACHE_DIR --target_sets $TARGET_SETS \
        --target_idx_file $found --num_targets $NUM_TARGETS \
        --out_dir $OUT_DIR --seed $SEED \
        --num_surrogates $NUM_SURROGATES \
        --batch_size $BATCH_SIZE --ntk_probes $NTK_PROBES --verify $VERIFY \
        --max_candidates $MAX_CANDIDATES \
        --lam $LAM --beta $BETA --budget $BUDGET $EXTRA"

    TOTAL=$((TOTAL + 1))
    if [ "$DRY_RUN" = "1" ]; then
        echo "    DRY_RUN: $(echo $CMD)"
        echo
        continue
    fi

    if [ "$SUBMIT" = "1" ]; then
        # slashes and dots are not usable in a job name; the pair may hold '-'
        safe="$(printf '%s_%s_%s_%s' "$ds" "$model" "$pair" "$atk" \
                | tr -c 'A-Za-z0-9_' '_')"
        job="$SB_DIR/align_${safe}.sh"
        mkdir -p "$SB_DIR" "$SB_LOGS"
        cat > "$job" <<EOF
#!/bin/bash
#SBATCH --account=$SB_ACCOUNT
#SBATCH --job-name=align_${safe}
#SBATCH --time=$SB_TIME
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=$SB_CPUS
#SBATCH --mem=$SB_MEM
#SBATCH --gres=$SB_GRES
#SBATCH --output=$SB_LOGS/align_${safe}-%j.out

set -Eeuo pipefail
source "$PYTHON_ENV/bin/activate"
cd "$PROJECT_ROOT"
$CMD
EOF
        chmod +x "$job"
        if out=$(sbatch "$job" 2>&1); then
            echo "    submitted: $out  ($job)"
            OK=$((OK + 1))
        else
            echo "!! sbatch FAILED for $tag: $out"
            FAILED="$FAILED\n    $tag"
        fi
        echo
        continue
    fi

    # One bad combo must not kill a long sweep.
    if eval "$CMD"; then
        OK=$((OK + 1))
    else
        echo "!! FAILED: $tag"
        FAILED="$FAILED\n    $tag"
    fi
    echo

done
done
done
done

if [ "$DRY_RUN" = "1" ]; then
    echo "=== DRY_RUN: $TOTAL combinations would run ==="
    exit 0
fi
if [ "$SUBMIT" = "1" ]; then
    echo "=== submitted $OK/$TOTAL jobs (scripts in $SB_DIR, logs in $SB_LOGS) ==="
else
    echo "=== done: $OK/$TOTAL combinations succeeded ==="
fi
if [ -n "$FAILED" ]; then
    printf "failed combinations:%b\n" "$FAILED"
    exit 1
fi
echo "results under $OUT_DIR (one directory per DATASET_MODEL_PAIR;"
echo "  meta.json, target_<idx>.npz, summary.csv, plus $OUT_DIR/manifest.jsonl)"
