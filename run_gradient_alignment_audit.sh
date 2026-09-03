#!/usr/bin/env bash
# Run the exact GRAFT/gradient-alignment export over the paper's cached models.
#
# Defaults cover:
#   * CIFAR-10: all three main architectures and both main class pairs;
#   * CIFAR-100, SVHN, Tiny ImageNet: the architecture/pair used in the paper.
#
# All jobs request K=20 and --train-missing, so an incomplete surrogate directory
# is filled before analysis. Existing valid output shards are resumed/skipped.
# No poisons or victims are produced.
#
# Common overrides:
#   DEVICE=cuda:0 OUTPUT_ROOT=/scratch/... ./run_gradient_alignment_audit.sh
#   JOB_INDEX=0 ./run_gradient_alignment_audit.sh       # one row (SLURM arrays)
#   JOBS_FILE=my_jobs.txt ./run_gradient_alignment_audit.sh
#   REPRESENTATION_NTK_MODE=trace-exact ./run_gradient_alignment_audit.sh
#   REPRESENTATION_NTK_MODE=trace-hutchinson NTK_TRACE_PROBES=32 ./run_gradient_alignment_audit.sh
#   REPRESENTATION_NTK_MODE=full-exact MAX_FULL_NTK_GB=200 ./run_gradient_alignment_audit.sh
#   MAX_TARGETS=1 MAX_CANDIDATES=16 SURROGATE_IDS=0 ./run_gradient_alignment_audit.sh
#
# "contracted" (default) saves the exact residual-weighted J_h(x_i)J_h(x_t)^T
# contraction A_i and exact all-parameter <g_i,g_t>. trace-hutchinson estimates
# the unweighted representation-NTK trace without W or residuals. trace-exact
# sums that trace exactly and is much slower. full-exact can consume many GiB
# for ONE target/surrogate.

set -euo pipefail

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
if [[ -x /home/mmoslem3/ENV/bin/python ]]; then
  DEFAULT_PYTHON=/home/mmoslem3/ENV/bin/python
else
  DEFAULT_PYTHON=python
fi
PYTHON_BIN=${PYTHON_BIN:-$DEFAULT_PYTHON}
DATA_PATH=${DATA_PATH:-$SCRIPT_DIR/data}
CACHE_DIR=${CACHE_DIR:-$SCRIPT_DIR/cache}
OUTPUT_ROOT=${OUTPUT_ROOT:-$SCRIPT_DIR/gradient_alignment_outputs}
DEVICE=${DEVICE:-auto}
NUM_SURROGATES=${NUM_SURROGATES:-20}
SEED=${SEED:-42}
FORWARD_BATCH_SIZE=${FORWARD_BATCH_SIZE:-512}
JACOBIAN_BATCH_SIZE=${JACOBIAN_BATCH_SIZE:-64}
REPRESENTATION_NTK_MODE=${REPRESENTATION_NTK_MODE:-contracted}
NTK_TRACE_PROBES=${NTK_TRACE_PROBES:-32}
MAX_FULL_NTK_GB=${MAX_FULL_NTK_GB:-50}
VALIDATION_SAMPLES=${VALIDATION_SAMPLES:-1}
MAX_TARGETS=${MAX_TARGETS:-0}
MAX_CANDIDATES=${MAX_CANDIDATES:-0}
CANDIDATE_SCOPE=${CANDIDATE_SCOPE:-adversarial-class}
TRAIN_MISSING=${TRAIN_MISSING:-1}
FORCE=${FORCE:-0}
ALLOW_CPU=${ALLOW_CPU:-0}

# dataset|model|class_pair|target_file
JOBS=(
  "CIFAR10|ConvNetBN|dog-bird|target_sets/ConvNetBN_gradmatch_dog-bird.json"
  "CIFAR10|ConvNetBN|frog-airplane|target_sets/ConvNetBN_gradmatch_frog-airplane.json"
  "CIFAR10|ResNet20BN|dog-bird|target_sets/ResNet20BN_gradmatch_dog-bird.json"
  "CIFAR10|ResNet20BN|frog-airplane|target_sets/ResNet20BN_gradmatch_frog-airplane.json"
  "CIFAR10|VGG13BN|dog-bird|target_sets/VGG13BN_gradmatch_dog-bird.json"
  "CIFAR10|VGG13BN|frog-airplane|target_sets/VGG13BN_gradmatch_frog-airplane.json"
  "CIFAR100|ResNet18BN|sea-willow_tree|logs-proposition-multidataset/targets/CIFAR100_ResNet18BN_sea-willow_tree_10.json"
  "SVHN|ConvNetBN|9-2|logs-proposition-multidataset/targets/SVHN_ConvNetBN_9-2_10.json"
  "TinyImageNet|ResNet18BN|n01443537-n01629819|logs-proposition-multidataset/targets/TinyImageNet_ResNet18BN_n01443537-n01629819_10.json"
)

# Optional custom matrix, one non-comment line per job using the same four-field
# format. This is the convenient way to request another dataset/architecture
# Cartesian product without editing the launcher.
if [[ -n ${JOBS_FILE:-} ]]; then
  if [[ ! -f $JOBS_FILE ]]; then
    echo "JOBS_FILE does not exist: $JOBS_FILE" >&2
    exit 2
  fi
  mapfile -t JOBS < <(sed -e 's/[[:space:]]*#.*$//' -e '/^[[:space:]]*$/d' "$JOBS_FILE")
  if (( ${#JOBS[@]} == 0 )); then
    echo "JOBS_FILE contains no jobs: $JOBS_FILE" >&2
    exit 2
  fi
fi

if [[ $ALLOW_CPU != 1 ]]; then
  if [[ $DEVICE == cpu ]]; then
    echo "Exact collection is GPU-scale work; DEVICE=cpu requires ALLOW_CPU=1." >&2
    exit 2
  fi
  if [[ $DEVICE == auto ]] && ! "$PYTHON_BIN" - <<'PY'
import sys
try:
    import torch
except Exception as exc:
    sys.exit("cannot import torch: %s" % exc)
sys.exit(0 if torch.cuda.is_available() else 1)
PY
  then
    echo "No CUDA GPU is visible. Start a DayBreak Blue GPU allocation, or set ALLOW_CPU=1 deliberately." >&2
    exit 2
  fi
fi

if [[ -n ${JOB_INDEX:-} ]]; then
  if ! [[ $JOB_INDEX =~ ^[0-9]+$ ]] || (( JOB_INDEX >= ${#JOBS[@]} )); then
    echo "JOB_INDEX must be in [0, $((${#JOBS[@]} - 1))]" >&2
    exit 2
  fi
  START_JOB=$JOB_INDEX
  END_JOB=$((JOB_INDEX + 1))
else
  START_JOB=0
  END_JOB=${#JOBS[@]}
fi

mkdir -p "$OUTPUT_ROOT/logs"

for ((job_id = START_JOB; job_id < END_JOB; job_id++)); do
  IFS='|' read -r dataset model class_pair target_rel <<< "${JOBS[$job_id]}"
  target_file=$SCRIPT_DIR/$target_rel
  log_file=$OUTPUT_ROOT/logs/job_${job_id}_${dataset}_${model}_${class_pair}.log

  command=(
    "$PYTHON_BIN" "$SCRIPT_DIR/gradient_alignment_audit.py"
    --dataset "$dataset"
    --model "$model"
    --class-pair "$class_pair"
    --pair-order poison-target
    --target-file "$target_file"
    --data-path "$DATA_PATH"
    --cache-dir "$CACHE_DIR"
    --output-root "$OUTPUT_ROOT"
    --device "$DEVICE"
    --seed "$SEED"
    --num-surrogates "$NUM_SURROGATES"
    --candidate-scope "$CANDIDATE_SCOPE"
    --forward-batch-size "$FORWARD_BATCH_SIZE"
    --jacobian-batch-size "$JACOBIAN_BATCH_SIZE"
    --representation-ntk-mode "$REPRESENTATION_NTK_MODE"
    --ntk-trace-probes "$NTK_TRACE_PROBES"
    --max-full-ntk-gb "$MAX_FULL_NTK_GB"
    --validation-samples "$VALIDATION_SAMPLES"
    --max-targets "$MAX_TARGETS"
    --max-candidates "$MAX_CANDIDATES"
  )
  if [[ -n ${SURROGATE_IDS:-} ]]; then
    command+=(--surrogate-ids "$SURROGATE_IDS")
  fi
  if [[ $TRAIN_MISSING == 1 ]]; then
    command+=(--train-missing)
  fi
  if [[ $FORCE == 1 ]]; then
    command+=(--force)
  fi

  echo "[$((job_id + 1))/${#JOBS[@]}] $dataset $model $class_pair" | tee "$log_file"
  "${command[@]}" 2>&1 | tee -a "$log_file"
done
