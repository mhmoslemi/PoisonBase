#!/usr/bin/env bash
# CPU-only analysis for tmp.md.  This script deliberately does not call sbatch.

[ -n "${BASH_VERSION:-}" ] || exec bash "$0" "$@"
set -Eeuo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
SOURCE_ROOT="${SOURCE_ROOT:-$SCRIPT_DIR}"
PYTHON_BIN="${PYTHON_BIN:-/home/mmoslem3/ENV/bin/python}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$SOURCE_ROOT/logs-proposiot}"
DATA_PATH="${DATA_PATH:-$SOURCE_ROOT/data}"
TARGET_IDX_FILE="${TARGET_IDX_FILE:-$SOURCE_ROOT/target_sets/ConvNetBN_gradmatch_dog-bird.json}"

MODEL="${MODEL:-ConvNetBN}"
CLASS_PAIR="${CLASS_PAIR:-dog-bird}"
SEED="${SEED:-42}"
TARGET_SELECT="${TARGET_SELECT:-70}"

[ -f "$SOURCE_ROOT/residual_suppression_experiment.py" ] || {
    printf 'ERROR: experiment runner missing under %s\n' "$SOURCE_ROOT" >&2; exit 1; }
[ -x "$PYTHON_BIN" ] || {
    printf 'ERROR: Python environment missing: %s\n' "$PYTHON_BIN" >&2; exit 1; }
mkdir -p "$OUTPUT_ROOT"
cd "$SOURCE_ROOT"
export MPLBACKEND=Agg

partial=()
[ "${ALLOW_PARTIAL:-0}" = 1 ] && partial+=(--allow-partial)

exec "$PYTHON_BIN" "$SOURCE_ROOT/residual_suppression_experiment.py" analyze \
    --output-root "$OUTPUT_ROOT" \
    --dataset CIFAR10 --data-path "$DATA_PATH" \
    --model "$MODEL" --class-pair "$CLASS_PAIR" --seed "$SEED" \
    --num-targets 5 --num-victims 5 --num-surrogates 5 \
    --target-select "$TARGET_SELECT" --target-idx-file "$TARGET_IDX_FILE" \
    --gpus none "${partial[@]}"
