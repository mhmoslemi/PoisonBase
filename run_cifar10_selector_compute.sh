#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
PYTHON_BIN=${PYTHON_BIN:-python}
DATA_PATH=${DATA_PATH:-$REPO_ROOT/data}
CACHE_DIR=${CACHE_DIR:-$REPO_ROOT/cache}
OUTPUT_ROOT=${OUTPUT_ROOT:-$REPO_ROOT}
DEVICE=${DEVICE:-cuda:0}

exec "$PYTHON_BIN" "$REPO_ROOT/analysis/benchmark_selector_compute.py" \
  --jobs-file "$REPO_ROOT/analysis/cifar10_selector_compute_jobs.tsv" \
  --data-path "$DATA_PATH" \
  --cache-dir "$CACHE_DIR" \
  --output-root "$OUTPUT_ROOT" \
  --device "$DEVICE" \
  --k 20 \
  --repeats 3 \
  --rand-inner-repeats 100 \
  --forward-batch-size "${FORWARD_BATCH_SIZE:-512}" \
  --gradient-batch-size "${GRADIENT_BATCH_SIZE:-64}" \
  --seed 42 \
  --surrogate-epochs 60 \
  --surrogate-lr 0.1 \
  --surrogate-bs 128 \
  --surrogate-decay 35 45 \
  --surrogate-wd 0.0
