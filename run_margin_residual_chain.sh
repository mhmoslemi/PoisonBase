#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
PYTHON_BIN=${PYTHON_BIN:-python}
DATA_PATH=${DATA_PATH:-$REPO_ROOT/data}
CACHE_DIR=${CACHE_DIR:-$REPO_ROOT/cache}
OUTPUT_ROOT=${OUTPUT_ROOT:-$REPO_ROOT}
DEVICE=${DEVICE:-cuda:0}
JOBS_FILE=${JOBS_FILE:-$REPO_ROOT/analysis/margin_residual_chain_jobs.tsv}

exec "$PYTHON_BIN" "$REPO_ROOT/analysis/margin_residual_chain.py" \
  --jobs-file "$JOBS_FILE" \
  --data-path "$DATA_PATH" \
  --cache-dir "$CACHE_DIR" \
  --output-root "$OUTPUT_ROOT" \
  --device "$DEVICE" \
  --seed 42 \
  --pair-order poison-target \
  --forward-batch-size "${FORWARD_BATCH_SIZE:-512}" \
  --bins 40 \
  --max-scatter 100000 \
  --u-t-threshold 1e-12 \
  --bound-atol 1e-10 \
  --bound-rtol 1e-8
