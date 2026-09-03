#!/usr/bin/env bash
# Collect the pure representation-Jacobian trace
#   T_i = tr(J_h(x_i) J_h(x_t)^T)
# for comparison with
#   R_i = h(x_i)^T h(x_t).
#
# This writes to a separate output tree and never overwrites the completed
# contracted-gradient audit.  The default Hutchinson estimator is appropriate
# for correlation/ranking analysis and is much faster than summing every output
# coordinate exactly.  Set TRACE_METHOD=trace-exact for the exact trace.

set -euo pipefail

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)

export OUTPUT_ROOT=${TRACE_OUTPUT_ROOT:-$SCRIPT_DIR/jacobian_feature_trace_outputs}
export REPRESENTATION_NTK_MODE=${TRACE_METHOD:-trace-hutchinson}
export NTK_TRACE_PROBES=${NTK_TRACE_PROBES:-32}
export VALIDATION_SAMPLES=0
export TRAIN_MISSING=${TRAIN_MISSING:-0}

if [[ $REPRESENTATION_NTK_MODE != trace-hutchinson && \
      $REPRESENTATION_NTK_MODE != trace-exact ]]; then
  echo "TRACE_METHOD must be trace-hutchinson or trace-exact" >&2
  exit 2
fi

exec bash "$SCRIPT_DIR/run_gradient_alignment_audit.sh"
