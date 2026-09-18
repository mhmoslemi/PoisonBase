#!/usr/bin/env bash
# Resume only unfinished shards from run_bench_missing_local_8gpu.sh.
# One attack process is allowed on each GPU because the first attempt showed
# that two ResNet workers do not fit beside the server's existing ~86 GiB GPU
# process. Completed target/victim rows are retained by final_update.py.

set -Eeuo pipefail

ROOT="${ROOT:-/home/ubuntu/PoisonBase}"
export ROOT
export RUN_MODE=resume
export MIN_FREE_MIB="${MIN_FREE_MIB:-7000}"
export GPU_POLL_SECONDS="${GPU_POLL_SECONDS:-30}"
export MAX_RETRIES="${MAX_RETRIES:-3}"
export BOOTSTRAP_CACHE="${BOOTSTRAP_CACHE:-1}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

exec bash "$ROOT/run_bench_missing_local_8gpu.sh"
