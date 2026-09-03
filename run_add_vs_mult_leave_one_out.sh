#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/mmoslem3/scratch/attack_if
source /home/mmoslem3/ENV/bin/activate
cd "$ROOT"

exec python add_vs_mult_leave_one_out.py \
    --jobs-file "$ROOT/add_vs_mult_leave_one_out_jobs.tsv" \
    --data-path /home/mmoslem3/scratch/data \
    --cache-dir "$ROOT/cache" \
    --output-root "$ROOT" \
    --device cuda:0 \
    --seed 42 \
    --rhos 0.001,0.002,0.005,0.01,0.02,0.04 \
    "$@"
