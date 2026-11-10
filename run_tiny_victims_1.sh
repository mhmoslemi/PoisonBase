#!/bin/sh
set -eu
: "${SLURM_JOB_ID:?Run this inside an interactive GPU allocation.}"
ROOT=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd -P)
cd "$ROOT"
echo "Group 1: 7 victims; rerun this same script to resume."
trap 'status=$?; if [ "$status" -eq 75 ]; then echo "Checkpoint saved. Rerun sh $0 in a new allocation."; fi' 0

echo "[1/7] bullfrog_dining_b0005_basis: target 4951, victim 0"
ALLOCATION_END=$(squeue -h -j "$SLURM_JOB_ID" -o "%e")
TASK_SECONDS=$(( $(date -d "$ALLOCATION_END" +%s) - $(date +%s) ))
if [ "$TASK_SECONDS" -le 180 ]; then echo "Allocation ending; rerun this script in a new allocation."; exit 0; fi
bash "$ROOT/experiments/rebuttal_tiny_gm/paired_job.sh" "$ROOT" "$ROOT/sbatch/rebuttal_tiny_gm_vgg13_5x5" "$TASK_SECONDS" victim_bullfrog_dining_b0005_basis_t4951_v0

echo "[2/7] alligator_tarantula_b0005_M: target 4531, victim 3"
bash "$ROOT/experiments/rebuttal_tiny_gm/interactive_craft.sh" M --target 4531 --victim 3

echo "[3/7] alligator_tarantula_b0005_R: target 8820, victim 3"
bash "$ROOT/experiments/rebuttal_tiny_gm/interactive_craft.sh" R --target 8820 --victim 3

echo "[4/7] alligator_tarantula_b0005_R: target 3271, victim 3"
bash "$ROOT/experiments/rebuttal_tiny_gm/interactive_craft.sh" R --target 3271 --victim 3

echo "[5/7] alligator_tarantula_b0005_R: target 6824, victim 3"
bash "$ROOT/experiments/rebuttal_tiny_gm/interactive_craft.sh" R --target 6824 --victim 3

echo "[6/7] alligator_tarantula_b0005_R: target 7813, victim 3"
bash "$ROOT/experiments/rebuttal_tiny_gm/interactive_craft.sh" R --target 7813 --victim 3

echo "[7/7] alligator_tarantula_b0005_R: target 4531, victim 3"
bash "$ROOT/experiments/rebuttal_tiny_gm/interactive_craft.sh" R --target 4531 --victim 3

echo "Group 1: all 7 victims complete."
