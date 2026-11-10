#!/bin/sh
set -eu
: "${SLURM_JOB_ID:?Run this inside an interactive GPU allocation.}"
ROOT=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd -P)
cd "$ROOT"
echo "Group 3: 6 victims; rerun this same script to resume."
trap 'status=$?; if [ "$status" -eq 75 ]; then echo "Checkpoint saved. Rerun sh $0 in a new allocation."; fi' 0

echo "[1/6] alligator_tarantula_b0005_M: target 4531, victim 0"
bash "$ROOT/experiments/rebuttal_tiny_gm/interactive_craft.sh" M --target 4531 --victim 0

echo "[2/6] alligator_tarantula_b0005_R: target 8820, victim 0"
bash "$ROOT/experiments/rebuttal_tiny_gm/interactive_craft.sh" R --target 8820 --victim 0

echo "[3/6] alligator_tarantula_b0005_R: target 3271, victim 0"
bash "$ROOT/experiments/rebuttal_tiny_gm/interactive_craft.sh" R --target 3271 --victim 0

echo "[4/6] alligator_tarantula_b0005_R: target 6824, victim 0"
bash "$ROOT/experiments/rebuttal_tiny_gm/interactive_craft.sh" R --target 6824 --victim 0

echo "[5/6] alligator_tarantula_b0005_R: target 7813, victim 0"
bash "$ROOT/experiments/rebuttal_tiny_gm/interactive_craft.sh" R --target 7813 --victim 0

echo "[6/6] alligator_tarantula_b0005_R: target 4531, victim 0"
bash "$ROOT/experiments/rebuttal_tiny_gm/interactive_craft.sh" R --target 4531 --victim 0

echo "Group 3: all 6 victims complete."
