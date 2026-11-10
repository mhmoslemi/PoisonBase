#!/bin/sh
set -eu
: "${SLURM_JOB_ID:?Run this inside your interactive GPU allocation.}"
ROOT=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd -P)
cd "$ROOT"
echo "Group 1: 34 victims. Completed trials are skipped; checkpoints resume."
trap 'status=$?; if [ "$status" -eq 75 ]; then echo "Checkpoint saved. Rerun sh $0 in a new allocation."; fi' 0
COUNT=0
while read -r VARIANT TARGET VICTIM CELL; do
    COUNT=$((COUNT + 1))
    echo "[$COUNT/34] $CELL: target $TARGET, victim $VICTIM"
    ALLOCATION_END=$(squeue -h -j "$SLURM_JOB_ID" -o '%e')
    TASK_SECONDS=$(( $(date -d "$ALLOCATION_END" +%s) - $(date +%s) ))
    if [ "$TASK_SECONDS" -le 180 ]; then echo "Allocation ending. Rerun this script to resume."; exit 75; fi
    if [ "$VARIANT" = original ]; then
        bash "$ROOT/experiments/rebuttal_tiny_gm/paired_job.sh" "$ROOT" "$ROOT/sbatch/rebuttal_tiny_gm_vgg13_5x5" "$TASK_SECONDS" "victim_${CELL}_t${TARGET}_v${VICTIM}" < /dev/null
    else
        bash "$ROOT/experiments/rebuttal_tiny_gm/interactive_craft.sh" "$VARIANT" --target "$TARGET" --victim "$VICTIM" < /dev/null
    fi
done <<'TRIALS'
R 6824 3 alligator_tarantula_b0005_R
R 4531 3 alligator_tarantula_b0005_R
R 7813 4 alligator_tarantula_b0005_R
R 7813 0 alligator_tarantula_b0005_R
R 6824 1 alligator_tarantula_b0005_R
R 4531 1 alligator_tarantula_b0005_R
R 8820 2 alligator_tarantula_b0005_R
R 6824 2 alligator_tarantula_b0005_R
R 4531 2 alligator_tarantula_b0005_R
basis_lam0p1 8820 3 alligator_tarantula_b0005_basis_lam0p1
basis_lam0p1 3271 0 alligator_tarantula_b0005_basis_lam0p1
basis_lam0p1 3271 2 alligator_tarantula_b0005_basis_lam0p1
basis_lam0p1 3271 4 alligator_tarantula_b0005_basis_lam0p1
basis_lam0p1 6824 1 alligator_tarantula_b0005_basis_lam0p1
basis_lam0p1 6824 3 alligator_tarantula_b0005_basis_lam0p1
basis_lam0p1 7813 0 alligator_tarantula_b0005_basis_lam0p1
basis_lam0p1 7813 2 alligator_tarantula_b0005_basis_lam0p1
basis_lam0p1 7813 4 alligator_tarantula_b0005_basis_lam0p1
basis_lam0p1 4531 1 alligator_tarantula_b0005_basis_lam0p1
basis_lam0p1 4531 3 alligator_tarantula_b0005_basis_lam0p1
basis_lam10 8820 0 alligator_tarantula_b0005_basis_lam10
basis_lam10 8820 2 alligator_tarantula_b0005_basis_lam10
basis_lam10 8820 4 alligator_tarantula_b0005_basis_lam10
basis_lam10 3271 1 alligator_tarantula_b0005_basis_lam10
basis_lam10 3271 3 alligator_tarantula_b0005_basis_lam10
basis_lam10 6824 0 alligator_tarantula_b0005_basis_lam10
basis_lam10 6824 2 alligator_tarantula_b0005_basis_lam10
basis_lam10 6824 4 alligator_tarantula_b0005_basis_lam10
basis_lam10 7813 1 alligator_tarantula_b0005_basis_lam10
basis_lam10 7813 3 alligator_tarantula_b0005_basis_lam10
basis_lam10 4531 0 alligator_tarantula_b0005_basis_lam10
basis_lam10 4531 2 alligator_tarantula_b0005_basis_lam10
basis_lam10 4531 4 alligator_tarantula_b0005_basis_lam10
original 4951 4 bullfrog_dining_b001_basis
TRIALS
echo "Group 1: all 34 victims complete."
