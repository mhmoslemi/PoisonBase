#!/usr/bin/env bash
#
# Everything still missing from defense_table.tex: the FRIENDS half of all four
# FC rows, 8 cells, 160 trials. Runs them one after another and stops on the
# first failure.
#
#     FC / dog-bird      / 2e-3 / FRIENDS   Random + DPP
#     FC / dog-bird      / 1e-2 / FRIENDS   Random + DPP
#     FC / frog-airplane / 2e-3 / FRIENDS   Random + DPP
#     FC / frog-airplane / 1e-2 / FRIENDS   Random + DPP
#
# 5 targets x 4 victims per cell, same protocol as the rest of the table.
# Nothing is crafted -- the saved perturbations are replayed from ours_result,
# only the victim's training changes.
#
# 7.2 h on an L40S.
#
# Resumes from each run's own results.csv, so if it dies part way just run it
# again and it picks up at the first missing trial.
#
#   sh dall.sh
#   DRY_RUN=1 sh dall.sh    # print the defense.py commands and stop

set -u

cd /home/mmoslem3/scratch/attack_if

sh ./preflight_cuda.sh || exit 1

run_row() {
    echo "=== dall | FC / $1 / budget $2 / FRIENDS | Random + DPP ==="
    MODEL="${MODEL:-ResNet20BN}" \
    ATTACK=fc \
    CLASS_PAIR="$1" \
    BUDGETS="$2" \
    SELS="random dpp" \
    DEFENSES="friends" \
    NUM_VICTIMS="${NUM_VICTIMS:-4}" \
    sh ./defense.sh || exit 1
    echo
}

run_row dog-bird      0.002
run_row dog-bird      0.01
run_row frog-airplane 0.002
run_row frog-airplane 0.01

echo "=== dall.sh finished ==="
