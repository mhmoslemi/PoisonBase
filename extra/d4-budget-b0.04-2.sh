#!/usr/bin/env bash
#
# app-base.tex tab:selection-ladder-budget -- the 4e-2 column, part 2 of 3.
#
# 2000 poisons a target puts crafting near 1900 s, so this column is split three ways.
#
# Estimated ~6 h on an L40S. Self-contained: it takes no arguments, skips any
# unit already on disk, and a killed allocation just needs this same file rerun.
#
#   sh appendix/final/d4-budget-b0.04-2.sh

set -u
cd /home/mmoslem3/scratch/attack_if
source /home/mmoslem3/ENV/bin/activate

if [ -z "${DRY_RUN:-}" ]; then
python -c 'import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)' || {
    echo "d4-budget-b0.04-2.sh: no CUDA device visible -- get a GPU allocation first"; exit 1; }
fi

BUDGETS=0.04 CRITS="first pixel" sh appendix/fin4-budget.sh
echo "=== d4-budget-b0.04-2.sh finished ==="
