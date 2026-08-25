#!/usr/bin/env bash
#
# app-base.tex tab:selection-ladder-budget -- the 1e-2 column.
#
# Five rules on ConvNetBN/GM at 500 poisons a target.
#
# Estimated ~5 h on an L40S. Self-contained: it takes no arguments, skips any
# unit already on disk, and a killed allocation just needs this same file rerun.
#
#   sh appendix/final/d2-budget-b0.01.sh

set -u
cd /home/mmoslem3/scratch/attack_if
source /home/mmoslem3/ENV/bin/activate

if [ -z "${DRY_RUN:-}" ]; then
python -c 'import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)' || {
    echo "d2-budget-b0.01.sh: no CUDA device visible -- get a GPU allocation first"; exit 1; }
fi

BUDGETS=0.01 sh appendix/fin4-budget.sh
echo "=== d2-budget-b0.01.sh finished ==="
