#!/usr/bin/env bash
#
# latex/greedy_table.tex -- GradMatch, bird->dog, 1e-3 through 1e-2.
#
# GM crafting is ~1.1 s a poison, so cost is set by the budget: 1 h a cell at
# 1e-3 rising to 6.9 h at 4e-2. That is why GM and SAPA are split by budget.
#
# Estimated ~6.1 h on an L40S. Self-contained, takes no arguments, skips finished
# cells, and resumes a killed one from its banked trials and crafted targets.
#
#   sh appendix/greedy/g3-gm-dog-bird-lo.sh

set -u
cd /home/mmoslem3/scratch/attack_if
source /home/mmoslem3/ENV/bin/activate

if [ -z "${DRY_RUN:-}" ]; then
python -c 'import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)' || {
    echo "g3-gm-dog-bird-lo.sh: no CUDA device visible -- get a GPU allocation first"; exit 1; }
fi

ATTACKS=gradmatch PAIRS=dog-bird BUDGETS="0.001 0.002 0.005 0.01" sh appendix/greedy-convnet.sh
echo "=== g3-gm-dog-bird-lo.sh finished ==="
