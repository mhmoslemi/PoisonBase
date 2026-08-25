#!/usr/bin/env bash
#
# latex/greedy_table.tex -- GradMatch, bird->dog, 4e-2.
#
# One cell. 2000 poisons a target puts crafting near 2200 s, so this will not
# finish in a 6 h allocation -- rerun this same file and it resumes per target.
#
# Estimated ~6.9 h on an L40S. Self-contained, takes no arguments, skips finished
# cells, and resumes a killed one from its banked trials and crafted targets.
#
#   sh appendix/greedy/g5-gm-dog-bird-4e-2.sh

set -u
cd /home/mmoslem3/scratch/attack_if
source /home/mmoslem3/ENV/bin/activate

if [ -z "${DRY_RUN:-}" ]; then
python -c 'import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)' || {
    echo "g5-gm-dog-bird-4e-2.sh: no CUDA device visible -- get a GPU allocation first"; exit 1; }
fi

ATTACKS=gradmatch PAIRS=dog-bird BUDGETS=0.04 sh appendix/greedy-convnet.sh
echo "=== g5-gm-dog-bird-4e-2.sh finished ==="
