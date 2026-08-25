#!/usr/bin/env bash
#
# latex/greedy_table.tex -- GradMatch, bird->dog at 2e-2 plus airplane->frog at 1e-3 and 2e-3.
#
# Three cells that pack into one allocation.
#
# Estimated ~6.0 h on an L40S. Self-contained, takes no arguments, skips finished
# cells, and resumes a killed one from its banked trials and crafted targets.
#
#   sh appendix/greedy/g4-gm-2e-2-and-frog-lo.sh

set -u
cd /home/mmoslem3/scratch/attack_if
source /home/mmoslem3/ENV/bin/activate

if [ -z "${DRY_RUN:-}" ]; then
python -c 'import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)' || {
    echo "g4-gm-2e-2-and-frog-lo.sh: no CUDA device visible -- get a GPU allocation first"; exit 1; }
fi

ATTACKS=gradmatch PAIRS=dog-bird BUDGETS=0.02 sh appendix/greedy-convnet.sh
ATTACKS=gradmatch PAIRS=frog-airplane BUDGETS="0.001 0.002" sh appendix/greedy-convnet.sh
echo "=== g4-gm-2e-2-and-frog-lo.sh finished ==="
