#!/usr/bin/env bash
#
# latex/greedy_table.tex -- GradMatch, airplane->frog, 4e-2.
#
# One cell, will need two allocations. Rerun to resume.
#
# Estimated ~6.9 h on an L40S. Self-contained, takes no arguments, skips finished
# cells, and resumes a killed one from its banked trials and crafted targets.
#
#   sh appendix/greedy/g8-gm-frog-4e-2.sh

set -u
cd /home/mmoslem3/scratch/attack_if
source /home/mmoslem3/ENV/bin/activate

if [ -z "${DRY_RUN:-}" ]; then
python -c 'import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)' || {
    echo "g8-gm-frog-4e-2.sh: no CUDA device visible -- get a GPU allocation first"; exit 1; }
fi

ATTACKS=gradmatch PAIRS=frog-airplane BUDGETS=0.04 sh appendix/greedy-convnet.sh
echo "=== g8-gm-frog-4e-2.sh finished ==="
