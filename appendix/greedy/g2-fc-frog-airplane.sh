#!/usr/bin/env bash
#
# latex/greedy_table.tex -- FC, airplane->frog, all six budgets.
#
# Independent of g1.
#
# Estimated ~5.3 h on an L40S. Self-contained, takes no arguments, skips finished
# cells, and resumes a killed one from its banked trials and crafted targets.
#
#   sh appendix/greedy/g2-fc-frog-airplane.sh

set -u
cd /home/mmoslem3/scratch/attack_if
source /home/mmoslem3/ENV/bin/activate

if [ -z "${DRY_RUN:-}" ]; then
python -c 'import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)' || {
    echo "g2-fc-frog-airplane.sh: no CUDA device visible -- get a GPU allocation first"; exit 1; }
fi

ATTACKS=fc PAIRS=frog-airplane sh appendix/greedy-convnet.sh
echo "=== g2-fc-frog-airplane.sh finished ==="
