#!/usr/bin/env bash
#
# latex/greedy_table.tex -- SAPA, bird->dog, 1e-3 through 1e-2.
#
# 1e-3, 5e-3 and 1e-2 already have banked trials from an earlier 5x4 run against
# the same pinned targets, so these resume rather than start cold.
#
# Estimated ~5.0 h on an L40S. Self-contained, takes no arguments, skips finished
# cells, and resumes a killed one from its banked trials and crafted targets.
#
#   sh appendix/greedy/g9-sapa-dog-bird-lo.sh

set -u
cd /home/mmoslem3/scratch/attack_if
source /home/mmoslem3/ENV/bin/activate

if [ -z "${DRY_RUN:-}" ]; then
python -c 'import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)' || {
    echo "g9-sapa-dog-bird-lo.sh: no CUDA device visible -- get a GPU allocation first"; exit 1; }
fi

ATTACKS=sapa PAIRS=dog-bird BUDGETS="0.001 0.002 0.005 0.01" sh appendix/greedy-convnet.sh
echo "=== g9-sapa-dog-bird-lo.sh finished ==="
