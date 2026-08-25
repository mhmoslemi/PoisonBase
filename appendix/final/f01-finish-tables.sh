#!/usr/bin/env bash
#
# Closes appendix.tex tab:augmentation-aware, tab:utility-defense and tab:computational-cost, then opens the budget table's 5e-3 column.
#
# The three defended sweeps here are resumes, not fresh runs: the RandAugment DPP
# arms stopped at 17/25 and 18/25 when their allocations expired, and the EPIC
# airplane->frog SAPA arm at 23/25. Each picks up from its own results_rank0.csv.
# Table work is ordered first so a short allocation still closes three tables; the
# 5e-3 budget column at the end is the part that can be dropped and rerun.
#
# Estimated ~5 h on an L40S. Self-contained: it takes no arguments, skips any
# unit already on disk, and a killed allocation just needs this same file rerun.
#
#   sh appendix/final/f01-finish-tables.sh

set -u
cd /home/mmoslem3/scratch/attack_if
source /home/mmoslem3/ENV/bin/activate

if [ -z "${DRY_RUN:-}" ]; then
python -c 'import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)' || {
    echo "f01-finish-tables.sh: no CUDA device visible -- get a GPU allocation first"; exit 1; }
fi

# tab:computational-cost -- the surrogate peak-memory cell. profile_selection.py
# cannot supply it: it reads training time off the cache without ever training.
python appendix/surrogate_mem.py --model ConvNetBN || echo "  (memory probe failed; not fatal)"

# tab:augmentation-aware -- the two RandAugment DPP cells (resumes at 17/25, 18/25)
STEP=ra_gm   SELS=dpp sh appendix/ap4-augaware.sh
STEP=ra_sapa SELS=dpp sh appendix/ap4-augaware.sh

# tab:utility-defense -- EPIC / airplane->frog / SAPA, the last row (resumes at 23/25)
PAIRS=frog-airplane sh appendix/ap5-b.sh

# tab:selection-ladder-budget -- the 5e-3 column. Random and DPP are already on
# disk from the appendix protocol and identical to what this would craft, so
# fin4 skips them and only these three run.
BUDGETS=0.005 CRITS="bottom first pixel" sh appendix/fin4-budget.sh
echo "=== f01-finish-tables.sh finished ==="
