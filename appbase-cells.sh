#!/usr/bin/env bash
#
# The experiments for the remaining cells of latex/app-base.tex. Nothing else.
#
#   sh appbase-cells.sh
#
#   tab:selection-ladder         ResNet20BN/GM  bottom, el2n, relevance  (23,22,22 /25)  ~10 min
#   tab:selection-ladder         ResNet20BN/FC  boundary (12/25), dpp (1/25)             ~1.5 h
#   tab:selection-ladder-budget  ConvNetBN/GM b0.01 bottom (5/25)                        ~1.5 h
#
# ~3 h on an L40S. Skips what is done, resumes what was killed, takes no arguments.
# The three ResNet/GM rules are nearly finished, so they run first: a 15-minute
# allocation still closes three cells.

set -u
cd /home/mmoslem3/scratch/attack_if
source /home/mmoslem3/ENV/bin/activate

if [ -z "${DRY_RUN:-}" ]; then
python -c 'import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)' || {
    echo "appbase-cells.sh: no CUDA device visible -- get a GPU allocation first"; exit 1; }
fi

echo "########## tab:selection-ladder -- ResNet20BN / GRADMATCH ##########"
MODELS=ResNet20BN ATTACKS=gradmatch CRITS="bottom el2n relevance" sh appendix/fin3-ladder.sh

echo
echo "########## tab:selection-ladder -- ResNet20BN / FC ##########"
MODELS=ResNet20BN ATTACKS=fc CRITS="boundary dpp" sh appendix/fin3-ladder.sh

echo
echo "########## tab:selection-ladder-budget -- ConvNetBN / GM at 1e-2 ##########"
BUDGETS=0.01 CRITS=bottom sh appendix/fin4-budget.sh

echo
echo "=== appbase-cells.sh finished ==="
