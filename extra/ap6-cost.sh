#!/usr/bin/env bash
#
# appendix.tex, tab:computational-cost -- selection cost and attack-construction cost
#
# ConvNetBN, K=20, the full adversarial-class candidate pool, averaged over the five
# dog -> bird targets.  Nothing is attacked and no victim is trained: the upper half
# of the table is pure selector timing, and the lower half is poison-CONSTRUCTION
# time, which is also measurable without training a victim.
#
#   upper half   train 20 surrogates (once)  -- read off the cache the runs share
#                candidate scoring           -- timed here
#                Greedy subset selection     -- timed here
#                DPP subset selection        -- timed here
#                peak GPU memory per stage   -- timed here
#
#   lower half   end-to-end attack construction for FC / GM / SAPA under Random and
#                DPP.  Poison optimization is identical between the two, so the DPP
#                column is the Random craft time plus that selection's time, and
#                Overhead is the ratio.  Craft times are mined from the ConvNetBN
#                logs already on disk -- nothing is re-crafted.
#
# Minutes, not hours.  Run this one first.
#
#   sh appendix/ap6-cost.sh

set -u

source /home/mmoslem3/ENV/bin/activate
cd /home/mmoslem3/scratch/attack_if

python -c 'import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)' || {
    echo "ap6: no CUDA device visible -- get a GPU allocation first"; exit 1; }

mkdir -p appendix/out

echo "=== upper half: selection cost (ConvNetBN, K=20, 5 targets) ==="
python appendix/profile_selection.py --model ConvNetBN --mode cost \
    --class_pair dog-bird --budget 0.005 --num_targets 5 \
    --out appendix/out/cost_ConvNetBN.json || exit 1

echo
echo "=== lower half: attack-construction time, from the ConvNetBN logs on disk ==="
python appendix/end_to_end_cost.py

echo
echo "=== ap6-cost.sh finished -- numbers in appendix/out/ ==="
