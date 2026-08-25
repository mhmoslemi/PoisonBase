#!/usr/bin/env bash
#
# appendix.tex, tab:utility-defense -- FRIENDS calibration, weaker settings
#
# Extends the ap5-a.sh sweep downward. That sweep covered noise_eps in {2,4,6,8,12}
# and every one of them missed the band: undefended ConvNetBN is 81.05, the weakest
# tested setting (eps=2) landed at 75.74, and the curve is nearly flat across the
# whole range (75.74 / 75.59 / 75.32 / 75.38 / 73.89). Flat means eps is not the
# dominant knob THERE -- it does not mean the band is unreachable, because as
# eps -> 0 friendly noise vanishes and FRIENDS must degenerate to ordinary
# training. So the band, if it is reachable at all, is below 2.
#
# Clean utility only: 1 target, no attack outcome is used. ~7 min per setting.
#
#   sh appendix/ap5-a2.sh
#
# Read off the printed "defended clean CTA" and pick the strongest setting still
# within two points of 81.05, i.e. >= 79.05. If neither reaches it, FRIENDS cannot
# be utility-matched under this protocol and the table says so.

set -u

NV="${NUM_VICTIMS:-5}"

source /home/mmoslem3/ENV/bin/activate
cd /home/mmoslem3/scratch/attack_if

python -c 'import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)' || {
    echo "ap5-a2: no CUDA device visible -- get a GPU allocation first"; exit 1; }

echo "=== FRIENDS clean-utility calibration below eps=2 (undefended 81.05, band >= 79.05) ==="
for E in 1 0.5; do
    echo "--- FRIENDS noise_eps=$E ---"
    MODEL=ConvNetBN ATTACK=gradmatch CLASS_PAIR="dog-bird" BUDGETS="0.02" \
        SELS="random" DEFENSES="friends" NUM_VICTIMS="$NV" NUM_TARGETS=1 \
        NOISE_EPS="$E" sh ./defense.sh 2>&1 | grep -i "defended clean" || true
done

echo
echo "already measured, for comparison:"
echo "    eps=2   75.74      eps=6   75.32      eps=12  73.89"
echo "    eps=4   75.59      eps=8   75.38      undefended 81.05"
echo
echo "=== ap5-a2.sh finished ==="
