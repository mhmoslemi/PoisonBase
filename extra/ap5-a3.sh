#!/usr/bin/env bash
#
# appendix.tex, tab:utility-defense -- FRIENDS calibration on the RIGHT knob
#
# ap5-a.sh swept --noise_eps over {2,4,6,8,12} and ap5-a2.sh extended it down to
# {1, 0.5}. Clean CTA never moved: 75.83 / 75.79 / 75.74 / 75.59 / 75.32 / 75.38 /
# 73.89 against an undefended 81.05. Flat all the way to eps=0.5 is the tell --
# if the loss came from the noise being added, it would vanish as eps -> 0.
#
# It does not, because --noise_eps scales only the RANDOM (bernoulli) half of
# FRIENDS. defense.py builds noise_type = ['friendly', 'bernoulli'] for
# --defense friends, then drops 'friendly' from the randomly-scaled kinds; the
# friendly noise is bounded by --friendly_clamp instead, 16/255 in every run
# above. That term is the defense proper and it is what costs the ~5.3 points.
#
# So this sweeps --friendly_clamp with --noise_eps held at the paper's default 8,
# i.e. it retunes the friendly-noise magnitude and leaves the random-noise
# component exactly as published.
#
# Clean utility only: 1 target, no attack outcome is used. ~7 min per setting.
#
#   sh appendix/ap5-a3.sh
#
# Pick the STRONGEST (largest) clamp still within two points of 81.05, i.e.
# >= 79.05, then run stage 2 with it.

set -u

NV="${NUM_VICTIMS:-5}"

source /home/mmoslem3/ENV/bin/activate
cd /home/mmoslem3/scratch/attack_if

python -c 'import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)' || {
    echo "ap5-a3: no CUDA device visible -- get a GPU allocation first"; exit 1; }

echo "=== FRIENDS clean-utility calibration over friendly_clamp (undefended 81.05, band >= 79.05) ==="
for C in 8 4 2 1; do
    echo "--- FRIENDS friendly_clamp=$C/255 (noise_eps=8) ---"
    MODEL=ConvNetBN ATTACK=gradmatch CLASS_PAIR="dog-bird" BUDGETS="0.02" \
        SELS="random" DEFENSES="friends" NUM_VICTIMS="$NV" NUM_TARGETS=1 \
        NOISE_EPS=8 FRIENDLY_CLAMP="$C" sh ./defense.sh 2>&1 | grep -i "defended clean" || true
done

echo
echo "for comparison, clamp=16 (the default) across the whole noise_eps sweep:"
echo "    eps=0.5  75.79     eps=4   75.59     eps=12  73.89"
echo "    eps=1    75.83     eps=6   75.32     undefended  81.05"
echo "    eps=2    75.74     eps=8   75.38"
echo
echo "=== ap5-a3.sh finished ==="
