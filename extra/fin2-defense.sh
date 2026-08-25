#!/usr/bin/env bash
#
# 2 of 5 -- appendix.tex tab:utility-defense (24 cells).
#
# 16 poisoned runs, 5 targets x 5 victims each. Every poison set is already on
# disk, so this is victim training only. Both defenses are calibrated on clean
# data and frozen:
#
#   EPIC     subset 0.05     clean CTA 80.33   (undefended 81.05, -0.72)
#   FRIENDS  clamp 4/255     clean CTA 80.02   (-1.03), noise_eps at the paper's 8
#
# ~10 h for EPIC and ~8.8 h for FRIENDS. Rerun until it finishes; completed runs
# skip themselves.
#
#   sh appendix/fin2-defense.sh
#
# Both halves pass TARGET_SELECT explicitly. sweep_config.json has no
# ConvNetBN/sapa entry, so without it the four SAPA rows die before running.

set -u
cd /home/mmoslem3/scratch/attack_if
source /home/mmoslem3/ENV/bin/activate

python -c 'import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)' || {
    echo "fin2: no CUDA device visible -- get a GPU allocation first"; exit 1; }

echo "########## EPIC half ##########"
sh appendix/ap5-b.sh || exit 1

echo
echo "########## FRIENDS half ##########"
sh appendix/ap5-c.sh || exit 1

echo "=== fin2-defense.sh finished ==="
