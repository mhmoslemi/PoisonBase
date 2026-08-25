#!/usr/bin/env bash
#
# 5 of 5 -- appendix.tex tab:cross-dataset TinyImageNet rows (6 cells) and the
#           TinyImageNet half of tab:additional-target-indices (4 cells).
#
# 20 attack runs: 5 class-pair instances x {GM, SAPA} x {Random, DPP}, one pinned
# target and five victims each, ResNet18BN at a poison budget of 1e-3.
#
# All 20 surrogates are cached, so no run here pays the 47-minute surrogate cost.
# One attack run is ~3.7 h (craft 1666 s + 5 victims at 2327 s), so ~74 h in total
# and about nine allocations. Every unit skips itself if already on disk -- a killed
# job just needs this file rerun. Needs --mem=32G.
#
#   sh appendix/fin5-tinyimagenet.sh

set -u
cd /home/mmoslem3/scratch/attack_if
source /home/mmoslem3/ENV/bin/activate

python -c 'import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)' || {
    echo "fin5: no CUDA device visible -- get a GPU allocation first"; exit 1; }
[ -s "/home/mmoslem3/scratch/data/tinyimagenet.pt" ] || {
    echo "TinyImageNet not prepared -- python appendix/prep_tinyimagenet.py --src tiny-imagenet-200.zip"
    exit 1; }

SDIR="./cache/surrogates/TinyImageNet_ResNet18BN_60ep_lr0.1_bs128_seed42"
HAVE=$(ls "$SDIR" 2>/dev/null | wc -l)
echo "TinyImageNet surrogates cached: $HAVE / 20"
[ "$HAVE" -ge 20 ] || echo "  (missing ones will be trained on demand, ~47 min each)"

sh appendix/ap2-4.sh || exit 1

echo "=== fin5-tinyimagenet.sh finished ==="
