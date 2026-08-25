#!/usr/bin/env bash
#
# app-base.tex tab:selection-ladder -- the ResNet20BN/GRADMATCH column, rules 1-4.
#
# Nothing of this column is on disk. GradMatch crafting on ResNet20BN is ~300 s a
# target against FC's 5 s, so a rule is ~1.4 h here and the column needs three
# shards. f04, f05 and f06 are independent of each other.
#
# Estimated ~5.6 h on an L40S. Self-contained: it takes no arguments, skips any
# unit already on disk, and a killed allocation just needs this same file rerun.
#
#   sh appendix/final/f04-ladder-resnet-gm-1.sh

set -u
cd /home/mmoslem3/scratch/attack_if
source /home/mmoslem3/ENV/bin/activate

if [ -z "${DRY_RUN:-}" ]; then
python -c 'import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)' || {
    echo "f04-ladder-resnet-gm-1.sh: no CUDA device visible -- get a GPU allocation first"; exit 1; }
fi

MODELS=ResNet20BN ATTACKS=gradmatch CRITS="bottom first random grand" sh appendix/fin3-ladder.sh
echo "=== f04-ladder-resnet-gm-1.sh finished ==="
