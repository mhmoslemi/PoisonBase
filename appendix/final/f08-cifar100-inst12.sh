#!/usr/bin/env bash
#
# appendix.tex tab:cross-dataset -- CIFAR-100 attack instances 1 and 2.
#
# 8 runs: GM and SAPA, Random and DPP, one pinned target and five victims each.
# The 20 CIFAR-100 surrogates are already cached, so this reads them and does not
# retrain. The probe step reprints the ConvNetBN-vs-ResNet18BN clean accuracies
# from cache in about seven minutes -- that pair of numbers is what justifies the
# ResNet18BN backbone in the text.
#
# Estimated ~6 h on an L40S. Self-contained: it takes no arguments, skips any
# unit already on disk, and a killed allocation just needs this same file rerun.
#
#   sh appendix/final/f08-cifar100-inst12.sh

set -u
cd /home/mmoslem3/scratch/attack_if
source /home/mmoslem3/ENV/bin/activate

if [ -z "${DRY_RUN:-}" ]; then
python -c 'import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)' || {
    echo "f08-cifar100-inst12.sh: no CUDA device visible -- get a GPU allocation first"; exit 1; }
fi

STEP=probe sh appendix/ap2-cifar100.sh || echo "  (probe failed; not fatal)"
STEP=runs INSTANCES="1 2" sh appendix/ap2-cifar100.sh
echo "=== f08-cifar100-inst12.sh finished ==="
