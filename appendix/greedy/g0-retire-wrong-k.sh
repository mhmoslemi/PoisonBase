#!/usr/bin/env bash
#
# latex/greedy_table.tex -- retire the 18 Greedy directories that were selected over K=5 or K=15.
#
# The published Random and DPP cells were selected over K=20 surrogates. Putting a
# K=5 Greedy row beside them is not a controlled comparison: Greedy averages its
# pointwise score over those K nets, so any Greedy-vs-DPP gap could be the surrogate
# count. Run this once, before anything else.
#
# Estimated ~1 s, no GPU on an L40S. Self-contained, takes no arguments, skips finished
# cells, and resumes a killed one from its banked trials and crafted targets.
#
#   sh appendix/greedy/g0-retire-wrong-k.sh

set -u
cd /home/mmoslem3/scratch/attack_if
source /home/mmoslem3/ENV/bin/activate

if [ -z "${DRY_RUN:-}" ]; then
python -c 'import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)' || {
    echo "g0-retire-wrong-k.sh: no CUDA device visible -- get a GPU allocation first"; exit 1; }
fi

# Renames, never deletes: every directory below keeps its log, its CSVs and its
# poison cache under a .K<n> suffix, so nothing is lost and any of it can be put
# back with the reverse mv. This has to run before g1-g13, because num_surrogates
# is not part of the run name -- without it those runs would reload the old K=5
# poison cache and silently keep the old selection while logging K=20.

retire() {
    D="ours_result/$1"
    [ -d "$D" ] || { echo "    absent: $1"; return 0; }
    [ -d "$D.K$2" ] && { echo "    already retired: $1"; return 0; }
    if [ -n "${DRY_RUN:-}" ]; then echo "    would retire K=$2: $1"; return 0; fi
    mv "$D" "$D.K$2" && echo "    retired K=$2: $1"
}

echo "=== retiring Greedy directories selected over K != 20 ==="
retire "CIFAR10_ConvNetBN_fc_ours_dog-bird_b0.001_eps8_seed42_lam1_cosine_ce5_tgt50" "5"
retire "CIFAR10_ConvNetBN_fc_ours_dog-bird_b0.002_eps8_seed42_lam1_cosine_ce5_tgt50" "5"
retire "CIFAR10_ConvNetBN_fc_ours_dog-bird_b0.005_eps8_seed42_lam1_cosine_ce5_tgt50" "5"
retire "CIFAR10_ConvNetBN_fc_ours_dog-bird_b0.01_eps8_seed42_lam1_cosine_ce5_tgt50" "5"
retire "CIFAR10_ConvNetBN_fc_ours_dog-bird_b0.02_eps8_seed42_lam1_cosine_ce5_tgt50" "5"
retire "CIFAR10_ConvNetBN_fc_ours_dog-bird_b0.04_eps8_seed42_lam1_cosine_ce5_tgt50" "5"
retire "CIFAR10_ConvNetBN_fc_ours_frog-airplane_b0.005_eps8_seed42_lam1_cosine_ce5_tgt20" "15"
retire "CIFAR10_ConvNetBN_gradmatch_ours_dog-bird_b0.001_eps8_seed42_lam1_cosine_ce5_tgt70" "5"
retire "CIFAR10_ConvNetBN_gradmatch_ours_dog-bird_b0.002_eps8_seed42_lam1_cosine_ce5_tgt70" "5"
retire "CIFAR10_ConvNetBN_gradmatch_ours_dog-bird_b0.005_eps8_seed42_lam1_cosine_ce5_tgt70" "5"
retire "CIFAR10_ConvNetBN_gradmatch_ours_dog-bird_b0.01_eps8_seed42_lam1_cosine_ce5_tgt70" "5"
retire "CIFAR10_ConvNetBN_gradmatch_ours_dog-bird_b0.02_eps8_seed42_lam1_cosine_ce5_tgt70" "5"
retire "CIFAR10_ConvNetBN_gradmatch_ours_frog-airplane_b0.001_eps8_seed42_lam1_cosine_ce5_tgt35" "5"
retire "CIFAR10_ConvNetBN_gradmatch_ours_frog-airplane_b0.002_eps8_seed42_lam1_cosine_ce5_tgt35" "5"
retire "CIFAR10_ConvNetBN_gradmatch_ours_frog-airplane_b0.005_eps8_seed42_lam1_cosine_ce5_tgt35" "5"
retire "CIFAR10_ConvNetBN_gradmatch_ours_frog-airplane_b0.01_eps8_seed42_lam1_cosine_ce5_tgt35" "5"
retire "CIFAR10_ConvNetBN_gradmatch_ours_frog-airplane_b0.02_eps8_seed42_lam1_cosine_ce5_tgt35" "5"
retire "CIFAR10_ConvNetBN_gradmatch_ours_frog-airplane_b0.04_eps8_seed42_lam1_cosine_ce5_tgt35" "5"

echo
echo 'To undo:  for d in ours_result/*.K5 ours_result/*.K15; do mv "$d" "${d%.K*}"; done'

echo "=== g0-retire-wrong-k.sh finished ==="
