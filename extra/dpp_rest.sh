#!/usr/bin/env bash
#
# Everything still missing from the dpp (SEL_ALPHA=2.0) column of table.tex,
# across fc AND gradmatch, all 12 model/pair/attack combos.
#
# What is NOT here, and why:
#   fc, every model/pair/budget          -- all 36 dpp cells already done
#   gradmatch, ConvNetBN both pairs      -- all 12 dpp cells already done
#   gradmatch b0.02 (4 combos)           -- complete on disk (their ASR/CTA are
#                                           already the 2e-2 cells in table.tex)
#   ResNet20BN gradmatch dog-bird b0.001 -- r6.sh is running it
#   ResNet20BN gradmatch dog-bird b0.002 -- r8.sh is running it
#   ResNet20BN gradmatch frog-air b0.005 -- r7.sh is running it
#   random base / smart variants         -- random is finished, smart is
#                                           deliberately partial
#
# That leaves the 5 runs below, all gradmatch, all from scratch (0/60 trials,
# no poison_cache), which is why they are expensive.
#
# Work, from the median victim / craft times these combos actually logged
# (ResNet20 victim ~131 s, VGG13 ~100 s; craft/target interpolated linearly in
# N_p from the b0.005 and b0.02 runs of the same combo):
#
#   #  combo                                  60 trials   10 crafts     total
#   1  ResNet20BN gradmatch dog-bird  b0.005    2.2 h    10x472 s      ~3.5 h
#   2  VGG13BN    gradmatch frog-air  b0.01     1.7 h    10x1090 s     ~4.7 h
#   3  VGG13BN    gradmatch dog-bird  b0.01     1.7 h    10x1090 s     ~4.7 h
#   4  ResNet20BN gradmatch frog-air  b0.01     2.2 h    10x1020 s     ~5.0 h
#   5  ResNet20BN gradmatch dog-bird  b0.01     2.2 h    10x1020 s     ~5.0 h
#   ----------------------------------------------------------------------
#   ~23 h if you run them one after another, so DO NOT do that on one GPU.
#
# Run them as separate shards, one per allocation/GPU -- JOBS picks which:
#
#     JOBS=1 sh dpp_rest.sh &   JOBS=2 sh dpp_rest.sh &   ...   # 5 GPUs, ~5 h
#     JOBS="1 2" sh dpp_rest.sh                                 # one GPU, ~8 h
#     sh dpp_rest.sh                                            # all 5, ~23 h
#
# Cheapest first, so a short allocation still lands a complete budget.
#
# Each job delegates to sel_dpp.sh with SELECT=dpp, so targets come from the
# pinned target_sets/<MODEL>_gradmatch_<PAIR>.json and the craft-memory flags
# come from sweep_config.json -- exactly the way every other dpp run in
# dpp_done.txt was produced. Nothing is duplicated here.
#
# A job whose run already has a final summary is skipped, so this is safe to
# re-run after a wall-clock kill; an interrupted job resumes at its first
# missing (target, victim) trial because sel_dpp.sh passes no --no_resume.

set -u

JOBS="${JOBS:-1 2 3 4 5}"
OUT_DIR=ours_result

cd /home/mmoslem3/scratch/attack_if

# cheapest first: "<model> <pair> <budget> <difficulty label>"
job_1="ResNet20BN dog-bird      0.005 14"
job_2="VGG13BN    frog-airplane 0.01  12"
job_3="VGG13BN    dog-bird      0.01  50"
job_4="ResNet20BN frog-airplane 0.01  10"
job_5="ResNet20BN dog-bird      0.01  14"

for j in $JOBS; do
    eval "spec=\${job_$j:-}"
    [ -n "$spec" ] || { echo "dpp_rest.sh: no such job '$j' (expected 1..5)"; exit 1; }
    # shellcheck disable=SC2086
    set -- $spec
    model=$1; pair=$2; bud=$3; tg=$4

    TAG="CIFAR10_${model}_gradmatch_ours_${pair}_b${bud}_eps8_seed42_lam1_cosine_seldpp2_ce5_tgt${tg}"
    if grep -q "==== $TAG : ASR = " "$OUT_DIR/$TAG/log.txt" 2>/dev/null; then
        echo "=== dpp_rest job $j | SKIP $model / gradmatch / $pair / b$bud -- already complete:"
        grep -o "ASR = .*====" "$OUT_DIR/$TAG/log.txt" | tail -1
        continue
    fi

    echo "=== dpp_rest job $j | dpp alpha=2.0 | $model / gradmatch / $pair | budget $bud ==="
    MODEL="$model" ATTACK=gradmatch CLASS_PAIR="$pair" BUDGETS="$bud" \
        SELECT=dpp SEL_ALPHA=2.0 TARGET_SELECT="" \
        sh sel_dpp.sh
done

echo "=== dpp_rest.sh finished (jobs: $JOBS) ==="
