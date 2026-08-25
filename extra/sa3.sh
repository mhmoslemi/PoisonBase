#!/usr/bin/env bash
#
# SAPA / random-base sweep shard -- the cells still missing from the
# "ResNet20 ... SAPA random" row(s) of table.tex.
#
# The 4e-2 runs are NOT here: they are already running. This shard covers
# budgets 0.001 0.01 for ResNet20BN / frog-airplane.
#
# Nothing on disk yet for these -- all 2 runs start from zero
# (0/60 trials, empty poison_cache), which is why they cost what they do.
#
# Cost, from rates measured in this project's own sapa logs
# (victim s/trial: ConvNet 50, ResNet20 130, VGG13 100; craft s/target fitted
# linearly in N_p per model from the b0.002/0.005/0.02/0.04 runs):
#
#   ResNet20 frog-airplane b0.001 10 crafts x   58 s + 60 trials x 130 s   ~2.33 h
#   ResNet20 frog-airplane b0.01 10 crafts x 1080 s + 60 trials x 130 s   ~5.17 h
#   ------------------------------------------------------------------
#   L40S  ~7.5 h      H100  ~4.2 h  (range 3.4-4.4 h, see note)
#
# The H100 number is an extrapolation, not measured. These runs are
# craft-dominated at b0.01 (large batched PGD -- scales well, nearer 2.2x) and
# victim-training-dominated at b0.001 (50 epochs at batch 125 -- launch-latency
# bound, nearer 1.5x), so treat ~1.8x as the blended figure and size the
# allocation off the L40S number if you want a safety margin.
#
# Targets come from the pinned target_sets/ResNet20BN_gradmatch_<pair>.json, which
# is what keeps sapa paired with gradmatch on identical images; TARGET_SELECT is
# passed empty and is ignored anyway because the combo is already pinned.
#
# sel_dpp.sh passes no --no_resume, so a wall-clock kill just needs a rerun.
#
#     sh sa3.sh
#     BUDGETS="0.001" sh sa3.sh     # narrow it if the allocation is short

set -u

MODEL_=ResNet20BN
PAIRS_="frog-airplane"
BUDGETS_="${BUDGETS:-0.001 0.01}"

cd /home/mmoslem3/scratch/attack_if

for pair in $PAIRS_; do
    echo "=== sa3 | sapa worst sigma=0.05 | $MODEL_ / random / $pair | budgets $BUDGETS_ ==="
    MODEL="$MODEL_" ATTACK=sapa CLASS_PAIR="$pair" \
        BUDGETS="$BUDGETS_" SELECT=random \
        SHARP_MODE=worst SHARP_SIGMA=0.05 TARGET_SELECT="" \
        sh sel_dpp.sh || exit 1
done

echo "=== sa3.sh finished ==="
