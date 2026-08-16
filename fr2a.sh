#!/usr/bin/env bash
#
# tab:defense-robustness -- FRIENDS / GM / frog-airplane / $2{\times}10^{-3}$ / Random + DPP
#
# Shard 1 of 3, split out of the old fr2.sh (which was 4 runs / ~12.8 h in one
# file). Together fr2a + fr2b + fr2c fill the four FRIENDS cells of the two
# frog-airplane GM rows:
#
#     frog--airplane & GM & $2{\times}10^{-3}$ & .. & .. & .. & .. & <Random> & <DPP>   <- fr2a
#     frog--airplane & GM & $2{\times}10^{-2}$ & .. & .. & .. & .. & <Random> & ..      <- fr2b
#     frog--airplane & GM & $2{\times}10^{-2}$ & .. & .. & .. & .. & ..       & <DPP>   <- fr2c
#
# Nothing is crafted -- the saved perturbations are replayed from ours_result/.
# The No Defense columns are not run here; they are recomputed from the same 5
# targets and victims 0-3.
#
# PROTOCOL: 5 targets x 4 victims = 20 trials per run, matching everything
# already in defense_result.txt (NUM_VICTIMS=4 is passed explicitly --
# defense.sh's own default is 6, which would be a different protocol).
#
# COST, from rates measured in defense_result.txt: FRIENDS is 517 s/trial plus a
# 61 s friendly-noise pass per trial = 578 s.
#   2 run(s) x 20 trials x 578 s
#   L40S ~6.4 h      H100 ~3.6 h  (at ~1.8x; extrapolated, not measured)
#
# defense.py resumes from results.csv, so a wall-clock kill just needs a rerun.
#
#   sh fr2a.sh
#   DRY_RUN=1 sh fr2a.sh    # print the defense.py command and stop

sh "$(dirname "$0")/preflight_cuda.sh" || exit 1

MODEL="${MODEL:-ResNet20BN}" \
ATTACK=gradmatch \
CLASS_PAIR="frog-airplane" \
BUDGETS="0.002" \
SELS="random dpp" \
DEFENSES="friends" \
NUM_VICTIMS="${NUM_VICTIMS:-4}" \
sh "$(dirname "$0")/defense.sh" || exit 1

echo "=== fr2a.sh finished ==="
