#!/usr/bin/env bash
#
# tab:defense-robustness -- FC / dog-bird / $2{x}10^{-3}$ / EPIC
#
# Fills this half-row of defense_table.tex (ResNet20BN, Random and DPP side by
# side, so the pair is judged on the same pinned targets):
#
#     dog--bird & FC & $2{\times}10^{-3}$ & .. & .. & <Random> & <DPP> & .. & ..
#
# The No Defense columns are NOT run here -- they are recomputed from
# ours_result over the same 5 targets and victims 0-3. Nothing is crafted: the
# saved perturbations are replayed straight out of ours_result/, so the only
# thing that differs from the undefended run is what the victim does while
# training.
#
# PROTOCOL: 5 targets x 4 victims = 20 trials per run, matching the runs already
# in defense_result.txt (defense.py's --num_targets default of 5, and
# NUM_VICTIMS=4 passed here -- defense.sh's own default is 6, which would
# silently give a different protocol than the rest of the table).
#
# COST, from rates measured in defense_result.txt itself:
#   epic    392 s/trial
#   friends 517 s/trial + a 61 s friendly-noise pass per trial = 578 s
#   -> 2 run(s) x 20 trials x 392 s
#      L40S ~4.4 h      H100 ~2.4 h  (at ~1.8x; extrapolated, not measured)
#
# defense.py resumes from results.csv, so a wall-clock kill just needs a rerun.
#
#   sh fcdef1.sh
#   DRY_RUN=1 sh fcdef1.sh    # print the defense.py commands and stop

sh "$(dirname "$0")/preflight_cuda.sh" || exit 1

MODEL="${MODEL:-ResNet20BN}" \
ATTACK=fc \
CLASS_PAIR="dog-bird" \
BUDGETS="0.002" \
SELS="random dpp" \
DEFENSES="epic" \
NUM_VICTIMS="${NUM_VICTIMS:-4}" \
sh "$(dirname "$0")/defense.sh" || exit 1

echo "=== fcdef1.sh finished ==="
