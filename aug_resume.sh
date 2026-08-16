#!/usr/bin/env bash
#
# Resume the four Crop+Flip (aug standard) runs killed in aug_crash.txt.
#
# All four died on the Slurm wall clock (jobs 4813194/98/99 and the fc b0.01
# shard, CANCELLED 12:37-12:39 on 2026-08-16), not on an error, so every trial
# already in defense_result/<tag>/results*.csv stays.
#
# Each job re-enters aug.sh with the SAME env the original shard used, so the
# target pinning is recomputed identically:
#
#   PAIR_SELS="random dpp"   the pinned target set is the intersection over BOTH
#                            selections, so it must stay "random dpp" even when
#                            SELS runs only one of them -- otherwise this shard
#                            would pin a different set than the row it belongs to
#   NUM_TARGETS=5            first 5 of that intersection
#   NUM_VICTIMS=4            victim ids 0..3
#   AUGS=standard            Crop+Flip only; the randaug/cutout cells are done
#
# aug.sh passes no --no_resume, so defense.py picks up at the first missing
# (target, victim) trial. Nothing is re-crafted -- the poisons are replayed from
# ours_result and the augmentation is resampled every epoch.
#
# Work left, counted from results*.csv restricted to the 5 pinned targets and
# victims 0-3 (~140 s/trial, measured from these runs' own logs):
#
#   #  combo                                sel      done   left    est
#   1  GM  / dog-bird      / b0.02          random   18/20    2    ~0.08 h
#   2  FC  / frog-airplane / b0.002         random   16/20    4    ~0.16 h
#   3  FC  / frog-airplane / b0.01          random   14/20    6    ~0.23 h
#   4  GM  / frog-airplane / b0.02          dpp       9/20   11    ~0.43 h
#   ---------------------------------------------------------------------
#   ~0.89 h for all four, so a single 1.5 h allocation covers it.
#
# Cheapest first. JOBS picks which to run:
#
#     sh aug_resume.sh                 # all four, ~0.9 h
#     JOBS="1 2" sh aug_resume.sh      # just those
#     DRY_RUN=1 sh aug_resume.sh       # print the defense.py commands and stop
#
# DRY_RUN is passed straight through to aug.sh.

set -u

JOBS="${JOBS:-1 2 3 4}"

cd /home/mmoslem3/scratch/attack_if

# "<attack> <pair> <budget> <selection>"
job_1="gradmatch dog-bird      0.02  random"
job_2="fc        frog-airplane 0.002 random"
job_3="fc        frog-airplane 0.01  random"
job_4="gradmatch frog-airplane 0.02  dpp"

for j in $JOBS; do
    eval "spec=\${job_$j:-}"
    [ -n "$spec" ] || { echo "aug_resume.sh: no such job '$j' (expected 1..4)"; exit 1; }
    # shellcheck disable=SC2086
    set -- $spec
    attack=$1; pair=$2; bud=$3; sel=$4

    echo "=== aug_resume job $j | ResNet20BN / $attack / $pair | budget $bud | aug standard | selection $sel ==="
    MODEL=ResNet20BN ATTACK="$attack" CLASS_PAIR="$pair" \
        BUDGETS="$bud" SELS="$sel" PAIR_SELS="random dpp" \
        AUGS="standard" NUM_TARGETS=5 NUM_VICTIMS=4 \
        sh ./aug.sh || exit 1
done

echo "=== aug_resume.sh finished (jobs: $JOBS) ==="
