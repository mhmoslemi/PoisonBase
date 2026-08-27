#!/bin/bash
#SBATCH --account=aip-boyuwang
#SBATCH --job-name=vdef_046_vgg13_sapa_epic_j
#SBATCH --time=0-04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=7G
#SBATCH --gpus-per-node=l40s:1
#SBATCH --signal=B:USR1@300
#SBATCH --output=/home/mmoslem3/scratch/PoisonBase/sbatch/logs2/%x_%j.out

# Finish the one blank defense.tex cell:
# VGG13BN / SAPA / dog--bird / b=0.01 / EPIC / Greedy_J.
#
# Submit on Vulcan with:
#   sbatch /home/mmoslem3/scratch/PoisonBase/sbatch/vulcan_remaining_20260827/defense_046_vgg13_sapa_b001_epic_ours_j_resume.sh
#
# The sourced runner stages from SOURCE_ROOT, merges results*.csv shards, and
# skips completed (target, victim) pairs. If no partial defense result was
# transferred, the four-hour allocation is long enough to rerun all 35 trials.

set -Eeuo pipefail

SOURCE_ROOT=/home/mmoslem3/scratch/PoisonBase
ATTACK_RUN=CIFAR10_VGG13BN_sapa_ours_dog-bird_b0.01_eps8_seed42_lam1_cosine_jacw1_worst0.05_ce5_tgt50
POISON_CACHE="$SOURCE_ROOT/ours_result/$ATTACK_RUN/poison_cache"

if [ ! -d "$POISON_CACHE" ]; then
    printf 'ERROR: poison cache is missing: %s\n' "$POISON_CACHE" >&2
    exit 1
fi

base_count=$(find "$POISON_CACHE" -maxdepth 1 -type f -name 'base_*.json' | wc -l | tr -d '[:space:]')
delta_count=$(find "$POISON_CACHE" -maxdepth 1 -type f -name 'delta_*.pt' | wc -l | tr -d '[:space:]')
if [ "$base_count" -lt 7 ] || [ "$delta_count" -lt 7 ]; then
    printf 'ERROR: incomplete poison cache at %s (bases=%s, deltas=%s; need at least 7 of each)\n' \
        "$POISON_CACHE" "$base_count" "$delta_count" >&2
    exit 1
fi

printf 'preflight: poison cache ready (bases=%s, deltas=%s)\n' "$base_count" "$delta_count"
source /home/mmoslem3/scratch/PoisonBase/sbatch/missing_tables_20260827/missing_defense_046_vgg13_sapa_dog_bird_b0_01_epic_ours_j_resume.sh
