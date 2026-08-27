#!/bin/sh
# Convenience wrapper: submit the last defense cell and all remaining
# RandAugment cells on Vulcan. Every sbatch still contains exactly one cell.

set -eu

SOURCE_ROOT="${SOURCE_ROOT:-/home/mmoslem3/scratch/PoisonBase}"
export SOURCE_ROOT

sh "$SOURCE_ROOT/submit_vulcan-remaining-defense.sh"
sh "$SOURCE_ROOT/submit_vulcan-remaining-augmentation.sh"
