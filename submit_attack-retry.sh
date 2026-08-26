#!/bin/sh
# The original 129 attack jobs are complete. Submit the one cache-building run
# still required by the failed ConvNet random-defense cells.
set -eu

ROOT="/home/mmoslem3/scratch/attack_if"
exec sbatch "$ROOT/sbatch/remaining/attack_missing_convnet_fc_dog_bird_b0_02_random_std.sh"
