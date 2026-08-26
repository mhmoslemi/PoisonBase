#!/bin/sh
# Build the one missing poison cache, then retry only its two failed defenses.
set -eu

ROOT="/home/mmoslem3/scratch/attack_if"
ATTACK="$ROOT/sbatch/remaining/attack_missing_convnet_fc_dog_bird_b0_02_random_std.sh"
EPIC="$ROOT/sbatch/defense/defense_004_convnet_fc_dog_bird_b0_02_epic_random_std.sh"
FRIENDS="$ROOT/sbatch/defense/defense_018_convnet_fc_dog_bird_b0_02_friends_random_std.sh"

mkdir -p "$ROOT/sbatch/logs"

attack_id=$(sbatch --parsable "$ATTACK")
attack_id=${attack_id%%;*}
echo "submitted missing poison-cache attack: $attack_id"

epic_id=$(sbatch --parsable --dependency="afterok:$attack_id" "$EPIC")
friends_id=$(sbatch --parsable --dependency="afterok:$attack_id" "$FRIENDS")
echo "submitted EPIC retry after $attack_id: ${epic_id%%;*}"
echo "submitted FRIENDS retry after $attack_id: ${friends_id%%;*}"
