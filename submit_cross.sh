#!/bin/sh
# Submit every still-blank cell in the three expanded cross-architecture tables.
set -eu

ROOT="/home/mmoslem3/scratch/attack_if"
sh "$ROOT/sbatch/cross_expanded/submit.sh"
