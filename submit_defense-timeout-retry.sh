#!/bin/sh
# Submit the one timed-out VGG13 SAPA/EPIC defense cell in resume mode.
set -eu

ROOT="/home/mmoslem3/scratch/attack_if"
exec sbatch "$ROOT/sbatch/missing_defense/missing_defense_046_vgg13_sapa_dog_bird_b0_01_epic_ours_j_resume.sh"
