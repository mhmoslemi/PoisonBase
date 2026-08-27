#!/bin/bash
#SBATCH --account=aip-boyuwang
#SBATCH --job-name=vdef_046_vgg13_sapa_epic_j
#SBATCH --time=0-01:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=7G
#SBATCH --gpus-per-node=l40s:1
#SBATCH --signal=B:USR1@300
#SBATCH --output=/home/mmoslem3/scratch/PoisonBase/sbatch/logs2/%x_%j.out

# One defense.tex cell only. Resume the 27/35 saved trials after the poison
# cache and partial state have been transferred from Killarney.
export EXPECTED_SAVED_TRIALS=27
source /home/mmoslem3/scratch/PoisonBase/sbatch/missing_tables_20260827/missing_defense_046_vgg13_sapa_dog_bird_b0_01_epic_ours_j_resume.sh
