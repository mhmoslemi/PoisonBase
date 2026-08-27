#!/bin/bash
#SBATCH --account=aip-boyuwang
#SBATCH --job-name=vdef_046_vgg13_sapa_epic_j
#SBATCH --time=0-03:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=7G
#SBATCH --gpus-per-node=l40s:1
#SBATCH --signal=B:USR1@300
#SBATCH --output=/home/mmoslem3/scratch/PoisonBase/sbatch/logs2/%x_%j.out

# One defense.tex cell only. Resume saved trials if available; otherwise rerun
# all 7 targets x 5 victims. Three hours covers the observed ~2 h workload plus
# roughly 45 minutes of staging/scheduler cushion.
export EXPECTED_SAVED_TRIALS=0
source /home/mmoslem3/scratch/PoisonBase/sbatch/missing_tables_20260827/missing_defense_046_vgg13_sapa_dog_bird_b0_01_epic_ours_j_resume.sh
