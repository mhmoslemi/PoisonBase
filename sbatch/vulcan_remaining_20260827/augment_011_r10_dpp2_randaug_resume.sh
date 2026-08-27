#!/bin/bash
#SBATCH --account=aip-boyuwang
#SBATCH --job-name=vaug_011_r10_d2_r
#SBATCH --time=0-01:15:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=7G
#SBATCH --gpus-per-node=l40s:1
#SBATCH --signal=B:USR1@300
#SBATCH --output=/home/mmoslem3/scratch/PoisonBase/sbatch/logs2/%x_%j.out

# One table cell only. Resume the saved RandAugment trials transferred from
# Killarney; the sourced template supplies this cell's exact run metadata.
export EXPECTED_SAVED_TRIALS=19
source /home/mmoslem3/scratch/PoisonBase/sbatch/augment_extra/augx_011_r10_d2_r.sh

