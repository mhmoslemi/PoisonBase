#!/bin/bash
#SBATCH --account=aip-boyuwang
#SBATCH --job-name=amr07_resnet20_fc_b0005_exact
#SBATCH --time=0-04:15:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=7G
#SBATCH --gpus-per-node=l40s:1
#SBATCH --signal=B:USR1@300
#SBATCH --output=/home/mmoslem3/scratch/PoisonBase/sbatch/logs/amr07_resnet20_fc_b0005_exact-%j.out

# Random sample seed 20260902, configuration 07/10; exact member of the pair.
export SOURCE_ROOT="${SOURCE_ROOT:-/home/mmoslem3/scratch/PoisonBase}"
export MODEL=ResNet20BN
export ATTACK=fc
export BUDGETS=0.005
export SELECT=exact
source "$SOURCE_ROOT/sbatch/exact_amr_dog_bird/_job_common.sh"
