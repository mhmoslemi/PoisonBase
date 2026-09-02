#!/bin/bash
#SBATCH --account=aip-boyuwang
#SBATCH --job-name=amr02_resnet20_sapa_b0001_a_mr
#SBATCH --time=0-03:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=7G
#SBATCH --gpus-per-node=l40s:1
#SBATCH --signal=B:USR1@300
#SBATCH --output=/home/mmoslem3/scratch/PoisonBase/sbatch/logs/amr02_resnet20_sapa_b0001_a_mr-%j.out

# Random sample seed 20260902, configuration 02/10; A-MR member of the pair.
export SOURCE_ROOT="${SOURCE_ROOT:-/home/mmoslem3/scratch/PoisonBase}"
export MODEL=ResNet20BN
export ATTACK=sapa
export BUDGETS=0.001
export SELECT=a-mr
source "$SOURCE_ROOT/sbatch/exact_amr_dog_bird/_job_common.sh"
