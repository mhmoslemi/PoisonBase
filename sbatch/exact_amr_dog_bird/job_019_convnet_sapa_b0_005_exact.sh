#!/bin/bash
#SBATCH --account=aip-boyuwang
#SBATCH --job-name=amr10_convnet_sapa_b0005_exact
#SBATCH --time=0-02:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=7G
#SBATCH --gpus-per-node=l40s:1
#SBATCH --signal=B:USR1@300
#SBATCH --output=/home/mmoslem3/scratch/PoisonBase/sbatch/logs/amr10_convnet_sapa_b0005_exact-%j.out

# Random sample seed 20260902, configuration 10/10; exact member of the pair.
export SOURCE_ROOT="${SOURCE_ROOT:-/home/mmoslem3/scratch/PoisonBase}"
export MODEL=ConvNetBN
export ATTACK=sapa
export BUDGETS=0.005
export SELECT=exact
source "$SOURCE_ROOT/sbatch/exact_amr_dog_bird/_job_common.sh"
