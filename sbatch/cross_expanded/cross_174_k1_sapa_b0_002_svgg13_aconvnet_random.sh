#!/bin/bash
#SBATCH --account=aip-boyuwang
#SBATCH --job-name=cross_174_k1_sapa_b0_002_svgg13_aconvnet_random
#SBATCH --time=0-01:45:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=7G
#SBATCH --gres=gpu:l40s:1
#SBATCH --signal=B:USR1@300
#SBATCH --output=/home/mmoslem3/scratch/attack_if/sbatch/logs/cross_174_k1_sapa_b0_002_svgg13_aconvnet_random-%j.out

# Exactly one table cell: SEL_K=1 BUDGET=0.002 ATTACKS=sapa MODELS=ConvNetBN SELECTOR_MODELS=VGG13BN SELECTIONS=random RUN_MATCHED=1 NUM_TARGETS=5 NUM_VICTIMS=4 sh cross_arch.sh
# Estimated L40S runtime 0-01:00:00 plus a 00:45 cushion.

export CROSS_MODEL=ConvNetBN
export CROSS_SELECTOR_MODEL=VGG13BN
export CROSS_ATTACK=sapa
export CROSS_SELECTION=random
export CROSS_BUDGET=0.002
export CROSS_K=1
export CROSS_NUM_TARGETS=5
export CROSS_NUM_VICTIMS=4
export CROSS_RUN_NAME=CIFAR10_ConvNetBN_sapa_random_dog-bird_b0.002_eps8_seed42_selarchVGG13BN_K1_worst0.05_ce5_tgt70
export ORIGINAL_COMMAND='SEL_K=1 BUDGET=0.002 ATTACKS=sapa MODELS=ConvNetBN SELECTOR_MODELS=VGG13BN SELECTIONS=random RUN_MATCHED=1 NUM_TARGETS=5 NUM_VICTIMS=4 sh cross_arch.sh'

source /home/mmoslem3/scratch/attack_if/sbatch/_cross_job_common.sh
