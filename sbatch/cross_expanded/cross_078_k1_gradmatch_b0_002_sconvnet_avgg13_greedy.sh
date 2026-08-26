#!/bin/bash
#SBATCH --account=aip-boyuwang
#SBATCH --job-name=cross_078_k1_gradmatch_b0_002_sconvnet_avgg13_greedy
#SBATCH --time=0-02:15:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=7G
#SBATCH --gres=gpu:l40s:1
#SBATCH --signal=B:USR1@300
#SBATCH --output=/home/mmoslem3/scratch/attack_if/sbatch/logs/cross_078_k1_gradmatch_b0_002_sconvnet_avgg13_greedy-%j.out

# Exactly one table cell: SEL_K=1 BUDGET=0.002 ATTACKS=gradmatch MODELS=VGG13BN SELECTOR_MODELS=ConvNetBN SELECTIONS=greedy RUN_MATCHED=1 NUM_TARGETS=5 NUM_VICTIMS=4 sh cross_arch.sh
# Estimated L40S runtime 0-01:30:00 plus a 00:45 cushion.

export CROSS_MODEL=VGG13BN
export CROSS_SELECTOR_MODEL=ConvNetBN
export CROSS_ATTACK=gradmatch
export CROSS_SELECTION=greedy
export CROSS_BUDGET=0.002
export CROSS_K=1
export CROSS_NUM_TARGETS=5
export CROSS_NUM_VICTIMS=4
export CROSS_RUN_NAME=CIFAR10_VGG13BN_gradmatch_ours_dog-bird_b0.002_eps8_seed42_lam1_cosine_selarchConvNetBN_K1_ce5_tgt50
export ORIGINAL_COMMAND='SEL_K=1 BUDGET=0.002 ATTACKS=gradmatch MODELS=VGG13BN SELECTOR_MODELS=ConvNetBN SELECTIONS=greedy RUN_MATCHED=1 NUM_TARGETS=5 NUM_VICTIMS=4 sh cross_arch.sh'

source /home/mmoslem3/scratch/attack_if/sbatch/_cross_job_common.sh
