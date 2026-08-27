#!/bin/bash
#SBATCH --account=aip-boyuwang
#SBATCH --job-name=cross_307_k1_gradmatch_b0_005_sresnet20_avgg13_random
#SBATCH --time=0-03:15:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=7G
#SBATCH --gpus-per-node=l40s:1
#SBATCH --signal=B:USR1@300
#SBATCH --output=/home/mmoslem3/scratch/PoisonBase/sbatch/logs/cross_307_k1_gradmatch_b0_005_sresnet20_avgg13_random-%j.out

# Exactly one table cell: SEL_K=1 BUDGET=0.005 ATTACKS=gradmatch MODELS=VGG13BN SELECTOR_MODELS=ResNet20BN SELECTIONS=random RUN_MATCHED=1 NUM_TARGETS=5 NUM_VICTIMS=4 sh cross_arch.sh
# Estimated L40S runtime 0-02:15:00 plus a 01:00 cushion.

export CROSS_MODEL=VGG13BN
export CROSS_SELECTOR_MODEL=ResNet20BN
export CROSS_ATTACK=gradmatch
export CROSS_SELECTION=random
export CROSS_BUDGET=0.005
export CROSS_K=1
export CROSS_NUM_TARGETS=5
export CROSS_NUM_VICTIMS=4
export CROSS_RUN_NAME=CIFAR10_VGG13BN_gradmatch_random_dog-bird_b0.005_eps8_seed42_selarchResNet20BN_K1_ce5_tgt50
export ORIGINAL_COMMAND='SEL_K=1 BUDGET=0.005 ATTACKS=gradmatch MODELS=VGG13BN SELECTOR_MODELS=ResNet20BN SELECTIONS=random RUN_MATCHED=1 NUM_TARGETS=5 NUM_VICTIMS=4 sh cross_arch.sh'

export SOURCE_ROOT=/home/mmoslem3/scratch/PoisonBase
export PERSIST_DATA_ROOT=/home/mmoslem3/scratch/PoisonBase/data
export PYTHON_ENV=/home/mmoslem3/ENV

source /home/mmoslem3/scratch/PoisonBase/sbatch/_cross_job_common.sh
