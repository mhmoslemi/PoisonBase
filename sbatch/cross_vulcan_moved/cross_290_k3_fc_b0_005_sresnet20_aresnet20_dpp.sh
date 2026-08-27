#!/bin/bash
#SBATCH --account=aip-boyuwang
#SBATCH --job-name=cross_290_k3_fc_b0_005_sresnet20_aresnet20_dpp
#SBATCH --time=0-02:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=7G
#SBATCH --gpus-per-node=l40s:1
#SBATCH --signal=B:USR1@300
#SBATCH --output=/home/mmoslem3/scratch/PoisonBase/sbatch/logs/cross_290_k3_fc_b0_005_sresnet20_aresnet20_dpp-%j.out

# Exactly one table cell: SEL_K=3 BUDGET=0.005 ATTACKS=fc MODELS=ResNet20BN SELECTOR_MODELS=ResNet20BN SELECTIONS=dpp RUN_MATCHED=1 NUM_TARGETS=5 NUM_VICTIMS=4 sh cross_arch.sh
# Estimated L40S runtime 0-01:30:00 plus a 01:00 cushion.

export CROSS_MODEL=ResNet20BN
export CROSS_SELECTOR_MODEL=ResNet20BN
export CROSS_ATTACK=fc
export CROSS_SELECTION=dpp
export CROSS_BUDGET=0.005
export CROSS_K=3
export CROSS_NUM_TARGETS=5
export CROSS_NUM_VICTIMS=4
export CROSS_RUN_NAME=CIFAR10_ResNet20BN_fc_ours_dog-bird_b0.005_eps8_seed42_lam1_cosine_seldpp2_K3_ce5_tgt10
export ORIGINAL_COMMAND='SEL_K=3 BUDGET=0.005 ATTACKS=fc MODELS=ResNet20BN SELECTOR_MODELS=ResNet20BN SELECTIONS=dpp RUN_MATCHED=1 NUM_TARGETS=5 NUM_VICTIMS=4 sh cross_arch.sh'

export SOURCE_ROOT=/home/mmoslem3/scratch/PoisonBase
export PERSIST_DATA_ROOT=/home/mmoslem3/scratch/PoisonBase/data
export PYTHON_ENV=/home/mmoslem3/ENV

source /home/mmoslem3/scratch/PoisonBase/sbatch/_cross_job_common.sh
