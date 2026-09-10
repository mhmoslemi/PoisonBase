#!/bin/bash
#SBATCH --account=aip-yiweilu
#SBATCH --job-name=bpdiag_b0005_rand
#SBATCH --time=0-02:40:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=7G
#SBATCH --gpus-per-node=l40s:1
#SBATCH --signal=B:USR1@300
#SBATCH --output=/home/mmoslem3/scratch/PoisonBase/sbatch/logs/bpdiag_b0005_rand-%j.out

# BP diagnostic only: rereads ASR and reruns poison construction; no victim training.
export SOURCE_ROOT="${SOURCE_ROOT:-/home/mmoslem3/scratch/PoisonBase}"
export ENV_ACTIVATE="${ENV_ACTIVATE:-/home/mmoslem3/ENV/bin/activate}"
export PERSIST_DATA_ROOT="${PERSIST_DATA_ROOT:-$SOURCE_ROOT/data}"
export DIAG_SELECTION=random
export DIAG_BUDGET=0.005
export DIAG_ASR_RUN_NAME=CIFAR10_ResNet20BN_fc_random_dog-bird_b0.005_eps8_seed42_ce5_tgt10
export ORIGINAL_COMMAND="model=ResNet20BN attack=BP selection=RANDOM budget=0.005 targets=10 curve_every=25 victim_training=none"
source "$SOURCE_ROOT/sbatch/bp_failure_diagnostic_20260910/_job_common.sh"
