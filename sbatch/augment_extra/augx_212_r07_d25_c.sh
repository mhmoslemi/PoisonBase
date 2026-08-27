#!/bin/bash
#SBATCH --account=aip-boyuwang
#SBATCH --job-name=augx_212_r07_d25_c
#SBATCH --time=0-02:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=7G
#SBATCH --gpus-per-node=l40s:1
#SBATCH --signal=B:USR1@300
#SBATCH --output=/home/mmoslem3/scratch/PoisonBase/sbatch/logs/%x_%j.out

export SOURCE_ROOT=/home/mmoslem3/scratch/PoisonBase
export PERSIST_DATA_ROOT=/home/mmoslem3/scratch/PoisonBase/data
export PYTHON_ENV=/home/mmoslem3/ENV
export ROW_ID=07
export MODEL=ResNet20BN
export ATTACK=gradmatch
export BUDGET=0.002
export TARGET_SELECT=14
export SELECTION=dpp025
export SEL_ALPHA=0.25
export AUGMENT=cutout
export ATTACK_RUN_NAME='CIFAR10_ResNet20BN_gradmatch_ours_dog-bird_b0.002_eps8_seed42_lam1_cosine_seldpp0.25_ce5_tgt14'
export RUN_RANDOM='CIFAR10_ResNet20BN_gradmatch_random_dog-bird_b0.002_eps8_seed42_ce5_tgt14'
export RUN_GREEDY='CIFAR10_ResNet20BN_gradmatch_ours_dog-bird_b0.002_eps8_seed42_lam1_cosine_ce5_tgt14'
export RUN_DPP2='CIFAR10_ResNet20BN_gradmatch_ours_dog-bird_b0.002_eps8_seed42_lam1_cosine_seldpp2_ce5_tgt14'
export RUN_DPP025='CIFAR10_ResNet20BN_gradmatch_ours_dog-bird_b0.002_eps8_seed42_lam1_cosine_seldpp0.25_ce5_tgt14'
export RUN_DPP01='CIFAR10_ResNet20BN_gradmatch_ours_dog-bird_b0.002_eps8_seed42_lam1_cosine_seldpp0.1_ce5_tgt14'
export EXPECTED_DEFENSE_RUN='CIFAR10_ResNet20BN_gradmatch_ours_dog-bird_b0.002_eps8_seed42_lam1_cosine_seldpp0.25_ce5_tgt14__def-none+aug-cutout'
export ORIGINAL_COMMAND='python defense.py --model ResNet20BN --attack gradmatch --selection dpp025 --budget 0.002 --victim_aug cutout --num_targets 5 --num_victims 4'

source "$SOURCE_ROOT/sbatch/_augment_extra_job_common.sh"

