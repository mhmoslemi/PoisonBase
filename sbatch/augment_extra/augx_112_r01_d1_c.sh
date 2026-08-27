#!/bin/bash
#SBATCH --account=aip-boyuwang
#SBATCH --job-name=augx_112_r01_d1_c
#SBATCH --time=0-02:00:00
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
export ROW_ID=01
export MODEL=ConvNetBN
export ATTACK=fc
export BUDGET=0.002
export TARGET_SELECT=50
export SELECTION=dpp01
export SEL_ALPHA=0.1
export AUGMENT=cutout
export ATTACK_RUN_NAME='CIFAR10_ConvNetBN_fc_ours_dog-bird_b0.002_eps8_seed42_lam1_cosine_seldpp0.1_ce5_tgt50'
export RUN_RANDOM='CIFAR10_ConvNetBN_fc_random_dog-bird_b0.002_eps8_seed42_ce5_tgt50'
export RUN_GREEDY='CIFAR10_ConvNetBN_fc_ours_dog-bird_b0.002_eps8_seed42_lam1_cosine_ce5_tgt50'
export RUN_DPP2='CIFAR10_ConvNetBN_fc_ours_dog-bird_b0.002_eps8_seed42_lam1_cosine_seldpp2_ce5_tgt50'
export RUN_DPP025='CIFAR10_ConvNetBN_fc_ours_dog-bird_b0.002_eps8_seed42_lam1_cosine_seldpp0.25_ce5_tgt50'
export RUN_DPP01='CIFAR10_ConvNetBN_fc_ours_dog-bird_b0.002_eps8_seed42_lam1_cosine_seldpp0.1_ce5_tgt50'
export EXPECTED_DEFENSE_RUN='CIFAR10_ConvNetBN_fc_ours_dog-bird_b0.002_eps8_seed42_lam1_cosine_seldpp0.1_ce5_tgt50__def-none+aug-cutout'
export ORIGINAL_COMMAND='python defense.py --model ConvNetBN --attack fc --selection dpp01 --budget 0.002 --victim_aug cutout --num_targets 5 --num_victims 4'

source "$SOURCE_ROOT/sbatch/_augment_extra_job_common.sh"

