#!/bin/bash
#SBATCH --account=aip-boyuwang
#SBATCH --job-name=augx_179_r05_d25_r
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
export ROW_ID=05
export MODEL=ConvNetBN
export ATTACK=sapa
export BUDGET=0.02
export TARGET_SELECT=70
export SELECTION=dpp025
export SEL_ALPHA=0.25
export AUGMENT=randaug
export ATTACK_RUN_NAME='CIFAR10_ConvNetBN_sapa_ours_dog-bird_b0.02_eps8_seed42_lam1_cosine_seldpp0.25_worst0.05_ce5_tgt70'
export RUN_RANDOM='CIFAR10_ConvNetBN_sapa_random_dog-bird_b0.02_eps8_seed42_worst0.05_ce5_tgt70'
export RUN_GREEDY='CIFAR10_ConvNetBN_sapa_ours_dog-bird_b0.02_eps8_seed42_lam1_cosine_worst0.05_ce5_tgt70'
export RUN_DPP2='CIFAR10_ConvNetBN_sapa_ours_dog-bird_b0.02_eps8_seed42_lam1_cosine_seldpp2_worst0.05_ce5_tgt70'
export RUN_DPP025='CIFAR10_ConvNetBN_sapa_ours_dog-bird_b0.02_eps8_seed42_lam1_cosine_seldpp0.25_worst0.05_ce5_tgt70'
export RUN_DPP01='CIFAR10_ConvNetBN_sapa_ours_dog-bird_b0.02_eps8_seed42_lam1_cosine_seldpp0.1_worst0.05_ce5_tgt70'
export EXPECTED_DEFENSE_RUN='CIFAR10_ConvNetBN_sapa_ours_dog-bird_b0.02_eps8_seed42_lam1_cosine_seldpp0.25_worst0.05_ce5_tgt70__def-none+aug-randaug'
export ORIGINAL_COMMAND='python defense.py --model ConvNetBN --attack sapa --selection dpp025 --budget 0.02 --victim_aug randaug --num_targets 5 --num_victims 4'

source "$SOURCE_ROOT/sbatch/_augment_extra_job_common.sh"

