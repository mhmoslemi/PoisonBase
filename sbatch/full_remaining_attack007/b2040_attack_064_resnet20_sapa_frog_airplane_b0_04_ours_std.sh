#!/bin/bash
#SBATCH --account=aip-boyuwang
#SBATCH --job-name=b2040_attack_064_resnet20_sapa_frog_airplane_b0_04_ours_std
#SBATCH --time=0-09:15:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=7G
#SBATCH --gres=gpu:l40s:1
#SBATCH --signal=B:USR1@300
#SBATCH --output=/home/mmoslem3/scratch/attack_if/sbatch/logs2/b2040_attack_064_resnet20_sapa_frog_airplane_b0_04_ours_std-%j.out

# Resume-only job: logs confirm 36/48 trials; targets 4639 and 2570 remain.
# The generous cap is retained because poison crafting is an atomic work unit.
# Exactly one table cell: USE_JACOBIAN_SCORE=0 JACOBIAN_WEIGHT=1.0 JACOBIAN_BATCH_SIZE=64 CLASS_PAIR=frog-airplane MODEL=ResNet20BN ATTACK=sapa SHARP_MODE=worst SHARP_SIGMA=0.05 BUDGETS=0.04 SELECT=ours SEL_ALPHA=2.0 NUM_TARGETS=8 NUM_VICTIMS=6 sh sel_dpp.sh

export JOB_KIND=attack
export ORIGINAL_COMMAND='USE_JACOBIAN_SCORE=0 JACOBIAN_WEIGHT=1.0 JACOBIAN_BATCH_SIZE=64 CLASS_PAIR=frog-airplane MODEL=ResNet20BN ATTACK=sapa SHARP_MODE=worst SHARP_SIGMA=0.05 BUDGETS=0.04 SELECT=ours SEL_ALPHA=2.0 NUM_TARGETS=8 NUM_VICTIMS=6 sh sel_dpp.sh'
export USE_JACOBIAN_SCORE=0
export JACOBIAN_WEIGHT=1.0
export JACOBIAN_BATCH_SIZE=64
export CLASS_PAIR=frog-airplane
export MODEL=ResNet20BN
export ATTACK=sapa
export BUDGETS=0.04
export SELECT=ours
export SEL_ALPHA=2.0
export SHARP_MODE=worst
export SHARP_SIGMA=0.05
export TARGET_SELECT=''
export NUM_TARGETS=8
export NUM_VICTIMS=6
export RECOMPUTE_DELTAS=0
export SOURCE_ROOT=/home/mmoslem3/scratch/attack_if
export RESUME_ONLY=1
export RESUME_RUN_NAME='CIFAR10_ResNet20BN_sapa_ours_frog-airplane_b0.04_eps8_seed42_lam1_cosine_worst0.05_ce5_tgt10'
export RESUME_MIN_COMPLETED=36
export RESUME_TOTAL_TRIALS=48
export RESUME_REMAINING_TRIALS=12

source /home/mmoslem3/scratch/attack_if/sbatch/_resume_attack_only.sh
source /home/mmoslem3/scratch/attack_if/sbatch/_job_common.sh
