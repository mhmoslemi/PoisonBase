#!/bin/bash
#SBATCH --account=aip-yiweilu
#SBATCH --job-name=k20v15_sapa_b0005_vr20_sconv_k20
#SBATCH --time=0-06:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=7G
#SBATCH --gres=gpu:l40s:1
#SBATCH --signal=B:USR1@300
#SBATCH --chdir=/home/mmoslem3/scratch/PoisonBase
#SBATCH --output=/home/mmoslem3/scratch/PoisonBase/sbatch/logs/k20v15_sapa_b0005_vr20_sconv_k20-%j.out

# Exactly one violating K=20 configuration; 10 targets x 6 victims.
export ROOT="${ROOT:-/home/mmoslem3/scratch/PoisonBase}"
export ENV_ACTIVATE="${ENV_ACTIVATE:-/home/mmoslem3/ENV/bin/activate}"
export DATA_ROOT="${DATA_ROOT:-$ROOT/data}"
export RESULT_ROOT="${RESULT_ROOT:-$ROOT/k20_violation_10x6_v70_20260919_result}"
export XFULL_INDEX=15
export XFULL_ATTACK=sapa
export XFULL_BUDGET=0.005
export XFULL_VICTIM_MODEL=ResNet20BN
export XFULL_SELECTOR_MODEL=ConvNetBN
export XFULL_K=20
export XFULL_TARGET_DEGREE=14
export XFULL_NUM_TARGETS=10
export XFULL_NUM_VICTIMS=6
export XFULL_BASE_DIST=cosine_norm
export XFULL_LAMBDA_MARGIN=1
export XFULL_VICTIM_EPOCHS=70
export XFULL_VICTIM_DECAY=50
export XFULL_RUN_NAME=CIFAR10_ResNet20BN_sapa_ours_dog-bird_b0.005_eps8_seed42_lam1_cosine_norm_selarchConvNetBN_K20_worst0.05_ce5_tgt14
export ORIGINAL_COMMAND="attack=sapa budget=0.005 V=ResNet20BN S=ConvNetBN K=20 base=ours base_dist=cosine_norm lambda_margin=1 jacobian=off craft_steps=250 victim_epochs=70 victim_decay=50 targets=10 victims=6; table lower-K=[65,80,66.7] old-K20=70"

source "$ROOT/sbatch/cross_arch_k_10x6_20260915/_job_common.sh"
