#!/bin/bash
#SBATCH --account=aip-boyuwang
#SBATCH --job-name=k20v11_gm_b0005_vconv_sr20_k20
#SBATCH --time=0-06:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=7G
#SBATCH --gres=gpu:l40s:1
#SBATCH --signal=B:USR1@300
#SBATCH --chdir=/home/mmoslem3/scratch/PoisonBase
#SBATCH --output=/home/mmoslem3/scratch/PoisonBase/sbatch/logs/k20v11_gm_b0005_vconv_sr20_k20-%j.out

# Exactly one violating K=20 configuration; 10 targets x 6 victims.
export ROOT="${ROOT:-/home/mmoslem3/scratch/PoisonBase}"
export ENV_ACTIVATE="${ENV_ACTIVATE:-/home/mmoslem3/ENV/bin/activate}"
export DATA_ROOT="${DATA_ROOT:-$ROOT/data}"
export RESULT_ROOT="${RESULT_ROOT:-$ROOT/k20_violation_10x6_v70_20260919_result}"
export XFULL_INDEX=11
export XFULL_ATTACK=gradmatch
export XFULL_BUDGET=0.005
export XFULL_VICTIM_MODEL=ConvNetBN
export XFULL_SELECTOR_MODEL=ResNet20BN
export XFULL_K=20
export XFULL_TARGET_DEGREE=70
export XFULL_NUM_TARGETS=10
export XFULL_NUM_VICTIMS=6
export XFULL_BASE_DIST=cosine_norm
export XFULL_LAMBDA_MARGIN=1
export XFULL_VICTIM_EPOCHS=70
export XFULL_VICTIM_DECAY=50
export XFULL_RUN_NAME=CIFAR10_ConvNetBN_gradmatch_ours_dog-bird_b0.005_eps8_seed42_lam1_cosine_norm_selarchResNet20BN_K20_ce5_tgt70
export ORIGINAL_COMMAND="attack=gradmatch budget=0.005 V=ConvNetBN S=ResNet20BN K=20 base=ours base_dist=cosine_norm lambda_margin=1 jacobian=off craft_steps=250 victim_epochs=70 victim_decay=50 targets=10 victims=6; table lower-K=[18.3,26.7,33.3] old-K20=30"

source "$ROOT/sbatch/cross_arch_k_10x6_20260915/_job_common.sh"
