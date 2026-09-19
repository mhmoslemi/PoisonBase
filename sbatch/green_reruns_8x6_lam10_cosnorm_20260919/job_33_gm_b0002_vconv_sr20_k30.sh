#!/bin/bash
#SBATCH --account=aip-yiweilu
#SBATCH --job-name=grn33_gm_b0002_vconv_sr20_k30
#SBATCH --time=0-03:10:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=7G
#SBATCH --gres=gpu:l40s:1
#SBATCH --signal=B:USR1@300
#SBATCH --chdir=/home/mmoslem3/scratch/PoisonBase
#SBATCH --output=/home/mmoslem3/scratch/PoisonBase/sbatch/logs/grn33_gm_b0002_vconv_sr20_k30-%j.out

# One green-highlighted table cell; 8 targets x 6 victims.
export ROOT="${ROOT:-/home/mmoslem3/scratch/PoisonBase}"
export ENV_ACTIVATE="${ENV_ACTIVATE:-/home/mmoslem3/ENV/bin/activate}"
export DATA_ROOT="${DATA_ROOT:-$ROOT/data}"
export RESULT_ROOT="$ROOT/green_rerun_8x6_lam10_cosnorm_20260919_result"
export XFULL_INDEX=33
export XFULL_ATTACK=gradmatch
export XFULL_BUDGET=0.002
export XFULL_VICTIM_MODEL=ConvNetBN
export XFULL_SELECTOR_MODEL=ResNet20BN
export XFULL_K=30
export XFULL_TARGET_DEGREE=70
export XFULL_NUM_TARGETS=8
export XFULL_NUM_VICTIMS=6
export XFULL_COMPONENT=
export XFULL_BASE_DIST=cosine_norm
export XFULL_LAMBDA_MARGIN=10
export XFULL_VICTIM_EPOCHS=50
export XFULL_VICTIM_DECAY=40
export XFULL_RUN_NAME=CIFAR10_ConvNetBN_gradmatch_ours_dog-bird_b0.002_eps8_seed42_lam10_cosine_norm_selarchResNet20BN_K30_ce5_tgt70
export ORIGINAL_COMMAND="attack=gradmatch budget=0.002 V=ConvNetBN S=ResNet20BN K=30 base=ours base_dist=cosine_norm lambda_margin=10 jacobian=off craft_steps=250 victim_epochs=50 victim_decay=40 targets=8 victims=6; green-table-ASR=10"

source "$ROOT/sbatch/cross_arch_k_10x6_20260915/_job_common.sh"
