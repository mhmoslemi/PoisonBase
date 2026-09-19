#!/bin/bash
#SBATCH --account=aip-boyuwang
#SBATCH --job-name=grn46_sapa_b0002_vr20_sconv_k1
#SBATCH --time=0-03:10:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=7G
#SBATCH --gres=gpu:l40s:1
#SBATCH --signal=B:USR1@300
#SBATCH --chdir=/home/mmoslem3/scratch/PoisonBase
#SBATCH --output=/home/mmoslem3/scratch/PoisonBase/sbatch/logs/grn46_sapa_b0002_vr20_sconv_k1-%j.out

# One green-highlighted table cell; 8 targets x 6 victims.
export ROOT="${ROOT:-/home/mmoslem3/scratch/PoisonBase}"
export ENV_ACTIVATE="${ENV_ACTIVATE:-/home/mmoslem3/ENV/bin/activate}"
export DATA_ROOT="${DATA_ROOT:-$ROOT/data}"
export RESULT_ROOT="$ROOT/green_rerun_8x6_lam10_cosnorm_20260919_result"
export XFULL_INDEX=46
export XFULL_ATTACK=sapa
export XFULL_BUDGET=0.002
export XFULL_VICTIM_MODEL=ResNet20BN
export XFULL_SELECTOR_MODEL=ConvNetBN
export XFULL_K=1
export XFULL_TARGET_DEGREE=14
export XFULL_NUM_TARGETS=8
export XFULL_NUM_VICTIMS=6
export XFULL_COMPONENT=
export XFULL_BASE_DIST=cosine_norm
export XFULL_LAMBDA_MARGIN=10
export XFULL_VICTIM_EPOCHS=50
export XFULL_VICTIM_DECAY=40
export XFULL_RUN_NAME=CIFAR10_ResNet20BN_sapa_ours_dog-bird_b0.002_eps8_seed42_lam10_cosine_norm_selarchConvNetBN_K1_worst0.05_ce5_tgt14
export ORIGINAL_COMMAND="attack=sapa budget=0.002 V=ResNet20BN S=ConvNetBN K=1 base=ours base_dist=cosine_norm lambda_margin=10 jacobian=off craft_steps=250 victim_epochs=50 victim_decay=40 targets=8 victims=6; green-table-ASR=51.7"

source "$ROOT/sbatch/cross_arch_k_10x6_20260915/_job_common.sh"
