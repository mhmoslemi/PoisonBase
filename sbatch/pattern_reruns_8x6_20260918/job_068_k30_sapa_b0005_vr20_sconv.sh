#!/bin/bash
#SBATCH --account=aip-boyuwang
#SBATCH --job-name=pr068_k30_sapa_b0005_vr20_sconv
#SBATCH --time=0-02:50:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=7G
#SBATCH --gres=gpu:l40s:1
#SBATCH --signal=B:USR1@300
#SBATCH --output=/home/mmoslem3/scratch/PoisonBase/sbatch/logs/pr068_k30_sapa_b0005_vr20_sconv-%j.out

export ROOT="${ROOT:-/home/mmoslem3/scratch/PoisonBase}"
export ENV_ACTIVATE="${ENV_ACTIVATE:-/home/mmoslem3/ENV/bin/activate}"
export DATA_ROOT="${DATA_ROOT:-$ROOT/data}"
export RESULT_ROOT="${RESULT_ROOT:-$ROOT/pattern_rerun_8x6_result}"
export XFULL_INDEX=68
export XFULL_ATTACK=sapa
export XFULL_BUDGET=0.005
export XFULL_VICTIM_MODEL=ResNet20BN
export XFULL_SELECTOR_MODEL=ConvNetBN
export XFULL_K=30
export XFULL_TARGET_DEGREE=14
export XFULL_NUM_TARGETS=8
export XFULL_NUM_VICTIMS=6
export XFULL_RUN_NAME=CIFAR10_ResNet20BN_sapa_ours_dog-bird_b0.005_eps8_seed42_lam1_cosine_selarchConvNetBN_K30_worst0.05_ce5_tgt14
export ORIGINAL_COMMAND="category=ensemble attack=sapa budget=0.005 V=ResNet20BN S=ConvNetBN K=30 component=basis base=ours jacobian=off craft_steps=250 targets=8 victims=6"

source "$ROOT/sbatch/cross_arch_k_10x6_20260915/_job_common.sh"
