#!/bin/bash
#SBATCH --account=aip-yiweilu
#SBATCH --job-name=ar048_k30_gm_b0005_vr20_sconv
#SBATCH --time=0-02:50:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=7G
#SBATCH --gres=gpu:l40s:1
#SBATCH --signal=B:USR1@300
#SBATCH --output=/home/mmoslem3/scratch/attack_if/sbatch/logs/ar048_k30_gm_b0005_vr20_sconv-%j.out

export SOURCE_ROOT="${SOURCE_ROOT:-/home/mmoslem3/scratch/attack_if}"
export ROOT="${ROOT:-$SOURCE_ROOT}"
export ENV_ACTIVATE="${ENV_ACTIVATE:-/home/mmoslem3/ENV/bin/activate}"
export DATA_ROOT="${DATA_ROOT:-/home/mmoslem3/scratch/data}"
export RESULT_ROOT="${RESULT_ROOT:-$ROOT/pattern_rerun_8x6_result}"
export RUN_ROOT="${RUN_ROOT:-${SLURM_TMPDIR:-}/attack_if_pattern_rerun_8x6}"
export XFULL_INDEX=48
export XFULL_ATTACK=gradmatch
export XFULL_BUDGET=0.005
export XFULL_VICTIM_MODEL=ResNet20BN
export XFULL_SELECTOR_MODEL=ConvNetBN
export XFULL_K=30
export XFULL_TARGET_DEGREE=14
export XFULL_NUM_TARGETS=8
export XFULL_NUM_VICTIMS=6
export XFULL_RUN_NAME=CIFAR10_ResNet20BN_gradmatch_ours_dog-bird_b0.005_eps8_seed42_lam1_cosine_selarchConvNetBN_K30_ce5_tgt14
export ORIGINAL_COMMAND="category=ensemble attack=gradmatch budget=0.005 V=ResNet20BN S=ConvNetBN K=30 component=basis base=ours jacobian=off craft_steps=250 targets=8 victims=6 profile=attack_if"

source "$ROOT/sbatch/cross_arch_k_10x6_20260915/_job_common.sh"
