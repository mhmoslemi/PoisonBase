#!/bin/bash
#SBATCH --account=aip-yiweilu
#SBATCH --job-name=ak177_sapa_b0002_vvgg_svgg_k1
#SBATCH --time=0-03:50:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=7G
#SBATCH --gres=gpu:l40s:1
#SBATCH --signal=B:USR1@300
#SBATCH --output=/home/mmoslem3/scratch/attack_if/sbatch/logs/ak177_sapa_b0002_vvgg_svgg_k1-%j.out

# Separate attack_if-path profile. One file = one configuration.
export SOURCE_ROOT="${SOURCE_ROOT:-/home/mmoslem3/scratch/attack_if}"
export ROOT="${ROOT:-$SOURCE_ROOT}"
export ENV_ACTIVATE="${ENV_ACTIVATE:-/home/mmoslem3/ENV/bin/activate}"
export DATA_ROOT="${DATA_ROOT:-/home/mmoslem3/scratch/data}"
export RUN_ROOT="${RUN_ROOT:-${SLURM_TMPDIR:-}/attack_if}"
export XFULL_INDEX=177
export XFULL_ATTACK=sapa
export XFULL_BUDGET=0.002
export XFULL_VICTIM_MODEL=VGG13BN
export XFULL_SELECTOR_MODEL=VGG13BN
export XFULL_K=1
export XFULL_TARGET_DEGREE=50
export XFULL_RUN_NAME=CIFAR10_VGG13BN_sapa_ours_dog-bird_b0.002_eps8_seed42_lam1_cosine_K1_worst0.05_ce5_tgt50
export ORIGINAL_COMMAND="attack=sapa budget=0.002 A=V=VGG13BN S=VGG13BN K=1 base=ours jacobian=off craft_steps=250 targets=10 victims=6 profile=attack_if"
source "$ROOT/sbatch/cross_arch_k_10x6_attack_if_20260915/_job_common.sh"
