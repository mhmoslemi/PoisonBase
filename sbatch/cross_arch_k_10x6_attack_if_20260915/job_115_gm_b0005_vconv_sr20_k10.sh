#!/bin/bash
#SBATCH --account=aip-boyuwang
#SBATCH --job-name=ak115_gm_b0005_vconv_sr20_k10
#SBATCH --time=0-03:10:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=7G
#SBATCH --gres=gpu:l40s:1
#SBATCH --signal=B:USR1@300
#SBATCH --output=/home/mmoslem3/scratch/attack_if/sbatch/logs/ak115_gm_b0005_vconv_sr20_k10-%j.out

# Separate attack_if-path profile. One file = one configuration.
export SOURCE_ROOT="${SOURCE_ROOT:-/home/mmoslem3/scratch/attack_if}"
export ROOT="${ROOT:-$SOURCE_ROOT}"
export ENV_ACTIVATE="${ENV_ACTIVATE:-/home/mmoslem3/ENV/bin/activate}"
export DATA_ROOT="${DATA_ROOT:-/home/mmoslem3/scratch/data}"
export RUN_ROOT="${RUN_ROOT:-${SLURM_TMPDIR:-}/attack_if}"
export XFULL_INDEX=115
export XFULL_ATTACK=gradmatch
export XFULL_BUDGET=0.005
export XFULL_VICTIM_MODEL=ConvNetBN
export XFULL_SELECTOR_MODEL=ResNet20BN
export XFULL_K=10
export XFULL_TARGET_DEGREE=70
export XFULL_RUN_NAME=CIFAR10_ConvNetBN_gradmatch_ours_dog-bird_b0.005_eps8_seed42_lam1_cosine_selarchResNet20BN_K10_ce5_tgt70
export ORIGINAL_COMMAND="attack=gradmatch budget=0.005 A=V=ConvNetBN S=ResNet20BN K=10 base=ours jacobian=off craft_steps=250 targets=10 victims=6 profile=attack_if"
source "$ROOT/sbatch/cross_arch_k_10x6_attack_if_20260915/_job_common.sh"
