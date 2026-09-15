#!/bin/bash
#SBATCH --account=aip-boyuwang
#SBATCH --job-name=xk155_sapa_b0002_vconv_svgg_k10
#SBATCH --time=0-03:10:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=7G
#SBATCH --gres=gpu:l40s:1
#SBATCH --signal=B:USR1@300
#SBATCH --output=/home/mmoslem3/scratch/PoisonBase/sbatch/logs/xk155_sapa_b0002_vconv_svgg_k10-%j.out

# One file = one attack/budget/victim/selector/K configuration.
export ROOT="${ROOT:-/home/mmoslem3/scratch/PoisonBase}"
export ENV_ACTIVATE="${ENV_ACTIVATE:-/home/mmoslem3/ENV/bin/activate}"
export DATA_ROOT="${DATA_ROOT:-$ROOT/data}"
export XFULL_INDEX=155
export XFULL_ATTACK=sapa
export XFULL_BUDGET=0.002
export XFULL_VICTIM_MODEL=ConvNetBN
export XFULL_SELECTOR_MODEL=VGG13BN
export XFULL_K=10
export XFULL_TARGET_DEGREE=70
export XFULL_RUN_NAME=CIFAR10_ConvNetBN_sapa_ours_dog-bird_b0.002_eps8_seed42_lam1_cosine_selarchVGG13BN_K10_worst0.05_ce5_tgt70
export ORIGINAL_COMMAND="attack=sapa budget=0.002 A=V=ConvNetBN S=VGG13BN K=10 base=ours jacobian=off craft_steps=250 targets=10 victims=6"
source "$ROOT/sbatch/cross_arch_k_10x6_20260915/_job_common.sh"
