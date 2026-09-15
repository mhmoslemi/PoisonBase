#!/bin/bash
#SBATCH --account=aip-yiweilu
#SBATCH --job-name=xk210_sapa_b0005_vvgg_sr20_k3
#SBATCH --time=0-03:50:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=7G
#SBATCH --gres=gpu:l40s:1
#SBATCH --signal=B:USR1@300
#SBATCH --output=/home/mmoslem3/scratch/PoisonBase/sbatch/logs/xk210_sapa_b0005_vvgg_sr20_k3-%j.out

# One file = one attack/budget/victim/selector/K configuration.
export ROOT="${ROOT:-/home/mmoslem3/scratch/PoisonBase}"
export ENV_ACTIVATE="${ENV_ACTIVATE:-/home/mmoslem3/ENV/bin/activate}"
export DATA_ROOT="${DATA_ROOT:-$ROOT/data}"
export XFULL_INDEX=210
export XFULL_ATTACK=sapa
export XFULL_BUDGET=0.005
export XFULL_VICTIM_MODEL=VGG13BN
export XFULL_SELECTOR_MODEL=ResNet20BN
export XFULL_K=3
export XFULL_TARGET_DEGREE=50
export XFULL_RUN_NAME=CIFAR10_VGG13BN_sapa_ours_dog-bird_b0.005_eps8_seed42_lam1_cosine_selarchResNet20BN_K3_worst0.05_ce5_tgt50
export ORIGINAL_COMMAND="attack=sapa budget=0.005 A=V=VGG13BN S=ResNet20BN K=3 base=ours jacobian=off craft_steps=250 targets=10 victims=6"
source "$ROOT/sbatch/cross_arch_k_10x6_20260915/_job_common.sh"
