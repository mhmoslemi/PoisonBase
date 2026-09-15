#!/bin/bash
#SBATCH --account=aip-boyuwang
#SBATCH --job-name=xb042_bp_b0005_vconv_sr20_k3
#SBATCH --time=0-03:10:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=7G
#SBATCH --gres=gpu:l40s:1
#SBATCH --signal=B:USR1@300
#SBATCH --output=/home/mmoslem3/scratch/PoisonBase/sbatch/logs/xb042_bp_b0005_vconv_sr20_k3-%j.out

# Boyu replacement for pending Yiwei job 927130 (xk042).
export ROOT="${ROOT:-/home/mmoslem3/scratch/PoisonBase}"
export ENV_ACTIVATE="${ENV_ACTIVATE:-/home/mmoslem3/ENV/bin/activate}"
export DATA_ROOT="${DATA_ROOT:-$ROOT/data}"
export XFULL_INDEX=42
export XFULL_ATTACK=fc
export XFULL_BUDGET=0.005
export XFULL_VICTIM_MODEL=ConvNetBN
export XFULL_SELECTOR_MODEL=ResNet20BN
export XFULL_K=3
export XFULL_TARGET_DEGREE=50
export XFULL_RUN_NAME=CIFAR10_ConvNetBN_fc_ours_dog-bird_b0.005_eps8_seed42_lam1_cosine_selarchResNet20BN_K3_ce5_tgt50
export ORIGINAL_COMMAND="attack=fc budget=0.005 A=V=ConvNetBN S=ResNet20BN K=3 base=ours jacobian=off craft_steps=250 targets=10 victims=6 replacement_for=927130"
source "$ROOT/sbatch/cross_arch_k_10x6_boyu_resubmit_20260915/_job_common.sh"
