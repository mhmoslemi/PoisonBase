#!/bin/bash
#SBATCH --account=aip-yiweilu
#SBATCH --job-name=bkr018_k10_sapa_b0002_vvgg_sr20
#SBATCH --time=0-03:50:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=7G
#SBATCH --gres=gpu:l40s:1
#SBATCH --signal=B:USR1@300
#SBATCH --output=/home/mmoslem3/scratch/PoisonBase/sbatch/logs/bkr018_k10_sapa_b0002_vvgg_sr20-%j.out

# One submission = one attack/budget/victim/selector/K configuration.
export ROOT="${ROOT:-/home/mmoslem3/scratch/PoisonBase}"
export ENV_ACTIVATE="${ENV_ACTIVATE:-/home/mmoslem3/ENV/bin/activate}"
export DATA_ROOT="${DATA_ROOT:-$ROOT/data}"
export RESULT_ROOT="${RESULT_ROOT:-$ROOT/basis_k_rand_rerun_20260919_result}"
export XFULL_INDEX=18
export XFULL_ATTACK=sapa
export XFULL_BUDGET=0.002
export XFULL_VICTIM_MODEL=VGG13BN
export XFULL_SELECTOR_MODEL=ResNet20BN
export XFULL_K=10
export XFULL_TARGET_DEGREE=50
export XFULL_NUM_TARGETS=10
export XFULL_NUM_VICTIMS=6
export XFULL_RUN_NAME=CIFAR10_VGG13BN_sapa_ours_dog-bird_b0.002_eps8_seed42_lam1_cosine_selarchResNet20BN_K10_worst0.05_ce5_tgt50
export ORIGINAL_COMMAND="category=k_pattern attack=sapa budget=0.002 V=VGG13BN S=ResNet20BN K=10 base=ours base_dist=cosine lambda_margin=1 jacobian=off craft_steps=250 targets=10 victims=6; table K10=88.3 exceeds fixed K20=75"

source "$ROOT/sbatch/cross_arch_k_10x6_20260915/_job_common.sh"
