#!/bin/bash
#SBATCH --account=aip-yiweilu
#SBATCH --job-name=bkr015_k3_gm_b0005_vconv_sr20
#SBATCH --time=0-03:10:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=7G
#SBATCH --gres=gpu:l40s:1
#SBATCH --signal=B:USR1@300
#SBATCH --output=/home/mmoslem3/scratch/PoisonBase/sbatch/logs/bkr015_k3_gm_b0005_vconv_sr20-%j.out

# One submission = one attack/budget/victim/selector/K configuration.
export ROOT="${ROOT:-/home/mmoslem3/scratch/PoisonBase}"
export ENV_ACTIVATE="${ENV_ACTIVATE:-/home/mmoslem3/ENV/bin/activate}"
export DATA_ROOT="${DATA_ROOT:-$ROOT/data}"
export RESULT_ROOT="${RESULT_ROOT:-$ROOT/basis_k_rand_rerun_20260919_result}"
export XFULL_INDEX=15
export XFULL_ATTACK=gradmatch
export XFULL_BUDGET=0.005
export XFULL_VICTIM_MODEL=ConvNetBN
export XFULL_SELECTOR_MODEL=ResNet20BN
export XFULL_K=3
export XFULL_TARGET_DEGREE=70
export XFULL_NUM_TARGETS=10
export XFULL_NUM_VICTIMS=6
export XFULL_RUN_NAME=CIFAR10_ConvNetBN_gradmatch_ours_dog-bird_b0.005_eps8_seed42_lam1_cosine_selarchResNet20BN_K3_ce5_tgt70
export ORIGINAL_COMMAND="category=k_pattern attack=gradmatch budget=0.005 V=ConvNetBN S=ResNet20BN K=3 base=ours base_dist=cosine lambda_margin=1 jacobian=off craft_steps=250 targets=10 victims=6; table K3=36.7 exceeds fixed K20=30"

source "$ROOT/sbatch/cross_arch_k_10x6_20260915/_job_common.sh"
