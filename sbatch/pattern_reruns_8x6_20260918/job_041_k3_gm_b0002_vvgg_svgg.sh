#!/bin/bash
#SBATCH --account=aip-boyuwang
#SBATCH --job-name=pr041_k3_gm_b0002_vvgg_svgg
#SBATCH --time=0-02:50:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=7G
#SBATCH --gres=gpu:l40s:1
#SBATCH --signal=B:USR1@300
#SBATCH --output=/home/mmoslem3/scratch/PoisonBase/sbatch/logs/pr041_k3_gm_b0002_vvgg_svgg-%j.out

export ROOT="${ROOT:-/home/mmoslem3/scratch/PoisonBase}"
export ENV_ACTIVATE="${ENV_ACTIVATE:-/home/mmoslem3/ENV/bin/activate}"
export DATA_ROOT="${DATA_ROOT:-$ROOT/data}"
export RESULT_ROOT="${RESULT_ROOT:-$ROOT/pattern_rerun_8x6_result}"
export XFULL_INDEX=41
export XFULL_ATTACK=gradmatch
export XFULL_BUDGET=0.002
export XFULL_VICTIM_MODEL=VGG13BN
export XFULL_SELECTOR_MODEL=VGG13BN
export XFULL_K=3
export XFULL_TARGET_DEGREE=50
export XFULL_NUM_TARGETS=8
export XFULL_NUM_VICTIMS=6
export XFULL_RUN_NAME=CIFAR10_VGG13BN_gradmatch_ours_dog-bird_b0.002_eps8_seed42_lam1_cosine_K3_ce5_tgt50
export ORIGINAL_COMMAND="category=ensemble attack=gradmatch budget=0.002 V=VGG13BN S=VGG13BN K=3 component=basis base=ours jacobian=off craft_steps=250 targets=8 victims=6"

source "$ROOT/sbatch/cross_arch_k_10x6_20260915/_job_common.sh"
