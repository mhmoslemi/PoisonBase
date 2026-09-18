#!/bin/bash
#SBATCH --account=aip-boyuwang
#SBATCH --job-name=ar034_k3_gm_b0002_vconv_sconv
#SBATCH --time=0-02:50:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=7G
#SBATCH --gres=gpu:l40s:1
#SBATCH --signal=B:USR1@300
#SBATCH --output=/home/mmoslem3/scratch/attack_if/sbatch/logs/ar034_k3_gm_b0002_vconv_sconv-%j.out

export SOURCE_ROOT="${SOURCE_ROOT:-/home/mmoslem3/scratch/attack_if}"
export ROOT="${ROOT:-$SOURCE_ROOT}"
export ENV_ACTIVATE="${ENV_ACTIVATE:-/home/mmoslem3/ENV/bin/activate}"
export DATA_ROOT="${DATA_ROOT:-/home/mmoslem3/scratch/data}"
export RESULT_ROOT="${RESULT_ROOT:-$ROOT/pattern_rerun_8x6_result}"
export RUN_ROOT="${RUN_ROOT:-${SLURM_TMPDIR:-}/attack_if_pattern_rerun_8x6}"
export XFULL_INDEX=34
export XFULL_ATTACK=gradmatch
export XFULL_BUDGET=0.002
export XFULL_VICTIM_MODEL=ConvNetBN
export XFULL_SELECTOR_MODEL=ConvNetBN
export XFULL_K=3
export XFULL_TARGET_DEGREE=70
export XFULL_NUM_TARGETS=8
export XFULL_NUM_VICTIMS=6
export XFULL_RUN_NAME=CIFAR10_ConvNetBN_gradmatch_ours_dog-bird_b0.002_eps8_seed42_lam1_cosine_K3_ce5_tgt70
export ORIGINAL_COMMAND="category=ensemble attack=gradmatch budget=0.002 V=ConvNetBN S=ConvNetBN K=3 component=basis base=ours jacobian=off craft_steps=250 targets=8 victims=6 profile=attack_if"

source "$ROOT/sbatch/cross_arch_k_10x6_20260915/_job_common.sh"
