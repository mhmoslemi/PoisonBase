#!/bin/bash
#SBATCH --account=aip-yiweilu
#SBATCH --job-name=ifb27_fus_gm_b0005_conv
#SBATCH --time=0-12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=7G
#SBATCH --gres=gpu:l40s:1
#SBATCH --signal=B:USR1@300
#SBATCH --output=/home/mmoslem3/scratch/PoisonBase/sbatch/logs/ifb27_fus_gm_b0005_conv-%j.out

# One file = one selector/attack/budget/model configuration.
export ROOT="${ROOT:-/home/mmoslem3/scratch/PoisonBase}"
export ENV_ACTIVATE="${ENV_ACTIVATE:-/home/mmoslem3/ENV/bin/activate}"
export DATA_ROOT="${DATA_ROOT:-$ROOT/data}"
export IFB_INDEX=27
export IFB_SELECTOR=fus
export IFB_ATTACK=gradmatch
export IFB_BUDGET=0.005
export IFB_MODEL=ConvNetBN
export IFB_K=20
export IFB_TARGET_DEGREE=70
export IFB_RUN_NAME=CIFAR10_ConvNetBN_gradmatch_ours_dog-bird_b0.005_eps8_seed42_lam1_cosine_selfus_i10_a0.5_e50_ps0_pr0_K20_ce5_tgt70
export ORIGINAL_COMMAND="selector=fus attack=gradmatch budget=0.005 model=ConvNetBN K=20 dog-bird targets=10 victims=6 craft_steps=250"
source "$ROOT/sbatch/influence_fus_baselines_20260917/_job_common.sh"
