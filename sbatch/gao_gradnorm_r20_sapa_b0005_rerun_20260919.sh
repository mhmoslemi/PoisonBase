#!/bin/bash
#SBATCH --account=aip-boyuwang
#SBATCH --job-name=gao_r20_sapa_b0005_rerun
#SBATCH --time=0-01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=7G
#SBATCH --gres=gpu:l40s:1
#SBATCH --signal=B:USR1@300
#SBATCH --output=/home/mmoslem3/scratch/PoisonBase/sbatch/logs/gao_r20_sapa_b0005_rerun-%j.out

# Resume/complete Gao gradient-norm, ResNet20, SAPA, rho=0.005.
export ROOT="${ROOT:-/home/mmoslem3/scratch/PoisonBase}"
export ENV_ACTIVATE="${ENV_ACTIVATE:-/home/mmoslem3/ENV/bin/activate}"
export DATA_ROOT="${DATA_ROOT:-$ROOT/data}"
export IFB_INDEX=16
export IFB_SELECTOR=gao-gradnorm
export IFB_ATTACK=sapa
export IFB_BUDGET=0.005
export IFB_MODEL=ResNet20BN
export IFB_K=20
export IFB_TARGET_DEGREE=14
export IFB_RUN_NAME=CIFAR10_ResNet20BN_sapa_ours_dog-bird_b0.005_eps8_seed42_lam1_cosine_selgao-gradnorm_ep10_K20_worst0.05_ce5_tgt14
export ORIGINAL_COMMAND="selector=gao-gradnorm attack=sapa budget=0.005 model=ResNet20BN K=20 dog-bird targets=10 victims=6 craft_steps=250 parts=1"

source "$ROOT/sbatch/influence_fus_baselines_20260917/_job_common.sh"
