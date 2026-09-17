#!/bin/bash
#SBATCH --account=aip-boyuwang
#SBATCH --job-name=ifb13_grad_sapa_b0002_conv
#SBATCH --time=0-02:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=7G
#SBATCH --gres=gpu:l40s:1
#SBATCH --signal=B:USR1@300
#SBATCH --output=/home/mmoslem3/scratch/PoisonBase/sbatch/logs/ifb13_grad_sapa_b0002_conv-%j.out

# One file = one selector/attack/budget/model configuration. FUS uses four
# target-partitioned array tasks; static Gao selectors use one task.
export ROOT="${ROOT:-/home/mmoslem3/scratch/PoisonBase}"
export ENV_ACTIVATE="${ENV_ACTIVATE:-/home/mmoslem3/ENV/bin/activate}"
export DATA_ROOT="${DATA_ROOT:-$ROOT/data}"
export IFB_INDEX=13
export IFB_SELECTOR=gao-gradnorm
export IFB_ATTACK=sapa
export IFB_BUDGET=0.002
export IFB_MODEL=ConvNetBN
export IFB_K=20
export IFB_TARGET_DEGREE=70
export IFB_RUN_NAME=CIFAR10_ConvNetBN_sapa_ours_dog-bird_b0.002_eps8_seed42_lam1_cosine_selgao-gradnorm_ep10_K20_worst0.05_ce5_tgt70
export ORIGINAL_COMMAND="selector=gao-gradnorm attack=sapa budget=0.002 model=ConvNetBN K=20 dog-bird targets=10 victims=6 craft_steps=250 parts=1"
source "$ROOT/sbatch/influence_fus_baselines_20260917/_job_common.sh"
