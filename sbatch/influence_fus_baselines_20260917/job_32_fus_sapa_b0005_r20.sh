#!/bin/bash
#SBATCH --account=aip-boyuwang
#SBATCH --job-name=ifb32_fus_sapa_b0005_r20
#SBATCH --time=0-04:00:00
#SBATCH --array=0-3
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=7G
#SBATCH --gres=gpu:l40s:1
#SBATCH --signal=B:USR1@300
#SBATCH --output=/home/mmoslem3/scratch/PoisonBase/sbatch/logs/ifb32_fus_sapa_b0005_r20-%A_%a.out

# One file = one selector/attack/budget/model configuration. FUS uses four
# target-partitioned array tasks; static Gao selectors use one task.
export ROOT="${ROOT:-/home/mmoslem3/scratch/PoisonBase}"
export ENV_ACTIVATE="${ENV_ACTIVATE:-/home/mmoslem3/ENV/bin/activate}"
export DATA_ROOT="${DATA_ROOT:-$ROOT/data}"
export IFB_INDEX=32
export IFB_SELECTOR=fus
export IFB_ATTACK=sapa
export IFB_BUDGET=0.005
export IFB_MODEL=ResNet20BN
export IFB_K=20
export IFB_TARGET_DEGREE=14
export IFB_RUN_NAME=CIFAR10_ResNet20BN_sapa_ours_dog-bird_b0.005_eps8_seed42_lam1_cosine_selfus_i10_a0.5_e50_ps0_pr0_K20_worst0.05_ce5_tgt14
export IFB_PART_COUNT=4
export ORIGINAL_COMMAND="selector=fus attack=sapa budget=0.005 model=ResNet20BN K=20 dog-bird targets=10 victims=6 craft_steps=250 parts=4"
source "$ROOT/sbatch/influence_fus_baselines_20260917/_job_common.sh"
