#!/bin/bash
#SBATCH --account=aip-boyuwang
#SBATCH --job-name=ifbm25_fus_gm_b0002_conv
#SBATCH --time=0-00:10:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --output=/home/mmoslem3/scratch/PoisonBase/sbatch/logs/ifbm25_fus_gm_b0002_conv-%j.out

export ROOT="${ROOT:-/home/mmoslem3/scratch/PoisonBase}"
export ENV_ACTIVATE="${ENV_ACTIVATE:-/home/mmoslem3/ENV/bin/activate}"
export RESULT_ROOT="${RESULT_ROOT:-$ROOT/influence_fus_baselines_result}"
export IFB_RUN_NAME=CIFAR10_ConvNetBN_gradmatch_ours_dog-bird_b0.002_eps8_seed42_lam1_cosine_selfus_i10_a0.5_e50_ps0_pr0_K20_ce5_tgt70
export IFB_MODEL=ConvNetBN
source "$ROOT/sbatch/influence_fus_baselines_20260917/_merge_fus_common.sh"
