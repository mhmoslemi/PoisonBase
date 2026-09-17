#!/bin/bash
#SBATCH --account=aip-yiweilu
#SBATCH --job-name=ifb_metric_r20
#SBATCH --array=0-19
#SBATCH --time=0-02:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=7G
#SBATCH --gres=gpu:l40s:1
#SBATCH --signal=B:USR1@300
#SBATCH --output=/home/mmoslem3/scratch/PoisonBase/sbatch/logs/ifb_metric_r20-%A_%a.out

export ROOT="${ROOT:-/home/mmoslem3/scratch/PoisonBase}"
export ENV_ACTIVATE="${ENV_ACTIVATE:-/home/mmoslem3/ENV/bin/activate}"
export DATA_ROOT="${DATA_ROOT:-$ROOT/data}"
export METRIC_MODEL=ResNet20BN
source "$ROOT/sbatch/influence_fus_baselines_20260917/_metric_job_common.sh"
