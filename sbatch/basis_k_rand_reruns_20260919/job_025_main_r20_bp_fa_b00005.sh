#!/bin/bash
#SBATCH --account=aip-boyuwang
#SBATCH --job-name=bkr025_main_r20_bp_fa_b00005
#SBATCH --time=0-03:50:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=7G
#SBATCH --gres=gpu:l40s:1
#SBATCH --signal=B:USR1@300
#SBATCH --output=/home/mmoslem3/scratch/PoisonBase/sbatch/logs/bkr025_main_r20_bp_fa_b00005-%j.out

# One submission = one main-table configuration.
export ROOT="${ROOT:-/home/mmoslem3/scratch/PoisonBase}"
export ENV_ACTIVATE="${ENV_ACTIVATE:-/home/mmoslem3/ENV/bin/activate}"
export DATA_ROOT="${DATA_ROOT:-$ROOT/data}"
export RESULT_ROOT="${RESULT_ROOT:-$ROOT/basis_k_rand_rerun_20260919_result}"
export MAIN_INDEX=25
export MAIN_MODEL=ResNet20BN
export MAIN_ATTACK=fc
export MAIN_CLASS_PAIR=frog-airplane
export MAIN_BUDGET=0.0005
export MAIN_TARGET_DEGREE=2
export MAIN_NUM_TARGETS=8
export MAIN_NUM_VICTIMS=6
export ORIGINAL_COMMAND="category=main_below_rand attack=fc budget=0.0005 V=S=ResNet20BN pair=frog-airplane K=20 base=ours base_dist=cosine lambda_margin=1 jacobian=off craft_steps=250 targets=8 victims=6; table RAND=5 BASIS=3.3"

source "$ROOT/sbatch/basis_k_rand_reruns_20260919/_main_job_common.sh"
