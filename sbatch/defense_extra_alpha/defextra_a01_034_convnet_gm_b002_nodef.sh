#!/bin/bash
#SBATCH --account=aip-boyuwang
#SBATCH --job-name=defextra_a01_034_convnet_gm_b002_nodef
#SBATCH --time=0-03:15:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=7G
#SBATCH --gpus-per-node=l40s:1
#SBATCH --signal=B:USR1@300
#SBATCH --output=/home/mmoslem3/scratch/PoisonBase/sbatch/logs/defextra_a01_034_convnet_gm_b002_nodef-%j.out

# Exactly one defense-extra table cell: alpha=0.1, No Defense,
# ConvNetBN / gradmatch / dog-bird / budget 0.02.
# Protocol: 5 targets x 1 victim; walltime includes a 00:45 cushion and 00:15 Vulcan buffer.

export EXTRA_ALPHA=0.1
export JOB_KIND=attack
export ORIGINAL_COMMAND='USE_JACOBIAN_SCORE=0 CLASS_PAIR=dog-bird MODEL=ConvNetBN ATTACK=gradmatch TARGET_SELECT=70 BUDGETS=0.02 SELECT=dpp SEL_ALPHA=0.1 NUM_TARGETS=5 NUM_VICTIMS=1 sh sel_dpp.sh'
export USE_JACOBIAN_SCORE=0
export JACOBIAN_WEIGHT=1.0
export JACOBIAN_BATCH_SIZE=64
export CLASS_PAIR=dog-bird
export MODEL=ConvNetBN
export ATTACK=gradmatch
export BUDGETS=0.02
export SELECT=dpp
export SEL_ALPHA=0.1
export SHARP_MODE=worst
export SHARP_SIGMA=0.05
export TARGET_SELECT=70
export NUM_TARGETS=5
export NUM_VICTIMS=1

export SOURCE_ROOT=/home/mmoslem3/scratch/PoisonBase
export PERSIST_DATA_ROOT=/home/mmoslem3/scratch/PoisonBase/data
export PYTHON_ENV=/home/mmoslem3/ENV

source /home/mmoslem3/scratch/PoisonBase/sbatch/_defense_extra_job_common.sh
