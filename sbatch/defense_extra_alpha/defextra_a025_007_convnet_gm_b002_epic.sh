#!/bin/bash
#SBATCH --account=aip-boyuwang
#SBATCH --job-name=defextra_a025_007_convnet_gm_b002_epic
#SBATCH --time=0-01:45:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=7G
#SBATCH --gres=gpu:l40s:1
#SBATCH --signal=B:USR1@300
#SBATCH --output=/home/mmoslem3/scratch/attack_if/sbatch/logs/defextra_a025_007_convnet_gm_b002_epic-%j.out

# Exactly one defense-extra table cell: alpha=0.25, EPIC,
# ConvNetBN / gradmatch / dog-bird / budget 0.02.
# Protocol: 5 targets x 4 victims; walltime includes a 00:45 cushion.

export EXTRA_ALPHA=0.25
export JOB_KIND=defense
export ORIGINAL_COMMAND='USE_JACOBIAN_SCORE=0 CLASS_PAIR=dog-bird MODEL=ConvNetBN ATTACK=gradmatch TARGET_SELECT=70 BUDGETS=0.02 SELS=dpp SEL_ALPHA=0.25 DEFENSES=epic NUM_TARGETS=5 NUM_VICTIMS=4 sh defense.sh'
export USE_JACOBIAN_SCORE=0
export JACOBIAN_WEIGHT=1.0
export JACOBIAN_BATCH_SIZE=64
export CLASS_PAIR=dog-bird
export MODEL=ConvNetBN
export ATTACK=gradmatch
export BUDGETS=0.02
export SELS=dpp
export SEL_ALPHA=0.25
export DEFENSES=epic
export TARGET_SELECT=70
export NUM_TARGETS=5
export NUM_VICTIMS=4
export EPIC_SUBSET=''
export NOISE_EPS=''
export FRIENDLY_CLAMP=''

source /home/mmoslem3/scratch/attack_if/sbatch/_defense_extra_job_common.sh
