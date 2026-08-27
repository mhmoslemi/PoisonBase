#!/bin/bash
#SBATCH --account=aip-boyuwang
#SBATCH --job-name=missing_defense_046_vgg13_sapa_b001_epic_ours_j_resume
#SBATCH --time=0-01:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=7G
#SBATCH --gpus-per-node=l40s:1
#SBATCH --signal=B:USR1@300
#SBATCH --output=/home/mmoslem3/scratch/PoisonBase/sbatch/logs2/missing_defense_046_vgg13_sapa_dog_bird_b0_01_epic_ours_j_resume-%j.out

# Resume exactly one defense-table cell. The previous run synchronized 27/35
# (target, victim) trials; defense.py merges its saved shards/results.csv and
# skips those trials automatically. Estimated remaining GPU work is ~25 min;
# the 01:30 request includes staging, final synchronization, and >45 min cushion.

export JOB_KIND=defense
export ORIGINAL_COMMAND='USE_JACOBIAN_SCORE=1 JACOBIAN_WEIGHT=1.0 JACOBIAN_BATCH_SIZE=64 CLASS_PAIR=dog-bird MODEL=VGG13BN ATTACK=sapa TARGET_SELECT=50 BUDGETS=0.01 SELS=ours SEL_ALPHA=2.0 DEFENSES=epic NUM_TARGETS=7 NUM_VICTIMS=5 sh defense.sh'
export USE_JACOBIAN_SCORE=1
export JACOBIAN_WEIGHT=1.0
export JACOBIAN_BATCH_SIZE=64
export CLASS_PAIR=dog-bird
export MODEL=VGG13BN
export ATTACK=sapa
export BUDGETS=0.01
export SELS=ours
export SEL_ALPHA=2.0
export DEFENSES=epic
export TARGET_SELECT=50
export NUM_TARGETS=7
export NUM_VICTIMS=5
export EPIC_SUBSET=''
export NOISE_EPS=''
export FRIENDLY_CLAMP=''

export SOURCE_ROOT=/home/mmoslem3/scratch/PoisonBase
export LEGACY_SOURCE_ROOT=/home/mmoslem3/scratch/attack_if
export PERSIST_DATA_ROOT=/home/mmoslem3/scratch/PoisonBase/data
export PYTHON_ENV=/home/mmoslem3/ENV

source /home/mmoslem3/scratch/PoisonBase/sbatch/_job_common.sh
