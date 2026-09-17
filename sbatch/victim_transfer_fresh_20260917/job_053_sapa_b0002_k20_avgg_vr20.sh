#!/bin/bash
#SBATCH --account=aip-boyuwang
#SBATCH --job-name=vxf053_sapa_b0002_k20_avgg_vr20
#SBATCH --time=0-02:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=7G
#SBATCH --gres=gpu:l40s:1
#SBATCH --signal=B:USR1@300
#SBATCH --output=/home/mmoslem3/scratch/PoisonBase/sbatch/logs/vxf053_sapa_b0002_k20_avgg_vr20-%j.out

# One file = one attack/budget/K/S=A/V configuration.
export ROOT="${ROOT:-/home/mmoslem3/scratch/PoisonBase}"
export ENV_ACTIVATE="${ENV_ACTIVATE:-/home/mmoslem3/ENV/bin/activate}"
export DATA_ROOT="${DATA_ROOT:-$ROOT/data}"
export VXF_INDEX=53
export VXF_ATTACK=sapa
export VXF_BUDGET=0.002
export VXF_K=20
export VXF_SOURCE_MODEL=VGG13BN
export VXF_VICTIM_MODEL=ResNet20BN
export VXF_TARGET_DEGREE=50
export VXF_RUN_NAME=CIFAR10_VGG13BN_sapa_ours_dog-bird_b0.002_eps8_seed42_lam1_cosine_victimarchResNet20BN_K20_worst0.05_ce5_tgt50
export ORIGINAL_COMMAND="attack=sapa budget=0.002 K=20 S=A=VGG13BN V=ResNet20BN base=ours jacobian=off craft_steps=250 targets=10 victims=6 FORCE=fresh"
source "$ROOT/sbatch/victim_transfer_fresh_20260917/_job_common.sh"
