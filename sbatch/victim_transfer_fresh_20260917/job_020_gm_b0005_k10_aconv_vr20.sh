#!/bin/bash
#SBATCH --account=aip-boyuwang
#SBATCH --job-name=vxf020_gm_b0005_k10_aconv_vr20
#SBATCH --time=0-02:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=7G
#SBATCH --gres=gpu:l40s:1
#SBATCH --signal=B:USR1@300
#SBATCH --output=/home/mmoslem3/scratch/PoisonBase/sbatch/logs/vxf020_gm_b0005_k10_aconv_vr20-%j.out

# One file = one attack/budget/K/S=A/V configuration.
export ROOT="${ROOT:-/home/mmoslem3/scratch/PoisonBase}"
export ENV_ACTIVATE="${ENV_ACTIVATE:-/home/mmoslem3/ENV/bin/activate}"
export DATA_ROOT="${DATA_ROOT:-$ROOT/data}"
export VXF_INDEX=20
export VXF_ATTACK=gradmatch
export VXF_BUDGET=0.005
export VXF_K=10
export VXF_SOURCE_MODEL=ConvNetBN
export VXF_VICTIM_MODEL=ResNet20BN
export VXF_TARGET_DEGREE=70
export VXF_RUN_NAME=CIFAR10_ConvNetBN_gradmatch_ours_dog-bird_b0.005_eps8_seed42_lam1_cosine_victimarchResNet20BN_K10_ce5_tgt70
export ORIGINAL_COMMAND="attack=gradmatch budget=0.005 K=10 S=A=ConvNetBN V=ResNet20BN base=ours jacobian=off craft_steps=250 targets=10 victims=6 FORCE=fresh"
source "$ROOT/sbatch/victim_transfer_fresh_20260917/_job_common.sh"
