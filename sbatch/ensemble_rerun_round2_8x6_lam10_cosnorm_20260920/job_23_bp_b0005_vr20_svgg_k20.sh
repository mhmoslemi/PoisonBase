#!/bin/bash
#SBATCH --account=aip-boyuwang
#SBATCH --job-name=enr223_bp_b0005_vr20_svgg_k20
#SBATCH --time=0-03:10:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=7G
#SBATCH --gres=gpu:l40s:1
#SBATCH --signal=B:USR1@300
#SBATCH --chdir=/home/mmoslem3/scratch/PoisonBase
#SBATCH --output=/home/mmoslem3/scratch/PoisonBase/sbatch/logs/enr223_bp_b0005_vr20_svgg_k20-%j.out

# One requested ensemble rerun; 8 targets x 6 victims.
export ROOT="${ROOT:-/home/mmoslem3/scratch/PoisonBase}"
export ENV_ACTIVATE="${ENV_ACTIVATE:-/home/mmoslem3/ENV/bin/activate}"
export DATA_ROOT="${DATA_ROOT:-$ROOT/data}"
export RESULT_ROOT="$ROOT/ensemble_rerun_round2_8x6_lam10_cosnorm_20260920_result"
export XFULL_INDEX=23
export XFULL_ATTACK=fc
export XFULL_BUDGET=0.005
export XFULL_VICTIM_MODEL=ResNet20BN
export XFULL_SELECTOR_MODEL=VGG13BN
export XFULL_K=20
export XFULL_TARGET_DEGREE=10
export XFULL_NUM_TARGETS=8
export XFULL_NUM_VICTIMS=6
export XFULL_COMPONENT=
export XFULL_BASE_DIST=cosine_norm
export XFULL_LAMBDA_MARGIN=10
export XFULL_VICTIM_EPOCHS=50
export XFULL_VICTIM_DECAY=40
export XFULL_RUN_NAME=CIFAR10_ResNet20BN_fc_ours_dog-bird_b0.005_eps8_seed42_lam10_cosine_norm_selarchVGG13BN_K20_ce5_tgt10
export ORIGINAL_COMMAND="attack=fc budget=0.005 V=ResNet20BN S=VGG13BN K=20 base=ours base_dist=cosine_norm lambda_margin=10 jacobian=off craft_steps=250 victim_epochs=50 victim_decay=40 targets=8 victims=6"

source "$ROOT/sbatch/cross_arch_k_10x6_20260915/_job_common.sh"
