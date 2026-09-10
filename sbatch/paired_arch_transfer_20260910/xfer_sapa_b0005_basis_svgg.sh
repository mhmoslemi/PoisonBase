#!/bin/bash
#SBATCH --account=aip-yiweilu
#SBATCH --job-name=xfer_sapa_b0005_basis_svgg
#SBATCH --time=0-02:40:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=7G
#SBATCH --gpus-per-node=l40s:1
#SBATCH --signal=B:USR1@300
#SBATCH --output=/home/mmoslem3/scratch/PoisonBase/sbatch/logs/xfer_sapa_b0005_basis_svgg-%j.out

# Only S changes; crafting and victims remain ConvNetBN with shared target/victim seeds.
export SOURCE_ROOT="${SOURCE_ROOT:-/home/mmoslem3/scratch/PoisonBase}"
export ENV_ACTIVATE="${ENV_ACTIVATE:-/home/mmoslem3/ENV/bin/activate}"
export PERSIST_DATA_ROOT="${PERSIST_DATA_ROOT:-$SOURCE_ROOT/data}"
export XFER_ATTACK=sapa
export XFER_BUDGET=0.005
export XFER_SELECTION=basis
export XFER_SELECTOR_MODEL=VGG13BN
export XFER_RUN_NAME=CIFAR10_ConvNetBN_sapa_ours_dog-bird_b0.005_eps8_seed42_lam1_cosine_selarchVGG13BN_K20_worst0.05_ce5_tgt70
export ORIGINAL_COMMAND="A=V=ConvNetBN S=VGG13BN K=20 attack=sapa selection=BASIS budget=0.005 targets=6 victims=6"
source "$SOURCE_ROOT/sbatch/paired_arch_transfer_20260910/_job_common.sh"
