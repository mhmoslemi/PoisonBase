#!/bin/bash
#SBATCH --account=aip-boyuwang
#SBATCH --job-name=vaug_vgg_bp002_graft_crop
#SBATCH --time=03:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=7G
#SBATCH --gpus-per-node=l40s:1
#SBATCH --signal=B:USR1@300
#SBATCH --output=/home/mmoslem3/scratch/attack_if/sbatch/logs/%x-%j.out

export SELECTION=greedy
export AUGMENT=standard
source /home/mmoslem3/scratch/attack_if/sbatch/victim_aug_vgg_fc_b0002/_job_common.sh
