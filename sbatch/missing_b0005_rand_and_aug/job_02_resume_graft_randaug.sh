#!/bin/bash
#SBATCH --account=aip-boyuwang
#SBATCH --job-name=vaug_bp002_graft_ra_resume
#SBATCH --time=00:01:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=7G
#SBATCH --gres=gpu:l40s:1
#SBATCH --signal=B:USR1@300
#SBATCH --output=/home/mmoslem3/scratch/attack_if/sbatch/logs/%x-%j1.out

export SELECTION=greedy
export AUGMENT=randaug
source /home/mmoslem3/scratch/attack_if/sbatch/victim_aug_vgg_fc_b0002/_job_common.sh
