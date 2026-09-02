#!/bin/bash
#SBATCH --account=aip-yiweilu
#SBATCH --job-name=redo_r20_fc_db_b004_gp_2restart-strong
#SBATCH --time=0-02:15:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=7G
#SBATCH --gpus-per-node=l40s:1
#SBATCH --signal=B:USR1@300
#SBATCH --output=/home/mmoslem3/scratch/PoisonBase/sbatch/logs/redo_r20_fc_db_b004_gp_2restart-strong-%j.out

# GRAFT+ redo: ResNet20BN / FC / dog-bird / budget 0.04.
# If you substantially increase crafting work in fc_hyperparameters.sh, also
# increase the #SBATCH --time value above.
export SOURCE_ROOT=/home/mmoslem3/scratch/PoisonBase
export BUDGETS=0.04
source "$SOURCE_ROOT/sbatch/redo_resnet20_fc_dog_bird_graftplus/_job_common.sh"
