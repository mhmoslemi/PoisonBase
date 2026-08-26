#!/bin/bash
#SBATCH --account=aip-boyuwang
#SBATCH --time=0-06:30:00       # Time limit (DD-HH:MM:SS)
#SBATCH --gpus-per-node=1  # Request 2 full H100 GPUs
#SBATCH --cpus-per-task=4      # Request CPU cores (adjust as needed; 12-24 is common for 2 GPUs)
#SBATCH --mem=7GB               # Request memory (adjust as needed)
#SBATCH --mail-user=mhmoslemi2338@gmail.com
#SBATCH --mail-type=ALL








module load python/3.11.5 cuda/12.6 cudnn

cp -r /home/mmoslem3/scratch/FairDD "$SLURM_TMPDIR"




# create the virtual environment on each node : 
srun --ntasks $SLURM_NNODES --tasks-per-node=1 bash << EOF
virtualenv --no-download $SLURM_TMPDIR/env
source $SLURM_TMPDIR/env/bin/activate

pip install --no-index --upgrade pip
pip install --no-index -r /home/mmoslem3/requirements.txt
EOF



# activate only on main node
source "$SLURM_TMPDIR/env/bin/activate"

for IPC in 10 50 100; do
# for IPC in 100; do
    echo "===== Running IPC=${IPC} ====="

    RUN_TAG="DC_${DATASET}_ipc${IPC}"
    TMP_LOG="$SLURM_TMPDIR/run_${RUN_TAG}.log"
    FINAL_LOG="/home/mmoslem3/scratch/FairDD/run_${RUN_TAG}.log"
    FINAL_RES="/home/mmoslem3/scratch/FairDD/results/${RUN_TAG}"

    mkdir -p "$FINAL_RES"

    # run
    srun python "$SLURM_TMPDIR/FairDD/main_DC.py" \
        --dataset "$DATASET" \
        --ipc "$IPC" \
        --Iteration 1000 \
        --save_path "$SLURM_TMPDIR/results" \
        2>&1 | tee "$TMP_LOG"

    # move log
    rm -f "$FINAL_LOG"
    mv "$TMP_LOG" "$FINAL_LOG"

    # move results (overwrite if same filename)
    mv -f "$SLURM_TMPDIR"/results/* "$FINAL_RES"/

    echo "===== DONE IPC=${IPC} ====="



done
