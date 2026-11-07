#!/bin/bash
# One sequential bundle; all resources are set by submit_saved_defenses_vulcan.sh.
set -Eeuo pipefail

ROOT="${DEFENSE_ROOT:-/scratch/mmoslem3/PoisonBase}"
BATCH="$ROOT/sbatch/rebuttal_saved_defenses"
INDEX="${1:?missing batch index}"
WORK="$SLURM_TMPDIR/rebuttal_saved_defenses"

export DEFENSE_ROOT="$ROOT"
export DEFENSE_DATA="$WORK/data"
export DEFENSE_DEADLINE=$(( $(date +%s) + 21300 ))
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export PYTHONUNBUFFERED=1
export HF_HOME="$ROOT/cache/runtime/huggingface"
export XDG_CACHE_HOME="$ROOT/cache/runtime"

module load StdEnv/2023 gcc/12.3 cuda/12.6 python/3.11.5 cudnn/9.10.0.56
source /home/mmoslem3/ENV/bin/activate

echo "job=$SLURM_JOB_ID batch=$INDEX node=$(hostname) start=$(date -u +%FT%TZ)"
(
    cd "$BATCH/code"
    sha256sum --check ../code.sha256
)
(
    cd "$BATCH"
    sha256sum --check plan.sha256
)
mkdir -p "$WORK/code" "$DEFENSE_DATA"
rsync -a "$BATCH/code/" "$WORK/code/"
rsync -a "$ROOT/data/cifar-10-batches-py" "$DEFENSE_DATA/"
export PYTHONPATH="$WORK/code${PYTHONPATH:+:$PYTHONPATH}"

# These checks run inside the allocation, never on the login node.
python -m py_compile "$WORK/code/"*.py
python -u "$WORK/code/validate.py" "$WORK/checks"
exec srun --ntasks=1 python -u "$WORK/code/run.py" --batch "$INDEX"
