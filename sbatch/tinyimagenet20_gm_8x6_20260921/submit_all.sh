#!/usr/bin/env bash
# One command submits all prerequisites and three methods with dependencies.
set -Eeuo pipefail
HERE=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
ACCOUNT="${TINYIMAGENET_SLURM_ACCOUNT:-aip-boyuwang}"
case "$ACCOUNT" in
    aip-boyuwang|aip-yiweilu) ;;
    *) echo "Unsupported account: $ACCOUNT" >&2; exit 1 ;;
esac
python3 "$HERE/validate.py"
if [[ "${DRY_RUN:-0}" != 1 ]]; then
    # Memory-map and validate the tensor archive before submitting any jobs.
    TINYIMAGENET_FILE=$("${PYTHON_BIN:-/home/mmoslem3/ENV/bin/python}" "$HERE/../../experiments/tinyimagenet20_data.py" \
        "${TINYIMAGENET_FILE:-/home/mmoslem3/scratch/PoisonBase/data/tinyimagenet.pt}")
    export TINYIMAGENET_FILE
    mkdir -p "${SBATCH_LOG_DIR:-/home/mmoslem3/scratch/PoisonBase/sbatch/logs}"
fi
submit() {
    local name="$1" dependency="${2:-}"
    local args=(--parsable --account="$ACCOUNT" --kill-on-invalid-dep=yes \
                --export="ALL,TINYIMAGENET_SLURM_ACCOUNT=$ACCOUNT")
    [[ -z "$dependency" ]] || args+=(--dependency="afterok:$dependency")
    if [[ "${DRY_RUN:-0}" == 1 ]]; then
        printf 'sbatch' >&2
        printf ' %q' "${args[@]}" "$HERE/job_$name.sh" >&2
        printf '\n' >&2
        printf '%s\n' "$name"
    else
        local id
        id=$(sbatch "${args[@]}" "$HERE/job_$name.sh")
        id=${id%%;*}
        printf '%s -> %s\n' "$name" "$id" >&2
        printf '%s\n' "$id"
    fi
}
prep=$(submit prepare_0)
s0=$(submit surrogate_0 "$prep")
s1=$(submit surrogate_1 "$prep")
s2=$(submit surrogate_2 "$prep")
targets=$(submit targets_0 "$s0:$s1:$s2")
submit experiment_random "$targets" >/dev/null
submit experiment_minus-m "$targets" >/dev/null
submit experiment_basis "$targets" >/dev/null
echo "Submitted TinyImageNet-20 (4,000 train images, 64x64, 30 epochs) on $ACCOUNT: RAND, M-only, BASIS; 8 targets x 6 victims each; shared 3-surrogate setup."
