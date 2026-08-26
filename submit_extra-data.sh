#!/usr/bin/env bash
# Submit every pending Greedy cell in extra-data.tex on Vulcan. Jobs are listed
# from the shortest estimated L40S runtime to the longest; Tiny ImageNet is
# deliberately last. DRY_RUN=1 prints the ordered commands without submitting.

set -eu

SOURCE_ROOT="${SOURCE_ROOT:-/home/mmoslem3/scratch/PoisonBase}"
ACCOUNT="${ACCOUNT:-aip-boyuwang}"
JOB_ROOT="$SOURCE_ROOT/sbatch/extra_data"
mkdir -p "$SOURCE_ROOT/sbatch/logs"

if [ "${DRY_RUN:-0}" != 1 ]; then
    case "$(hostname -s)" in
        vulcan*) ;;
        *)
            echo "ERROR: submit_extra-data.sh must be run on vulcan.alliancecan.ca" >&2
            echo "Current host: $(hostname -s)" >&2
            exit 1
            ;;
    esac
fi

COUNT=$(find "$JOB_ROOT" -maxdepth 1 -type f -name 'xdata_*.sh' -print | wc -l | tr -d ' ')
[ "$COUNT" -eq 54 ] || {
    echo "expected 54 one-cell jobs under $JOB_ROOT; found $COUNT" >&2
    exit 1
}

echo "extra-data on Vulcan: $COUNT one-cell jobs, shortest first; Tiny ImageNet last"
find "$JOB_ROOT" -maxdepth 1 -type f -name 'xdata_*.sh' -print | sort | while IFS= read -r job; do
    if [ "${DRY_RUN:-0}" = 1 ]; then
        echo "sbatch --account=$ACCOUNT $job"
    else
        sbatch --account="$ACCOUNT" "$job"
    fi
done
