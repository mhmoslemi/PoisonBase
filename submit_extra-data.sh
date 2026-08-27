#!/usr/bin/env bash
# Submit every pending Greedy cell in extra-data.tex on Vulcan. Jobs are sorted
# by their actual #SBATCH --time request, shortest first. DRY_RUN=1 prints the
# ordered commands without submitting them.

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

ordered_jobs() {
    find "$JOB_ROOT" -maxdepth 1 -type f -name 'xdata_*.sh' -print |
    while IFS= read -r job; do
        requested=$(sed -n 's/^#SBATCH --time=//p' "$job")
        [ -n "$requested" ] || {
            echo "missing #SBATCH --time in $job" >&2
            exit 1
        }
        seconds=$(printf '%s\n' "$requested" | awk -F '[-:]' \
            '{ print (($1 * 24 + $2) * 60 + $3) * 60 + $4 }')
        printf '%010d\t%s\n' "$seconds" "$job"
    done | sort -n -k1,1 -k2,2 | cut -f2-
}

echo "extra-data on Vulcan: $COUNT one-cell jobs, sorted by requested time"
ordered_jobs | while IFS= read -r job; do
    if [ "${DRY_RUN:-0}" = 1 ]; then
        echo "sbatch --account=$ACCOUNT $job"
    else
        sbatch --account="$ACCOUNT" "$job"
    fi
done
