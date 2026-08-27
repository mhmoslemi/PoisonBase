#!/usr/bin/env bash
# Submit missing poison generators, then the affected one-cell augmentation
# evaluations with row-scoped afterok dependencies.

set -Eeuo pipefail

ROOT=/home/mmoslem3/scratch/PoisonBase
ACCOUNT=aip-boyuwang
PRE_DIR="$ROOT/sbatch/augment_missing_prereqs_20260827"
EVAL_DIR="$ROOT/sbatch/vulcan_remaining_20260827"
mkdir -p "$ROOT/sbatch/logs2"

declare -A ACTIVE ROW_DEPS
while IFS='|' read -r name jid; do
    [ -n "$name" ] && ACTIVE["$name"]="$jid"
done < <(squeue -h -u "${USER:-mmoslem3}" -o '%.200j|%A')

append_dependency() {
    local row="$1" jid="$2"
    if [ -n "${ROW_DEPS[$row]:-}" ]; then
        ROW_DEPS[$row]="${ROW_DEPS[$row]}:$jid"
    else
        ROW_DEPS[$row]="$jid"
    fi
}

shopt -s nullglob
prereqs=()
while IFS= read -r filename; do
    [ -n "$filename" ] && prereqs+=("$PRE_DIR/$filename")
done < "$PRE_DIR/current_jobs.txt"
printf 'Submitting %s missing poison prerequisite job(s).\n' "${#prereqs[@]}"
for job in "${prereqs[@]}"; do
    name=$(sed -n 's/^#SBATCH --job-name=//p' "$job")
    row=$(sed -n 's/^export PREREQ_ROW=//p' "$job")
    if [ -n "${ACTIVE[$name]:-}" ]; then
        jid="${ACTIVE[$name]}"
        printf 'reuse active prerequisite %s -> %s\n' "$name" "$jid"
    else
        output=$(sbatch --parsable --account="$ACCOUNT" "$job")
        jid=${output%%;*}
        case "$jid" in *[!0-9]*|'') echo "bad sbatch output: $output" >&2; exit 1 ;; esac
        ACTIVE["$name"]="$jid"
        printf 'submitted prerequisite %s -> %s\n' "$name" "$jid"
    fi
    append_dependency "$row" "$jid"
done

evals=("$EVAL_DIR"/augment_*_resume.sh)
[ "${#evals[@]}" -eq 21 ] || {
    printf 'ERROR: expected 21 augmentation evaluation jobs; found %s\n' "${#evals[@]}" >&2
    exit 1
}

for job in "${evals[@]}"; do
    name=$(sed -n 's/^#SBATCH --job-name=//p' "$job")
    template=$(sed -n 's|^source \(.*sbatch/augment_extra/.*\.sh\)$|\1|p' "$job")
    row=$(sed -n 's/^export ROW_ID=//p' "$template")
    if [ -n "${ACTIVE[$name]:-}" ]; then
        printf 'skip active evaluation %s -> %s\n' "$name" "${ACTIVE[$name]}"
        continue
    fi
    dependency="${ROW_DEPS[$row]:-}"
    if [ -n "$dependency" ]; then
        output=$(sbatch --parsable --account="$ACCOUNT" \
            --dependency="afterok:$dependency" "$job")
    else
        output=$(sbatch --parsable --account="$ACCOUNT" "$job")
    fi
    jid=${output%%;*}
    case "$jid" in *[!0-9]*|'') echo "bad sbatch output: $output" >&2; exit 1 ;; esac
    ACTIVE["$name"]="$jid"
    if [ -n "$dependency" ]; then
        printf 'submitted evaluation %s -> %s afterok:%s\n' "$name" "$jid" "$dependency"
    else
        printf 'submitted evaluation %s -> %s\n' "$name" "$jid"
    fi
done
