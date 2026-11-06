#!/bin/bash
# Submit the approved four blocks; every task uses job.sh's 1 CPU / 6G / 1 L40S.
set -Eeuo pipefail
ROOT="${REVIEW_ROOT:-/scratch/mmoslem3/PoisonBase}"
BATCH="${REVIEW_BATCH:-$ROOT/sbatch/reviewer_20261106}"
MANIFEST="${REVIEW_MANIFEST:-$BATCH/manifest.tsv}"
JOB_SCRIPT="${REVIEW_JOB_SCRIPT:-$BATCH/job.sh}"
ACCOUNT="${REVIEW_ACCOUNT:-aip-yiweilu}"
PREFIX="${REVIEW_PREFIX:-rv}"
cd "$ROOT"
CHECK_JOB="${CHECK_JOB:-}"
if [[ "${HOLD_JOBS:-0}" != 1 && -z "$CHECK_JOB" && ! -s "$ROOT/reviewer_20261106_result/checks_passed.json" ]]; then
    echo 'Validation has not passed' >&2; exit 1
fi
mkdir -p "$BATCH/logs"
exec 9>"$BATCH/submit.lock"
flock -n 9 || { echo 'Another submitter is active' >&2; exit 1; }
LEDGER="$BATCH/submissions.tsv"
touch "$LEDGER"
submit() {
    local label="$1" dependency="$2" time="$3" array="$4"
    shift 4
    local old jid
    old=$(awk -F '\t' -v key="$label" '$1==key {print $2}' "$LEDGER")
    if [[ -n "$old" ]]; then printf '%s\n' "$old"; return; fi
    local opts=(--account="$ACCOUNT" --parsable --job-name="${PREFIX}_$label" --time="$time"
                --chdir="$ROOT" --output="$BATCH/logs/%x-%A_%a.out" --export=ALL)
    [[ "${HOLD_JOBS:-0}" != 1 ]] || opts+=(--hold)
    if [[ -n "$dependency" ]]; then
        [[ "$dependency" == after* ]] || dependency="afterok:$dependency"
        opts+=(--dependency="$dependency")
    fi
    [[ -z "$array" ]] || opts+=(--array="$array")
    jid=$(sbatch "${opts[@]}" "$JOB_SCRIPT" "$@")
    jid=${jid%%;*}
    [[ "$jid" =~ ^[0-9]+$ ]] || { echo "Invalid job response: $jid" >&2; exit 1; }
    printf '%s\t%s\t%s\t%s\t%s\n' "$label" "$jid" "$dependency" "$time" "$array" >> "$LEDGER"
    printf '%s\n' "$jid"
}
declare -A POOLS CONTROLS ATTACKS
for model in ConvNetBN ResNet20BN; do
    short=c; [[ "$model" == ConvNetBN ]] || short=r
    POOLS[$model]="$CHECK_JOB"
    for fraction in 0.2 0.5; do
        for repeat in 0 1 2; do
            duration=00:10:00
            [[ "$model" == ConvNetBN ]] || duration=00:15:00
            id=$(submit "pool_${short}_${fraction}_${repeat}" "$CHECK_JOB" "$duration" 0-3 \
                pretrain --model "$model" --fraction "$fraction" --repeat "$repeat")
            POOLS[$model]="${POOLS[$model]}${POOLS[$model]:+:}$id"
        done
    done
    for kind in plain asd asd_supervised telltale; do
        duration=00:10:00
        if [[ "$kind" == asd ]]; then duration=01:00:00; fi
        if [[ "$kind" == asd_supervised ]]; then duration=00:30:00; fi
        if [[ "$kind" == telltale ]]; then duration=01:00:00; fi
        id=$(submit "clean_${short}_${kind}" "$CHECK_JOB" "$duration" 0-2 \
            control --model "$model" --kind "$kind")
        CONTROLS[$model,$kind]="$id"
    done
done
WO_PRE=$(submit wo_pretrained "$CHECK_JOB" 00:20:00 '' wo_pretrained)
WO_OOD=$(submit wo_ood "$CHECK_JOB" 02:00:00 '' wo_ood)
# Manifest is written as part of source preparation; no Python on the login node.
while IFS=$'\t' read -r id model budget selector fraction pool defense attack_id minutes; do
    [[ "$id" == id ]] && continue
    deps="${CONTROLS[$model,plain]}"
    [[ -z "$CHECK_JOB" ]] || deps="$CHECK_JOB:$deps"
    if [[ "$defense" == none ]]; then
        if [[ "$fraction" != 1 ]]; then deps="$deps:${POOLS[$model]}"; fi
        [[ "$selector" != wo_pretrained ]] || deps="$deps:$WO_PRE"
        [[ "$selector" != wo_ood ]] || deps="$deps:$WO_OOD"
        tasks=0-4
    else
        deps="$deps:${ATTACKS[$attack_id]}:${CONTROLS[$model,$defense]}"
        tasks=0-9
    fi
    time=$(printf '%02d:%02d:00' "$((minutes/60))" "$((minutes%60))")
    jid=$(submit "cell_$id" "$deps" "$time" "$tasks" cell --cell "$id")
    ATTACKS[$id]="$jid"
done < "$MANIFEST"
all_ids=$(awk -F '\t' '$1!="aggregate" {print $2}' "$LEDGER" | paste -sd:)
submit aggregate "afterany:$all_ids" 00:05:00 '' aggregate
echo "Recorded all submissions in $LEDGER"
