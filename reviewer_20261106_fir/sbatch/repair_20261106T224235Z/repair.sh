#!/bin/bash
set -Eeuo pipefail
repair_dir=$(cd -- "$(dirname -- "$0")" && pwd -P)
batch=$(dirname -- "$repair_dir")
export REVIEW_ROOT=$(dirname -- "$batch")
export REVIEW_CODE=/scratch/mmoslem3/PoisonBase/sbatch/reviewer_20261106/code
export ENV_ACTIVATE=/home/mmoslem3/ENV/bin/activate
exec 8>"$REVIEW_ROOT/setup.lock"
flock -n 8 || { echo 'Another launcher is active' >&2; exit 1; }
seconds() { local h m s; IFS=: read -r h m s <<< "$1"; echo $((10#$h*3600+10#$m*60+10#$s)); }
duration() { printf '%02d:%02d:%02d' "$(( $1/3600 ))" "$(( $1/60%60 ))" "$(( $1%60 ))"; }
declare -A estimates memory state mapped new_deps new_time old_group
while IFS='|' read -r label sec mem; do estimates[$label]=$sec; memory[$label]=$mem; done < <(tr '\t' '|' < "$repair_dir/logical_estimates.tsv")
while IFS='|' read -r id status elapsed limit; do state[$id]=${status%% *}; done < "$repair_dir/states.before.tsv"
: > "$repair_dir/plan.tsv"
while IFS='|' read -r group id dep oldtime members; do
    [[ -n "$group" && -n "$id" ]] || exit 1
    old_group[$id]=$group
    oldseconds=$(seconds "$oldtime")
    total=0; mem=6G; known=1
    IFS=, read -ra labels <<< "$members"
    for label in "${labels[@]}"; do
        [[ -n "${estimates[$label]:-}" ]] || { known=0; continue; }
        total=$((total+estimates[$label]))
        [[ ${memory[$label]} != 15G ]] || mem=15G
    done
    ((known)) || total=$oldseconds
    time=$(duration "$total")
    action=replace
    case ${state[$id]:-UNKNOWN} in
        COMPLETED) action=keep; time=$oldtime ;;
        RUNNING)
            action=keep; time=$oldtime
            if ((total>oldseconds)); then
                action=continue
                remaining=$((total-oldseconds*8/10))
                ((remaining>=1800)) || remaining=1800
                time=$(duration "$(((remaining+899)/900*900))")
            fi ;;
        PENDING|FAILED|OUT_OF_MEMORY|CANCELLED|TIMEOUT) ;;
        *) echo "Unexpected state $id ${state[$id]:-UNKNOWN}" >&2; exit 1 ;;
    esac
    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$group" "$id" "$action" "$mem" "$time" "$dep" "$members" >> "$repair_dir/plan.tsv"
done < <(tr '\t' '|' < "$repair_dir/groups.before.tsv")
awk -F '\t' 'NF!=7 {exit 1} $4=="15G" {tt++} $4!="15G" && $4!="6G" {exit 1} END {if(NR!=115 || tt!=20) exit 1}' "$repair_dir/plan.tsv"
awk -F '\t' '{count[$3" "$4]++} END {for(k in count) print k,count[k]}' "$repair_dir/plan.tsv"
[[ ${1:-plan} == apply ]] || exit 0
[[ ! -e "$repair_dir/new_jobs.tsv" ]] || { echo 'Already started; inspect new_jobs.tsv before recovery' >&2; exit 1; }
: > "$repair_dir/new_jobs.tsv"
: > "$repair_dir/groups.new.tsv"
remap_dependency() {
    local dep=$1 parent parents=() result=''
    [[ -n "$dep" ]] || { printf ''; return; }
    IFS=: read -ra parents <<< "${dep#*:}"
    for parent in "${parents[@]}"; do
        [[ ${state[$parent]} != COMPLETED ]] || continue
        [[ -n ${mapped[$parent]:-} ]] || { echo "Unmapped dependency $parent" >&2; return 1; }
        result+=${result:+:}${mapped[$parent]}
    done
    [[ -z "$result" ]] || printf '%s:%s' "${dep%%:*}" "$result"
    return 0
}
while IFS='|' read -r group id action mem time dep members; do
    if [[ $action == keep ]]; then
        jid=$id; dependency=$dep
    else
        dependency=$(remap_dependency "$dep")
        [[ $action != continue ]] || dependency="afterany:$id"
        opts=(--parsable --hold --account=def-bghaddar --nodes=1 --ntasks=1 --cpus-per-task=1 --mem="$mem" --gpus-per-node=h100:1 --time="$time" --job-name="rv_fir_$group" --chdir="$REVIEW_ROOT" --output="$batch/logs/%x-%j.out" --export="ALL,REVIEW_BUNDLE_FILE=$batch/groups/$group.sh")
        [[ -z "$dependency" ]] || opts+=(--dependency="$dependency")
        jid=$(sbatch "${opts[@]}" "$batch/job.sh")
        jid=${jid%%;*}
        [[ $jid =~ ^[0-9]+$ ]] || { echo "Invalid sbatch reply $jid" >&2; exit 1; }
        printf '%s\t%s\t%s\t%s\t%s\t%s\n' "$id" "$jid" "$group" "$mem" "$time" "$dependency" >> "$repair_dir/new_jobs.tsv"
        printf 'Submitted %s -> %s %s %s %s\n' "$id" "$jid" "$group" "$mem" "$time"
    fi
    mapped[$id]=$jid; new_deps[$id]=$dependency; new_time[$id]=$time
    printf '%s\t%s\t%s\t%s\t%s\n' "$group" "$jid" "$dependency" "$time" "$members" >> "$repair_dir/groups.new.tsv"
done < <(tr '\t' '|' < "$repair_dir/plan.tsv")
# Validate held allocations before replacing the old queued jobs.
while IFS='|' read -r old jid group mem time dep; do
    detail=$(scontrol show job -o "$jid")
    [[ $detail == *'Account=def-bghaddar_gpu '* && $detail == *"MinMemoryNode=$mem "* && $detail == *'JobState=PENDING '* ]] || { echo "Unexpected allocation $jid: $detail" >&2; exit 1; }
done < <(tr '\t' '|' < "$repair_dir/new_jobs.tsv")
: > "$repair_dir/submissions.new.tsv"
while IFS='|' read -r label id dep time range; do
    [[ -n ${mapped[$id]:-} ]] || exit 1
    printf '%s\t%s\t%s\t%s\t%s\n' "$label" "${mapped[$id]}" "${new_deps[$id]}" "${new_time[$id]}" "$range" >> "$repair_dir/submissions.new.tsv"
done < <(tr '\t' '|' < "$repair_dir/submissions.before.tsv")
[[ $(wc -l < "$repair_dir/groups.new.tsv") == 115 && $(wc -l < "$repair_dir/submissions.new.tsv") == 143 ]] || exit 1
while read -r old; do
    [[ ${state[$old]} == PENDING && ${mapped[$old]} != "$old" ]] || exit 1
    scancel "$old"
done < "$repair_dir/old_pending.txt"
cp "$repair_dir/groups.new.tsv" "$batch/groups.tsv"
cp "$repair_dir/submissions.new.tsv" "$batch/submissions.tsv"
while IFS='|' read -r old jid group mem time dep; do scontrol release "$jid"; done < <(tr '\t' '|' < "$repair_dir/new_jobs.tsv")
printf 'Repair complete: %s new jobs; all released.\n' "$(wc -l < "$repair_dir/new_jobs.tsv")"
