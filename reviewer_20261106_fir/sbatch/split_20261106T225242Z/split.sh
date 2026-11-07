#!/bin/bash
set -Eeuo pipefail
shopt -s nullglob
split_dir=$(cd -- "$(dirname -- "$0")" && pwd -P)
batch=$(dirname -- "$split_dir")
export REVIEW_ROOT=$(dirname -- "$batch")
export REVIEW_CODE=/scratch/mmoslem3/PoisonBase/sbatch/reviewer_20261106/code
export ENV_ACTIVATE=/home/mmoslem3/ENV/bin/activate
out="$REVIEW_ROOT/reviewer_20261106_result"
exec 8>"$REVIEW_ROOT/setup.lock"
flock -n 8 || exit 1
declare -A commands defense mapped
record_request() { local id=$1; shift; printf -v "commands[$id]" '%q ' "$@"; }
source "$split_dir/original/requests.sh"
while IFS='|' read -r id model budget selector fraction pool kind attack minutes; do
    [[ $id == id ]] || defense[$id]=$kind
done < <(tr '\t' '|' < /scratch/mmoslem3/PoisonBase/sbatch/reviewer_20261106/manifest.tsv)
targets=(843 1257 8252 8730 8055 5859 8954 821 1547 4427)
part_done() {
    local label=$1 part=$2 model kind fraction repeat cell target first last index i dirs=()
    case "$label" in
        pool_*)
            [[ $label =~ ^pool_([cr])_([^_]+)_([0-2])$ ]] || return 1
            model=ConvNetBN; [[ ${BASH_REMATCH[1]} != r ]] || model=ResNet20BN
            fraction=${BASH_REMATCH[2]}; repeat=${BASH_REMATCH[3]}
            dirs=("$out/surrogates/${model}_f${fraction}_r${repeat}_"*)
            ((${#dirs[@]}==1)) || return 1
            for ((i=part*5;i<(part+1)*5;i++)); do [[ -s "${dirs[0]}/net_$i.pt" ]] || return 1; done ;;
        clean_*)
            [[ $label =~ ^clean_([cr])_(.*)$ ]] || return 1
            model=ConvNetBN; [[ ${BASH_REMATCH[1]} != r ]] || model=ResNet20BN
            kind=${BASH_REMATCH[2]}
            [[ -s "$out/controls/$model/${kind}_r$part/result.json" ]] || return 1 ;;
        cell_*)
            cell=${label#cell_}; printf -v folder '%03d' "$cell"
            first=$part; last=$part
            if [[ ${defense[$cell]} == none ]]; then first=$((part*2)); last=$((first+1)); fi
            for ((index=first;index<=last;index++)); do
                target=${targets[$index]}
                for repeat in 0 1 2; do [[ -s "$out/cells/$folder/t${target}_r$repeat/result.json" ]] || return 1; done
            done ;;
        wo_pretrained|wo_ood) [[ -s "$out/wicked/$label.pt" ]] || return 1 ;;
        *) return 1 ;;
    esac
    return 0
}
: > "$split_dir/plan.tsv"
while IFS='|' read -r label id dep time range; do
    pending=''; first=0; last=0; mem=6G
    [[ -z "$range" ]] || { first=${range%-*}; last=${range#*-}; }
    for ((part=first;part<=last;part++)); do
        if ! part_done "$label" "$part"; then pending+=${pending:+,}$part; fi
    done
    if [[ $label == clean_*_telltale ]]; then mem=15G; fi
    if [[ $label == cell_* && ${defense[${label#cell_}]} == telltale ]]; then mem=15G; fi
    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$label" "$id" "$dep" "$time" "$range" "$pending" "$mem" >> "$split_dir/plan.tsv"
done < <(tr '\t' '|' < "$split_dir/original/submissions.tsv")
awk -F '\t' 'NF!=7 {exit 1} $6!="" {jobs++; tasks+=split($6,p,","); if($7=="15G") tt++} END {if(NR!=143 || tt!=20)exit 1; print "Arrays/singletons",jobs,"unfinished tasks",tasks,"Telltale arrays",tt}' "$split_dir/plan.tsv"
[[ ${1:-plan} == apply ]] || exit 0
[[ ! -e "$split_dir/new_jobs.tsv" ]] || { echo 'Submission already started; inspect the ledger' >&2; exit 1; }
squeue -u "$USER" -h -o '%i|%j|%T' | awk -F '|' '$2~/^rv_fir_group_/ {print}' > "$split_dir/old_active.tsv"
while IFS='|' read -r id name status; do [[ $status != PENDING ]] || scontrol hold "$id"; done < "$split_dir/old_active.tsv"
: > "$split_dir/new_jobs.tsv"
: > "$split_dir/submissions.new.tsv"
while IFS='|' read -r label id dep time range pending mem; do
    if [[ -z "$pending" ]]; then
        mapped[$id]=''
        printf '%s\tCOMPLETED\t\t%s\t\n' "$label" "$time" >> "$split_dir/submissions.new.tsv"
        continue
    fi
    dependency=''; parents=()
    [[ -z "$dep" ]] || IFS=: read -ra parents <<< "${dep#afterok:}"
    # The original graph uses bare afterok IDs, and explicit afterany for aggregation.
    type=afterok
    if [[ $dep == afterany:* ]]; then type=afterany; IFS=: read -ra parents <<< "${dep#afterany:}"; fi
    for parent in "${parents[@]}"; do
        [[ -n ${mapped[$parent]+yes} ]] || { echo "Unmapped parent $parent" >&2; exit 1; }
        [[ -z ${mapped[$parent]} ]] || dependency+=${dependency:+:}${mapped[$parent]}
    done
    [[ -z "$dependency" ]] || dependency="$type:$dependency"
    opts=(--parsable --hold --account=def-bghaddar --nodes=1 --ntasks=1 --cpus-per-task=1 --mem="$mem" --gpus-per-node=h100:1 --time="$time" --job-name="rv_fir_$label" --chdir="$REVIEW_ROOT" --output="$batch/logs/%x-%A_%a.out" --export=ALL)
    [[ -z "$range" ]] || opts+=(--array="$pending")
    [[ -z "$dependency" ]] || opts+=(--dependency="$dependency")
    eval "args=( ${commands[$id]} )"
    jid=$(sbatch "${opts[@]}" "$batch/array_job.sh" "${args[@]}"); jid=${jid%%;*}
    [[ $jid =~ ^[0-9]+$ ]] || exit 1
    mapped[$id]=$jid
    printf '%s\t%s\t%s\t%s\t%s\t%s\n' "$label" "$jid" "$dependency" "$time" "$pending" "$mem" >> "$split_dir/new_jobs.tsv"
    printf '%s\t%s\t%s\t%s\t%s\n' "$label" "$jid" "$dependency" "$time" "$pending" >> "$split_dir/submissions.new.tsv"
    printf 'Submitted %s %s parts=%s time=%s mem=%s\n' "$jid" "$label" "$pending" "$time" "$mem"
done < <(tr '\t' '|' < "$split_dir/plan.tsv")
[[ $(wc -l < "$split_dir/submissions.new.tsv") == 143 ]] || exit 1
printf 'All replacement arrays submitted held. Verify before canceling old jobs and releasing these.\n'
