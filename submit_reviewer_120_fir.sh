#!/bin/sh
# Run on Fir from /home/mmoslem3/scratch/PoisonBase.
[ -n "${BASH_VERSION:-}" ] || exec bash "$0" "$@"
set -Eeuo pipefail
SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd -P)
SOURCE_ROOT="${SOURCE_ROOT:-$SCRIPT_DIR}"
SOURCE_ROOT=$(CDPATH= cd -- "$SOURCE_ROOT" && pwd -P)
ASSETS="$SCRIPT_DIR/sbatch/reviewer_20261106"
export REVIEW_ROOT="$SOURCE_ROOT/reviewer_20261106_fir"
export REVIEW_BATCH="$REVIEW_ROOT/sbatch"
if [[ -s "$REVIEW_BATCH/arrays_layout.txt" ]]; then
    printf "Fir already uses separate array tasks. Ledger: %s\n" "$REVIEW_BATCH/submissions.tsv"
    exit 0
fi
export REVIEW_CODE="$ASSETS/code"
export REVIEW_MANIFEST="$ASSETS/manifest.tsv"
export REVIEW_JOB_SCRIPT="$REVIEW_BATCH/job.sh"
export REVIEW_ACCOUNT=def-bghaddar
export REVIEW_PREFIX=rv_fir
export ENV_ACTIVATE="${ENV_ACTIVATE:-/home/mmoslem3/ENV/bin/activate}"
unset CHECK_JOB HOLD_JOBS

die() { printf 'ERROR: %s\n' "$*" >&2; exit 1; }
for command in sbatch sinfo scontrol sha256sum flock module; do
    command -v "$command" >/dev/null 2>&1 || die "$command is unavailable; run on the Fir login node"
done
[[ -f "$ENV_ACTIVATE" ]] || die "Missing environment: $ENV_ACTIVATE"
[[ -f "$ASSETS/validated_on_vulcan.json" ]] || die 'Copy sbatch/reviewer_20261106 with this launcher'
[[ $(awk 'END {print NR-1}' "$REVIEW_MANIFEST") == 120 ]] || die 'Expected exactly 120 configurations'
(cd "$REVIEW_CODE" && sha256sum --check --status "$ASSETS/validated_source.sha256") || die 'Runtime differs from the validated snapshot'
(cd "$SOURCE_ROOT" && sha256sum --check --status "$ASSETS/validated_surrogates.sha256") || die 'Copy the two exact 60-epoch surrogate pools from Vulcan; see validated_surrogates.sha256'

DATA_ROOT="${DATA_ROOT:-$SOURCE_ROOT/data}"
DATA_ROOT=$(CDPATH= cd -- "$DATA_ROOT" && pwd -P)
for file in data_batch_1 data_batch_2 data_batch_3 data_batch_4 data_batch_5 test_batch batches.meta; do
    [[ -f "$DATA_ROOT/cifar-10-batches-py/$file" ]] || die "Missing CIFAR-10 input: $DATA_ROOT/cifar-10-batches-py/$file"
done
TINY_ARCHIVE="$DATA_ROOT/tiny-imagenet-200.zip"
[[ -f "$TINY_ARCHIVE" ]] || die "Missing $TINY_ARCHIVE"

gres=$(sinfo -h -o '%G' --Node)
if [[ "$gres" == *gpu:h100:* ]]; then
    gpu_type=h100
elif [[ "$gres" == *gpu:h100_80gb:* ]]; then
    gpu_type=h100_80gb
else
    die 'No full H100 GPU advertised by Fir; check sinfo -h -o %G --Node'
fi
cluster=$(scontrol show config | awk '$1=="ClusterName" {print $3}')
[[ -n "$cluster" ]] || die 'Could not identify the Slurm cluster'
for spec in StdEnv/2023 gcc/12.3 python/3.11.5 cuda/12.6 cudnn/9.10.0.56; do
    module spider "$spec" >/dev/null 2>&1 || die "Required module unavailable: $spec"
done

mkdir -p "$REVIEW_BATCH" "$REVIEW_ROOT/data" "$REVIEW_ROOT/reviewer_20261106_result"
exec 8>"$REVIEW_ROOT/setup.lock"
flock -n 8 || die 'Another Fir launcher is active'
identity="$cluster def-bghaddar $SOURCE_ROOT"
if [[ -f "$REVIEW_BATCH/cluster.txt" ]]; then
    [[ $(cat "$REVIEW_BATCH/cluster.txt") == "$identity" ]] || die 'This ledger belongs to another cluster/project; do not reuse its job IDs'
else
    printf '%s\n' "$identity" > "$REVIEW_BATCH/cluster.txt"
fi
[[ "$REVIEW_ROOT/cache" -ef "$SOURCE_ROOT/cache" ]] || ln -sT "$SOURCE_ROOT/cache" "$REVIEW_ROOT/cache"
[[ "$REVIEW_ROOT/data/cifar-10-batches-py" -ef "$DATA_ROOT/cifar-10-batches-py" ]] || \
    ln -sT "$DATA_ROOT/cifar-10-batches-py" "$REVIEW_ROOT/data/cifar-10-batches-py"
[[ "$REVIEW_ROOT/data/tiny-imagenet-200.zip" -ef "$TINY_ARCHIVE" ]] || \
    ln -sT "$TINY_ARCHIVE" "$REVIEW_ROOT/data/tiny-imagenet-200.zip"
if [[ ! -s "$REVIEW_ROOT/reviewer_20261106_result/checks_passed.json" ]]; then
    cp "$ASSETS/validated_on_vulcan.json" "$REVIEW_ROOT/reviewer_20261106_result/checks_passed.json.tmp"
    mv "$REVIEW_ROOT/reviewer_20261106_result/checks_passed.json.tmp" "$REVIEW_ROOT/reviewer_20261106_result/checks_passed.json"
fi
# The job body stages data once, then runs every member and its former array parts.
sed -e 's/^#SBATCH --account=.*/#SBATCH --account=def-bghaddar/' \
    -e "s/^#SBATCH --gres=.*/#SBATCH --gpus-per-node=$gpu_type:1/" \
    -e '/^srun /,$d' "$ASSETS/job.sh" > "$REVIEW_JOB_SCRIPT.tmp"
cat >> "$REVIEW_JOB_SCRIPT.tmp" <<'RUN'
run_parts() {
    local task_range="$1" first_part last_part part
    shift
    [[ "$task_range" =~ ^([0-9]+)-([0-9]+)$ ]] || { echo 'Invalid part range' >&2; return 1; }
    first_part=${BASH_REMATCH[1]}
    last_part=${BASH_REMATCH[2]}
    for ((part=first_part; part<=last_part; part++)); do
        srun --ntasks=1 --cpus-per-task=1 python "$REVIEW_CODE/experiments/reviewer_sweep/run.py" "$@" --part "$part"
    done
}
source "${REVIEW_BUNDLE_FILE:?Missing Fir bundle file}"
RUN
mv "$REVIEW_JOB_SCRIPT.tmp" "$REVIEW_JOB_SCRIPT"
mkdir -p "$REVIEW_BATCH/logs" "$REVIEW_BATCH/groups"

# Collect the shared submitter's exact commands without contacting Slurm.
export REVIEW_PLAN_DIR
REVIEW_PLAN_DIR=$(mktemp -d "$REVIEW_BATCH/plan.XXXXXX")
trap 'rm -rf -- "$REVIEW_PLAN_DIR"' EXIT
sbatch() {
    local id=700000000 task_range=0-0
    [[ ! -f "$REVIEW_PLAN_DIR/counter" ]] || id=$(cat "$REVIEW_PLAN_DIR/counter")
    id=$((id+1))
    printf '%s\n' "$id" > "$REVIEW_PLAN_DIR/counter"
    while (($#)); do
        case "$1" in
            --array=*) task_range="${1#--array=}"; shift ;;
            --*) shift ;;
            *) shift; break ;;
        esac
    done
    { printf '%q ' record_request "$id" "$task_range" "$@"; printf '\n'; } >> "$REVIEW_PLAN_DIR/requests.sh"
    printf '%s\n' "$id"
}
export -f sbatch
REVIEW_BATCH="$REVIEW_PLAN_DIR" bash "$ASSETS/submit_all.sh" > "$REVIEW_PLAN_DIR/submit.log"
unset -f sbatch

declare -A task_commands labels durations dependencies dependency_types ranges request_group group_job
request_order=()
record_request() {
    local id="$1"
    shift
    printf -v "task_commands[$id]" '%q ' run_parts "$@"
}
source "$REVIEW_PLAN_DIR/requests.sh"
while IFS='|' read -r label id dependency duration range; do
    request_order+=("$id")
    labels[$id]="$label"
    IFS=: read -r hours minutes seconds <<< "$duration"
    durations[$id]=$((10#$hours*3600+10#$minutes*60+10#$seconds))
    ranges[$id]="$range"
    dependency_types[$id]="${dependency%%:*}"
    if [[ -n "$dependency" ]]; then
        dependencies[$id]="${dependency#*:}"
    else
        dependency_types[$id]=afterok
        dependencies[$id]=''
    fi
done < <(tr '\t' '|' < "$REVIEW_PLAN_DIR/submissions.tsv")
[[ ${#request_order[@]} == 143 ]] || die 'Expected 143 logical jobs including setup and aggregation'

# A group only includes ready nodes, so bundling cannot create dependency cycles.
ready() {
    local id="$1" completed_group="$2" parent
    local parents=()
    IFS=: read -ra parents <<< "${dependencies[$id]}"
    for parent in "${parents[@]}"; do
        [[ -n "${request_group[$parent]:-}" ]] || return 1
        ((request_group[$parent]<=completed_group)) || return 1
    done
}
GROUP_LEDGER="$REVIEW_BATCH/groups.tsv"
touch "$GROUP_LEDGER"
remaining=${#request_order[@]}
group_no=0
while ((remaining)); do
    first=''
    for id in "${request_order[@]}"; do
        [[ -z "${request_group[$id]:-}" ]] || continue
        if ready "$id" "$group_no"; then first="$id"; break; fi
    done
    [[ -n "$first" ]] || die 'Dependency graph cannot be scheduled'
    group_no=$((group_no+1))
    members=("$first")
    request_group[$first]=$group_no
    if ((durations[$first]<1800)) && [[ "${dependency_types[$first]}" == afterok ]]; then
        for id in "${request_order[@]}"; do
            [[ -z "${request_group[$id]:-}" ]] || continue
            ((durations[$id]<1800)) || continue
            [[ "${dependency_types[$id]}" == afterok ]] || continue
            ready "$id" "$((group_no-1))" || continue
            members+=("$id")
            request_group[$id]=$group_no
            ((${#members[@]}<5)) || break
        done
    fi
    printf -v group_name 'group_%03d' "$group_no"
    group_file="$REVIEW_BATCH/groups/$group_name.sh"
    : > "$group_file.tmp"
    total_seconds=0
    member_labels=''
    group_dependencies=''
    unset seen_dependencies
    declare -A seen_dependencies=()
    for id in "${members[@]}"; do
        printf '%s\n' "${task_commands[$id]}" >> "$group_file.tmp"
        total_seconds=$((total_seconds+durations[$id]))
        member_labels+="${member_labels:+,}${labels[$id]}"
        IFS=: read -ra parents <<< "${dependencies[$id]}"
        for parent in "${parents[@]}"; do
            parent_job=${group_job[${request_group[$parent]}]}
            [[ -z "${seen_dependencies[$parent_job]:-}" ]] || continue
            seen_dependencies[$parent_job]=1
            group_dependencies+="${group_dependencies:+:}$parent_job"
        done
    done
    mv "$group_file.tmp" "$group_file"
    printf -v walltime '%02d:%02d:%02d' "$((total_seconds/3600))" "$((total_seconds/60%60))" "$((total_seconds%60))"
    dependency_arg=''
    [[ -z "$group_dependencies" ]] || dependency_arg="${dependency_types[$first]}:$group_dependencies"
    old=$(awk -F '\t' -v name="$group_name" '$1==name {print $2"|"$3"|"$4"|"$5}' "$GROUP_LEDGER")
    if [[ -n "$old" ]]; then
        IFS='|' read -r jid old_dependency old_walltime old_members <<< "$old"
        [[ "$old_dependency" == "$dependency_arg" && "$old_walltime" == "$walltime" && "$old_members" == "$member_labels" ]] || die "Existing $group_name differs from the current plan"
    else
        options=(--parsable --account=def-bghaddar --nodes=1 --ntasks=1 --cpus-per-task=1 --mem=6G
            --gpus-per-node="$gpu_type:1" --time="$walltime" --job-name="rv_fir_$group_name"
            --chdir="$REVIEW_ROOT" --output="$REVIEW_BATCH/logs/%x-%j.out"
            --export="ALL,REVIEW_BUNDLE_FILE=$group_file")
        [[ -z "$dependency_arg" ]] || options+=(--dependency="$dependency_arg")
        jid=$(command sbatch "${options[@]}" "$REVIEW_JOB_SCRIPT")
        jid=${jid%%;*}
        [[ "$jid" =~ ^[0-9]+$ ]] || die "Invalid job response: $jid"
        printf '%s\t%s\t%s\t%s\t%s\n' "$group_name" "$jid" "$dependency_arg" "$walltime" "$member_labels" >> "$GROUP_LEDGER"
        printf 'Submitted %s: %s (%s; %s)\n' "$jid" "$group_name" "$walltime" "$member_labels"
    fi
    group_job[$group_no]="$jid"
    remaining=$((remaining-${#members[@]}))
done

# Keep the original logical-job ledger, now pointing to their combined allocations.
: > "$REVIEW_BATCH/submissions.tsv.tmp"
for id in "${request_order[@]}"; do
    mapped_dependencies=''
    IFS=: read -ra parents <<< "${dependencies[$id]}"
    for parent in "${parents[@]}"; do
        mapped_dependencies+="${mapped_dependencies:+:}${group_job[${request_group[$parent]}]}"
    done
    [[ -z "$mapped_dependencies" ]] || mapped_dependencies="${dependency_types[$id]}:$mapped_dependencies"
    printf -v original_time '%02d:%02d:%02d' "$((durations[$id]/3600))" "$((durations[$id]/60%60))" "$((durations[$id]%60))"
    printf '%s\t%s\t%s\t%s\t%s\n' "${labels[$id]}" "${group_job[${request_group[$id]}]}" "$mapped_dependencies" "$original_time" "${ranges[$id]}" >> "$REVIEW_BATCH/submissions.tsv.tmp"
done
mv "$REVIEW_BATCH/submissions.tsv.tmp" "$REVIEW_BATCH/submissions.tsv"
printf 'Ready: %s Fir jobs for all 120 configurations plus setup/controls. No arrays.\n' "$group_no"
printf 'Account: def-bghaddar. Each job: 1 CPU, 6 GB RAM, 1 %s. Ledger: %s\n' "$gpu_type" "$GROUP_LEDGER"
