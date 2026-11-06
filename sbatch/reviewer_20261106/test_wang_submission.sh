#!/bin/bash
# Shell-only integration check: every Slurm command is mocked; submits no jobs.
set -Eeuo pipefail
ROOT=$(CDPATH= cd -- "$(dirname -- "$0")/../.." && pwd -P)
TEST_ROOT=$(mktemp -d /tmp/reviewer-wang-test.XXXXXX)
trap 'rm -rf -- "$TEST_ROOT"' EXIT
export TEST_ROOT
case "${1:-wang}" in
    wang) LAUNCHER=submit_reviewer_120_boyuwang.sh; PROFILE=wang; TEST_ACCOUNT=aip-boyuwang; TEST_GPU=l40s; TEST_CLUSTER=killarney ;;
    fir) LAUNCHER=submit_reviewer_120_fir.sh; PROFILE=fir; TEST_ACCOUNT=def-bghaddar; TEST_GPU=h100; TEST_CLUSTER=fir ;;
    *) echo 'Expected wang or fir' >&2; exit 1 ;;
esac
export TEST_ACCOUNT TEST_GPU TEST_CLUSTER

mkdir -p "$TEST_ROOT/bin" "$TEST_ROOT/project/sbatch"
cp "$ROOT/$LAUNCHER" "$TEST_ROOT/project/"
ln -s "$ROOT/sbatch/reviewer_20261106" "$TEST_ROOT/project/sbatch/reviewer_20261106"
ln -s "$ROOT/cache" "$TEST_ROOT/project/cache"
ln -s "$ROOT/data" "$TEST_ROOT/project/data"
touch "$TEST_ROOT/activate"
cat > "$TEST_ROOT/bashenv" <<'MOCK'
module() { [[ "$1" == spider || "$1" == load ]]; }
MOCK
cat > "$TEST_ROOT/bin/sinfo" <<'MOCK'
#!/bin/bash
printf '%s\n' "${MOCK_GPUS:-gpu:$TEST_GPU:4}"
MOCK
cat > "$TEST_ROOT/bin/sacctmgr" <<'MOCK'
#!/bin/bash
printf '%s\n' "${MOCK_ACCOUNT:-$TEST_ACCOUNT}"
MOCK
cat > "$TEST_ROOT/bin/scontrol" <<'MOCK'
#!/bin/bash
[[ "$*" == 'show config' ]] || exit 1
printf 'ClusterName = %s\n' "$TEST_CLUSTER"
MOCK
cat > "$TEST_ROOT/bin/sbatch" <<'MOCK'
#!/bin/bash
set -eu
printf '%s\n' "$*" >> "$TEST_ROOT/calls"
id=900000
[[ ! -f "$TEST_ROOT/counter" ]] || id=$(cat "$TEST_ROOT/counter")
id=$((id+1)); printf '%s\n' "$id" > "$TEST_ROOT/counter"
printf '%s;%s\n' "$id" "$TEST_CLUSTER"
MOCK
chmod +x "$TEST_ROOT/bin/"*
export PATH="$TEST_ROOT/bin:$PATH" BASH_ENV="$TEST_ROOT/bashenv" ENV_ACTIVATE="$TEST_ROOT/activate"
unset SOURCE_ROOT DATA_ROOT REVIEW_ROOT REVIEW_BATCH REVIEW_CODE CHECK_JOB HOLD_JOBS
sh "$TEST_ROOT/project/$LAUNCHER" > "$TEST_ROOT/launch.log" 2>&1
LEDGER="$TEST_ROOT/project/reviewer_20261106_${PROFILE}/sbatch/submissions.tsv"
awk -F '\t' '
    NR==FNR {time[$1]=$4; tasks[$1]=$5; next}
    {if (!($1 in time) || time[$1]!=$4 || tasks[$1]!=$5) exit 1;
     if (seen[$1]++) exit 1;
     split($3,deps,":"); for (i in deps) if (deps[i] ~ /^[0-9]+$/ && !(deps[i] in ids)) exit 1;
     ids[$2]=1; n=1; if ($5 ~ /-/) {split($5,a,"-"); n=a[2]-a[1]+1} total+=n;
     if ($1 ~ /^cell_/) cells++; count++}
    END {if (count!=143 || cells!=120 || total!=855) exit 1}
' "$ROOT/sbatch/reviewer_20261106/submissions.tsv" "$LEDGER"
expected_jobs=143
if [[ "$PROFILE" == fir ]]; then
    GROUP_LEDGER="$TEST_ROOT/project/reviewer_20261106_fir/sbatch/groups.tsv"
    expected_jobs=$(wc -l < "$GROUP_LEDGER")
    ((expected_jobs<143))
    awk -F '\t' '
        function seconds(value,p) {split(value,p,":"); return p[1]*3600+p[2]*60+p[3]}
        NR==FNR {limit[$1]=seconds($4); next}
        {n=split($5,members,","); if(n<1 || n>5) exit 1; total=0;
         for(i=1;i<=n;i++) {if (!(members[i] in limit) || used[members[i]]++) exit 1;
             if(n>1 && limit[members[i]]>=1800) exit 1; total+=limit[members[i]]}
         if(total!=seconds($4)) exit 1;
         split($3,deps,":"); for(i in deps) if(deps[i] ~ /^[0-9]+$/ && !(deps[i] in ids)) exit 1;
         ids[$2]=1; if(n==5) full_packs++; count+=n}
        END {if(count!=143 || full_packs<1) exit 1}
    ' "$LEDGER" "$GROUP_LEDGER"
fi
[[ $(wc -l < "$TEST_ROOT/calls") == "$expected_jobs" ]]
[[ $(grep -c -- "--account=$TEST_ACCOUNT" "$TEST_ROOT/calls") == "$expected_jobs" ]]
! grep -qE -- '--hold|aip-yiweilu|216[0-9]{4}' "$TEST_ROOT/calls"
JOB_SCRIPT="$ROOT/sbatch/reviewer_20261106/job.sh"
[[ "$PROFILE" != fir ]] || JOB_SCRIPT="$TEST_ROOT/project/reviewer_20261106_fir/sbatch/job.sh"
gpu_flag="--gres=gpu:$TEST_GPU:1"
[[ "$PROFILE" != fir ]] || gpu_flag="--gpus-per-node=$TEST_GPU:1"
for directive in '--cpus-per-task=1' '--mem=6G' "$gpu_flag"; do
    grep -qF -- "#SBATCH $directive" "$JOB_SCRIPT"
done
if [[ "$PROFILE" == fir ]]; then
    ! grep -q -- '--array=' "$TEST_ROOT/calls"
    [[ $(grep -c -- '--export=ALL,REVIEW_BUNDLE_FILE=' "$TEST_ROOT/calls") == "$expected_jobs" ]]
    [[ $(grep -c -- '--gpus-per-node=h100:1' "$TEST_ROOT/calls") == "$expected_jobs" ]]
    ! grep -q '700000' "$TEST_ROOT/calls"
    # Execute only the shell loop: srun and rsync are mocks, with no Python work.
    cat > "$TEST_ROOT/bin/srun" <<'MOCK'
#!/bin/bash
printf '%s\n' "$*" >> "$TEST_ROOT/steps"
MOCK
    cat > "$TEST_ROOT/bin/rsync" <<'MOCK'
#!/bin/bash
exit 0
MOCK
    chmod +x "$TEST_ROOT/bin/srun" "$TEST_ROOT/bin/rsync"
    REVIEW_ROOT="$TEST_ROOT/project/reviewer_20261106_fir" \
        REVIEW_CODE="$ROOT/sbatch/reviewer_20261106/code" \
        REVIEW_BUNDLE_FILE="$TEST_ROOT/project/reviewer_20261106_fir/sbatch/groups/group_001.sh" \
        SLURM_TMPDIR="$TEST_ROOT/local" bash "$JOB_SCRIPT"
    [[ $(wc -l < "$TEST_ROOT/steps") == 20 ]]
    awk '{if ($NF!=(NR-1)%4 || $(NF-1)!="--part") exit 1}' "$TEST_ROOT/steps"
    defense_group=$(awk -F '\t' '$5=="cell_85" {print $1}' "$GROUP_LEDGER")
    : > "$TEST_ROOT/steps"
    REVIEW_ROOT="$TEST_ROOT/project/reviewer_20261106_fir" \
        REVIEW_CODE="$ROOT/sbatch/reviewer_20261106/code" \
        REVIEW_BUNDLE_FILE="$TEST_ROOT/project/reviewer_20261106_fir/sbatch/groups/$defense_group.sh" \
        SLURM_TMPDIR="$TEST_ROOT/local" bash "$JOB_SCRIPT"
    [[ $(wc -l < "$TEST_ROOT/steps") == 10 ]]
    awk '{if ($NF!=NR-1 || $(NF-1)!="--part") exit 1}' "$TEST_ROOT/steps"
else
    [[ $(grep -c -- '--array=' "$TEST_ROOT/calls") == 140 ]]
fi
MOCK_ACCOUNT="$TEST_ACCOUNT|" sh "$TEST_ROOT/project/$LAUNCHER" > "$TEST_ROOT/relaunch.log" 2>&1
[[ $(wc -l < "$TEST_ROOT/calls") == "$expected_jobs" ]]
if MOCK_GPUS=gpu:unsupported:8 sh "$TEST_ROOT/project/$LAUNCHER" > "$TEST_ROOT/rejected.log" 2>&1; then
    echo 'Missing-GPU check did not reject the launch' >&2; exit 1
fi
[[ $(wc -l < "$TEST_ROOT/calls") == "$expected_jobs" ]]
printf 'PASS (%s): 120 cells, %s allocations, 855 execution parts; correct account/GPU; unchanged walltimes; valid dependencies and array policy; idempotent rerun; missing GPU rejected.\n' "$PROFILE" "$expected_jobs"
