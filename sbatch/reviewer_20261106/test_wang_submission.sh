#!/bin/bash
# Shell-only integration check: every Slurm command is mocked; submits no jobs.
set -Eeuo pipefail
ROOT=$(CDPATH= cd -- "$(dirname -- "$0")/../.." && pwd -P)
TEST_ROOT=$(mktemp -d /tmp/reviewer-wang-test.XXXXXX)
trap 'rm -rf -- "$TEST_ROOT"' EXIT
export TEST_ROOT
mkdir -p "$TEST_ROOT/bin" "$TEST_ROOT/project/sbatch"
cp "$ROOT/submit_reviewer_120_boyuwang.sh" "$TEST_ROOT/project/"
ln -s "$ROOT/sbatch/reviewer_20261106" "$TEST_ROOT/project/sbatch/reviewer_20261106"
ln -s "$ROOT/cache" "$TEST_ROOT/project/cache"
ln -s "$ROOT/data" "$TEST_ROOT/project/data"
touch "$TEST_ROOT/activate"
cat > "$TEST_ROOT/bashenv" <<'MOCK'
module() { [[ "$1" == spider ]]; }
MOCK
cat > "$TEST_ROOT/bin/sinfo" <<'MOCK'
#!/bin/bash
printf '%s\n' "${MOCK_GPUS:-gpu:l40s:4}"
MOCK
cat > "$TEST_ROOT/bin/sacctmgr" <<'MOCK'
#!/bin/bash
printf '%s\n' "${MOCK_ACCOUNT:-aip-boyuwang}"
MOCK
cat > "$TEST_ROOT/bin/scontrol" <<'MOCK'
#!/bin/bash
[[ "$*" == 'show config' ]] || exit 1
printf 'ClusterName = killarney\n'
MOCK
cat > "$TEST_ROOT/bin/sbatch" <<'MOCK'
#!/bin/bash
set -eu
printf '%s\n' "$*" >> "$TEST_ROOT/calls"
id=900000
[[ ! -f "$TEST_ROOT/counter" ]] || id=$(cat "$TEST_ROOT/counter")
id=$((id+1)); printf '%s\n' "$id" > "$TEST_ROOT/counter"
printf '%s;killarney\n' "$id"
MOCK
chmod +x "$TEST_ROOT/bin/"*
export PATH="$TEST_ROOT/bin:$PATH" BASH_ENV="$TEST_ROOT/bashenv" ENV_ACTIVATE="$TEST_ROOT/activate"
unset SOURCE_ROOT DATA_ROOT REVIEW_ROOT REVIEW_BATCH REVIEW_CODE CHECK_JOB HOLD_JOBS
sh "$TEST_ROOT/project/submit_reviewer_120_boyuwang.sh" > "$TEST_ROOT/launch.log" 2>&1
LEDGER="$TEST_ROOT/project/reviewer_20261106_wang/sbatch/submissions.tsv"
awk -F '\t' '
    NR==FNR {time[$1]=$4; tasks[$1]=$5; next}
    {if (!($1 in time) || time[$1]!=$4 || tasks[$1]!=$5) exit 1;
     if (seen[$1]++) exit 1;
     split($3,deps,":"); for (i in deps) if (deps[i] ~ /^[0-9]+$/ && !(deps[i] in ids)) exit 1;
     ids[$2]=1; n=1; if ($5 ~ /-/) {split($5,a,"-"); n=a[2]-a[1]+1} total+=n;
     if ($1 ~ /^cell_/) cells++; count++}
    END {if (count!=143 || cells!=120 || total!=855) exit 1}
' "$ROOT/sbatch/reviewer_20261106/submissions.tsv" "$LEDGER"
[[ $(wc -l < "$TEST_ROOT/calls") == 143 ]]
[[ $(grep -c -- '--account=aip-boyuwang' "$TEST_ROOT/calls") == 143 ]]
! grep -qE -- '--hold|aip-yiweilu|216[0-9]{4}' "$TEST_ROOT/calls"
for directive in '--cpus-per-task=1' '--mem=6G' '--gres=gpu:l40s:1'; do
    grep -qF -- "#SBATCH $directive" "$ROOT/sbatch/reviewer_20261106/job.sh"
done
MOCK_ACCOUNT="aip-boyuwang|" sh "$TEST_ROOT/project/submit_reviewer_120_boyuwang.sh" > "$TEST_ROOT/relaunch.log" 2>&1
[[ $(wc -l < "$TEST_ROOT/calls") == 143 ]]
if MOCK_GPUS=gpu:h100:8 sh "$TEST_ROOT/project/submit_reviewer_120_boyuwang.sh" > "$TEST_ROOT/rejected.log" 2>&1; then
    echo 'Missing-L40S check did not reject the launch' >&2; exit 1
fi
[[ $(wc -l < "$TEST_ROOT/calls") == 143 ]]
printf 'PASS: 120 cells, 143 submissions, 855 tasks; Wang account; unchanged times/arrays; valid dependencies; idempotent rerun; missing GPU rejected.\n'
