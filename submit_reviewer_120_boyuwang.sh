#!/bin/sh
# Run with sh on Killarney in ~/scratch/attack_if, or in the local project copy.
[ -n "${BASH_VERSION:-}" ] || exec bash "$0" "$@"
set -Eeuo pipefail
SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd -P)
SOURCE_ROOT="${SOURCE_ROOT:-$SCRIPT_DIR}"
SOURCE_ROOT=$(CDPATH= cd -- "$SOURCE_ROOT" && pwd -P)
ASSETS="$SCRIPT_DIR/sbatch/reviewer_20261106"
# A separate workspace prevents collisions with the Lu jobs and their job IDs.
export REVIEW_ROOT="$SOURCE_ROOT/reviewer_20261106_wang"
export REVIEW_BATCH="$REVIEW_ROOT/sbatch"
export REVIEW_CODE="$ASSETS/code"
export REVIEW_MANIFEST="$ASSETS/manifest.tsv"
export REVIEW_JOB_SCRIPT="$ASSETS/job.sh"
export REVIEW_ACCOUNT=aip-boyuwang
export REVIEW_PREFIX=rv_wang
export ENV_ACTIVATE="${ENV_ACTIVATE:-/home/mmoslem3/ENV/bin/activate}"
unset CHECK_JOB HOLD_JOBS

die() { printf 'ERROR: %s\n' "$*" >&2; exit 1; }
for command in sbatch sinfo sacctmgr scontrol sha256sum flock module; do
    command -v "$command" >/dev/null 2>&1 || die "$command is unavailable; run on the cluster login node"
done
[[ -f "$ENV_ACTIVATE" ]] || die "Missing environment: $ENV_ACTIVATE"
[[ -f "$ASSETS/validated_on_vulcan.json" ]] || die 'Copy sbatch/reviewer_20261106 with this launcher'
[[ $(awk 'END {print NR-1}' "$REVIEW_MANIFEST") == 120 ]] || die 'Expected exactly 120 configurations'
(cd "$REVIEW_CODE" && sha256sum --check --status "$ASSETS/validated_source.sha256") || die 'Runtime differs from the validated snapshot'
(cd "$SOURCE_ROOT" && sha256sum --check --status "$ASSETS/validated_surrogates.sha256") || die 'Copy the two exact 60-epoch surrogate pools from Vulcan; see validated_surrogates.sha256'

if [[ -z "${DATA_ROOT:-}" ]]; then
    DATA_ROOT="$SOURCE_ROOT/data"
    [[ -f "$DATA_ROOT/cifar-10-batches-py/data_batch_1" ]] || DATA_ROOT=/home/mmoslem3/scratch/data
fi
DATA_ROOT=$(CDPATH= cd -- "$DATA_ROOT" && pwd -P)
for file in data_batch_1 data_batch_2 data_batch_3 data_batch_4 data_batch_5 test_batch batches.meta; do
    [[ -f "$DATA_ROOT/cifar-10-batches-py/$file" ]] || die "Missing CIFAR-10 input: $DATA_ROOT/cifar-10-batches-py/$file"
done
TINY_ARCHIVE="$DATA_ROOT/tiny-imagenet-200.zip"
[[ -f "$TINY_ARCHIVE" ]] || TINY_ARCHIVE="$SOURCE_ROOT/data/tiny-imagenet-200.zip"
[[ -f "$TINY_ARCHIVE" ]] || die 'Missing tiny-imagenet-200.zip in the data directory'

gres=$(sinfo -h -o '%G' --Node)
[[ "$gres" == *gpu:l40s:* ]] || die 'No L40S GPUs advertised by this cluster'
sacctmgr -nP show assoc user="$USER" account=aip-boyuwang format=Account | awk -F '|' '$1=="aip-boyuwang" {found=1} END {exit !found}' || die 'Wang account is not available to this user'
cluster=$(scontrol show config | awk '$1=="ClusterName" {print $3}')
[[ -n "$cluster" ]] || die 'Could not identify the Slurm cluster'
for spec in StdEnv/2023 gcc/12.3 python/3.11.5 cuda/12.6 cudnn/9.10.0.56; do
    module spider "$spec" >/dev/null 2>&1 || die "Required module unavailable: $spec"
done

mkdir -p "$REVIEW_BATCH" "$REVIEW_ROOT/data" "$REVIEW_ROOT/reviewer_20261106_result"
exec 8>"$REVIEW_ROOT/setup.lock"
flock -n 8 || die 'Another Wang launcher is active'
identity="$cluster aip-boyuwang $SOURCE_ROOT"
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
printf 'Submitting the same 120 configurations and preparation/controls on %s, account aip-boyuwang.\n' "$cluster"
printf 'Resources per task: 1 CPU, 6 GB RAM, 1 L40S. Results: %s/reviewer_20261106_result\n' "$REVIEW_ROOT"
bash "$ASSETS/submit_all.sh"
