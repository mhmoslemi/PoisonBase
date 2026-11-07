#!/bin/bash
[ -n "${BASH_VERSION:-}" ] || exec bash "$0" "$@"
set -Eeuo pipefail

ROOT=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd -P)
BATCH="$ROOT/sbatch/rebuttal_access_conv_7x8_vulcan"
ACCOUNT=aip-bghaddar
LEGACY="$ROOT/rebuttal_access_conv_5x4_result"
BASE="$ROOT/experiments/rebuttal_access_conv/runner.py"

die() { echo "ERROR: $*" >&2; exit 1; }
command -v sbatch >/dev/null || die "Run this script on Vulcan."
sinfo -h -o '%G' --Node | sort -u | rg -q 'gpu:l40s:' || die "Vulcan L40S GPUs were not found."
sacctmgr -n -P show assoc where user="$USER" format=Account |
    awk -F '|' '$1=="aip-bghaddar" {found=1} END {exit !found}' || die "$ACCOUNT is not available."

for path in "$BASE" /home/mmoslem3/ENV/bin/activate \
    "$ROOT"/sbatch/reviewer_20261106/code/{final_update,networks,utils}.py \
    "$ROOT"/data/cifar-10-batches-py/{data_batch_1,data_batch_2,data_batch_3,data_batch_4,data_batch_5,test_batch,batches.meta}; do
    [ -s "$path" ] || die "Missing $path"
done

for ((i=0; i<30; i++)); do
    path="$ROOT/cache/surrogates/ConvNetBN_60ep_lr0.1_bs128_seed42/net_$i.pt"
    [ -s "$path" ] || die "Missing $path"
done
for fraction in 0.2 0.5; do
    for draw in 0 1 2; do
        pools=("$ROOT"/reviewer_20261106_fir/reviewer_20261106_result/surrogates/ConvNetBN_f${fraction}_r${draw}_*)
        [ "${#pools[@]}" -eq 1 ] && [ -d "${pools[0]}" ] || die "Missing/ambiguous surrogate pool f=$fraction r=$draw"
        for ((i=0; i<20; i++)); do
            [ -s "${pools[0]}/net_$i.pt" ] || die "Missing ${pools[0]}/net_$i.pt"
        done
    done
done

# Existing victims can only be extended with their original poison perturbations.
missing=0
first_missing=""
for method in random basis; do
    if [ "$method" = random ]; then conditions=(1 3 4); draws=(fixed); else conditions=(1 2 3 4 5 6); draws=(0 1 2); fi
    for condition in "${conditions[@]}"; do
        for surrogate in "${draws[@]}"; do
            for candidate in 0 1 2; do
                for target in 843 1257 8252 8730 8055; do
                    path="$LEGACY/$method/condition_$condition/poisons/s${surrogate}_c${candidate}_t${target}.pt"
                    if [ ! -s "$path" ]; then
                        missing=$((missing+1))
                        [ -n "$first_missing" ] || first_missing="$path"
                    fi
                    for victim in 0 1 2 3; do
                        path="$LEGACY/$method/condition_$condition/trials/s${surrogate}_c${candidate}_t${target}/victim_$victim.json"
                        [ -s "$path" ] || die "Missing completed result $path"
                    done
                done
            done
        done
    done
done
[ "$missing" -eq 0 ] || die "Missing $missing original GM poison caches; first: $first_missing. Copy the original poisons/ directories from Fir into $LEGACY before submitting. No jobs submitted."

mkdir -p "$BATCH/code" "$BATCH/logs"
exec 9>"$BATCH/submit.lock"
flock -n 9 || die "Another submission is already running."
# Freeze the existing runner so later edits do not change queued jobs.
[ -f "$BATCH/code/access_base.py" ] || cp "$BASE" "$BATCH/code/access_base.py"

MANIFEST="$BATCH/manifest.tsv"
if [ ! -f "$MANIFEST" ]; then
    printf 'id\tattack\tmethod\tcondition\tstart\tstop\n' > "$MANIFEST.tmp"
    task=0
    add() {
        printf '%s\t%s\t%s\t%s\t%s\t%s\n' "$task" "$1" "$2" "$3" "$4" "$5" >> "$MANIFEST.tmp"
        task=$((task+1))
    }
    for condition in 1 3 4; do add gradmatch random "$condition" 0 3; done
    for condition in 1 2 3 4 5 6; do add gradmatch basis "$condition" 0 9; done
    add gradmatch basis 8 0 4
    add gradmatch basis 8 4 9
    for condition in 1 3 4; do add fc random "$condition" 0 3; done
    for condition in 1 2 3 4 5 6 8; do
        add fc basis "$condition" 0 4
        add fc basis "$condition" 4 9
    done
    [ "$task" -eq 28 ] || die "Expected exactly 28 tasks."
    mv "$MANIFEST.tmp" "$MANIFEST"
fi

if [ ! -f "$BATCH/code/worker.py" ]; then
cat > "$BATCH/code/worker.py" <<'PY'
"""Extend W1 to seven targets and eight victims, retaining completed GM trials."""
import csv
import json
import os
from pathlib import Path
import sys

import access_base as base

batch = Path(os.environ["ACCESS_BATCH"])
with (batch / "manifest.tsv").open() as handle:
    tasks = list(csv.DictReader(handle, delimiter="\t"))
assert len(tasks) == 28
task = tasks[int(sys.argv[1])]
attack, method = task["attack"], task["method"]
condition_id = int(task["condition"])
start, stop = int(task["start"]), int(task["stop"])
assert attack in ("gradmatch", "fc") and method in ("random", "basis")
assert condition_id in (1, 2, 3, 4, 5, 6, 8)
assert method != "random" or condition_id in (1, 3, 4)

base.TARGETS = {
    "gradmatch": (843, 1257, 8252, 8730, 8055, 5859, 8954),
    "fc": (2540, 8129, 3725, 5705, 7663, 3875, 5169),
}[attack]
base.VICTIMS = range(8)
# Append the full-access condition without changing the original condition seeds.
base.CONDITIONS += ((0.002, 1.0, 1.0),)
if attack == "fc":
    base.OUT = base.ROOT / "rebuttal_access_conv_7x8_result" / "bp"
condition = base.CONDITIONS[condition_id]
all_pairs = list(base.combinations(method))
assert 0 <= start < stop <= len(all_pairs)
pairs = all_pairs[start:stop]

if attack == "gradmatch" and condition_id != 8:
    for sd, cd in pairs:
        for target in base.TARGETS[:5]:
            path = base.poison_path(method, condition_id, sd, cd, target)
            if not path.is_file():
                raise RuntimeError(f"Original poison cache missing; refusing to recraft: {path}")


def collect(draws, required):
    rows = []
    for sd, cd in draws:
        sname = "fixed" if sd is None else str(sd)
        for target in base.TARGETS:
            directory = (base.OUT / method / f"condition_{condition_id}" /
                         "trials" / f"s{sname}_c{cd}_t{target}")
            for victim in base.VICTIMS:
                path = directory / f"victim_{victim}.json"
                if not path.is_file():
                    if required:
                        raise RuntimeError(f"Missing completed trial: {path}")
                    return None
                row = json.loads(path.read_text())
                assert row["method"] == method and row["target"] == target
                assert row["victim"] == victim and row["epochs"] == 50
                assert tuple(row["condition"]) == tuple(condition)
                assert row["seed"] == 4200000 + target * 100 + victim
                assert row["surrogate_draw"] == sd and row["candidate_draw"] == cd
                assert row["success"] in (0, 1)
                rows.append(row)
    assert len(rows) == len(draws) * 7 * 8
    return rows


def summarize(_method, _condition):
    directory = base.OUT / method / f"condition_{condition_id}"

    def save(name, rows, draws):
        base.atomic_json(directory / name, {
            "attack": attack, "method": method, "condition_id": condition_id,
            "condition": condition, "targets": base.TARGETS,
            "victims": list(base.VICTIMS), "draws": draws, "trials": len(rows),
            "asr": sum(row["success"] for row in rows) / len(rows),
            "clean_accuracy": sum(row["clean_accuracy"] for row in rows) / len(rows),
        })

    save(f"summary_7x8_draws_{start}_{stop}.json", collect(pairs, True), pairs)
    rows = collect(all_pairs, False)
    if rows is not None:
        save("summary_7x8.json", rows, all_pairs)
        print(f"CELL COMPLETE: attack={attack} method={method} condition={condition_id} "
              f"trials={len(rows)}", flush=True)


base.combinations = lambda _method: iter(pairs)
base.summarize = summarize
print(f"W1 task={task['id']} attack={attack} method={method} condition={condition} "
      f"draws={pairs} targets={base.TARGETS} victims=0..7 epochs=50", flush=True)
args, ctx = base.context()
args.attack = attack
args.num_targets = 7
args.num_victims = 8
base.run_cell(args, ctx, method, condition_id)
PY
fi

if [ ! -f "$BATCH/job.sh" ]; then
cat > "$BATCH/job.sh" <<'SH'
#!/bin/bash
set -Eeuo pipefail
cd "$ACCESS_ROOT"
module load StdEnv/2023 gcc/12.3 python/3.11.5 cuda/12.6 cudnn/9.10.0.56
source /home/mmoslem3/ENV/bin/activate
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
export HF_HOME="$ACCESS_ROOT/rebuttal_access_conv_7x8_result/downloads/hf"
export TORCH_HOME="$ACCESS_ROOT/rebuttal_access_conv_7x8_result/downloads/torch"
export XDG_CACHE_HOME="$ACCESS_ROOT/rebuttal_access_conv_7x8_result/downloads/cache"
export MPLCONFIGDIR="${SLURM_TMPDIR:?}/matplotlib"
mkdir -p "$SLURM_TMPDIR/data"
rsync -a "$ACCESS_ROOT/data/cifar-10-batches-py/" "$SLURM_TMPDIR/data/cifar-10-batches-py/"
export REVIEW_DATA="$SLURM_TMPDIR/data"
srun --ntasks=1 --cpus-per-task=1 python "$ACCESS_BATCH/code/worker.py" "$1"
SH
fi

LEDGER="$BATCH/submissions.tsv"
touch "$LEDGER"
while IFS=$'\t' read -r task attack method condition start stop; do
    [ "$task" != id ] || continue
    old=$(awk -F '\t' -v task="$task" '$1==task {print $2}' "$LEDGER")
    if [ -n "$old" ]; then
        echo "Already submitted task $task: $old"
        continue
    fi
    job=$(sbatch --parsable --account="$ACCOUNT" \
        --nodes=1 --ntasks=1 --cpus-per-task=1 --mem=6G \
        --gres=gpu:l40s:1 --time=04:00:00 \
        --job-name="w1_${attack}_${method}_${condition}_${start}" \
        --chdir="$ROOT" --output="$BATCH/logs/%x-%j.out" \
        --export="ALL,ACCESS_ROOT=$ROOT,ACCESS_BATCH=$BATCH" "$BATCH/job.sh" "$task")
    job=${job%%;*}
    [[ "$job" =~ ^[0-9]+$ ]] || die "Invalid sbatch response: $job"
    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
        "$task" "$job" "$attack" "$method" "$condition" "$start" "$stop" >> "$LEDGER"
    echo "Submitted task $task: job $job ($attack $method condition=$condition draws=$start:$stop)"
done < "$MANIFEST"
echo "28 jobs recorded under $ACCOUNT; 1 CPU, 6G RAM, 1 L40S, 4h each. Ledger: $LEDGER"
