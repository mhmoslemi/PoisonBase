#!/usr/bin/env python3
"""Generate missing augmentation poison caches on Vulcan, then evaluate them."""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parent
VULCAN_HOST = os.environ.get("VULCAN_HOST", "mmoslem3@vulcan.alliancecan.ca")
SSH_OPTIONS = [
    "-o", "ControlMaster=no",
    "-o", "ControlPersist=no",
    "-o", "ControlPath=none",
    "-o", "ServerAliveInterval=30",
    "-o", "ServerAliveCountMax=20",
]
SENTINEL = "__END_MISSING__"


def die(message: str) -> "NoReturn":
    raise SystemExit(f"ERROR: {message}")


def exports(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    for line in path.read_text().splitlines():
        match = re.match(
            r"export ([A-Z0-9_]+)=(?:'([^']*)'|([^\s]+))", line
        )
        if match:
            values[match.group(1)] = match.group(2) or match.group(3)
    return values


def poison_configs() -> dict[str, dict[str, str]]:
    wrappers = sorted(
        (ROOT / "sbatch" / "vulcan_remaining_20260827").glob(
            "augment_*_resume.sh"
        )
    )
    if len(wrappers) != 21:
        die(f"expected 21 augmentation wrappers; found {len(wrappers)}")

    selections = {
        "RUN_RANDOM": ("random", "2", "random"),
        "RUN_GREEDY": ("ours", "2", "greedy"),
        "RUN_DPP2": ("dpp", "2", "dpp2"),
        "RUN_DPP025": ("dpp", "0.25", "dpp025"),
        "RUN_DPP01": ("dpp", "0.1", "dpp01"),
    }
    configs: dict[str, dict[str, str]] = {}
    for wrapper in wrappers:
        match = re.search(
            r"^source .*/sbatch/augment_extra/([^/\s]+\.sh)$",
            wrapper.read_text(),
            re.MULTILINE,
        )
        if not match:
            die(f"source template missing from {wrapper}")
        template = ROOT / "sbatch" / "augment_extra" / match.group(1)
        values = exports(template)
        for key in ("ROW_ID", "MODEL", "ATTACK", "BUDGET", "TARGET_SELECT"):
            if not values.get(key):
                die(f"{key} missing from {template}")
        for run_key, (select, alpha, label) in selections.items():
            run = values.get(run_key)
            if not run:
                die(f"{run_key} missing from {template}")
            config = {
                "run": run,
                "row": values["ROW_ID"],
                "model": values["MODEL"],
                "attack": values["ATTACK"],
                "budget": values["BUDGET"],
                "target_select": values["TARGET_SELECT"],
                "select": select,
                "alpha": alpha,
                "label": label,
            }
            old = configs.setdefault(run, config)
            if old != config:
                die(f"inconsistent metadata for {run}")
    if len(configs) != 25:
        die(f"expected 25 distinct poison configurations; found {len(configs)}")
    return configs


def requested_time(config: dict[str, str]) -> str:
    if config["model"] == "ConvNetBN":
        return "0-02:00:00" if config["attack"] == "fc" else "0-03:00:00"
    if config["attack"] == "fc":
        return "0-03:00:00"
    if config["attack"] == "gradmatch":
        return "0-04:30:00"
    return "0-05:30:00"


def prerequisite_script(index: int, config: dict[str, str]) -> str:
    name = f"vaugpre_{index:02d}_r{config['row']}_{config['label']}"
    original = (
        f"USE_JACOBIAN_SCORE=0 CLASS_PAIR=dog-bird "
        f"MODEL={config['model']} ATTACK={config['attack']} "
        f"TARGET_SELECT={config['target_select']} "
        f"BUDGETS={config['budget']} SELECT={config['select']} "
        f"SEL_ALPHA={config['alpha']} NUM_TARGETS=5 NUM_VICTIMS=1 "
        f"RECOMPUTE_DELTAS=1 sh sel_dpp.sh"
    )
    return f"""#!/bin/bash
#SBATCH --account=aip-boyuwang
#SBATCH --job-name={name}
#SBATCH --time={requested_time(config)}
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=15G
#SBATCH --gpus-per-node=l40s:1
#SBATCH --signal=B:USR1@300
#SBATCH --output=/home/mmoslem3/scratch/PoisonBase/sbatch/logs2/%x_%j.out

# Generate exactly one missing poison cache.
export JOB_KIND=attack
export EXTRA_ALPHA={config['alpha']}
export ORIGINAL_COMMAND='{original}'
export USE_JACOBIAN_SCORE=0
export JACOBIAN_WEIGHT=1.0
export JACOBIAN_BATCH_SIZE=64
export CLASS_PAIR=dog-bird
export MODEL={config['model']}
export ATTACK={config['attack']}
export BUDGETS={config['budget']}
export SELECT={config['select']}
export SEL_ALPHA={config['alpha']}
export SHARP_MODE=worst
export SHARP_SIGMA=0.05
export TARGET_SELECT={config['target_select']}
export NUM_TARGETS=5
export REQUIRED_CACHED_TARGETS=5
export NUM_VICTIMS=1
export RECOMPUTE_DELTAS=1
export PREREQ_ROW={config['row']}
export EXPECTED_RUN_NAME='{config['run']}'

export SOURCE_ROOT=/home/mmoslem3/scratch/PoisonBase
export PERSIST_DATA_ROOT=/home/mmoslem3/scratch/PoisonBase/data
export PYTHON_ENV=/home/mmoslem3/ENV

source "$SOURCE_ROOT/sbatch/_defense_extra_job_common.sh"
"""


VULCAN_REMOTE = r"""
set -eu
root='/home/mmoslem3/scratch/PoisonBase'
while IFS= read -r artifact_path; do
    [ "$artifact_path" = '__END_MANIFEST__' ] && break
    count=$(find "$root/$artifact_path" -maxdepth 1 -type f -name 'delta_*.pt' 2>/dev/null | wc -l | tr -d '[:space:]')
    if [ "$count" -lt 5 ]; then
        printf 'MISSING\t%s\n' "$artifact_path"
    else
        printf 'Vulcan present: %s (%s perturbations)\n' "$artifact_path" "$count" >&2
    fi
done
printf '__END_MISSING__\n'

mkdir -p "$root"
tar -xf - -C "$root"
cd "$root"
bash submit_vulcan-augmentation-with-prereqs.sh
"""


def create_stage(stage: Path, missing: list[dict[str, str]]) -> None:
    pre_dir = stage / "sbatch" / "augment_missing_prereqs_20260827"
    pre_dir.mkdir(parents=True)
    filenames: list[str] = []
    for index, config in enumerate(missing, 1):
        filename = (
            f"vaugpre_{index:02d}_r{config['row']}_{config['label']}.sh"
        )
        filenames.append(filename)
        (pre_dir / filename).write_text(prerequisite_script(index, config))
    (pre_dir / "current_jobs.txt").write_text(
        "".join(f"{filename}\n" for filename in filenames)
    )

    shutil.copytree(
        ROOT / "sbatch" / "vulcan_remaining_20260827",
        stage / "sbatch" / "vulcan_remaining_20260827",
        dirs_exist_ok=True,
    )
    shutil.copy2(ROOT / "sel_dpp.sh", stage / "sel_dpp.sh")
    shutil.copy2(
        ROOT / "submit_vulcan-augmentation-with-prereqs.sh",
        stage / "submit_vulcan-augmentation-with-prereqs.sh",
    )


def tar_command(stage: Path) -> list[str]:
    gtar = shutil.which("gtar")
    if gtar:
        return [
            gtar,
            "--checkpoint=500",
            "--checkpoint-action=dot",
            "-cf", "-",
            "-C", str(stage),
            ".",
        ]
    return ["tar", "-cf", "-", "-C", str(stage), "."]


def run() -> int:
    configs = poison_configs()
    paths = {
        f"ours_result/{run}/poison_cache": config
        for run, config in configs.items()
    }

    print("Authenticate to Vulcan once...", flush=True)
    process = subprocess.Popen(
        ["ssh", *SSH_OPTIONS, VULCAN_HOST, VULCAN_REMOTE],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=None,
    )
    assert process.stdin is not None
    assert process.stdout is not None

    try:
        manifest = "".join(f"{path}\n" for path in sorted(paths))
        process.stdin.write((manifest + "__END_MANIFEST__\n").encode())
        process.stdin.flush()

        missing_paths: list[str] = []
        while True:
            raw = process.stdout.readline()
            if not raw:
                die("Vulcan connection ended before returning its cache list")
            line = raw.decode(errors="replace").rstrip("\n")
            if line == SENTINEL:
                break
            if line.startswith("MISSING\t"):
                missing_paths.append(line.split("\t", 1)[1])
            elif line:
                print(line, flush=True)

        print(f"Vulcan is missing {len(missing_paths)} poison cache(s):")
        for path in missing_paths:
            print(f"  {path}")

        with tempfile.TemporaryDirectory(prefix="augment-generate.") as tmp:
            stage = Path(tmp)
            create_stage(stage, [paths[path] for path in missing_paths])
            print(
                "Uploading poison-generator jobs and 21 one-cell evaluations...",
                flush=True,
            )
            tar_proc = subprocess.Popen(
                tar_command(stage), stdout=subprocess.PIPE, stderr=None
            )
            assert tar_proc.stdout is not None
            shutil.copyfileobj(
                tar_proc.stdout, process.stdin, length=1024 * 1024
            )
            tar_proc.stdout.close()
            if tar_proc.wait() != 0:
                die("local tar process failed")

        process.stdin.close()
        for raw in process.stdout:
            sys.stdout.write(raw.decode(errors="replace"))
            sys.stdout.flush()
        code = process.wait()
        if code:
            die(f"Vulcan prerequisite submission failed with status {code}")
        print("Missing-cache generation and augmentation submission complete.")
        return 0
    except BaseException:
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
        raise


if __name__ == "__main__":
    raise SystemExit(run())
