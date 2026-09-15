#!/usr/bin/env python3
"""Generate the separate attack_if-path copy of the 216 Slurm jobs.

The canonical manifest from the PoisonBase-path set is used as the configuration
source, ensuring this profile changes paths only. This script never calls sbatch.
"""

import csv
from pathlib import Path


JOB_DIR = Path(__file__).resolve().parent
SOURCE_MANIFEST = JOB_DIR.parent / "cross_arch_k_10x6_20260915" / "MANIFEST.tsv"
MODEL_TAG = {"ConvNetBN": "conv", "ResNet20BN": "r20", "VGG13BN": "vgg"}
ATTACK_TAG = {"fc": "bp", "gradmatch": "gm", "sapa": "sapa"}
BUDGET_TAG = {"0.002": "b0002", "0.005": "b0005"}
TARGET_DEGREES = {
    ("ConvNetBN", "fc"): 50,
    ("ConvNetBN", "gradmatch"): 70,
    ("ConvNetBN", "sapa"): 70,
    ("ResNet20BN", "fc"): 10,
    ("ResNet20BN", "gradmatch"): 14,
    ("ResNet20BN", "sapa"): 14,
    ("VGG13BN", "fc"): 3,
    ("VGG13BN", "gradmatch"): 50,
    ("VGG13BN", "sapa"): 50,
}


def main() -> None:
    if not SOURCE_MANIFEST.is_file():
        raise SystemExit(f"source manifest missing: {SOURCE_MANIFEST}")
    with SOURCE_MANIFEST.open(newline="") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    if len(rows) != 216:
        raise SystemExit(f"source manifest has {len(rows)} rows, expected 216")

    for old in JOB_DIR.glob("job_*.sh"):
        old.unlink()

    for row in rows:
        index = int(row["index"])
        attack = row["attack"]
        budget = row["budget"]
        victim = row["victim_model"]
        selector = row["selector_model"]
        k = int(row["K"])
        if (row["targets"], row["victims"], row["craft_ensemble"],
                row["num_surrogates"], row["craft_steps"]) != (
                    "10", "6", "5", "30", "250"):
            raise SystemExit(f"unexpected protocol in manifest row {index}")

        attack_tag = ATTACK_TAG[attack]
        budget_tag = BUDGET_TAG[budget]
        victim_tag = MODEL_TAG[victim]
        selector_tag = MODEL_TAG[selector]
        short_name = (
            f"ak{index:03d}_{attack_tag}_{budget_tag}_"
            f"v{victim_tag}_s{selector_tag}_k{k}"
        )
        filename = (
            f"job_{index:03d}_{attack_tag}_{budget_tag}_"
            f"v{victim_tag}_s{selector_tag}_k{k}.sh"
        )
        body = f'''#!/bin/bash
#SBATCH --account={row["account"]}
#SBATCH --job-name={short_name}
#SBATCH --time={row["time_limit"]}
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=7G
#SBATCH --gres=gpu:l40s:1
#SBATCH --signal=B:USR1@300
#SBATCH --output=/home/mmoslem3/scratch/attack_if/sbatch/logs/{short_name}-%j.out

# Separate attack_if-path profile. One file = one configuration.
export SOURCE_ROOT="${{SOURCE_ROOT:-/home/mmoslem3/scratch/attack_if}}"
export ROOT="${{ROOT:-$SOURCE_ROOT}}"
export ENV_ACTIVATE="${{ENV_ACTIVATE:-/home/mmoslem3/ENV/bin/activate}}"
export DATA_ROOT="${{DATA_ROOT:-/home/mmoslem3/scratch/data}}"
export RUN_ROOT="${{RUN_ROOT:-${{SLURM_TMPDIR:-}}/attack_if}}"
export XFULL_INDEX={index}
export XFULL_ATTACK={attack}
export XFULL_BUDGET={budget}
export XFULL_VICTIM_MODEL={victim}
export XFULL_SELECTOR_MODEL={selector}
export XFULL_K={k}
export XFULL_TARGET_DEGREE={TARGET_DEGREES[(victim, attack)]}
export XFULL_RUN_NAME={row["run_name"]}
export ORIGINAL_COMMAND="attack={attack} budget={budget} A=V={victim} S={selector} K={k} base=ours jacobian=off craft_steps=250 targets=10 victims=6 profile=attack_if"
source "$ROOT/sbatch/cross_arch_k_10x6_attack_if_20260915/_job_common.sh"
'''
        (JOB_DIR / filename).write_text(body)
        row["job_file"] = filename

    with (JOB_DIR / "MANIFEST.tsv").open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=list(rows[0]), delimiter="\t", lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(rows)
    print(f"generated {len(rows)} attack_if jobs in {JOB_DIR}")


if __name__ == "__main__":
    main()
