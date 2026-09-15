#!/usr/bin/env python3
"""Generate Boyu copies of the 54 pasted pending Yiwei jobs.

This writes job files and a manifest only. It never invokes scancel or sbatch.
"""

import csv
from pathlib import Path


JOB_DIR = Path(__file__).resolve().parent
SOURCE_MANIFEST = JOB_DIR.parent / "cross_arch_k_10x6_20260915" / "MANIFEST.tsv"

# original configuration index -> pending Yiwei Slurm job ID from the pasted list
REPLACED_JOBS = {
    42: 927130,
    45: 927133,
    48: 927136,
    51: 927139,
    54: 927142,
    57: 927145,
    60: 927148,
    63: 927151,
    66: 927154,
    69: 927157,
    72: 927160,
    75: 927163,
    78: 927166,
    81: 927169,
    84: 927172,
    87: 927175,
    90: 927178,
    93: 927181,
    96: 927184,
    99: 927187,
    102: 927190,
    105: 927193,
    108: 927196,
    111: 927199,
    114: 927202,
    117: 927205,
    120: 927208,
    123: 927211,
    126: 927214,
    129: 927217,
    132: 927220,
    135: 927223,
    138: 927226,
    141: 927229,
    144: 927232,
    147: 927235,
    165: 927253,
    168: 927256,
    171: 927259,
    174: 927262,
    177: 927265,
    180: 927268,
    183: 927271,
    186: 927274,
    189: 927277,
    192: 927280,
    195: 927283,
    198: 927286,
    201: 927289,
    204: 927292,
    207: 927295,
    210: 927298,
    213: 927301,
    216: 927304,
}

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
    with SOURCE_MANIFEST.open(newline="") as handle:
        source_rows = list(csv.DictReader(handle, delimiter="\t"))
    source_by_index = {int(row["index"]): row for row in source_rows}
    if set(REPLACED_JOBS) - set(source_by_index):
        raise SystemExit("one or more pasted indices are absent from the canonical manifest")

    for old in JOB_DIR.glob("job_*.sh"):
        old.unlink()

    output_rows = []
    for index in sorted(REPLACED_JOBS):
        row = dict(source_by_index[index])
        if row["account"] != "aip-yiweilu":
            raise SystemExit(f"canonical job {index} is not assigned to Yiwei")
        attack = row["attack"]
        budget = row["budget"]
        victim = row["victim_model"]
        selector = row["selector_model"]
        k = int(row["K"])
        attack_tag = ATTACK_TAG[attack]
        budget_tag = BUDGET_TAG[budget]
        victim_tag = MODEL_TAG[victim]
        selector_tag = MODEL_TAG[selector]
        short_name = (
            f"xb{index:03d}_{attack_tag}_{budget_tag}_"
            f"v{victim_tag}_s{selector_tag}_k{k}"
        )
        filename = (
            f"job_{index:03d}_{attack_tag}_{budget_tag}_"
            f"v{victim_tag}_s{selector_tag}_k{k}.sh"
        )
        body = f'''#!/bin/bash
#SBATCH --account=aip-boyuwang
#SBATCH --job-name={short_name}
#SBATCH --time={row["time_limit"]}
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=7G
#SBATCH --gres=gpu:l40s:1
#SBATCH --signal=B:USR1@300
#SBATCH --output=/home/mmoslem3/scratch/PoisonBase/sbatch/logs/{short_name}-%j.out

# Boyu replacement for pending Yiwei job {REPLACED_JOBS[index]} (xk{index:03d}).
export ROOT="${{ROOT:-/home/mmoslem3/scratch/PoisonBase}}"
export ENV_ACTIVATE="${{ENV_ACTIVATE:-/home/mmoslem3/ENV/bin/activate}}"
export DATA_ROOT="${{DATA_ROOT:-$ROOT/data}}"
export XFULL_INDEX={index}
export XFULL_ATTACK={attack}
export XFULL_BUDGET={budget}
export XFULL_VICTIM_MODEL={victim}
export XFULL_SELECTOR_MODEL={selector}
export XFULL_K={k}
export XFULL_TARGET_DEGREE={TARGET_DEGREES[(victim, attack)]}
export XFULL_RUN_NAME={row["run_name"]}
export ORIGINAL_COMMAND="attack={attack} budget={budget} A=V={victim} S={selector} K={k} base=ours jacobian=off craft_steps=250 targets=10 victims=6 replacement_for={REPLACED_JOBS[index]}"
source "$ROOT/sbatch/cross_arch_k_10x6_boyu_resubmit_20260915/_job_common.sh"
'''
        (JOB_DIR / filename).write_text(body)

        row["account"] = "aip-boyuwang"
        row["job_file"] = filename
        row["replaces_job_id"] = str(REPLACED_JOBS[index])
        output_rows.append(row)

    if len(output_rows) != 54:
        raise SystemExit(f"generated {len(output_rows)} rows, expected 54")
    fields = ["replaces_job_id"] + list(source_rows[0])
    with (JOB_DIR / "MANIFEST.tsv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(output_rows)
    (JOB_DIR / "CANCEL_JOB_IDS.txt").write_text(
        " ".join(str(REPLACED_JOBS[i]) for i in sorted(REPLACED_JOBS)) + "\n"
    )
    print(f"generated {len(output_rows)} Boyu replacement jobs in {JOB_DIR}")


if __name__ == "__main__":
    main()
