#!/usr/bin/env python3
"""Generate the 55 green-highlighted reruns from the supplied table image."""

from __future__ import annotations

from pathlib import Path


OUT = Path(__file__).resolve().parent

MODELS = {
    "ConvNet": "ConvNetBN",
    "ResNet20": "ResNet20BN",
    "VGG13": "VGG13BN",
}
SHORT = {"ConvNet": "conv", "ResNet20": "r20", "VGG13": "vgg"}
ATTACKS = {"BP": "fc", "GM": "gradmatch", "SAPA": "sapa"}
ATTACK_TAG = {"BP": "bp", "GM": "gm", "SAPA": "sapa"}

# attack, rho, selector S, victim V, K, displayed ASR.
# This list follows the green markings in the screenshot row by row.  The
# partially highlighted BP/.005/S=VGG13/V=VGG13/K=3 cell is included.
GREEN = [
    # BP, rho=.002
    ("BP", "0.002", "ConvNet", "ResNet20", 3, 40.0),
    ("BP", "0.002", "ConvNet", "ResNet20", 20, 35.0),
    ("BP", "0.002", "ConvNet", "ResNet20", 30, 31.7),
    ("BP", "0.002", "ConvNet", "VGG13", 1, 70.0),
    ("BP", "0.002", "ConvNet", "VGG13", 3, 58.3),
    ("BP", "0.002", "ConvNet", "VGG13", 10, 78.3),
    ("BP", "0.002", "ConvNet", "VGG13", 20, 70.0),
    ("BP", "0.002", "ConvNet", "VGG13", 30, 71.7),
    ("BP", "0.002", "ResNet20", "ConvNet", 20, 45.0),
    ("BP", "0.002", "ResNet20", "ResNet20", 3, 25.0),
    ("BP", "0.002", "ResNet20", "ResNet20", 20, 25.0),
    ("BP", "0.002", "ResNet20", "ResNet20", 30, 31.7),
    ("BP", "0.002", "VGG13", "ResNet20", 1, 20.0),
    ("BP", "0.002", "VGG13", "ResNet20", 10, 25.0),
    ("BP", "0.002", "VGG13", "ResNet20", 20, 40.0),
    ("BP", "0.002", "VGG13", "ResNet20", 30, 23.3),
    ("BP", "0.002", "VGG13", "VGG13", 1, 46.7),

    # BP, rho=.005
    ("BP", "0.005", "ConvNet", "ResNet20", 10, 43.3),
    ("BP", "0.005", "ConvNet", "ResNet20", 20, 65.0),
    ("BP", "0.005", "ConvNet", "ResNet20", 30, 58.3),
    ("BP", "0.005", "ConvNet", "VGG13", 3, 85.0),
    ("BP", "0.005", "ResNet20", "ResNet20", 1, 45.0),
    ("BP", "0.005", "ResNet20", "ResNet20", 3, 36.7),
    ("BP", "0.005", "ResNet20", "VGG13", 20, 65.0),
    ("BP", "0.005", "ResNet20", "VGG13", 30, 48.3),
    ("BP", "0.005", "VGG13", "ConvNet", 1, 56.7),
    ("BP", "0.005", "VGG13", "ResNet20", 20, 30.0),
    ("BP", "0.005", "VGG13", "ResNet20", 30, 38.3),
    ("BP", "0.005", "VGG13", "VGG13", 1, 55.0),
    ("BP", "0.005", "VGG13", "VGG13", 3, 50.0),
    ("BP", "0.005", "VGG13", "VGG13", 10, 60.0),

    # GM, rho=.002
    ("GM", "0.002", "ConvNet", "ConvNet", 1, 23.3),
    ("GM", "0.002", "ResNet20", "ConvNet", 30, 10.0),
    ("GM", "0.002", "ResNet20", "ResNet20", 3, 55.0),
    ("GM", "0.002", "VGG13", "ResNet20", 20, 35.0),
    ("GM", "0.002", "VGG13", "ResNet20", 30, 43.3),

    # GM, rho=.005
    ("GM", "0.005", "ConvNet", "ConvNet", 30, 45.0),
    ("GM", "0.005", "ConvNet", "ResNet20", 20, 85.0),
    ("GM", "0.005", "ConvNet", "ResNet20", 30, 61.7),
    ("GM", "0.005", "ResNet20", "ConvNet", 20, 30.0),
    ("GM", "0.005", "ResNet20", "ConvNet", 30, 50.0),
    ("GM", "0.005", "VGG13", "ConvNet", 20, 25.0),
    ("GM", "0.005", "VGG13", "ConvNet", 30, 40.0),
    ("GM", "0.005", "VGG13", "ResNet20", 20, 95.0),

    # SAPA, rho=.002
    ("SAPA", "0.002", "ConvNet", "ConvNet", 1, 28.3),
    ("SAPA", "0.002", "ConvNet", "ResNet20", 1, 51.7),
    ("SAPA", "0.002", "ConvNet", "ResNet20", 3, 46.7),
    ("SAPA", "0.002", "ResNet20", "ConvNet", 20, 25.0),
    ("SAPA", "0.002", "ResNet20", "ConvNet", 30, 11.7),
    ("SAPA", "0.002", "ResNet20", "VGG13", 20, 75.0),
    ("SAPA", "0.002", "ResNet20", "VGG13", 30, 81.7),
    ("SAPA", "0.002", "VGG13", "ResNet20", 20, 65.0),
    ("SAPA", "0.002", "VGG13", "ResNet20", 30, 60.0),

    # SAPA, rho=.005
    ("SAPA", "0.005", "ConvNet", "ResNet20", 3, 80.0),
    ("SAPA", "0.005", "ConvNet", "ResNet20", 30, 66.7),
]


def target_degree(victim: str, attack: str) -> int:
    if victim == "ConvNet":
        return 50 if attack == "BP" else 70
    if victim == "ResNet20":
        return 10 if attack == "BP" else 14
    return 3 if attack == "BP" else 50


def account(index: int) -> str:
    # Same approximately 2:1 allocation used by the other generated batches.
    return "aip-yiweilu" if index % 3 == 0 else "aip-boyuwang"


def walltime(victim: str) -> str:
    return "0-03:10:00" if victim == "ConvNet" else "0-03:50:00"


def run_name(attack: str, rho: str, selector: str,
             victim: str, k: int) -> str:
    name = (f"CIFAR10_{MODELS[victim]}_{ATTACKS[attack]}_ours_dog-bird_"
            f"b{rho}_eps8_seed42_lam10_cosine_norm")
    if selector != victim:
        name += f"_selarch{MODELS[selector]}"
    name += f"_K{k}"
    if attack == "SAPA":
        name += "_worst0.05"
    return name + f"_ce5_tgt{target_degree(victim, attack)}"


def write_job(index: int, row: tuple[str, str, str, str, int, float]) -> tuple[str, str]:
    attack, rho, selector, victim, k, old_asr = row
    budget_tag = rho.replace("0.", "b0")
    label = (f"{ATTACK_TAG[attack]}_{budget_tag}_v{SHORT[victim]}_"
             f"s{SHORT[selector]}_k{k}")
    job_name = f"grn{index:02d}_{label}"
    filename = f"job_{index:02d}_{label}.sh"
    description = (
        f"attack={ATTACKS[attack]} budget={rho} V={MODELS[victim]} "
        f"S={MODELS[selector]} K={k} base=ours base_dist=cosine_norm "
        "lambda_margin=10 jacobian=off craft_steps=250 "
        "victim_epochs=50 victim_decay=40 targets=8 victims=6; "
        f"green-table-ASR={old_asr:g}"
    )
    script = f"""#!/bin/bash
#SBATCH --account={account(index)}
#SBATCH --job-name={job_name}
#SBATCH --time={walltime(victim)}
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=7G
#SBATCH --gres=gpu:l40s:1
#SBATCH --signal=B:USR1@300
#SBATCH --chdir=/home/mmoslem3/scratch/PoisonBase
#SBATCH --output=/home/mmoslem3/scratch/PoisonBase/sbatch/logs/{job_name}-%j.out

# One green-highlighted table cell; 8 targets x 6 victims.
export ROOT="${{ROOT:-/home/mmoslem3/scratch/PoisonBase}}"
export ENV_ACTIVATE="${{ENV_ACTIVATE:-/home/mmoslem3/ENV/bin/activate}}"
export DATA_ROOT="${{DATA_ROOT:-$ROOT/data}}"
export RESULT_ROOT="$ROOT/green_rerun_8x6_lam10_cosnorm_20260919_result"
export XFULL_INDEX={index}
export XFULL_ATTACK={ATTACKS[attack]}
export XFULL_BUDGET={rho}
export XFULL_VICTIM_MODEL={MODELS[victim]}
export XFULL_SELECTOR_MODEL={MODELS[selector]}
export XFULL_K={k}
export XFULL_TARGET_DEGREE={target_degree(victim, attack)}
export XFULL_NUM_TARGETS=8
export XFULL_NUM_VICTIMS=6
export XFULL_COMPONENT=
export XFULL_BASE_DIST=cosine_norm
export XFULL_LAMBDA_MARGIN=10
export XFULL_VICTIM_EPOCHS=50
export XFULL_VICTIM_DECAY=40
export XFULL_RUN_NAME={run_name(attack, rho, selector, victim, k)}
export ORIGINAL_COMMAND="{description}"

source "$ROOT/sbatch/cross_arch_k_10x6_20260915/_job_common.sh"
"""
    path = OUT / filename
    path.write_text(script)
    path.chmod(0o755)
    return filename, description


def main() -> None:
    if len(GREEN) != 55:
        raise RuntimeError(f"expected 55 green cells, found {len(GREEN)}")
    keys = [row[:5] for row in GREEN]
    if len(set(keys)) != len(keys):
        raise RuntimeError("duplicate green configuration")

    manifest = [
        "index\tjob_file\taccount\tattack\trho\tselector\tvictim\tK\t"
        "targets\tvictims\told_ASR\tconfiguration"
    ]
    for index, row in enumerate(GREEN, 1):
        attack, rho, selector, victim, k, old_asr = row
        filename, description = write_job(index, row)
        manifest.append(
            f"{index}\t{filename}\t{account(index)}\t{attack}\t{rho}\t"
            f"{selector}\t{victim}\t{k}\t8\t6\t{old_asr}\t{description}"
        )
    (OUT / "manifest.tsv").write_text("\n".join(manifest) + "\n")
    print("generated 55 green-cell jobs; every job is 8 targets x 6 victims")


if __name__ == "__main__":
    main()
