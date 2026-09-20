#!/usr/bin/env python3
"""Generate the requested second-round ensemble reruns."""

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

# attack, rho, selector architecture S, victim architecture V, selector K.
# The order follows the user's numbered list, expanding K from left to right.
CONFIGS = [
    *(('SAPA', '0.005', 'ResNet20', 'ConvNet', k) for k in (10, 20, 30)),
    *(('SAPA', '0.002', 'ConvNet', 'ConvNet', k) for k in (1, 3, 10)),
    *(('SAPA', '0.005', 'ConvNet', 'ResNet20', k) for k in (3, 10)),
    *(('SAPA', '0.002', 'ResNet20', 'VGG13', k) for k in (10, 20)),
    *(('SAPA', '0.005', 'VGG13', 'ResNet20', k) for k in (20, 30)),
    *(('BP', '0.002', 'ResNet20', 'ConvNet', k) for k in (20, 30)),
    *(('BP', '0.002', 'ResNet20', 'ResNet20', k) for k in (10, 20)),
    *(('BP', '0.005', 'ConvNet', 'ResNet20', k) for k in (3, 10, 20)),
    *(('BP', '0.005', 'ResNet20', 'VGG13', k) for k in (20, 30)),
    *(('BP', '0.005', 'VGG13', 'ResNet20', k) for k in (10, 20, 30)),
    *(('BP', '0.002', 'ConvNet', 'VGG13', k) for k in (1, 10, 30)),
    *(('GM', '0.005', 'VGG13', 'ResNet20', k) for k in (10, 20, 30)),
    *(('GM', '0.005', 'VGG13', 'ConvNet', k) for k in (20, 30)),
    ('GM', '0.005', 'ConvNet', 'ConvNet', 30),
    ('GM', '0.005', 'ResNet20', 'ConvNet', 20),
]


def target_degree(victim: str, attack: str) -> int:
    if victim == "ConvNet":
        return 50 if attack == "BP" else 70
    if victim == "ResNet20":
        return 10 if attack == "BP" else 14
    return 3 if attack == "BP" else 50


def account(index: int) -> str:
    # Keep the prior approximately 2:1 Boyu/Yiweilu allocation.
    return "aip-yiweilu" if index % 3 == 0 else "aip-boyuwang"


def run_name(attack: str, rho: str, selector: str,
             victim: str, k: int) -> str:
    name = (
        f"CIFAR10_{MODELS[victim]}_{ATTACKS[attack]}_ours_dog-bird_"
        f"b{rho}_eps8_seed42_lam10_cosine_norm"
    )
    if selector != victim:
        name += f"_selarch{MODELS[selector]}"
    name += f"_K{k}"
    if attack == "SAPA":
        name += "_worst0.05"
    return name + f"_ce5_tgt{target_degree(victim, attack)}"


def write_job(index: int, row: tuple[str, str, str, str, int]) -> tuple[str, str]:
    attack, rho, selector, victim, k = row
    budget_tag = rho.replace("0.", "b0")
    label = (
        f"{ATTACK_TAG[attack]}_{budget_tag}_v{SHORT[victim]}_"
        f"s{SHORT[selector]}_k{k}"
    )
    job_name = f"enr2{index:02d}_{label}"
    filename = f"job_{index:02d}_{label}.sh"
    description = (
        f"attack={ATTACKS[attack]} budget={rho} V={MODELS[victim]} "
        f"S={MODELS[selector]} K={k} base=ours base_dist=cosine_norm "
        "lambda_margin=10 jacobian=off craft_steps=250 "
        "victim_epochs=50 victim_decay=40 targets=8 victims=6"
    )
    script = f"""#!/bin/bash
#SBATCH --account={account(index)}
#SBATCH --job-name={job_name}
#SBATCH --time=0-03:10:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=7G
#SBATCH --gres=gpu:l40s:1
#SBATCH --signal=B:USR1@300
#SBATCH --chdir=/home/mmoslem3/scratch/PoisonBase
#SBATCH --output=/home/mmoslem3/scratch/PoisonBase/sbatch/logs/{job_name}-%j.out

# One requested ensemble rerun; 8 targets x 6 victims.
export ROOT="${{ROOT:-/home/mmoslem3/scratch/PoisonBase}}"
export ENV_ACTIVATE="${{ENV_ACTIVATE:-/home/mmoslem3/ENV/bin/activate}}"
export DATA_ROOT="${{DATA_ROOT:-$ROOT/data}}"
export RESULT_ROOT="$ROOT/ensemble_rerun_round2_8x6_lam10_cosnorm_20260920_result"
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
    if len(CONFIGS) != 34:
        raise RuntimeError(f"expected 34 configurations, found {len(CONFIGS)}")
    if len(set(CONFIGS)) != len(CONFIGS):
        raise RuntimeError("duplicate requested configuration")

    manifest = [
        "index\tjob_file\taccount\tattack\trho\tselector\tvictim\tK\t"
        "targets\tvictims\tconfiguration"
    ]
    for index, row in enumerate(CONFIGS, 1):
        attack, rho, selector, victim, k = row
        filename, description = write_job(index, row)
        manifest.append(
            f"{index}\t{filename}\t{account(index)}\t{attack}\t{rho}\t"
            f"{selector}\t{victim}\t{k}\t8\t6\t{description}"
        )
    (OUT / "manifest.tsv").write_text("\n".join(manifest) + "\n")
    print("generated 34 fresh ensemble rerun jobs; each is 8 targets x 6 victims")


if __name__ == "__main__":
    main()
