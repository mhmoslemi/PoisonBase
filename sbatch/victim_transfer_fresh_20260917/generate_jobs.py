#!/usr/bin/env python3
"""Generate fresh S=A -> V Slurm jobs.

Each generated file contains exactly one attack/budget/K/source/victim
configuration.  This program only writes files; it never submits a job.
"""

from pathlib import Path


JOB_DIR = Path(__file__).resolve().parent
ATTACKS = (("gradmatch", "gm"), ("sapa", "sapa"))
BUDGETS = (("0.002", "b0002"), ("0.005", "b0005"))
KS = (10, 20)
MODELS = (
    ("ConvNetBN", "conv"),
    ("ResNet20BN", "r20"),
    ("VGG13BN", "vgg"),
)
TARGET_DEGREES = {
    "ConvNetBN": 70,
    "ResNet20BN": 14,
    "VGG13BN": 50,
}


def run_name(source: str, victim: str, attack: str, budget: str, k: int) -> str:
    """Mirror final_update.build_run_name for this experiment."""
    name = (
        f"CIFAR10_{source}_{attack}_ours_dog-bird_b{budget}"
        "_eps8_seed42_lam1_cosine"
    )
    if victim != source:
        name += f"_victimarch{victim}"
    name += f"_K{k}"
    if attack == "sapa":
        name += "_worst0.05"
    name += f"_ce5_tgt{TARGET_DEGREES[source]}"
    return name


def main() -> None:
    for old in JOB_DIR.glob("job_*.sh"):
        old.unlink()

    manifest = [
        "index\taccount\tattack\tbudget\tK\tselection_model\tcraft_model\t"
        "victim_model\ttargets\tvictims\tcraft_ensemble\tnum_surrogates\t"
        "craft_steps\ttime_limit\trun_name\tjob_file"
    ]

    index = 0
    for attack, attack_tag in ATTACKS:
        for budget, budget_tag in BUDGETS:
            for k in KS:
                for source, source_tag in MODELS:
                    for victim, victim_tag in MODELS:
                        index += 1
                        account = (
                            "aip-boyuwang"
                            if (index - 1) % 3 in (0, 1)
                            else "aip-yiweilu"
                        )
                        time_limit = "0-02:30:00"
                        short_name = (
                            f"vxf{index:03d}_{attack_tag}_{budget_tag}_k{k}_"
                            f"a{source_tag}_v{victim_tag}"
                        )
                        filename = (
                            f"job_{index:03d}_{attack_tag}_{budget_tag}_k{k}_"
                            f"a{source_tag}_v{victim_tag}.sh"
                        )
                        name = run_name(source, victim, attack, budget, k)
                        body = f'''#!/bin/bash
#SBATCH --account={account}
#SBATCH --job-name={short_name}
#SBATCH --time={time_limit}
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=7G
#SBATCH --gres=gpu:l40s:1
#SBATCH --signal=B:USR1@300
#SBATCH --output=/home/mmoslem3/scratch/PoisonBase/sbatch/logs/{short_name}-%j.out

# One file = one attack/budget/K/S=A/V configuration.
export ROOT="${{ROOT:-/home/mmoslem3/scratch/PoisonBase}}"
export ENV_ACTIVATE="${{ENV_ACTIVATE:-/home/mmoslem3/ENV/bin/activate}}"
export DATA_ROOT="${{DATA_ROOT:-$ROOT/data}}"
export VXF_INDEX={index}
export VXF_ATTACK={attack}
export VXF_BUDGET={budget}
export VXF_K={k}
export VXF_SOURCE_MODEL={source}
export VXF_VICTIM_MODEL={victim}
export VXF_TARGET_DEGREE={TARGET_DEGREES[source]}
export VXF_RUN_NAME={name}
export ORIGINAL_COMMAND="attack={attack} budget={budget} K={k} S=A={source} V={victim} base=ours jacobian=off craft_steps=250 targets=10 victims=6 FORCE=fresh"
source "$ROOT/sbatch/victim_transfer_fresh_20260917/_job_common.sh"
'''
                        path = JOB_DIR / filename
                        path.write_text(body)
                        path.chmod(0o755)
                        manifest.append(
                            "\t".join(
                                map(
                                    str,
                                    (
                                        index,
                                        account,
                                        attack,
                                        budget,
                                        k,
                                        source,
                                        source,
                                        victim,
                                        10,
                                        6,
                                        5,
                                        20,
                                        250,
                                        time_limit,
                                        name,
                                        filename,
                                    ),
                                )
                            )
                        )

    if index != 72:
        raise SystemExit(f"generated {index} jobs; expected 72")
    (JOB_DIR / "MANIFEST.tsv").write_text("\n".join(manifest) + "\n")
    print(f"generated {index} jobs in {JOB_DIR}")


if __name__ == "__main__":
    main()
