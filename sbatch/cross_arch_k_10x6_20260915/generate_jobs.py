#!/usr/bin/env python3
"""Generate the 216 one-cell Slurm jobs for the 10x6 K ablation.

This only writes shell scripts and a manifest. It never calls sbatch.
"""

from pathlib import Path


JOB_DIR = Path(__file__).resolve().parent
ATTACKS = (("fc", "bp"), ("gradmatch", "gm"), ("sapa", "sapa"))
BUDGETS = (("0.002", "b0002"), ("0.005", "b0005"))
MODELS = (
    ("ConvNetBN", "conv"),
    ("ResNet20BN", "r20"),
    ("VGG13BN", "vgg"),
)
KS = (1, 3, 10, 30)

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


def run_name(victim: str, selector: str, attack: str, budget: str, k: int) -> str:
    name = (
        f"CIFAR10_{victim}_{attack}_ours_dog-bird_b{budget}"
        "_eps8_seed42_lam1_cosine"
    )
    if selector != victim:
        name += f"_selarch{selector}"
    name += f"_K{k}"
    if attack == "sapa":
        name += "_worst0.05"
    name += f"_ce5_tgt{TARGET_DEGREES[(victim, attack)]}"
    return name


def main() -> None:
    for old in JOB_DIR.glob("job_*.sh"):
        old.unlink()

    manifest = [
        "index\taccount\tattack\tbudget\tvictim_model\tselector_model\tK\t"
        "targets\tvictims\tcraft_ensemble\tnum_surrogates\tcraft_steps\t"
        "time_limit\trun_name\tjob_file"
    ]
    index = 0
    for attack, attack_tag in ATTACKS:
        for budget, budget_tag in BUDGETS:
            for victim, victim_tag in MODELS:
                for selector, selector_tag in MODELS:
                    for k in KS:
                        index += 1
                        # Balanced 2:1 throughout the ordered grid, not one account
                        # getting all of one attack/model block.
                        account = (
                            "aip-boyuwang" if (index - 1) % 3 in (0, 1)
                            else "aip-yiweilu"
                        )
                        time_limit = (
                            "0-03:10:00" if victim == "ConvNetBN" else "0-03:50:00"
                        )
                        short_name = (
                            f"xk{index:03d}_{attack_tag}_{budget_tag}_"
                            f"v{victim_tag}_s{selector_tag}_k{k}"
                        )
                        filename = f"job_{index:03d}_{attack_tag}_{budget_tag}_v{victim_tag}_s{selector_tag}_k{k}.sh"
                        name = run_name(victim, selector, attack, budget, k)
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

# One file = one attack/budget/victim/selector/K configuration.
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
export XFULL_RUN_NAME={name}
export ORIGINAL_COMMAND="attack={attack} budget={budget} A=V={victim} S={selector} K={k} base=ours jacobian=off craft_steps=250 targets=10 victims=6"
source "$ROOT/sbatch/cross_arch_k_10x6_20260915/_job_common.sh"
'''
                        (JOB_DIR / filename).write_text(body)
                        manifest.append(
                            "\t".join(
                                map(
                                    str,
                                    (
                                        index,
                                        account,
                                        attack,
                                        budget,
                                        victim,
                                        selector,
                                        k,
                                        10,
                                        6,
                                        5,
                                        30,
                                        250,
                                        time_limit,
                                        name,
                                        filename,
                                    ),
                                )
                            )
                        )

    if index != 216:
        raise SystemExit(f"internal error: generated {index} jobs, expected 216")
    (JOB_DIR / "MANIFEST.tsv").write_text("\n".join(manifest) + "\n")
    print(f"generated {index} jobs in {JOB_DIR}")


if __name__ == "__main__":
    main()
