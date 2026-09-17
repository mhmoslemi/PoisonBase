#!/usr/bin/env python3
"""Generate Gao-selector and adapted-FUS experiment wrappers; never submit."""

from pathlib import Path


JOB_DIR = Path(__file__).resolve().parent
SELECTORS = (
    ("gao-loss", "loss"),
    ("gao-gradnorm", "grad"),
    ("gao-forgetting", "forget"),
    ("fus", "fus"),
)
ATTACKS = (("gradmatch", "gm"), ("sapa", "sapa"))
BUDGETS = (("0.002", "b0002"), ("0.005", "b0005"))
MODELS = (
    ("ConvNetBN", "conv", 70),
    ("ResNet20BN", "r20", 14),
)
K = 20


def run_name(model: str, selector: str, attack: str, budget: str, degree: int) -> str:
    name = (
        f"CIFAR10_{model}_{attack}_ours_dog-bird_b{budget}"
        f"_eps8_seed42_lam1_cosine_sel{selector}"
    )
    if selector.startswith("gao-"):
        name += "_ep10"
    elif selector == "fus":
        name += "_i10_a0.5_e50_ps0_pr0"
    name += f"_K{K}"
    if attack == "sapa":
        name += "_worst0.05"
    name += f"_ce5_tgt{degree}"
    return name


def main() -> None:
    for old in JOB_DIR.glob("job_*.sh"):
        old.unlink()

    manifest = [
        "index\taccount\tselector\tattack\tbudget\tmodel\tK\ttargets\t"
        "victims\tcraft_steps\tfus_iters\tfus_alpha\tfus_search_epochs\t"
        "time_limit\trun_name\tjob_file"
    ]
    index = 0
    for selector, selector_tag in SELECTORS:
        for attack, attack_tag in ATTACKS:
            for budget, budget_tag in BUDGETS:
                for model, model_tag, degree in MODELS:
                    index += 1
                    account = (
                        "aip-boyuwang" if (index - 1) % 3 in (0, 1)
                        else "aip-yiweilu"
                    )
                    # FUS performs ten additional craft/train/update rounds for
                    # every target; giving it the normal 2:30 limit would create
                    # jobs that predictably time out before victim evaluation.
                    time_limit = "0-12:00:00" if selector == "fus" else "0-02:30:00"
                    short_name = (
                        f"ifb{index:02d}_{selector_tag}_{attack_tag}_"
                        f"{budget_tag}_{model_tag}"
                    )
                    filename = (
                        f"job_{index:02d}_{selector_tag}_{attack_tag}_"
                        f"{budget_tag}_{model_tag}.sh"
                    )
                    name = run_name(model, selector, attack, budget, degree)
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

# One file = one selector/attack/budget/model configuration.
export ROOT="${{ROOT:-/home/mmoslem3/scratch/PoisonBase}}"
export ENV_ACTIVATE="${{ENV_ACTIVATE:-/home/mmoslem3/ENV/bin/activate}}"
export DATA_ROOT="${{DATA_ROOT:-$ROOT/data}}"
export IFB_INDEX={index}
export IFB_SELECTOR={selector}
export IFB_ATTACK={attack}
export IFB_BUDGET={budget}
export IFB_MODEL={model}
export IFB_K={K}
export IFB_TARGET_DEGREE={degree}
export IFB_RUN_NAME={name}
export ORIGINAL_COMMAND="selector={selector} attack={attack} budget={budget} model={model} K={K} dog-bird targets=10 victims=6 craft_steps=250"
source "$ROOT/sbatch/influence_fus_baselines_20260917/_job_common.sh"
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
                                    selector,
                                    attack,
                                    budget,
                                    model,
                                    K,
                                    10,
                                    6,
                                    250,
                                    10 if selector == "fus" else 0,
                                    0.5 if selector == "fus" else "",
                                    50 if selector == "fus" else "",
                                    time_limit,
                                    name,
                                    filename,
                                ),
                            )
                        )
                    )

    if index != 32:
        raise SystemExit(f"generated {index} jobs; expected 32")
    (JOB_DIR / "MANIFEST.tsv").write_text("\n".join(manifest) + "\n")
    print(f"generated {index} jobs in {JOB_DIR}")


if __name__ == "__main__":
    main()
