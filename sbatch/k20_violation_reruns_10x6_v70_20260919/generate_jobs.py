#!/usr/bin/env python3
"""Generate only the full-protocol K=20 reruns for current table violations."""

from __future__ import annotations

import re
from pathlib import Path


OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
ENSEMBLE = ROOT / "ensemble.tex"

MODELS = {
    "ConvNet": "ConvNetBN",
    "ResNet20": "ResNet20BN",
    "VGG13": "VGG13BN",
}
SHORT = {"ConvNet": "conv", "ResNet20": "r20", "VGG13": "vgg"}
ATTACKS = {"BP": "fc", "GM": "gradmatch", "SAPA": "sapa"}
ATTACK_TAG = {"BP": "bp", "GM": "gm", "SAPA": "sapa"}

# Guard against accidentally regenerating the old lower-K batch.  These are the
# current ensemble.tex cells where K=20 is below max(K=1, K=3, K=10).
EXPECTED_CONFIGS = [
    ("BP", "0.002", "ConvNet", "ResNet20"),
    ("BP", "0.002", "ConvNet", "VGG13"),
    ("BP", "0.002", "ResNet20", "ResNet20"),
    ("BP", "0.002", "VGG13", "VGG13"),
    ("BP", "0.005", "VGG13", "ConvNet"),
    ("BP", "0.005", "VGG13", "ResNet20"),
    ("BP", "0.005", "VGG13", "VGG13"),
    ("GM", "0.002", "ConvNet", "ResNet20"),
    ("GM", "0.002", "ResNet20", "VGG13"),
    ("GM", "0.002", "VGG13", "ResNet20"),
    ("GM", "0.005", "ResNet20", "ConvNet"),
    ("GM", "0.005", "VGG13", "ConvNet"),
    ("SAPA", "0.002", "ConvNet", "ConvNet"),
    ("SAPA", "0.002", "ResNet20", "VGG13"),
    ("SAPA", "0.005", "ConvNet", "ResNet20"),
    ("SAPA", "0.005", "ConvNet", "VGG13"),
    ("SAPA", "0.005", "ResNet20", "ConvNet"),
    ("SAPA", "0.005", "VGG13", "ConvNet"),
]


def parse_ensemble() -> list[tuple[str, str, str, str, list[float]]]:
    """Return attack, rho, selector, victim and [K1,K3,K10,K20,K30]."""
    chunks: list[str] = []
    buffer: list[str] = []
    inside = False
    for line in ENSEMBLE.read_text().splitlines():
        if r"\midrule" in line:
            inside = True
            continue
        if not inside:
            continue
        if r"\bottomrule" in line:
            break
        if not line.strip() or line.lstrip().startswith(r"\cmidrule"):
            continue
        buffer.append(line.strip())
        if line.rstrip().endswith(r"\\"):
            chunks.append(" ".join(buffer))
            buffer = []

    attack = rho = None
    number = re.compile(r"(?<![A-Za-z])(?:100|\d+\.\d+)")
    records: list[tuple[str, str, str, str, list[float]]] = []
    for line in chunks:
        if not any(token in line for token in
                   ("& ConvNet", "& & ResNet20", "& & VGG13")):
            continue
        match = re.search(r"\\textbf\{(BP|GM|SAPA)\}", line)
        if match:
            attack = match.group(1)
        match = re.search(r"\\multirow\{3\}\{\*\}\{(0\.\d+)\}", line)
        if match:
            rho = match.group(1)
        fields = [field.strip() for field in line.split("&")]
        selector = re.sub(r"\\.*", "", fields[2]).strip()
        for victim, start in zip(("ConvNet", "ResNet20", "VGG13"),
                                 (3, 9, 15)):
            values = [float(number.findall(fields[index])[-1])
                      for index in range(start + 1, start + 6)]
            if attack is None or rho is None:
                raise RuntimeError("failed to recover attack/rho from ensemble.tex")
            records.append((attack, rho, selector, victim, values))
    if len(records) != 54:
        raise RuntimeError(f"expected 54 ensemble conditions, parsed {len(records)}")
    return records


def violations() -> list[tuple[str, str, str, str, list[float], float]]:
    rows = []
    for attack, rho, selector, victim, values in parse_ensemble():
        if max(values[:3]) > values[3]:
            rows.append((attack, rho, selector, victim, values[:3], values[3]))
    found = [row[:4] for row in rows]
    if found != EXPECTED_CONFIGS:
        raise RuntimeError(
            "ensemble.tex violation set changed; refusing to generate jobs\n"
            f"expected={EXPECTED_CONFIGS!r}\nfound={found!r}"
        )
    return rows


def target_degree(victim: str, attack: str) -> int:
    if victim == "ConvNet":
        return 50 if attack == "BP" else 70
    if victim == "ResNet20":
        return 10 if attack == "BP" else 14
    return 3 if attack == "BP" else 50


def account(index: int) -> str:
    # Requested historical split: two thirds boyuwang, one third yiweilu.
    return "aip-yiweilu" if index % 3 == 0 else "aip-boyuwang"


def walltime(victim: str, selector: str) -> str:
    # The longer 70-epoch victim schedule needs more headroom than the original
    # 50-epoch table jobs.  VGG selection/crafting also gets the larger limit.
    if victim == "ConvNet" and selector == "ConvNet":
        return "0-05:00:00"
    return "0-06:00:00"


def run_name(attack: str, rho: str, selector: str, victim: str) -> str:
    name = (f"CIFAR10_{MODELS[victim]}_{ATTACKS[attack]}_ours_dog-bird_"
            f"b{rho}_eps8_seed42_lam1_cosine_norm")
    if selector != victim:
        name += f"_selarch{MODELS[selector]}"
    name += "_K20"
    if attack == "SAPA":
        name += "_worst0.05"
    return name + f"_ce5_tgt{target_degree(victim, attack)}"


def write_job(index: int, row: tuple[str, str, str, str,
                                      list[float], float]) -> tuple[str, str]:
    attack, rho, selector, victim, lower_values, old_k20 = row
    budget_tag = rho.replace("0.", "b0")
    label = (f"{ATTACK_TAG[attack]}_{budget_tag}_"
             f"v{SHORT[victim]}_s{SHORT[selector]}_k20")
    job_name = f"k20v{index:02d}_{label}"
    filename = f"job_{index:02d}_{label}.sh"
    lower_text = ",".join(f"{value:g}" for value in lower_values)
    original = (
        f"attack={ATTACKS[attack]} budget={rho} V={MODELS[victim]} "
        f"S={MODELS[selector]} K=20 base=ours base_dist=cosine_norm "
        "lambda_margin=1 jacobian=off craft_steps=250 "
        "victim_epochs=70 victim_decay=50 targets=10 victims=6; "
        f"table lower-K=[{lower_text}] old-K20={old_k20:g}"
    )
    script = f"""#!/bin/bash
#SBATCH --account={account(index)}
#SBATCH --job-name={job_name}
#SBATCH --time={walltime(victim, selector)}
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=7G
#SBATCH --gres=gpu:l40s:1
#SBATCH --signal=B:USR1@300
#SBATCH --chdir=/home/mmoslem3/scratch/PoisonBase
#SBATCH --output=/home/mmoslem3/scratch/PoisonBase/sbatch/logs/{job_name}-%j.out

# Exactly one violating K=20 configuration; 10 targets x 6 victims.
export ROOT="${{ROOT:-/home/mmoslem3/scratch/PoisonBase}}"
export ENV_ACTIVATE="${{ENV_ACTIVATE:-/home/mmoslem3/ENV/bin/activate}}"
export DATA_ROOT="${{DATA_ROOT:-$ROOT/data}}"
export RESULT_ROOT="${{RESULT_ROOT:-$ROOT/k20_violation_10x6_v70_20260919_result}}"
export XFULL_INDEX={index}
export XFULL_ATTACK={ATTACKS[attack]}
export XFULL_BUDGET={rho}
export XFULL_VICTIM_MODEL={MODELS[victim]}
export XFULL_SELECTOR_MODEL={MODELS[selector]}
export XFULL_K=20
export XFULL_TARGET_DEGREE={target_degree(victim, attack)}
export XFULL_NUM_TARGETS=10
export XFULL_NUM_VICTIMS=6
export XFULL_BASE_DIST=cosine_norm
export XFULL_LAMBDA_MARGIN=1
export XFULL_VICTIM_EPOCHS=70
export XFULL_VICTIM_DECAY=50
export XFULL_RUN_NAME={run_name(attack, rho, selector, victim)}
export ORIGINAL_COMMAND="{original}"

source "$ROOT/sbatch/cross_arch_k_10x6_20260915/_job_common.sh"
"""
    path = OUT / filename
    path.write_text(script)
    path.chmod(0o755)
    return filename, original


def main() -> None:
    rows = violations()
    manifest = [
        "index\tjob_file\taccount\tattack\trho\tselector\tvictim\tK\t"
        "targets\tvictims\tlower_K_values\told_K20\tconfiguration"
    ]
    for index, row in enumerate(rows, 1):
        attack, rho, selector, victim, lower_values, old_k20 = row
        filename, description = write_job(index, row)
        manifest.append(
            f"{index}\t{filename}\t{account(index)}\t{attack}\t{rho}\t"
            f"{selector}\t{victim}\t20\t10\t6\t"
            f"{','.join(map(str, lower_values))}\t{old_k20}\t{description}"
        )
    if len(rows) != 18:
        raise RuntimeError(f"expected 18 K=20 jobs, generated {len(rows)}")
    (OUT / "manifest.tsv").write_text("\n".join(manifest) + "\n")
    print("generated 18 jobs; every job is K=20 with 10 targets x 6 victims")


if __name__ == "__main__":
    main()
