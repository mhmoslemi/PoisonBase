#!/usr/bin/env python3
"""Generate 8-target x 6-victim reruns for the anomalous table cells."""

from __future__ import annotations

import re
from collections import defaultdict
from pathlib import Path


OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
ENSEMBLE = ROOT / "ensemble.tex"

MODEL = {"ConvNet": "ConvNetBN", "ResNet20": "ResNet20BN", "VGG13": "VGG13BN"}
SHORT = {"ConvNet": "conv", "ResNet20": "r20", "VGG13": "vgg"}
ATTACK = {"BP": "fc", "GM": "gradmatch", "SAPA": "sapa"}
ATTACK_TAG = {"BP": "bp", "GM": "gm", "SAPA": "sapa"}


def parse_ensemble() -> list[tuple[str, str, str, str, list[float]]]:
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
    records: list[tuple[str, str, str, str, list[float]]] = []
    number = re.compile(r"(?<![A-Za-z])(?:100|\d+\.\d+)")
    for line in chunks:
        if not any(token in line for token in ("& ConvNet", "& & ResNet20", "& & VGG13")):
            continue
        match = re.search(r"\\textbf\{(BP|GM|SAPA)\}", line)
        if match:
            attack = match.group(1)
        match = re.search(r"\\multirow\{3\}\{\*\}\{(0\.\d+)\}", line)
        if match:
            rho = match.group(1)
        fields = [field.strip() for field in line.split("&")]
        selector = re.sub(r"\\.*", "", fields[2]).strip()
        for victim, start in zip(("ConvNet", "ResNet20", "VGG13"), (3, 9, 15)):
            values = [float(number.findall(fields[index])[-1])
                      for index in range(start + 1, start + 6)]
            assert attack is not None and rho is not None
            records.append((attack, rho, selector, victim, values))
    if len(records) != 54:
        raise RuntimeError(f"expected 54 ensemble rows, parsed {len(records)}")
    return records


def cross_reruns() -> list[tuple[str, str, str, str, int]]:
    """Choose the later non-K20 cell at each downward adjacent step."""
    selected: set[tuple[str, str, str, str, int]] = set()
    for attack, rho, selector, victim, values in parse_ensemble():
        if values[0] > values[1]:
            selected.add((attack, rho, selector, victim, 3))
        if values[1] > values[2] or values[2] > values[3]:
            selected.add((attack, rho, selector, victim, 10))
        if values[3] > values[4]:
            selected.add((attack, rho, selector, victim, 30))
    if len(selected) != 67:
        raise RuntimeError(f"expected 67 unique K reruns, found {len(selected)}")
    return sorted(selected, key=lambda row: (
        ("BP", "GM", "SAPA").index(row[0]), float(row[1]),
        ("ConvNet", "ResNet20", "VGG13").index(row[3]),
        ("ConvNet", "ResNet20", "VGG13").index(row[2]), row[4]))


def target_degree(victim: str, attack: str) -> int:
    if victim == "ConvNet":
        return 50 if attack == "BP" else 70
    if victim == "ResNet20":
        return 10 if attack == "BP" else 14
    return 3 if attack == "BP" else 50


def run_name(attack: str, rho: str, selector: str, victim: str, k: int,
             component: str = "") -> str:
    name = (f"CIFAR10_{MODEL[victim]}_{ATTACK[attack]}_ours_dog-bird_"
            f"b{rho}_eps8_seed42_lam1_cosine")
    if component:
        name += "_selMinusM"
    if selector != victim:
        name += f"_selarch{MODEL[selector]}"
    name += f"_K{k}"
    if attack == "SAPA":
        name += "_worst0.05"
    return name + f"_ce5_tgt{target_degree(victim, attack)}"


def account(index: int) -> str:
    return "aip-yiweilu" if index % 3 == 0 else "aip-boyuwang"


def write_job(index: int, *, category: str, attack: str, rho: str,
              selector: str, victim: str, k: int, component: str = "") -> tuple[str, str]:
    budget_tag = rho.replace("0.", "b0")
    if category == "component":
        label = f"m_{SHORT[victim]}_{ATTACK_TAG[attack]}_{budget_tag}"
    else:
        label = (f"k{k}_{ATTACK_TAG[attack]}_{budget_tag}_"
                 f"v{SHORT[victim]}_s{SHORT[selector]}")
    job_name = f"pr{index:03d}_{label}"
    filename = f"job_{index:03d}_{label}.sh"
    component_export = f"export XFULL_COMPONENT={component}\n" if component else ""
    original = (f"category={category} attack={ATTACK[attack]} budget={rho} "
                f"V={MODEL[victim]} S={MODEL[selector]} K={k} component={component or 'basis'} "
                "base=ours jacobian=off craft_steps=250 targets=8 victims=6")
    script = f"""#!/bin/bash
#SBATCH --account={account(index)}
#SBATCH --job-name={job_name}
#SBATCH --time=0-02:50:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=7G
#SBATCH --gres=gpu:l40s:1
#SBATCH --signal=B:USR1@300
#SBATCH --output=/home/mmoslem3/scratch/PoisonBase/sbatch/logs/{job_name}-%j.out

export ROOT="${{ROOT:-/home/mmoslem3/scratch/PoisonBase}}"
export ENV_ACTIVATE="${{ENV_ACTIVATE:-/home/mmoslem3/ENV/bin/activate}}"
export DATA_ROOT="${{DATA_ROOT:-$ROOT/data}}"
export RESULT_ROOT="${{RESULT_ROOT:-$ROOT/pattern_rerun_8x6_result}}"
export XFULL_INDEX={index}
export XFULL_ATTACK={ATTACK[attack]}
export XFULL_BUDGET={rho}
export XFULL_VICTIM_MODEL={MODEL[victim]}
export XFULL_SELECTOR_MODEL={MODEL[selector]}
export XFULL_K={k}
export XFULL_TARGET_DEGREE={target_degree(victim, attack)}
export XFULL_NUM_TARGETS=8
export XFULL_NUM_VICTIMS=6
{component_export}export XFULL_RUN_NAME={run_name(attack, rho, selector, victim, k, component)}
export ORIGINAL_COMMAND="{original}"

source "$ROOT/sbatch/cross_arch_k_10x6_20260915/_job_common.sh"
"""
    path = OUT / filename
    path.write_text(script)
    path.chmod(0o755)
    return filename, original


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for old in OUT.glob("job_*.sh"):
        old.unlink()

    # Five cells where margin-only M exceeded BASIS in bench.tex.
    component_jobs = [
        ("GM", "0.002", "ConvNet", "ConvNet", 20),
        ("SAPA", "0.002", "ConvNet", "ConvNet", 20),
        ("SAPA", "0.005", "ConvNet", "ConvNet", 20),
        ("GM", "0.005", "ResNet20", "ResNet20", 20),
        ("SAPA", "0.002", "ResNet20", "ResNet20", 20),
    ]

    manifest = ["index\tcategory\tjob_file\tconfiguration"]
    index = 1
    for attack, rho, selector, victim, k in component_jobs:
        filename, description = write_job(
            index, category="component", attack=attack, rho=rho,
            selector=selector, victim=victim, k=k, component="minus-m")
        manifest.append(f"{index}\tcomponent\t{filename}\t{description}")
        index += 1
    for attack, rho, selector, victim, k in cross_reruns():
        filename, description = write_job(
            index, category="ensemble", attack=attack, rho=rho,
            selector=selector, victim=victim, k=k)
        manifest.append(f"{index}\tensemble\t{filename}\t{description}")
        index += 1

    if index - 1 != 72:
        raise RuntimeError(f"expected 72 jobs, generated {index - 1}")
    (OUT / "manifest.tsv").write_text("\n".join(manifest) + "\n")
    print("generated 72 jobs: 5 M-ablation reruns + 67 ensemble-K reruns")


if __name__ == "__main__":
    main()
