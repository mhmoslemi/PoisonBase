#!/usr/bin/env python3
"""Generate fresh reruns for the K-pattern and BASIS<RAND anomalies.

The K=20 value is the fixed reference.  For every ensemble-table condition in
which a smaller ensemble beats K=20, this generator reruns the strongest
offending K<20 setting (ties go to the larger K).  It also adds the one
non-overlapping ensemble cell below RAND and the four valid main-table
BASIS<RAND cells.  The erroneous VGG13/BP/0.002 RAND=41.7 comparison is not
included as a main-table anomaly.
"""

from __future__ import annotations

import re
from pathlib import Path


OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
ENSEMBLE = ROOT / "ensemble.tex"

MODEL = {"ConvNet": "ConvNetBN", "ResNet20": "ResNet20BN", "VGG13": "VGG13BN"}
SHORT = {"ConvNet": "conv", "ResNet20": "r20", "VGG13": "vgg"}
ATTACK = {"BP": "fc", "GM": "gradmatch", "SAPA": "sapa"}
ATTACK_TAG = {"BP": "bp", "GM": "gm", "SAPA": "sapa"}
KS = (1, 3, 10, 20, 30)


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
            if attack is None or rho is None:
                raise RuntimeError("failed to recover attack/rho while parsing ensemble.tex")
            records.append((attack, rho, selector, victim, values))
    if len(records) != 54:
        raise RuntimeError(f"expected 54 ensemble rows, parsed {len(records)}")
    return records


def k_issues() -> list[tuple[str, str, str, str, int, float, float]]:
    """One strongest offending K<20 rerun for each K20 monotonicity issue."""
    selected: list[tuple[str, str, str, str, int, float, float]] = []
    for attack, rho, selector, victim, values in parse_ensemble():
        best_k, best_value = max(zip(KS[:3], values[:3]), key=lambda item: (item[1], item[0]))
        k20_value = values[3]
        if best_value > k20_value:
            selected.append((attack, rho, selector, victim,
                             best_k, best_value, k20_value))
    selected.sort(key=lambda row: (
        ("BP", "GM", "SAPA").index(row[0]), float(row[1]),
        ("ConvNet", "ResNet20", "VGG13").index(row[3]),
        ("ConvNet", "ResNet20", "VGG13").index(row[2])))
    if len(selected) != 22:
        raise RuntimeError(f"expected 22 K<20>K20 conditions, found {len(selected)}")
    return selected


def target_degree(victim: str, attack: str) -> int:
    if victim == "ConvNet":
        return 50 if attack == "BP" else 70
    if victim == "ResNet20":
        return 10 if attack == "BP" else 14
    return 3 if attack == "BP" else 50


def account(index: int) -> str:
    # Preserve the requested 2:1 split: 18 boyuwang and 9 yiweilu for 27 jobs.
    return "aip-yiweilu" if index % 3 == 0 else "aip-boyuwang"


def walltime(victim: str) -> str:
    return "0-03:10:00" if victim == "ConvNet" else "0-03:50:00"


def cross_run_name(attack: str, rho: str, selector: str,
                   victim: str, k: int) -> str:
    name = (f"CIFAR10_{MODEL[victim]}_{ATTACK[attack]}_ours_dog-bird_"
            f"b{rho}_eps8_seed42_lam1_cosine")
    if selector != victim:
        name += f"_selarch{MODEL[selector]}"
    name += f"_K{k}"
    if attack == "SAPA":
        name += "_worst0.05"
    return name + f"_ce5_tgt{target_degree(victim, attack)}"


def write_cross_job(index: int, *, category: str, attack: str, rho: str,
                    selector: str, victim: str, k: int,
                    note: str) -> tuple[str, str]:
    budget_tag = rho.replace("0.", "b0")
    label = (f"k{k}_{ATTACK_TAG[attack]}_{budget_tag}_"
             f"v{SHORT[victim]}_s{SHORT[selector]}")
    job_name = f"bkr{index:03d}_{label}"
    filename = f"job_{index:03d}_{label}.sh"
    original = (f"category={category} attack={ATTACK[attack]} budget={rho} "
                f"V={MODEL[victim]} S={MODEL[selector]} K={k} base=ours "
                "base_dist=cosine lambda_margin=1 jacobian=off craft_steps=250 "
                f"targets=10 victims=6; {note}")
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
#SBATCH --output=/home/mmoslem3/scratch/PoisonBase/sbatch/logs/{job_name}-%j.out

# One submission = one attack/budget/victim/selector/K configuration.
export ROOT="${{ROOT:-/home/mmoslem3/scratch/PoisonBase}}"
export ENV_ACTIVATE="${{ENV_ACTIVATE:-/home/mmoslem3/ENV/bin/activate}}"
export DATA_ROOT="${{DATA_ROOT:-$ROOT/data}}"
export RESULT_ROOT="${{RESULT_ROOT:-$ROOT/basis_k_rand_rerun_20260919_result}}"
export XFULL_INDEX={index}
export XFULL_ATTACK={ATTACK[attack]}
export XFULL_BUDGET={rho}
export XFULL_VICTIM_MODEL={MODEL[victim]}
export XFULL_SELECTOR_MODEL={MODEL[selector]}
export XFULL_K={k}
export XFULL_TARGET_DEGREE={target_degree(victim, attack)}
export XFULL_NUM_TARGETS=10
export XFULL_NUM_VICTIMS=6
export XFULL_RUN_NAME={cross_run_name(attack, rho, selector, victim, k)}
export ORIGINAL_COMMAND="{original}"

source "$ROOT/sbatch/cross_arch_k_10x6_20260915/_job_common.sh"
"""
    path = OUT / filename
    path.write_text(script)
    path.chmod(0o755)
    return filename, original


def write_main_job(index: int, *, class_pair: str, rho: str,
                   num_targets: int, rand_asr: float,
                   basis_asr: float) -> tuple[str, str]:
    victim = "ResNet20"
    attack = "BP"
    pair_tag = "db" if class_pair == "dog-bird" else "fa"
    budget_tag = rho.replace("0.", "b0")
    label = f"main_{SHORT[victim]}_bp_{pair_tag}_{budget_tag}"
    job_name = f"bkr{index:03d}_{label}"
    filename = f"job_{index:03d}_{label}.sh"
    degree = 10 if class_pair == "dog-bird" else 2
    original = (f"category=main_below_rand attack=fc budget={rho} "
                f"V=S=ResNet20BN pair={class_pair} K=20 base=ours "
                "base_dist=cosine lambda_margin=1 jacobian=off craft_steps=250 "
                f"targets={num_targets} victims=6; table RAND={rand_asr:g} "
                f"BASIS={basis_asr:g}")
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
#SBATCH --output=/home/mmoslem3/scratch/PoisonBase/sbatch/logs/{job_name}-%j.out

# One submission = one main-table configuration.
export ROOT="${{ROOT:-/home/mmoslem3/scratch/PoisonBase}}"
export ENV_ACTIVATE="${{ENV_ACTIVATE:-/home/mmoslem3/ENV/bin/activate}}"
export DATA_ROOT="${{DATA_ROOT:-$ROOT/data}}"
export RESULT_ROOT="${{RESULT_ROOT:-$ROOT/basis_k_rand_rerun_20260919_result}}"
export MAIN_INDEX={index}
export MAIN_MODEL=ResNet20BN
export MAIN_ATTACK=fc
export MAIN_CLASS_PAIR={class_pair}
export MAIN_BUDGET={rho}
export MAIN_TARGET_DEGREE={degree}
export MAIN_NUM_TARGETS={num_targets}
export MAIN_NUM_VICTIMS=6
export ORIGINAL_COMMAND="{original}"

source "$ROOT/sbatch/basis_k_rand_reruns_20260919/_main_job_common.sh"
"""
    path = OUT / filename
    path.write_text(script)
    path.chmod(0o755)
    return filename, original


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for old in OUT.glob("job_*.sh"):
        old.unlink()

    manifest = ["index\tcategory\tjob_file\tconfiguration"]
    index = 1
    for attack, rho, selector, victim, k, smaller_value, k20_value in k_issues():
        note = f"table K{k}={smaller_value:g} exceeds fixed K20={k20_value:g}"
        filename, description = write_cross_job(
            index, category="k_pattern", attack=attack, rho=rho,
            selector=selector, victim=victim, k=k, note=note)
        manifest.append(f"{index}\tk_pattern\t{filename}\t{description}")
        index += 1

    # The only additional ensemble BASIS<RAND cell not already represented by
    # the 22 K-pattern jobs: BP/.002, V=ResNet20, S=VGG13, K=3 (13.3 < 16.7).
    filename, description = write_cross_job(
        index, category="ensemble_below_rand", attack="BP", rho="0.002",
        selector="VGG13", victim="ResNet20", k=3,
        note="table BASIS=13.3 below RAND=16.7")
    manifest.append(f"{index}\tensemble_below_rand\t{filename}\t{description}")
    index += 1

    # Main-table anomalies after removing the erroneous VGG13/BP/.002 RAND=41.7
    # comparison.  The paper's stated low-budget protocol uses eight targets at
    # rho=.0005; the other entries are completed to ten targets.  All use 6 victims.
    main_anomalies = [
        ("dog-bird", "0.02", 10, 55.0, 33.3),
        ("frog-airplane", "0.0005", 8, 5.0, 3.3),
        ("frog-airplane", "0.001", 10, 8.3, 3.3),
        ("frog-airplane", "0.002", 10, 8.3, 6.7),
    ]
    for class_pair, rho, num_targets, rand_asr, basis_asr in main_anomalies:
        filename, description = write_main_job(
            index, class_pair=class_pair, rho=rho, num_targets=num_targets,
            rand_asr=rand_asr, basis_asr=basis_asr)
        manifest.append(f"{index}\tmain_below_rand\t{filename}\t{description}")
        index += 1

    if index - 1 != 27:
        raise RuntimeError(f"expected 27 jobs, generated {index - 1}")
    (OUT / "manifest.tsv").write_text("\n".join(manifest) + "\n")
    print("generated 27 jobs: 22 K-pattern + 1 ensemble<RAND + 4 main<RAND")


if __name__ == "__main__":
    main()
