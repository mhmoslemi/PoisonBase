#!/usr/bin/env python3
"""Generate one SLURM job for each unfilled EPIC/FRIENDS cell in defense.tex."""

from __future__ import annotations

import json
import re
import shlex
from pathlib import Path

from generate_jobs import ROOT, SBATCH, make_script


OUT = SBATCH / "missing_defense"
METHODS = ("random", "ours", "dpp", "ours_j", "dpp_j")
ATTACK_NAMES = {"BP": "fc", "GM": "gradmatch", "SAPA": "sapa"}
MODEL_NAMES = {
    "ConvNet": "ConvNetBN",
    "ResNet20": "ResNet20BN",
    "VGG13": "VGG13BN",
}

# These two cells already have a separate dependency chain in sbatch/remaining:
# first create their shared random-poison cache, then retry both defenses.
RETRY_KEYS = {
    ("ConvNetBN", "fc", "dog-bird", "0.02", "epic", "random", "0"),
    ("ConvNetBN", "fc", "dog-bird", "0.02", "friends", "random", "0"),
}


def fmt_budget(value: int) -> str:
    return f"{value / 1000:g}"


def missing_cells():
    lines = (ROOT / "defense.tex").read_text().splitlines()
    difficulty = json.loads((ROOT / "sweep_config.json").read_text())["difficulty"]
    model = None
    for index, line in enumerate(lines):
        model_match = re.search(r"\\textbf\{(ConvNet|ResNet20|VGG13)\}", line)
        if model_match and re.search(r"&\s*(BP|GM|SAPA)\s*&", line):
            model = MODEL_NAMES[model_match.group(1)]
        row = re.search(
            r"&\s*(BP|GM|SAPA)\s*&\s*(dog--bird|frog--airplane)"
            r"\s*&\s*(\d+)\s*&",
            line,
        )
        if not row:
            continue
        attack = ATTACK_NAMES[row.group(1)]
        pair = row.group(2).replace("--", "-")
        budget = fmt_budget(int(row.group(3)))
        cells = re.findall(r"(?:\\scoreT\{[^}]+\}\{[^}]+\}|--)", lines[index + 1])
        if len(cells) != 15:
            raise ValueError(f"expected 15 cells on defense row after line {index + 1}")
        for defense_index, defense in enumerate(("epic", "friends"), 1):
            for method_index, method in enumerate(METHODS):
                if cells[defense_index * 5 + method_index] != "--":
                    continue
                jacobian = "1" if method.endswith("_j") else "0"
                selection = method.removesuffix("_j")
                key = (model, attack, pair, budget, defense, selection, jacobian)
                if key in RETRY_KEYS:
                    continue
                target_select = ""
                if attack == "sapa":
                    target_select = str(difficulty[model]["gradmatch"][pair])
                env = {
                    "USE_JACOBIAN_SCORE": jacobian,
                    "JACOBIAN_WEIGHT": "1.0",
                    "JACOBIAN_BATCH_SIZE": "64",
                    "CLASS_PAIR": pair,
                    "MODEL": model,
                    "ATTACK": attack,
                    "BUDGETS": budget,
                    "SELS": selection,
                    "SEL_ALPHA": "2.0",
                    "DEFENSES": defense,
                    "TARGET_SELECT": target_select,
                    "NUM_TARGETS": "7",
                    "NUM_VICTIMS": "5",
                }
                assignments = [
                    f"USE_JACOBIAN_SCORE={jacobian}",
                    'JACOBIAN_WEIGHT=1.0',
                    'JACOBIAN_BATCH_SIZE=64',
                    f"CLASS_PAIR={shlex.quote(pair)}",
                    f"MODEL={model}",
                    f"ATTACK={attack}",
                ]
                if target_select:
                    assignments.append(f"TARGET_SELECT={target_select}")
                assignments.extend((
                    f"BUDGETS={budget}",
                    f"SELS={selection}",
                    "SEL_ALPHA=2.0",
                    f"DEFENSES={defense}",
                    "NUM_TARGETS=7",
                    "NUM_VICTIMS=5",
                    "sh defense.sh",
                ))
                yield env, " ".join(assignments), key


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    for old in OUT.glob("*.sh"):
        old.unlink()
    specs = list(missing_cells())
    names = []
    manifest = ["index\tfile\tmodel\tattack\tpair\tbudget\tdefense\tselection\tjacobian\tcommand"]
    for index, (env, command, key) in enumerate(specs, 1):
        filename, text, *_ = make_script(
            "defense", index, command, command, env, name_prefix="missing_"
        )
        (OUT / filename).write_text(text)
        names.append(filename)
        model, attack, pair, budget, defense, selection, jacobian = key
        manifest.append("\t".join((
            str(index), filename, model, attack, pair, budget, defense,
            selection, jacobian, command,
        )))
    (OUT / "manifest.tsv").write_text("\n".join(manifest) + "\n")
    submit = [
        "#!/bin/sh",
        "# Submit defense cells that were absent from the original 62-job set.",
        "set -eu",
        'ROOT="/home/mmoslem3/scratch/attack_if"',
        'mkdir -p "$ROOT/sbatch/logs"',
    ]
    for name in names:
        submit.extend((f'sbatch "$ROOT/sbatch/missing_defense/{name}"', "sleep 1"))
    submit.append("")
    (OUT / "submit.sh").write_text("\n".join(submit))
    for path in OUT.glob("*.sh"):
        path.chmod(0o755)
    print(f"generated {len(names)} newly missing defense jobs")


if __name__ == "__main__":
    main()
