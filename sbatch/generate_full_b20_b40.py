#!/usr/bin/env python3
"""Generate one attack job for each blank ResNet20/VGG13 b20/b40 cell."""

from __future__ import annotations

import re
import shlex
from pathlib import Path

from generate_jobs import ROOT, SBATCH, make_script


OUT = SBATCH / "full_b20_b40"
MODELS = {"ResNet20": "ResNet20BN", "VGG13": "VGG13BN"}
PAIRS = ("dog-bird", "frog-airplane")
ATTACKS = ("fc", "gradmatch", "sapa")
METHODS = (
    ("random", "0", "random"),
    ("greedy", "0", "ours"),
    ("dpp", "0", "dpp"),
    ("greedy_j", "1", "ours"),
    ("dpp_j", "1", "dpp"),
)


def missing_cells():
    lines = (ROOT / "full.tex").read_text().splitlines()
    model = None
    for index, line in enumerate(lines):
        model_match = re.search(r"\\textbf\{(ResNet20|VGG13)\}", line)
        if model_match and "multirow" in line:
            model = MODELS[model_match.group(1)]
        budget_match = re.match(r"\s*&\s*(20|40)\s*&", line)
        if model is None or not budget_match:
            continue
        budget_int = int(budget_match.group(1))
        budget = f"{budget_int / 1000:g}"
        cells = re.findall(
            r"(?:\\score(?:TT|B)\{[^}]+\}\{[^}]+\}|--)",
            " ".join(lines[index + 1:index + 3]),
        )
        if len(cells) != 30:
            raise ValueError(
                f"expected 30 cells for {model} budget {budget_int}, got {len(cells)}"
            )
        for cell_index, cell in enumerate(cells):
            if cell != "--":
                continue
            pair = PAIRS[cell_index // 15]
            within_pair = cell_index % 15
            attack = ATTACKS[within_pair // 5]
            method, jacobian, selection = METHODS[within_pair % 5]
            env = {
                "USE_JACOBIAN_SCORE": jacobian,
                "JACOBIAN_WEIGHT": "1.0",
                "JACOBIAN_BATCH_SIZE": "64",
                "CLASS_PAIR": pair,
                "MODEL": model,
                "ATTACK": attack,
                "BUDGETS": budget,
                "SELECT": selection,
                "SEL_ALPHA": "2.0",
                "SHARP_MODE": "worst",
                "SHARP_SIGMA": "0.05",
                "NUM_TARGETS": "8",
                "NUM_VICTIMS": "6",
            }
            assignments = [
                f"USE_JACOBIAN_SCORE={jacobian}",
                "JACOBIAN_WEIGHT=1.0",
                "JACOBIAN_BATCH_SIZE=64",
                f"CLASS_PAIR={shlex.quote(pair)}",
                f"MODEL={model}",
                f"ATTACK={attack}",
            ]
            if attack == "sapa":
                assignments.extend(("SHARP_MODE=worst", "SHARP_SIGMA=0.05"))
            assignments.extend((
                f"BUDGETS={budget}",
                f"SELECT={selection}",
                "SEL_ALPHA=2.0",
                "NUM_TARGETS=8",
                "NUM_VICTIMS=6",
                "sh sel_dpp.sh",
            ))
            command = " ".join(assignments)
            yield {
                "model": model,
                "budget_int": budget_int,
                "pair": pair,
                "attack": attack,
                "method": method,
                "env": env,
                "command": command,
            }


def priority(spec):
    """BP first; within BP/non-BP groups, submit b20 before b40."""
    attack_order = {"fc": 0, "gradmatch": 1, "sapa": 2}
    model_order = {"ResNet20BN": 0, "VGG13BN": 1}
    pair_order = {"dog-bird": 0, "frog-airplane": 1}
    method_order = {"greedy": 0, "greedy_j": 1, "dpp_j": 2}
    return (
        0 if spec["attack"] == "fc" else 1,
        spec["budget_int"],
        attack_order[spec["attack"]],
        model_order[spec["model"]],
        pair_order[spec["pair"]],
        method_order[spec["method"]],
    )


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    for old in OUT.glob("*.sh"):
        old.unlink()
    specs = sorted(missing_cells(), key=priority)
    names = []
    manifest = [
        "index\tfile\tmodel\tattack\tpair\tbudget\tmethod\testimate\twalltime\tcommand"
    ]
    for index, spec in enumerate(specs, 1):
        filename, text, estimate, wall, *_ = make_script(
            "attack",
            index,
            spec["command"],
            spec["command"],
            spec["env"],
            name_prefix="b2040_",
        )
        (OUT / filename).write_text(text)
        names.append(filename)
        manifest.append("\t".join((
            str(index), filename, spec["model"], spec["attack"], spec["pair"],
            f"{spec['budget_int'] / 1000:g}", spec["method"], str(estimate),
            str(wall), spec["command"],
        )))
    (OUT / "manifest.tsv").write_text("\n".join(manifest) + "\n")

    submit = [
        "#!/bin/sh",
        "# BP jobs first; b20 precedes b40 within each priority group.",
        "set -eu",
        'ROOT="/home/mmoslem3/scratch/attack_if"',
        'mkdir -p "$ROOT/sbatch/logs"',
    ]
    for name in names:
        submit.extend((f'sbatch "$ROOT/sbatch/full_b20_b40/{name}"', "sleep 1"))
    submit.append("")
    (OUT / "submit.sh").write_text("\n".join(submit))
    for path in OUT.glob("*.sh"):
        path.chmod(0o755)
    print(f"generated {len(names)} ResNet20/VGG13 b20/b40 jobs")


if __name__ == "__main__":
    main()
