#!/usr/bin/env python3
"""Expand the three cross-architecture tables and generate one job per blank."""

from __future__ import annotations

import math
import re
import shlex
from pathlib import Path

from generate_jobs import ROOT, SBATCH, attack_minutes, duration


TABLE = ROOT / "cross-table.tex"
OUT = SBATCH / "cross_expanded"
KS = (1, 3, 20)
BUDGETS = (0.002, 0.005)
ATTACKS = ("fc", "gradmatch", "sapa")
ATTACK_LABEL = {"fc": "BP", "gradmatch": "GM", "sapa": "SAPA"}
MODELS = ("ConvNetBN", "ResNet20BN", "VGG13BN")
DISPLAY = {"ConvNetBN": "ConvNet", "ResNet20BN": "ResNet20", "VGG13BN": "VGG13"}
DISPLAY_TO_MODEL = {value: key for key, value in DISPLAY.items()}
METHODS = ("random", "greedy", "dpp")
SCORE_RE = re.compile(r"\\score\{([0-9.]+)\}\{([0-9.]+)\}")


def parse_old_table(text: str):
    """Read the user's original Random/DPP, b=0.005 three-table layout."""
    values = {}
    for block in text.split(r"\begin{table*}")[1:]:
        if r"\label{tab:cross-architecture-k1}" in block:
            k = 1
        elif r"\label{tab:cross-architecture-k3}" in block:
            k = 3
        elif r"\label{tab:cross-architecture}" in block:
            k = 20
        else:
            continue
        attack = None
        lines = block.splitlines()
        i = 0
        while i < len(lines):
            stripped = lines[i].strip()
            match = re.search(r"\\textbf\{(BP|GM|SAPA)\}", stripped)
            if match:
                attack = {"BP": "fc", "GM": "gradmatch", "SAPA": "sapa"}[match.group(1)]
            if stripped in {"ConvNet &", "ResNet20 &", "VGG13 &"}:
                selector = DISPLAY_TO_MODEL[stripped[:-2].strip()]
                cell_text = []
                i += 1
                while i < len(lines) and lines[i].strip() != r"\\":
                    cell_text.append(lines[i])
                    i += 1
                tokens = re.findall(
                    r"\\score\{[0-9.]+\}\{[0-9.]+\}|--", " ".join(cell_text)
                )
                if len(tokens) != 6:
                    raise ValueError(
                        f"old K={k} {attack} {selector}: expected 6 cells, got {len(tokens)}"
                    )
                for index, token in enumerate(tokens):
                    if token == "--":
                        continue
                    victim = MODELS[index // 2]
                    method = ("random", "dpp")[index % 2]
                    score = SCORE_RE.fullmatch(token)
                    values[(k, attack, 0.005, selector, victim, method)] = score.groups()
            i += 1
    return values


def parse_expanded_table(text: str):
    """Read an already expanded table so regeneration preserves filled cells."""
    values = {}
    lines = text.splitlines()
    for index, line in enumerate(lines):
        match = re.fullmatch(
            r"% CELLROW K=(1|3|20) attack=(fc|gradmatch|sapa) "
            r"budget=(0\.002|0\.005) selector=(ConvNetBN|ResNet20BN|VGG13BN)",
            line.strip(),
        )
        if not match:
            continue
        k, attack, budget, selector = match.groups()
        row = []
        cursor = index + 1
        while cursor < len(lines):
            row.append(lines[cursor].strip())
            if lines[cursor].strip().endswith(r"\\"):
                break
            cursor += 1
        parts = " ".join(row)[:-2].split("&")
        cells = parts[-9:]
        if len(cells) != 9:
            raise ValueError(f"expanded row has {len(cells)} cells: {line}")
        for cell_index, cell in enumerate(cells):
            score = SCORE_RE.search(cell)
            if not score:
                continue
            victim = MODELS[cell_index // 3]
            method = METHODS[cell_index % 3]
            values[(int(k), attack, float(budget), selector, victim, method)] = score.groups()
    return values


def current_values():
    text = TABLE.read_text()
    if "% CELLROW " in text:
        return parse_expanded_table(text)
    return parse_old_table(text)


def format_cell(value, matched: bool):
    inner = "--" if value is None else rf"\score{{{value[0]}}}{{{value[1]}}}"
    return rf"\matched{{{inner}}}" if matched else inner


def render_table(k: int, values: dict):
    k_text = "one selector surrogate" if k == 1 else f"{k} selector surrogates"
    exception = (
        " The two completed off-diagonal VGG13 BP/DPP cells at budget 5 use "
        "16 trials after one pinned baseline free-win target was excluded."
        if k == 1 else ""
    )
    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\caption{",
        "Cross-architecture transfer of base selection for dog--bird at budgets",
        rf"$\epsilon\in\{{2,5\}}\times10^{{-3}}$ using {k_text} ($K={k}$).",
        "Rows identify the selector architecture $S$; column groups identify the",
        "attack/victim architecture $A=V$. Shaded cells are matched architectures.",
        "Entries report ASR (\%) with clean test accuracy in parentheses; ``--''",
        f"denotes an incomplete or unevaluated five-target, four-victim cell.{exception}",
        r"}",
        r"\vspace{-0.5em}",
        (r"\label{tab:cross-architecture}"
         if k == 20 else rf"\label{{tab:cross-architecture-k{k}}}"),
        "",
        r"\setlength{\tabcolsep}{3pt}",
        r"\renewcommand{\arraystretch}{1.08}",
        "",
        r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{@{}c c l ccc ccc ccc@{}}",
        r"\toprule",
        r"\textbf{Attack} &",
        r"$\boldsymbol{\epsilon}$ &",
        r"\textbf{Selection $S$} &",
        r"\multicolumn{3}{c}{\textbf{ConvNet}} &",
        r"\multicolumn{3}{c}{\textbf{ResNet20}} &",
        r"\multicolumn{3}{c}{\textbf{VGG13}} \\",
        r"\cmidrule(lr){4-6} \cmidrule(lr){7-9} \cmidrule(l){10-12}",
        r"& {\tiny $(\times10^{-3})$} & &",
        r"Random & Greedy & DPP & Random & Greedy & DPP & Random & Greedy & DPP \\",
        r"\midrule",
    ]
    for attack_index, attack in enumerate(ATTACKS):
        for budget_index, budget in enumerate(BUDGETS):
            for selector_index, selector in enumerate(MODELS):
                lines.append(
                    f"% CELLROW K={k} attack={attack} budget={budget:.3f} selector={selector}"
                )
                prefix = []
                if budget_index == 0 and selector_index == 0:
                    prefix.append(rf"\multirow{{6}}{{*}}{{\textbf{{{ATTACK_LABEL[attack]}}}}}")
                else:
                    prefix.append("")
                if selector_index == 0:
                    prefix.append(rf"\multirow{{3}}{{*}}{{{int(round(budget * 1000))}}}")
                else:
                    prefix.append("")
                prefix.append(DISPLAY[selector])
                cells = []
                for victim in MODELS:
                    for method in METHODS:
                        key = (k, attack, budget, selector, victim, method)
                        cells.append(format_cell(values.get(key), selector == victim))
                lines.append(" & ".join(prefix + cells) + r" \\")
            if budget_index == 0:
                lines.append(r"\cmidrule(lr){2-12}")
        if attack_index != len(ATTACKS) - 1:
            lines.append(r"\midrule")
    lines.extend((r"\bottomrule", r"\end{tabular}%", r"}", r"\end{table*}", ""))
    return "\n".join(lines)


def render_all(values: dict):
    return "\n\n\n".join(render_table(k, values) for k in KS)


def slug(value: str):
    return re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")


def run_name(spec):
    base = "random" if spec["method"] == "random" else "ours"
    name = (
        f"CIFAR10_{spec['victim']}_{spec['attack']}_{base}_dog-bird_"
        f"b{spec['budget']:g}_eps8_seed42"
    )
    if base == "ours":
        name += "_lam1_cosine"
        if spec["method"] == "dpp":
            name += "_seldpp2"
    if spec["selector"] != spec["victim"]:
        name += f"_selarch{spec['selector']}"
    if spec["k"] != 20:
        name += f"_K{spec['k']}"
    if spec["attack"] == "sapa":
        name += "_worst0.05"
    degree = {
        ("ConvNetBN", "fc"): 50, ("ConvNetBN", "gradmatch"): 70,
        ("ConvNetBN", "sapa"): 70, ("ResNet20BN", "fc"): 10,
        ("ResNet20BN", "gradmatch"): 14, ("ResNet20BN", "sapa"): 14,
        ("VGG13BN", "fc"): 3, ("VGG13BN", "gradmatch"): 50,
        ("VGG13BN", "sapa"): 50,
    }[(spec["victim"], spec["attack"])]
    return name + f"_ce5_tgt{degree}"


def blank_specs(values: dict):
    for k in KS:
        for budget in BUDGETS:
            for attack in ATTACKS:
                for selector in MODELS:
                    for victim in MODELS:
                        for method in METHODS:
                            key = (k, attack, budget, selector, victim, method)
                            if key not in values:
                                yield {
                                    "k": k, "budget": budget, "attack": attack,
                                    "selector": selector, "victim": victim, "method": method,
                                }


INTERRUPTED = {
    (1, "gradmatch", 0.005, "ConvNetBN", "ResNet20BN", "dpp"),
    (1, "gradmatch", 0.005, "VGG13BN", "ResNet20BN", "dpp"),
    (3, "fc", 0.005, "VGG13BN", "ConvNetBN", "dpp"),
    (3, "fc", 0.005, "ConvNetBN", "ResNet20BN", "dpp"),
    (3, "fc", 0.005, "VGG13BN", "ResNet20BN", "dpp"),
}


def priority(spec):
    key = (spec["k"], spec["attack"], spec["budget"], spec["selector"],
           spec["victim"], spec["method"])
    return (
        0 if key in INTERRUPTED else 1,
        spec["budget"],
        {"fc": 0, "gradmatch": 1, "sapa": 2}[spec["attack"]],
        {1: 0, 3: 1, 20: 2}[spec["k"]],
        {"random": 0, "greedy": 1, "dpp": 2}[spec["method"]],
        MODELS.index(spec["selector"]), MODELS.index(spec["victim"]),
    )


def make_job(index: int, spec):
    env_for_time = {
        "MODEL": spec["victim"], "ATTACK": spec["attack"],
        "BUDGETS": f"{spec['budget']:g}", "NUM_TARGETS": "5",
        "NUM_VICTIMS": "4", "SELECT": spec["method"],
    }
    estimate = attack_minutes(env_for_time)
    wall = estimate + 45 if estimate > 7 * 60 else min(estimate + 45, 7 * 60)
    budget_slug = slug(f"{spec['budget']:g}")
    label = (
        f"cross_{index:03d}_k{spec['k']}_{slug(spec['attack'])}_"
        f"b{budget_slug}_s{slug(DISPLAY[spec['selector']])}_"
        f"a{slug(DISPLAY[spec['victim']])}_{spec['method']}"
    )
    command = (
        f"SEL_K={spec['k']} BUDGET={spec['budget']:g} "
        f"ATTACKS={spec['attack']} MODELS={spec['victim']} "
        f"SELECTOR_MODELS={spec['selector']} SELECTIONS={spec['method']} "
        "RUN_MATCHED=1 NUM_TARGETS=5 NUM_VICTIMS=4 sh cross_arch.sh"
    )
    body = [
        "#!/bin/bash",
        "#SBATCH --account=aip-boyuwang",
        f"#SBATCH --job-name={label}",
        f"#SBATCH --time={duration(wall)}",
        "#SBATCH --nodes=1",
        "#SBATCH --ntasks=1",
        "#SBATCH --cpus-per-task=1",
        "#SBATCH --mem=7G",
        "#SBATCH --gres=gpu:l40s:1",
        "#SBATCH --signal=B:USR1@300",
        f"#SBATCH --output=/home/mmoslem3/scratch/attack_if/sbatch/logs/{label}-%j.out",
        "",
        f"# Exactly one table cell: {command}",
        f"# Estimated L40S runtime {duration(estimate)} plus a 00:45 cushion.",
        "",
        f"export CROSS_MODEL={shlex.quote(spec['victim'])}",
        f"export CROSS_SELECTOR_MODEL={shlex.quote(spec['selector'])}",
        f"export CROSS_ATTACK={shlex.quote(spec['attack'])}",
        f"export CROSS_SELECTION={shlex.quote(spec['method'])}",
        f"export CROSS_BUDGET={spec['budget']:g}",
        f"export CROSS_K={spec['k']}",
        "export CROSS_NUM_TARGETS=5",
        "export CROSS_NUM_VICTIMS=4",
        f"export CROSS_RUN_NAME={shlex.quote(run_name(spec))}",
        f"export ORIGINAL_COMMAND={shlex.quote(command)}",
        "",
        "source /home/mmoslem3/scratch/attack_if/sbatch/_cross_job_common.sh",
        "",
    ]
    return label + ".sh", "\n".join(body), estimate, wall, command


def generate_jobs(values: dict):
    OUT.mkdir(parents=True, exist_ok=True)
    for old in OUT.glob("*.sh"):
        old.unlink()
    specs = sorted(blank_specs(values), key=priority)
    manifest = [
        "index\tfile\tK\tbudget\tattack\tselector\tvictim\tmethod\trun_name\t"
        "estimate\twalltime\tcommand"
    ]
    names = []
    for index, spec in enumerate(specs, 1):
        filename, text, estimate, wall, command = make_job(index, spec)
        (OUT / filename).write_text(text)
        names.append(filename)
        manifest.append("\t".join((
            str(index), filename, str(spec["k"]), f"{spec['budget']:g}",
            spec["attack"], spec["selector"], spec["victim"], spec["method"],
            run_name(spec), duration(estimate), duration(wall), command,
        )))
    (OUT / "manifest.tsv").write_text("\n".join(manifest) + "\n")
    submit = [
        "#!/bin/sh", "# Interrupted H200 cells first, then b0.002 before b0.005.",
        "set -eu", 'ROOT="/home/mmoslem3/scratch/attack_if"',
        'mkdir -p "$ROOT/sbatch/logs"',
    ]
    for name in names:
        submit.extend((f'sbatch "$ROOT/sbatch/cross_expanded/{name}"', "sleep 1"))
    submit.append("")
    (OUT / "submit.sh").write_text("\n".join(submit))
    for path in OUT.glob("*.sh"):
        path.chmod(0o755)
    return len(names)


def main():
    values = current_values()
    TABLE.write_text(render_all(values))
    count = generate_jobs(values)
    print(f"preserved {len(values)} complete cells; generated {count} one-cell jobs")


if __name__ == "__main__":
    main()
