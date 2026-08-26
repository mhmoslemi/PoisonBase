#!/usr/bin/env python3
"""Regenerate one SLURM script per command in the two root command lists."""

from __future__ import annotations

import math
import re
import shlex
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
SBATCH = ROOT / "sbatch"


def parse_commands(path: Path):
    for line_number, raw in enumerate(path.read_text().splitlines(), 1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        env = {}
        for token in shlex.split(line):
            if "=" in token and not token.startswith("./"):
                key, value = token.split("=", 1)
                env[key] = value
        yield line_number, line, env


def slug(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")


def attack_minutes(env: dict[str, str]) -> int:
    model = env["MODEL"]
    attack = env["ATTACK"]
    jacobian = env.get("USE_JACOBIAN_SCORE", "0") == "1"
    budgets = [float(x) for x in env["BUDGETS"].split()]

    fc = {"ConvNetBN": 6.0, "ResNet20BN": 14.0, "VGG13BN": 11.0}
    intercept = {"ConvNetBN": 4.5, "ResNet20BN": 14.0, "VGG13BN": 10.0}
    slope = {
        ("ConvNetBN", "gradmatch"): 925.0,
        ("ConvNetBN", "sapa"): 925.0,
        ("ResNet20BN", "gradmatch"): 1600.0,
        ("ResNet20BN", "sapa"): 1100.0,
        ("VGG13BN", "gradmatch"): 2400.0,
        ("VGG13BN", "sapa"): 2000.0,
    }
    jacobian_observed = {
        "ConvNetBN": {0.001: 6, 0.002: 7, 0.005: 10, 0.01: 31, 0.02: 50, 0.04: 90},
        "ResNet20BN": {0.001: 18, 0.002: 20, 0.005: 24, 0.01: 90, 0.02: 130, 0.04: 210},
        "VGG13BN": {0.001: 13, 0.002: 14, 0.005: 19, 0.01: 83, 0.02: 130, 0.04: 210},
    }

    per_target = []
    for budget in budgets:
        if attack == "fc":
            base = fc[model]
        else:
            base = intercept[model] + slope[(model, attack)] * budget
        if jacobian:
            base = max(base, jacobian_observed[model].get(budget, base * 1.5))
        per_target.append(base)
    estimate = 8 * sum(per_target) + 15
    if env.get("SELECT") == "dpp":
        estimate += 5 * len(budgets)
    return max(60, int(math.ceil(estimate / 15.0) * 15))


def defense_minutes(env: dict[str, str]) -> int:
    model = env["MODEL"]
    defense = env["DEFENSES"]
    per_trial = {
        ("ConvNetBN", "epic"): 2.0,
        ("ConvNetBN", "friends"): 2.0,
        ("ResNet20BN", "epic"): 3.0,
        ("ResNet20BN", "friends"): 2.7,
        ("VGG13BN", "epic"): 3.0,
        ("VGG13BN", "friends"): 2.7,
    }[(model, defense)]
    cells = len(env["BUDGETS"].split()) * len(env["SELS"].split())
    estimate = cells * 7 * 5 * per_trial + 5 * per_trial + 10
    return max(60, int(math.ceil(estimate / 15.0) * 15))


def duration(minutes: int) -> str:
    days, rem = divmod(minutes, 24 * 60)
    hours, mins = divmod(rem, 60)
    return f"{days}-{hours:02d}:{mins:02d}:00"


def replace_assignment(command: str, key: str, value: str) -> str:
    pattern = rf"\b{re.escape(key)}=(?:\"[^\"]*\"|'[^']*'|\S+)"
    replacement = f"{key}={shlex.quote(value)}"
    updated, count = re.subn(pattern, replacement, command, count=1)
    if count != 1:
        raise ValueError(f"could not replace {key} in: {command}")
    return updated


def expand_cells(kind: str, command: str, env: dict[str, str]):
    """Split grouped launcher commands into one table cell per SLURM job."""
    for budget in env["BUDGETS"].split():
        selections = [env["SELECT"]] if kind == "attack" else env["SELS"].split()
        for selection in selections:
            cell = dict(env)
            cell["BUDGETS"] = budget
            effective = replace_assignment(command, "BUDGETS", budget)
            if kind == "attack":
                cell["SELECT"] = selection
            else:
                cell["SELS"] = selection
                effective = replace_assignment(effective, "SELS", selection)
                effective = replace_assignment(effective, "NUM_TARGETS", "7")
                effective = replace_assignment(effective, "NUM_VICTIMS", "5")
            yield cell, effective


def exports(env: dict[str, str], kind: str) -> list[str]:
    if kind == "attack":
        values = {
            "USE_JACOBIAN_SCORE": env.get("USE_JACOBIAN_SCORE", "0"),
            "JACOBIAN_WEIGHT": env.get("JACOBIAN_WEIGHT", "1.0"),
            "JACOBIAN_BATCH_SIZE": env.get("JACOBIAN_BATCH_SIZE", "64"),
            "CLASS_PAIR": env["CLASS_PAIR"],
            "MODEL": env["MODEL"],
            "ATTACK": env["ATTACK"],
            "BUDGETS": env["BUDGETS"],
            "SELECT": env["SELECT"],
            "SEL_ALPHA": env.get("SEL_ALPHA", "2.0"),
            "SHARP_MODE": env.get("SHARP_MODE", "worst"),
            "SHARP_SIGMA": env.get("SHARP_SIGMA", "0.05"),
            "TARGET_SELECT": env.get("TARGET_SELECT", ""),
        }
    else:
        values = {
            "USE_JACOBIAN_SCORE": env.get("USE_JACOBIAN_SCORE", "0"),
            "JACOBIAN_WEIGHT": env.get("JACOBIAN_WEIGHT", "1.0"),
            "JACOBIAN_BATCH_SIZE": env.get("JACOBIAN_BATCH_SIZE", "64"),
            "CLASS_PAIR": env["CLASS_PAIR"],
            "MODEL": env["MODEL"],
            "ATTACK": env["ATTACK"],
            "BUDGETS": env["BUDGETS"],
            "SELS": env["SELS"],
            "SEL_ALPHA": env.get("SEL_ALPHA", "2.0"),
            "DEFENSES": env["DEFENSES"],
            "TARGET_SELECT": env.get("TARGET_SELECT", ""),
            "NUM_TARGETS": "7",
            "NUM_VICTIMS": "5",
            "EPIC_SUBSET": env.get("EPIC_SUBSET", ""),
            "NOISE_EPS": env.get("NOISE_EPS", ""),
            "FRIENDLY_CLAMP": env.get("FRIENDLY_CLAMP", ""),
        }
    return [f"export {key}={shlex.quote(value)}" for key, value in values.items()]


def make_script(kind: str, index: int, source_command: str,
                effective_command: str, env: dict[str, str]):
    estimate = attack_minutes(env) if kind == "attack" else defense_minutes(env)
    long_attack = kind == "attack" and estimate > 7 * 60
    wall = estimate + 45 if long_attack else min(estimate + 45, 7 * 60)
    cushion_capped = not long_attack and estimate + 45 > 7 * 60
    method = env.get("SELECT", env.get("SELS", "selection"))
    jac = "j" if env.get("USE_JACOBIAN_SCORE", "0") == "1" else "std"
    parts = [kind, f"{index:03d}", slug(env["MODEL"].replace("BN", "")),
             slug(env["ATTACK"]), slug(env["CLASS_PAIR"]),
             "b" + slug(env["BUDGETS"])]
    if kind == "defense":
        parts.extend((slug(env["DEFENSES"]), slug(method)))
    else:
        parts.append(slug(method))
    parts.append(jac)
    label = "_".join(parts)
    filename = label + ".sh"
    if long_attack:
        time_note = (f"estimated {duration(estimate)}; long attack gets its full "
                     "estimate plus the 00:45 cushion")
    elif cushion_capped:
        time_note = (f"estimated {duration(estimate)}; requested at the standard "
                     "0-07:00:00 maximum")
    else:
        time_note = f"estimated {duration(estimate)}; includes the 00:45 cushion"
    body = [
        "#!/bin/bash",
        "#SBATCH --account=aip-boyuwang",
        f"#SBATCH --job-name={label[:80]}",
        f"#SBATCH --time={duration(wall)}",
        "#SBATCH --nodes=1",
        "#SBATCH --ntasks=1",
        "#SBATCH --cpus-per-task=1",
        "#SBATCH --mem=7G",
        "#SBATCH --gres=gpu:l40s:1",
        "#SBATCH --signal=B:USR1@300",
        f"#SBATCH --output=/home/mmoslem3/scratch/attack_if/sbatch/logs/{label}-%j.out",
        "",
        f"# L40S walltime: {time_note}.",
        f"# Grouped source command: {source_command}",
        f"# This-cell command: {effective_command}",
        "",
        f"export JOB_KIND={kind}",
        f"export ORIGINAL_COMMAND={shlex.quote(effective_command)}",
        *exports(env, kind),
        "",
        'source /home/mmoslem3/scratch/attack_if/sbatch/_job_common.sh',
        "",
    ]
    return (filename, "\n".join(body), estimate, wall, long_attack,
            cushion_capped, effective_command)


def write_submitter(kind: str, names: list[str]):
    body = [
        "#!/bin/sh",
        "set -eu",
        'ROOT="/home/mmoslem3/scratch/attack_if"',
        'mkdir -p "$ROOT/sbatch/logs"',
        *[line for name in names
          for line in (f'sbatch "$ROOT/sbatch/{kind}/{name}"', "sleep 1")],
        "",
    ]
    (SBATCH / f"submit_{kind}.sh").write_text("\n".join(body))


def main():
    manifests = ["kind\tindex\tfile\testimate\twalltime\tschedule_class\tcommand"]
    all_names = {}
    specs = (("attack", ROOT / "remaining_full_tex_commands.txt"),
             ("defense", ROOT / "run_defense.txt"))
    for kind, source in specs:
        out = SBATCH / kind
        out.mkdir(parents=True, exist_ok=True)
        for old in out.glob("*.sh"):
            old.unlink()
        names = []
        cells = []
        for _line_number, command, env in parse_commands(source):
            cells.extend((command, cell_env, effective)
                         for cell_env, effective in expand_cells(kind, command, env))
        if kind == "attack":
            # Stable partition: normal cells retain table order, while genuinely
            # >7h cells are submitted last with their uncapped requested time.
            cells.sort(key=lambda item: attack_minutes(item[1]) > 7 * 60)
        for index, (source_command, env, effective) in enumerate(cells, 1):
            (filename, text, estimate, wall, long_attack, cushion_capped,
             adjusted) = make_script(
                kind, index, source_command, effective, env)
            (out / filename).write_text(text)
            names.append(filename)
            manifests.append("\t".join((kind, str(index), f"{kind}/{filename}",
                                         duration(estimate), duration(wall),
                                         "long_attack" if long_attack else
                                         "standard_7h" if cushion_capped else
                                         "standard", adjusted)))
        all_names[kind] = names
        write_submitter(kind, names)

    (SBATCH / "manifest.tsv").write_text("\n".join(manifests) + "\n")
    for path in SBATCH.rglob("*.sh"):
        path.chmod(0o755)
    print(f"generated {len(all_names['attack'])} attack-cell and "
          f"{len(all_names['defense'])} defense-cell jobs")


if __name__ == "__main__":
    main()
