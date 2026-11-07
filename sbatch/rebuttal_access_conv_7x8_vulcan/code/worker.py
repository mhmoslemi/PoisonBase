"""Extend W1 to seven targets and eight victims, retaining completed GM trials."""
import csv
import json
import os
from pathlib import Path
import sys

import access_base as base

batch = Path(os.environ["ACCESS_BATCH"])
with (batch / "manifest.tsv").open() as handle:
    tasks = list(csv.DictReader(handle, delimiter="\t"))
assert len(tasks) == 28
task = tasks[int(sys.argv[1])]
attack, method = task["attack"], task["method"]
condition_id = int(task["condition"])
start, stop = int(task["start"]), int(task["stop"])
assert attack in ("gradmatch", "fc") and method in ("random", "basis")
assert condition_id in (1, 2, 3, 4, 5, 6, 8)
assert method != "random" or condition_id in (1, 3, 4)

base.TARGETS = {
    "gradmatch": (843, 1257, 8252, 8730, 8055, 5859, 8954),
    "fc": (2540, 8129, 3725, 5705, 7663, 3875, 5169),
}[attack]
base.VICTIMS = range(8)
# Append the full-access condition without changing the original condition seeds.
base.CONDITIONS += ((0.002, 1.0, 1.0),)
if attack == "fc":
    base.OUT = base.ROOT / "rebuttal_access_conv_7x8_result" / "bp"
condition = base.CONDITIONS[condition_id]
all_pairs = list(base.combinations(method))
assert 0 <= start < stop <= len(all_pairs)
pairs = all_pairs[start:stop]

if attack == "gradmatch" and condition_id != 8:
    for sd, cd in pairs:
        for target in base.TARGETS[:5]:
            path = base.poison_path(method, condition_id, sd, cd, target)
            if not path.is_file():
                raise RuntimeError(f"Original poison cache missing; refusing to recraft: {path}")


def collect(draws, required):
    rows = []
    for sd, cd in draws:
        sname = "fixed" if sd is None else str(sd)
        for target in base.TARGETS:
            directory = (base.OUT / method / f"condition_{condition_id}" /
                         "trials" / f"s{sname}_c{cd}_t{target}")
            for victim in base.VICTIMS:
                path = directory / f"victim_{victim}.json"
                if not path.is_file():
                    if required:
                        raise RuntimeError(f"Missing completed trial: {path}")
                    return None
                row = json.loads(path.read_text())
                assert row["method"] == method and row["target"] == target
                assert row["victim"] == victim and row["epochs"] == 50
                assert tuple(row["condition"]) == tuple(condition)
                assert row["seed"] == 4200000 + target * 100 + victim
                assert row["surrogate_draw"] == sd and row["candidate_draw"] == cd
                assert row["success"] in (0, 1)
                rows.append(row)
    assert len(rows) == len(draws) * 7 * 8
    return rows


def summarize(_method, _condition):
    directory = base.OUT / method / f"condition_{condition_id}"

    def save(name, rows, draws):
        base.atomic_json(directory / name, {
            "attack": attack, "method": method, "condition_id": condition_id,
            "condition": condition, "targets": base.TARGETS,
            "victims": list(base.VICTIMS), "draws": draws, "trials": len(rows),
            "asr": sum(row["success"] for row in rows) / len(rows),
            "clean_accuracy": sum(row["clean_accuracy"] for row in rows) / len(rows),
        })

    save(f"summary_7x8_draws_{start}_{stop}.json", collect(pairs, True), pairs)
    rows = collect(all_pairs, False)
    if rows is not None:
        save("summary_7x8.json", rows, all_pairs)
        print(f"CELL COMPLETE: attack={attack} method={method} condition={condition_id} "
              f"trials={len(rows)}", flush=True)


base.combinations = lambda _method: iter(pairs)
base.summarize = summarize
print(f"W1 task={task['id']} attack={attack} method={method} condition={condition} "
      f"draws={pairs} targets={base.TARGETS} victims=0..7 epochs=50", flush=True)
args, ctx = base.context()
args.attack = attack
args.num_targets = 7
args.num_victims = 8
base.run_cell(args, ctx, method, condition_id)
