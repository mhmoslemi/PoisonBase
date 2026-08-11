#!/usr/bin/env python3
import csv
import glob
import json
import statistics
import os

ROOT = "/root/mohammad/DD-attack"
CACHE_GLOB = os.path.join(ROOT, "cache", "clean_baseline", "CIFAR10_*_seed42_v*.json")
OUT_DIR = os.path.join(ROOT, "ours_result")

by_model = {}
for path in sorted(glob.glob(CACHE_GLOB)):
    with open(path) as f:
        d = json.load(f)
    by_model.setdefault(d["model"], []).append(d)

rows = []
json_out = {}
for model in sorted(by_model):
    entries = sorted(by_model[model], key=lambda d: d["victim_id"])
    accs = [e["test_acc"] for e in entries]
    mean_acc = statistics.mean(accs)
    std_acc = statistics.stdev(accs) if len(accs) > 1 else 0.0
    rows.append({
        "model": model,
        "num_checkpoints": len(accs),
        "mean_acc": mean_acc,
        "std_acc": std_acc,
    })
    json_out[model] = {
        "num_checkpoints": len(accs),
        "mean_acc": mean_acc,
        "std_acc": std_acc,
        "checkpoints": [{"victim_id": e["victim_id"], "test_acc": e["test_acc"]} for e in entries],
    }

csv_path = os.path.join(OUT_DIR, "clean_baseline_summary.csv")
with open(csv_path, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["model", "num_checkpoints", "mean_acc", "std_acc"])
    w.writeheader()
    for r in rows:
        w.writerow(r)

json_path = os.path.join(OUT_DIR, "clean_baseline_summary.json")
with open(json_path, "w") as f:
    json.dump(json_out, f, indent=2)

print(f"wrote {csv_path}")
print(f"wrote {json_path}")
print()
for r in rows:
    print(f"{r['model']}: mean={r['mean_acc']:.4f} std={r['std_acc']:.4f} n={r['num_checkpoints']}")
