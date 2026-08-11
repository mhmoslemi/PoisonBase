#!/usr/bin/env python3
"""Stop the sweep cleanly and report exactly what's done/partial/not-started, so a
future resume knows precisely what's left. Safe to run any time -- main_new.py's own
results.csv resume logic means nothing is corrupted or double-counted by a hard kill."""
import glob
import json
import os
import subprocess
import time

ROOT = "/root/mohammad/DD-attack"
NUM_SURROGATE = 5
MODELS = ["ConvNetBN", "VGG13BN", "ResNet20BN"]
BASES = ["random", "ours"]
BUDGETS = [0.05, 0.02, 0.01, 0.005, 0.002, 0.001]
CLASS_PAIRS = ["dog-bird", "frog-airplane"]
TRIALS_TOTAL = 10 * 6  # num_targets x num_victims


def make_configs(attack):
    cfgs = []
    for cp in CLASS_PAIRS:
        for model in MODELS:
            for base in BASES:
                for budget in BUDGETS:
                    cfgs.append(dict(model=model, attack=attack, base=base, budget=budget, class_pair=cp))
    return cfgs


ALL_CONFIGS = make_configs("FC") + make_configs("GRAD")


def run_name(cfg):
    name = (f"CIFAR10_{cfg['model']}_{cfg['attack']}_{cfg['base']}_{cfg['class_pair']}"
            f"_budget{cfg['budget']:g}_seed42")
    if cfg['base'] == 'ours':
        name += f"_K{NUM_SURROGATE}_coef1.0"
    return name


print("[stop] killing sweep driver...", flush=True)
subprocess.run(["pkill", "-9", "-f", "run_sweep.py"])
time.sleep(1)

print("[stop] killing in-flight worker jobs...", flush=True)
subprocess.run(["pkill", "-9", "-f", "main_new.py --model"])
time.sleep(2)

remaining = subprocess.run(["pgrep", "-f", "main_new.py --model"], capture_output=True, text=True).stdout.strip()
if remaining:
    print(f"[stop] WARNING: processes still alive after kill: {remaining}", flush=True)
else:
    print("[stop] confirmed: no main_new.py / run_sweep.py processes remain", flush=True)

report = []
for cfg in ALL_CONFIGS:
    rn = run_name(cfg)
    run_dir = os.path.join(ROOT, 'ours_result', rn)
    entry = dict(cfg)
    entry['run_name'] = rn
    summary_path = os.path.join(run_dir, 'summary.json')
    results_path = os.path.join(run_dir, 'results.csv')
    if os.path.exists(summary_path):
        entry['status'] = 'complete'
        with open(summary_path) as f:
            s = json.load(f)
        entry['asr_mean'] = s.get('asr_mean')
        entry['cta_drop_mean'] = s.get('cta_drop_mean')
    elif os.path.exists(results_path):
        with open(results_path) as f:
            n = max(sum(1 for _ in f) - 1, 0)
        entry['status'] = 'partial'
        entry['trials_done'] = n
        entry['trials_total'] = TRIALS_TOTAL
    else:
        entry['status'] = 'not_started'
    report.append(entry)

complete = [r for r in report if r['status'] == 'complete']
partial = [r for r in report if r['status'] == 'partial']
not_started = [r for r in report if r['status'] == 'not_started']

out_path = os.path.join(ROOT, 'ours_result', 'sweep_status_at_cutoff.json')
with open(out_path, 'w') as f:
    json.dump({
        'complete': complete, 'partial': partial, 'not_started': not_started,
        'summary': {'complete': len(complete), 'partial': len(partial),
                    'not_started': len(not_started), 'total': len(report)},
    }, f, indent=2)

print(f"\n[stop] complete={len(complete)} partial={len(partial)} "
      f"not_started={len(not_started)} total={len(report)}", flush=True)
print(f"[stop] status written to {out_path}", flush=True)
if partial:
    total_partial_trials = sum(p['trials_done'] for p in partial)
    print(f"[stop] {len(partial)} configs mid-flight with {total_partial_trials} completed "
          f"trials between them -- these will resume from exactly that point next run, "
          f"nothing lost.", flush=True)
