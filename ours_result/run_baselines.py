#!/usr/bin/env python3
import itertools
import os
import subprocess
import sys
import time

ROOT = "/root/mohammad/DD-attack"
NUM_GPUS = 8
MODELS = ["ConvNetBN", "VGG13BN", "ResNet20BN"]
VICTIMS = list(range(6))

jobs = list(itertools.product(MODELS, VICTIMS))  # 18 jobs

def make_cmd(model, v):
    log_path = os.path.join(ROOT, "ours_result", "logs", "clean_baseline", f"{model}_v{v}.log")
    cmd = (
        f"python main_new.py "
        f"--baseline_only --model {model} --baseline_victim_id {v} "
        f"--dataset CIFAR10 --seed 42 --num_victims 6 "
        f"--victim_epochs 60 --victim_lr 0.1 --victim_bs 256 --victim_decay 35 45 --no_augmentation "
        f"--cache_dir ./cache --data_path /root/mohammad/MKDT/data "
        f"> {log_path} 2>&1"
    )
    return cmd, log_path

running = {}  # gpu_id -> (proc, model, v, log_path)
pending = list(jobs)
results = []  # (model, v, returncode, log_path)
free_gpus = list(range(NUM_GPUS))

def launch(gpu_id, model, v):
    cmd, log_path = make_cmd(model, v)
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    proc = subprocess.Popen(cmd, shell=True, cwd=ROOT, env=env, executable="/bin/bash")
    print(f"[launch] gpu={gpu_id} {model} v{v} pid={proc.pid}", flush=True)
    running[gpu_id] = (proc, model, v, log_path)

while pending or running:
    while pending and free_gpus:
        gpu_id = free_gpus.pop(0)
        model, v = pending.pop(0)
        launch(gpu_id, model, v)

    time.sleep(5)
    finished_gpus = []
    for gpu_id, (proc, model, v, log_path) in running.items():
        rc = proc.poll()
        if rc is not None:
            print(f"[done] gpu={gpu_id} {model} v{v} rc={rc}", flush=True)
            results.append((model, v, rc, log_path))
            finished_gpus.append(gpu_id)
    for gpu_id in finished_gpus:
        del running[gpu_id]
        free_gpus.append(gpu_id)

failures = [r for r in results if r[2] != 0]
print("\n=== SUMMARY ===")
for model, v, rc, log_path in sorted(results):
    print(f"{model} v{v}: rc={rc} log={log_path}")

if failures:
    print(f"\n{len(failures)} job(s) FAILED:")
    for model, v, rc, log_path in failures:
        print(f"\n--- tail of {log_path} ({model} v{v}, rc={rc}) ---")
        subprocess.run(["tail", "-n", "60", log_path])
    sys.exit(1)
else:
    print(f"\nAll {len(results)} jobs succeeded.")
    sys.exit(0)
