#!/usr/bin/env python3
import itertools
import os
import subprocess
import sys
import time

ROOT = "/root/mohammad/DD-attack"
NUM_GPUS = 8
JOBS_PER_GPU = 5          # 40 concurrent slots; bump later if GPUs show headroom
NUM_SURROGATE = 5

MODELS = ["ConvNetBN", "VGG13BN", "ResNet20BN"]
BASES = ["random", "ours"]
BUDGETS = [0.05, 0.02, 0.01, 0.005, 0.002, 0.001]
CLASS_PAIRS = ["dog-bird", "frog-airplane"]

# ResNet20BN's default-margin [0.05, 0.5] targets turned out too hard (near-zero ASR even at
# budget=0.05) -- gap distribution check against the cached reference model showed most
# candidates sit near gap~1.0 (very confidently correct), so shift to the easiest accessible
# band [0.0, 0.05] (87/1000 dog-bird, 38/1000 frog-airplane candidates, both >> num_targets=10).
MARGIN_OVERRIDE = {
    'ResNet20BN': (0.0, 0.05),
}
DEFAULT_MARGIN = (0.05, 0.5)


def make_configs(attack):
    cfgs = []
    for cp in CLASS_PAIRS:
        for model in MODELS:
            for base in BASES:
                for budget in BUDGETS:
                    lo, hi = MARGIN_OVERRIDE.get(model, DEFAULT_MARGIN)
                    cfgs.append(dict(model=model, attack=attack, base=base, budget=budget,
                                      class_pair=cp, margin_low=lo, margin_high=hi))
    return cfgs


ALL_CONFIGS = make_configs("FC") + make_configs("GRAD")   # FC first, as requested
for i, cfg in enumerate(ALL_CONFIGS):
    cfg['idx'] = i
    cfg['retries'] = 0

MAX_RETRIES = 1


def group_key(cfg):
    return (cfg['model'], cfg['class_pair'])


def surrogates_warm(cfg):
    K = NUM_SURROGATE if cfg['base'] == 'ours' else 1
    for k in range(K):
        p = os.path.join(ROOT, 'cache', 'surrogates',
                          f"CIFAR10_{cfg['model']}_{cfg['class_pair']}_seed42_k{k}.pt")
        if not os.path.exists(p):
            return False
    return True


def refkey(cfg):
    return cfg['class_pair']


def reference_warm(cfg):
    ref_path = os.path.join(ROOT, 'cache', 'reference_model',
                             f"CIFAR10_{cfg['class_pair']}_seed42.pt")
    return os.path.exists(ref_path)


def targetkey(cfg):
    return (cfg['class_pair'], cfg['margin_low'], cfg['margin_high'])


def targets_warm(cfg):
    targets_path = os.path.join(
        ROOT, 'cache', 'targets',
        f"CIFAR10_{cfg['class_pair']}_seed42_margin{cfg['margin_low']:g}-{cfg['margin_high']:g}.json")
    return os.path.exists(targets_path)


def run_name(cfg):
    name = (f"CIFAR10_{cfg['model']}_{cfg['attack']}_{cfg['base']}_{cfg['class_pair']}"
            f"_budget{cfg['budget']:g}_seed42")
    if cfg['base'] == 'ours':
        name += f"_K{NUM_SURROGATE}_coef1.0"
    return name


def log_path(cfg):
    return os.path.join(ROOT, 'ours_result', 'logs', 'sweep', f"{cfg['idx']:03d}_{run_name(cfg)}.log")


def make_cmd(cfg, gpu_id):
    lp = log_path(cfg)
    cmd = (
        f"python main_new.py --model {cfg['model']} --attack {cfg['attack']} --base {cfg['base']} "
        f"--class_pair {cfg['class_pair']} --budget {cfg['budget']} "
        f"--dataset CIFAR10 --seed 42 --num_victims 6 --num_targets 10 "
        f"--victim_epochs 60 --victim_lr 0.1 --victim_bs 256 --victim_decay 35 45 --no_augmentation "
        f"--num_surrogate {NUM_SURROGATE} --coef 1.0 --epsilon 0.0313 --craft_steps 250 --craft_lr 0.01 "
        f"--ref_model ResNet20BN --target_margin_low {cfg['margin_low']:g} "
        f"--target_margin_high {cfg['margin_high']:g} "
        f"--cache_dir ./cache --data_path /root/mohammad/MKDT/data --out_dir ours_result "
        f"> {lp} 2>&1"
    )
    return cmd, lp


os.makedirs(os.path.join(ROOT, 'ours_result', 'logs', 'sweep'), exist_ok=True)


def already_complete(cfg):
    return os.path.exists(os.path.join(ROOT, 'ours_result', run_name(cfg), 'summary.json'))


skipped = [c for c in ALL_CONFIGS if already_complete(c)]
pending = [c for c in ALL_CONFIGS if not already_complete(c)]
if skipped:
    print(f"[resume] skipping {len(skipped)} already-complete configs (summary.json exists), "
          f"{len(pending)} remaining to dispatch", flush=True)
running = {}          # slot_id -> (proc, cfg, log_path, cold_group, cold_ref, cold_targets, start_ts)
results = []          # (cfg, rc, log_path)
groups_running = set()   # (model, class_pair) with a currently-running cold surrogate job
refs_running = set()     # class_pair with a currently-running cold reference-model job
targets_running = set()  # (class_pair, margin_low, margin_high) with a currently-running cold target-selection job

slots = [(g, s) for g in range(NUM_GPUS) for s in range(JOBS_PER_GPU)]
free_slots = list(slots)


def dispatchable(cfg):
    g = group_key(cfg)
    r = refkey(cfg)
    t = targetkey(cfg)
    ref_ok = reference_warm(cfg) or r not in refs_running
    tgt_ok = targets_warm(cfg) or t not in targets_running
    grp_ok = surrogates_warm(cfg) or g not in groups_running
    return ref_ok and tgt_ok and grp_ok


def pick_next_job():
    for i, cfg in enumerate(pending):
        if dispatchable(cfg):
            return i
    return None


def launch(cfg, slot):
    gpu_id, _ = slot
    cmd, lp = make_cmd(cfg, gpu_id)
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    proc = subprocess.Popen(cmd, shell=True, cwd=ROOT, env=env, executable="/bin/bash",
                             start_new_session=True)
    g = group_key(cfg)
    r = refkey(cfg)
    t = targetkey(cfg)
    cold_group = not surrogates_warm(cfg)
    cold_ref = not reference_warm(cfg)
    cold_targets = not targets_warm(cfg)
    if cold_group:
        groups_running.add(g)
    if cold_ref:
        refs_running.add(r)
    if cold_targets:
        targets_running.add(t)
    print(f"[launch] gpu={gpu_id} idx={cfg['idx']:03d} {run_name(cfg)} pid={proc.pid} "
          f"cold_ref={cold_ref} cold_targets={cold_targets} cold_group={cold_group}", flush=True)
    return proc, lp, cold_group, cold_ref, cold_targets, time.time()


total = len(ALL_CONFIGS)
t0 = time.time()

while pending or running:
    while free_slots and pending:
        idx = pick_next_job()
        if idx is None:
            break
        cfg = pending.pop(idx)
        slot = free_slots.pop(0)
        proc, lp, cold_group, cold_ref, cold_targets, start_ts = launch(cfg, slot)
        running[slot] = (proc, cfg, lp, cold_group, cold_ref, cold_targets, start_ts)

    time.sleep(10)
    finished_slots = []
    for slot, (proc, cfg, lp, cold_group, cold_ref, cold_targets, start_ts) in running.items():
        rc = proc.poll()
        if rc is not None:
            dur = time.time() - start_ts
            print(f"[done] idx={cfg['idx']:03d} {run_name(cfg)} rc={rc} dur={dur/60:.1f}min", flush=True)
            if cold_group:
                groups_running.discard(group_key(cfg))
            if cold_ref:
                refs_running.discard(refkey(cfg))
            if cold_targets:
                targets_running.discard(targetkey(cfg))
            if rc != 0 and cfg['retries'] < MAX_RETRIES:
                cfg['retries'] += 1
                print(f"[retry] idx={cfg['idx']:03d} {run_name(cfg)} rc={rc}, "
                      f"requeuing (attempt {cfg['retries']}) -- main_new.py's own resume logic "
                      f"will skip already-completed trials", flush=True)
                pending.append(cfg)
            else:
                results.append((cfg, rc, lp))
            finished_slots.append(slot)
    for slot in finished_slots:
        del running[slot]
        free_slots.append(slot)

    done_n = len(results)
    if done_n and done_n % 5 == 0:
        elapsed = (time.time() - t0) / 60
        print(f"[progress] {done_n}/{total} done, {elapsed:.1f}min elapsed", flush=True)

failures = [r for r in results if r[1] != 0]
print("\n=== SWEEP SUMMARY ===", flush=True)
print(f"{len(results)}/{total} configs finished, {len(failures)} failed", flush=True)
for cfg, rc, lp in failures:
    print(f"FAILED idx={cfg['idx']:03d} {run_name(cfg)} rc={rc} log={lp}", flush=True)
