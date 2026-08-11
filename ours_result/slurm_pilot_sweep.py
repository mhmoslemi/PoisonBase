#!/usr/bin/env python3
"""
Concurrent per-config SLURM orchestrator for the clean-label poisoning sweep on vulcan.
Up to MAX_CONCURRENT sbatch jobs in flight at once across all 12 (attack, class_pair,
model) groups. Each group: pilot-check ASR sanity at high/mid/low budget (5 targets x 3
victims) before committing to the full 60-trial run (10 targets x 6 victims) across all 6
budgets. Cold-cache builds (reference model / surrogates / targets) are gated to at most
one in-flight job per cache key so concurrent jobs never race-write the same cache file.

Runs as a persistent loop -- launch it inside tmux on vulcan so it survives disconnect.
State persisted to pilot_state.json after every transition; safe to kill and rerun.
"""
import json
import os
import subprocess
import time

ROOT = "/home/mmoslem3/scratch/attack_if"
OUT_DIR = os.path.join(ROOT, "ours_result")
CACHE_DIR = os.path.join(ROOT, "cache")
DATA_PATH = "/home/mmoslem3/scratch/data"
ENV_ACTIVATE = "/home/mmoslem3/ENV/bin/activate"
ACCOUNT = "aip-boyuwang"
PARTITION = "gpubase_bygpu_b2"   # 12h cap, 1 GPU per job
NUM_SURROGATE = 5

MODELS = ["ConvNetBN", "VGG13BN", "ResNet20BN"]
BASES = ["random", "ours"]
ALL_BUDGETS = [0.05, 0.02, 0.01, 0.005, 0.002, 0.001]
CLASS_PAIRS = ["dog-bird", "frog-airplane"]
ATTACKS = ["FC", "GRAD"]

MARGIN_OVERRIDE = {"ResNet20BN": (0.0, 0.05)}
DEFAULT_MARGIN = (0.05, 0.5)

HIGH_BUDGET_ASR_FLOOR = 0.15   # below this at budget=0.05 -> bad / retry with wider margin
MID_BUDGET_ASR_FLOOR = 0.07    # below this at budget=0.01 -> stop pilot-checking, go full
MAX_MARGIN_ATTEMPTS = 2
MAX_JOB_RETRIES = 1

MAX_CONCURRENT = 50
POLL_SECS = 25

STATE_PATH = os.path.join(OUT_DIR, "pilot_state.json")
BAD_GROUPS_PATH = os.path.join(OUT_DIR, "bad_groups.json")
SLURM_LOG_DIR = os.path.join(OUT_DIR, "logs", "slurm")
os.makedirs(SLURM_LOG_DIR, exist_ok=True)


def log(msg):
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}", flush=True)


def load_state():
    if os.path.exists(STATE_PATH):
        with open(STATE_PATH) as f:
            return json.load(f)
    return {"groups": {}}


def save_state(state):
    tmp = STATE_PATH + ".tmp"
    with open(tmp, "w") as f:
        json.dump(state, f, indent=2)
    os.replace(tmp, STATE_PATH)


def save_bad(bad):
    with open(BAD_GROUPS_PATH, "w") as f:
        json.dump(bad, f, indent=2)


def load_bad():
    return json.load(open(BAD_GROUPS_PATH)) if os.path.exists(BAD_GROUPS_PATH) else {}


def run_name(model, attack, base, class_pair, budget):
    name = f"CIFAR10_{model}_{attack}_{base}_{class_pair}_budget{budget:g}_seed42"
    if base == "ours":
        name += f"_K{NUM_SURROGATE}_coef1.0"
    return name


def summary_path(model, attack, base, class_pair, budget):
    return os.path.join(OUT_DIR, run_name(model, attack, base, class_pair, budget), "summary.json")


def config_done(model, attack, base, class_pair, budget):
    return os.path.exists(summary_path(model, attack, base, class_pair, budget))


def read_asr(model, attack, base, class_pair, budget):
    p = summary_path(model, attack, base, class_pair, budget)
    if not os.path.exists(p):
        return None
    with open(p) as f:
        return json.load(f).get("asr_mean")


def surrogates_warm(model, class_pair):
    for k in range(NUM_SURROGATE):   # gate on the stricter K=5 ('ours') requirement
        p = os.path.join(CACHE_DIR, "surrogates", f"CIFAR10_{model}_{class_pair}_seed42_k{k}.pt")
        if not os.path.exists(p):
            return False
    return True


def reference_warm(class_pair):
    return os.path.exists(os.path.join(CACHE_DIR, "reference_model", f"CIFAR10_{class_pair}_seed42.pt"))


def targets_warm(class_pair, lo, hi):
    return os.path.exists(os.path.join(
        CACHE_DIR, "targets", f"CIFAR10_{class_pair}_seed42_margin{lo:g}-{hi:g}.json"))


def pilot_time_for(model, attack):
    return "02:00:00" if model == "VGG13BN" else "01:30:00"


def full_time_for(model, attack, budget):
    base = {"ConvNetBN": "04:00:00", "ResNet20BN": "04:30:00", "VGG13BN": "05:30:00"}[model]
    if attack == "GRAD" and budget >= 0.02:
        return "07:00:00" if model == "VGG13BN" else "06:00:00"
    return base


def submit_job(spec):
    model, attack, base, class_pair, budget = (spec["model"], spec["attack"], spec["base"],
                                                 spec["class_pair"], spec["budget"])
    lo, hi = spec["margin"]
    tag = spec["tag"]
    lp = os.path.join(SLURM_LOG_DIR, f"{tag}_{run_name(model, attack, base, class_pair, budget)}.log")
    sbatch_script = f"""#!/bin/bash
#SBATCH --account={ACCOUNT}
#SBATCH --partition={PARTITION}
#SBATCH --gpus=1
#SBATCH --cpus-per-task=1
#SBATCH --mem={spec.get('mem', 8)}G
#SBATCH --time={spec['time_limit']}
#SBATCH --output={lp}
#SBATCH --job-name={tag[:3]}_{model[:4]}_{attack[:2]}
set -e
source {ENV_ACTIVATE}
cd {ROOT}
python main_new.py --model {model} --attack {attack} --base {base} \\
    --class_pair {class_pair} --budget {budget} \\
    --dataset CIFAR10 --seed 42 --num_victims {spec['num_victims']} --num_targets {spec['num_targets']} \\
    --victim_epochs 60 --victim_lr 0.1 --victim_bs 256 --victim_decay 35 45 --no_augmentation \\
    --num_surrogate {NUM_SURROGATE} --coef 1.0 --epsilon 0.0313 --craft_steps 250 --craft_lr 0.01 \\
    --ref_model ResNet20BN --target_margin_low {lo:g} --target_margin_high {hi:g} \\
    --cache_dir {CACHE_DIR} --data_path {DATA_PATH} --out_dir {OUT_DIR}
"""
    script_path = os.path.join(SLURM_LOG_DIR, f"{tag}_{run_name(model, attack, base, class_pair, budget)}.sbatch")
    with open(script_path, "w") as f:
        f.write(sbatch_script)
    result = subprocess.run(["sbatch", "--parsable", script_path], capture_output=True, text=True, cwd=ROOT)
    if result.returncode != 0:
        raise RuntimeError(f"sbatch failed: {result.stderr}")
    job_id = result.stdout.strip().split(";")[0]
    log(f"[submit] job={job_id} {spec['tag']} {run_name(model, attack, base, class_pair, budget)} "
        f"time={spec['time_limit']}")
    return job_id


def batch_job_states(job_ids):
    if not job_ids:
        return {}
    r = subprocess.run(["sacct", "-j", ",".join(job_ids), "-o", "JobID,State", "-n", "-P"],
                        capture_output=True, text=True)
    states = {}
    for line in r.stdout.splitlines():
        line = line.strip()
        if not line or "." in line.split("|")[0]:   # skip .batch/.extern sub-steps
            continue
        parts = line.split("|")
        if len(parts) == 2:
            jid, st = parts
            states[jid] = st.split()[0]
    return states


TERMINAL = {"COMPLETED", "FAILED", "CANCELLED", "TIMEOUT", "NODE_FAIL", "OUT_OF_MEMORY", "PREEMPTED"}


def make_spec(gkey, model, attack, base, class_pair, budget, margin, num_targets, num_victims, tag, time_limit):
    return dict(gkey=gkey, model=model, attack=attack, base=base, class_pair=class_pair,
                budget=budget, margin=list(margin), num_targets=num_targets, num_victims=num_victims,
                tag=tag, time_limit=time_limit, retries=0)


class Scheduler:
    def __init__(self, state):
        self.state = state
        self.pending = []          # list of specs not yet submitted
        self.running = {}          # job_id -> spec
        self.groups_running = set()   # (model, class_pair) cold surrogate build in flight
        self.refs_running = set()     # class_pair cold reference build in flight
        self.targets_running = set()  # (class_pair, lo, hi) cold target build in flight
        # (gkey, budget) -> {'ours': asr_or_pending, 'random': asr_or_pending}
        self.pilot_wait = {}

    def group_state(self, gkey):
        return self.state["groups"].setdefault(gkey, {"status": "not_started", "margin": None, "pilot_asr": {}})

    def margin_for(self, gkey, model):
        g = self.group_state(gkey)
        return tuple(g["margin"]) if g["margin"] else MARGIN_OVERRIDE.get(model, DEFAULT_MARGIN)

    def init_group(self, model, attack, class_pair):
        gkey = f"{model}|{attack}|{class_pair}"
        g = self.group_state(gkey)
        if g["status"] in ("done", "bad"):
            return
        all_cfgs = [(b, budget) for budget in ALL_BUDGETS for b in BASES]
        done_cfgs = [(b, budget) for (b, budget) in all_cfgs
                     if config_done(model, attack, b, class_pair, budget)]
        if len(done_cfgs) == len(all_cfgs):
            g["status"] = "done"
            log(f"[init] {gkey} already fully complete")
            return
        lo, hi = self.margin_for(gkey, model)
        if done_cfgs and g["status"] == "not_started":
            log(f"[init] {gkey} has {len(done_cfgs)}/12 configs already complete -- "
                f"pre-validated, skipping pilot")
            g["status"] = "full"
            g["margin"] = [lo, hi]
            self.enqueue_full(gkey, model, attack, class_pair, lo, hi)
            return
        if g["status"] == "not_started":
            g["status"] = "pilot_high"
            g["margin"] = [lo, hi]
            self.enqueue_pilot(gkey, model, attack, class_pair, lo, hi, 0.05)
        elif g["status"] in ("pilot_high", "pilot_mid", "pilot_low"):
            budget = {"pilot_high": 0.05, "pilot_mid": 0.01, "pilot_low": 0.001}[g["status"]]
            self.enqueue_pilot(gkey, model, attack, class_pair, lo, hi, budget)
        elif g["status"] == "full":
            self.enqueue_full(gkey, model, attack, class_pair, lo, hi)

    def enqueue_pilot(self, gkey, model, attack, class_pair, lo, hi, budget):
        # NOTE: select_targets() in main_new.py caches its target list keyed only on
        # (class_pair, margin_low, margin_high) -- it ignores num_targets once that cache
        # file exists. So pilot MUST request num_targets=10 (same as full), otherwise a
        # pilot job that's first to build a fresh margin's cache would permanently pin it
        # to 5 targets, silently capping the later "full" run at 5 targets too. Only
        # num_victims (3 vs 6) is actually safe to shrink for the pilot, since victims
        # aren't cached.
        self.pilot_wait[(gkey, budget)] = {}
        for base in BASES:
            self.pending.append(make_spec(gkey, model, attack, base, class_pair, budget, (lo, hi),
                                           10, 3, "pilot", pilot_time_for(model, attack)))

    def enqueue_full(self, gkey, model, attack, class_pair, lo, hi):
        n = 0
        for budget in ALL_BUDGETS:
            for base in BASES:
                if config_done(model, attack, base, class_pair, budget):
                    continue
                self.pending.append(make_spec(gkey, model, attack, base, class_pair, budget, (lo, hi),
                                               10, 6, "full", full_time_for(model, attack, budget)))
                n += 1
        if n == 0:
            self.group_state(gkey)["status"] = "done"
            log(f"[group done] {gkey} (no remaining full configs)")

    def dispatchable(self, spec):
        cp, lo, hi, model = spec["class_pair"], spec["margin"][0], spec["margin"][1], spec["model"]
        ref_ok = reference_warm(cp) or cp not in self.refs_running
        tgt_ok = targets_warm(cp, lo, hi) or (cp, lo, hi) not in self.targets_running
        grp_ok = surrogates_warm(model, cp) or (model, cp) not in self.groups_running
        return ref_ok and tgt_ok and grp_ok

    def dispatch(self, spec):
        cp, lo, hi, model = spec["class_pair"], spec["margin"][0], spec["margin"][1], spec["model"]
        cold_ref = not reference_warm(cp)
        cold_tgt = not targets_warm(cp, lo, hi)
        cold_grp = not surrogates_warm(model, cp)
        job_id = submit_job(spec)
        spec["_cold_ref"], spec["_cold_tgt"], spec["_cold_grp"] = cold_ref, cold_tgt, cold_grp
        if cold_ref:
            self.refs_running.add(cp)
        if cold_tgt:
            self.targets_running.add((cp, lo, hi))
        if cold_grp:
            self.groups_running.add((model, cp))
        self.running[job_id] = spec

    def on_finish(self, job_id, spec, ok, state=None):
        cp, lo, hi, model = spec["class_pair"], spec["margin"][0], spec["margin"][1], spec["model"]
        if spec.get("_cold_ref"):
            self.refs_running.discard(cp)
        if spec.get("_cold_tgt"):
            self.targets_running.discard((cp, lo, hi))
        if spec.get("_cold_grp"):
            self.groups_running.discard((model, cp))

        if not ok and spec["retries"] < MAX_JOB_RETRIES:
            spec["retries"] += 1
            if state and "OUT_OF_MEM" in state:
                old_mem = spec.get("mem", 8)
                spec["mem"] = old_mem + 8
                log(f"[retry-oom] job={job_id} {spec['tag']} "
                    f"{run_name(model, spec['attack'], spec['base'], cp, spec['budget'])} "
                    f"OOM'd at {old_mem}G, bumping to {spec['mem']}G and requeuing")
            else:
                log(f"[retry] job={job_id} {spec['tag']} {run_name(model, spec['attack'], spec['base'], cp, spec['budget'])} "
                    f"failed ({state}), requeuing (attempt {spec['retries']})")
            self.pending.append(spec)
            return

        if spec["tag"] == "pilot":
            self.handle_pilot_finish(spec)
        # "full" jobs: nothing to do per-job; group completion checked in sweep loop

    def handle_pilot_finish(self, spec):
        gkey, budget, base = spec["gkey"], spec["budget"], spec["base"]
        model, attack, class_pair = spec["model"], spec["attack"], spec["class_pair"]
        lo, hi = spec["margin"]
        asr = read_asr(model, attack, base, class_pair, budget)
        key = (gkey, budget)
        self.pilot_wait.setdefault(key, {})[base] = asr
        g = self.group_state(gkey)
        g["pilot_asr"].setdefault(f"{budget:g}", {})[base] = asr
        g["pilot_asr"][f"{budget:g}"]["margin"] = [lo, hi]
        save_state(self.state)
        log(f"[pilot result] {gkey} budget={budget:g} base={base} asr={asr}")

        wait = self.pilot_wait[key]
        if "ours" not in wait or "random" not in wait:
            return   # still waiting on the partner base job
        asr_smart, asr_random = wait["ours"], wait["random"]
        del self.pilot_wait[key]

        if budget == 0.05:
            if asr_smart is not None and asr_smart >= HIGH_BUDGET_ASR_FLOOR:
                g["status"] = "pilot_mid"
                save_state(self.state)
                self.enqueue_pilot(gkey, model, attack, class_pair, lo, hi, 0.01)
            else:
                attempts = g.get("margin_attempts", 0) + 1
                g["margin_attempts"] = attempts
                if attempts > MAX_MARGIN_ATTEMPTS:
                    g["status"] = "bad"
                    save_state(self.state)
                    bad = load_bad()
                    bad[gkey] = {"reason": "high-budget ASR still below floor after margin widening",
                                 "pilot_asr": g["pilot_asr"], "last_margin_tried": [lo, hi]}
                    save_bad(bad)
                    log(f"[BAD] {gkey} flagged for manual review after {attempts-1} margin-widening attempts")
                else:
                    new_hi = min(1.0, hi * 2 if hi > 0 else 0.1)
                    new_lo = 0.0
                    log(f"[margin-widen] {gkey} ASR too low at margin=[{lo:g},{hi:g}] (smart_asr={asr_smart}), "
                        f"retrying with margin=[{new_lo:g},{new_hi:g}]")
                    g["margin"] = [new_lo, new_hi]
                    save_state(self.state)
                    self.enqueue_pilot(gkey, model, attack, class_pair, new_lo, new_hi, 0.05)
        elif budget == 0.01:
            if asr_smart is not None and asr_smart < MID_BUDGET_ASR_FLOOR:
                log(f"[pilot] {gkey} already low at mid budget (expected trend) -- skipping low-budget "
                    f"pilot check, proceeding to full sweep")
                g["status"] = "full"
                save_state(self.state)
                self.enqueue_full(gkey, model, attack, class_pair, lo, hi)
            else:
                g["status"] = "pilot_low"
                save_state(self.state)
                self.enqueue_pilot(gkey, model, attack, class_pair, lo, hi, 0.001)
        elif budget == 0.001:
            g["status"] = "full"
            save_state(self.state)
            self.enqueue_full(gkey, model, attack, class_pair, lo, hi)

    def check_group_completion(self):
        for gkey, g in self.state["groups"].items():
            if g["status"] != "full":
                continue
            model, attack, class_pair = gkey.split("|")
            lo, hi = tuple(g["margin"])
            still_pending = any(s["gkey"] == gkey for s in self.pending)
            still_running = any(s["gkey"] == gkey for s in self.running.values())
            if still_pending or still_running:
                continue
            all_done = all(config_done(model, attack, b, class_pair, budget)
                            for budget in ALL_BUDGETS for b in BASES)
            if all_done:
                g["status"] = "done"
                save_state(self.state)
                log(f"[group done] {gkey} margin=[{lo:g},{hi:g}]")

    def run(self):
        for class_pair in CLASS_PAIRS:
            for attack in ATTACKS:
                for model in MODELS:
                    self.init_group(model, attack, class_pair)
        save_state(self.state)

        while self.pending or self.running:
            free = MAX_CONCURRENT - len(self.running)
            i = 0
            launched_this_round = 0
            while i < len(self.pending) and launched_this_round < free:
                spec = self.pending[i]
                if self.dispatchable(spec):
                    self.pending.pop(i)
                    self.dispatch(spec)
                    launched_this_round += 1
                else:
                    i += 1

            time.sleep(POLL_SECS)
            if self.running:
                states = batch_job_states(list(self.running.keys()))
                for job_id in list(self.running.keys()):
                    st = states.get(job_id)
                    if st and any(st.startswith(t) for t in TERMINAL):
                        spec = self.running.pop(job_id)
                        ok = st.startswith("COMPLETED")
                        log(f"[finish] job={job_id} state={st} {spec['tag']} "
                            f"{run_name(spec['model'], spec['attack'], spec['base'], spec['class_pair'], spec['budget'])}")
                        self.on_finish(job_id, spec, ok, state=st)
            self.check_group_completion()

        log("=== ALL GROUPS PROCESSED ===")
        bad = load_bad()
        if bad:
            log(f"{len(bad)} group(s) need manual review: {list(bad.keys())}")


def main():
    state = load_state()
    Scheduler(state).run()


if __name__ == "__main__":
    main()
