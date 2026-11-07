"""Fill the 16 Wicked Oddities cells in the rebuttal Part 2 table."""
import argparse
import fcntl
import gc
import json
import os
from pathlib import Path
import resource
import shutil
import sys
import time

ROOT = Path(os.environ.get("WO_ROOT", Path(__file__).resolve().parents[2]))
SWEEP = ROOT / "experiments" / "reviewer_sweep"
sys.path[:0] = [str(ROOT), str(SWEEP)]

import torch
import final_update as fu
from protocol import atomic_json
from defenses import save_torch

OUT = ROOT / "rebuttal_wo_table_result"
TARGETS = {
    "ConvNetBN": (843, 1257, 8252, 8730, 8055, 5859, 8954, 821),
    "ResNet20BN": (5118, 8740, 5324, 4630, 833, 2065, 7056, 9059),
}
GROUPS = {
    "conv-all": [("ConvNetBN", selector, attack, budget)
                 for selector in ("wo_knn", "wo_ood")
                 for attack in ("gradmatch", "sapa")
                 for budget in (0.002, 0.005)],
    "res-b002": [("ResNet20BN", selector, attack, 0.002)
                  for selector in ("wo_knn", "wo_ood")
                  for attack in ("gradmatch", "sapa")],
    "res-b005-knn": [("ResNet20BN", "wo_knn", attack, 0.005)
                     for attack in ("gradmatch", "sapa")],
    "res-b005-ood": [("ResNet20BN", "wo_ood", attack, 0.005)
                     for attack in ("gradmatch", "sapa")],
}


def attack_args(model, attack):
    values = [
        "--model", model, "--dataset", "CIFAR10",
        "--data_path", os.environ.get("WO_DATA", str(ROOT / "data")),
        "--seed", "42", "--attack", attack, "--class_pair", "dog-bird",
        "--num_surrogates", "20", "--craft_ensemble", "5",
        "--craft_steps", "250", "--restarts", "8",
        "--epsilon", str(8 / 255), "--craft_alpha", str(1 / 255),
        "--base_dist", "cosine", "--lambda_margin", "1",
        "--surrogate_epochs", "60", "--surrogate_bs", "128",
        "--surrogate_decay", "35", "45",
        "--victim_epochs", "50", "--victim_bs", "125",
        "--victim_decay", "40", "--num_victims", "5",
        "--num_targets", "8", "--gpus", "0",
        "--no_parallel_targets", "--no_parallel_pretrain",
    ]
    if attack == "sapa":
        values += ["--sharp_mode", "worst", "--sharp_sigma", "0.05"]
    return fu.parse_args(values)


def context(model, attack):
    args = attack_args(model, attack)
    ctx = fu.build_context(args, torch.device("cuda:0"))
    assert len(ctx["train_labs"]) == 50000
    assert all(int(ctx["test_labs"][target]) == 2 for target in TARGETS[model])
    return args, ctx


def load_pool(model, ctx):
    directory = ROOT / "cache" / "surrogates" / f"{model}_60ep_lr0.1_bs128_seed42"
    nets = []
    for index in range(20):
        net = fu.build_network(model, 3, 10, (32, 32), ctx["device"], seed=1042 + index)
        state = torch.load(directory / f"net_{index}.pt", map_location=ctx["device"],
                           weights_only=True)
        net.load_state_dict(state)
        nets.append(net.eval())
    return nets


def validate_ranking(path):
    value = torch.load(path, map_location="cpu", weights_only=False)
    indices = torch.as_tensor(value["indices"])
    scores = torch.as_tensor(value["scores"])
    assert indices.ndim == scores.ndim == 1 and len(indices) == len(scores) == 5000
    assert len(indices.unique()) == 5000 and int(indices.min()) >= 0 and int(indices.max()) < 50000
    assert torch.isfinite(scores).all()
    return value


def prep():
    sources = {
        "wo_knn.pt": ROOT / "reviewer_20261106_result" / "wicked" / "wo_pretrained.pt",
        "wo_ood.pt": ROOT / "reviewer_20261106_fir" / "reviewer_20261106_result" / "wicked" / "wo_ood.pt",
    }
    destination = OUT / "rankings"
    destination.mkdir(parents=True, exist_ok=True)
    for name, source in sources.items():
        validate_ranking(source)
        temporary = destination / f".{name}.{os.getpid()}.tmp"
        shutil.copyfile(source, temporary)
        os.replace(temporary, destination / name)
        validate_ranking(destination / name)
        print("STAGED", name, "from", source, flush=True)


def cell_directory(model, selector, attack, budget):
    return OUT / "cells" / model / selector / attack / f"rho_{budget:g}"


def poison_for(cell, tidx, args, ctx, craft_nets, ranking):
    model, selector, attack, budget = cell
    directory = cell_directory(*cell) / "poisons"
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"target_{tidx}.pt"
    if path.exists():
        return torch.load(path, map_location="cpu", weights_only=False)

    n = round(50000 * budget)
    indices = ranking["indices"].to(ctx["device"])
    scores = ranking["scores"].to(ctx["device"])
    order = torch.argsort(scores, descending=True, stable=True)
    selected = indices[order[:n]]
    assert len(selected.unique()) == n
    assert bool((ctx["train_labs"][selected] == 5).all())

    seed = 42 * 100003 + tidx
    fu.set_seed(seed)
    before = time.monotonic()
    base, adversarial, objective = fu.craft_selected_bases(
        args, ctx, craft_nets[:5], selected, ctx["test_imgs"][tidx], 5)
    delta = (adversarial - base).detach().cpu()
    assert torch.isfinite(delta).all() and float(delta.abs().max()) <= 8 / 255 + 1e-6
    value = {
        "model": model, "selector": selector, "attack": attack, "budget": budget,
        "target": tidx, "indices": selected.cpu(), "delta": delta,
        "objective": float(objective), "seconds": time.monotonic() - before,
    }
    save_torch(path, value)
    print("CRAFTED", model, selector, attack, budget, tidx,
          f"{value['seconds']:.1f}s", flush=True)
    return value


def summarize(directory):
    rows = [json.loads(path.read_text()) for path in sorted(directory.glob("targets/*/victim_*.json"))]
    summary = {
        "completed": len(rows), "expected": 40,
        "asr_percent": 100 * sum(row["success"] for row in rows) / len(rows) if rows else None,
        "mean_clean_accuracy": sum(row["clean_accuracy"] for row in rows) / len(rows) if rows else None,
    }
    atomic_json(directory / "summary.json", summary)
    return summary


def run_cell(cell, args, ctx, nets, ranking):
    model, selector, attack, budget = cell
    directory = cell_directory(*cell)
    directory.mkdir(parents=True, exist_ok=True)
    atomic_json(directory / "protocol.json", {
        "model": model, "selector": selector, "attack": attack, "budget": budget,
        "targets": TARGETS[model], "victims_per_target": 5,
        "victim_epochs": 50, "victim_batch_size": 125, "victim_decay": [40],
        "surrogates": 20, "crafting_surrogates": 5,
        "surrogate_epochs": 60, "craft_steps": 250, "restarts": 8,
        "epsilon": 8 / 255, "craft_alpha": 1 / 255,
        "sapa_mode": "worst" if attack == "sapa" else None,
        "sapa_sigma": 0.05 if attack == "sapa" else None,
    })
    args.budget = budget

    for tidx in TARGETS[model]:
        target_dir = directory / "targets" / str(tidx)
        target_dir.mkdir(parents=True, exist_ok=True)
        pending = [victim for victim in range(5)
                   if not (target_dir / f"victim_{victim}.json").exists()]
        if not pending:
            print("DONE TARGET", model, selector, attack, budget, tidx, flush=True)
            continue
        poison = poison_for(cell, tidx, args, ctx, nets, ranking)
        selected = poison["indices"].to(ctx["device"])
        clean = ctx["train_imgs"][selected].clone()
        original = ctx["denorm"](clean).clamp(0, 1)
        ctx["train_imgs"][selected] = ctx["norm"](
            (original + poison["delta"].to(ctx["device"])).clamp(0, 1))
        try:
            for victim in pending:
                before = time.monotonic()
                seed = 42 * 100000 + tidx * 100 + victim
                net = fu.build_network(model, 3, 10, (32, 32), ctx["device"], seed=seed)

                def progress(epoch, total):
                    if epoch == 1 or epoch % 10 == 0 or epoch == total:
                        print("EPOCH", model, selector, attack, budget, tidx,
                              f"victim={victim}", f"{epoch}/{total}", flush=True)

                fu.train_from_scratch(net, ctx["train_imgs"], ctx["train_labs"],
                                      50, 0.1, 125, [40], ctx["device"],
                                      progress_callback=progress)
                prediction = fu.predict_target(net, ctx["test_imgs"][tidx])
                result = {
                    "model": model, "selector": selector, "attack": attack,
                    "budget": budget, "target": tidx, "victim": victim,
                    "seed": seed, "prediction": prediction,
                    "success": int(prediction == 5),
                    "clean_accuracy": fu.test_acc(net, ctx["test_imgs"], ctx["test_labs"]),
                    "seconds": time.monotonic() - before,
                    "max_rss_kb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                }
                atomic_json(target_dir / f"victim_{victim}.json", result)
                del net
                torch.cuda.empty_cache()
                summary = summarize(directory)
                print("RESULT", model, selector, attack, budget, tidx, victim,
                      result["success"], f"ASR={summary['asr_percent']:.2f}", flush=True)
        finally:
            ctx["train_imgs"][selected] = clean
        gc.collect()
    print("CELL COMPLETE", cell, summarize(directory), flush=True)


def run_group(name):
    lock_path = OUT / "locks" / f"{name}.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        current = None
        args = ctx = nets = None
        rankings = {}
        for cell in GROUPS[name]:
            model, selector, attack, _ = cell
            if current != (model, attack):
                del nets, ctx, args
                gc.collect()
                torch.cuda.empty_cache()
                args, ctx = context(model, attack)
                nets = load_pool(model, ctx)
                current = (model, attack)
            if selector not in rankings:
                rankings[selector] = validate_ranking(OUT / "rankings" / f"{selector}.pt")
            run_cell(cell, args, ctx, nets, rankings[selector])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["prep", *GROUPS])
    values = parser.parse_args()
    assert os.environ.get("SLURM_JOB_ID"), "Run only inside a Slurm job"
    if values.mode == "prep":
        prep()
        return
    assert torch.cuda.is_available(), "A full GPU is required"
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.backends.cudnn.benchmark = True
    OUT.mkdir(parents=True, exist_ok=True)
    print("START", values.mode, torch.__version__, torch.cuda.get_device_name(), flush=True)
    run_group(values.mode)


if __name__ == "__main__":
    main()
