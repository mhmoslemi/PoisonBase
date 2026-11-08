"""Focused ConvNet experiment for the first rebuttal table."""
import argparse
import gc
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(os.environ.get("ACCESS_ROOT", Path(__file__).resolve().parents[2]))
CODE = Path(__file__).resolve().parent
sys.path.insert(0, str(CODE))

import numpy as np
import torch
import final_update as fu

OUT = ROOT / "rebuttal_access_conv_5x4_result"
MODEL = "ConvNetBN"
TARGETS = (843, 1257, 8252, 8730, 8055)
VICTIMS = range(4)
SURROGATE_DRAWS = range(3)
CANDIDATE_DRAWS = range(3)

# (rho, surrogate-data fraction, candidate-pool fraction)
CONDITIONS = (
    (0.001, 0.2, 0.2),
    (0.002, 0.5, 1.0),
    (0.002, 0.2, 1.0),
    (0.002, 1.0, 0.5),
    (0.002, 1.0, 0.2),
    (0.002, 0.5, 0.5),
    (0.002, 0.2, 0.2),
    (0.005, 0.2, 0.2),
)


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".{os.getpid()}.tmp")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(tmp, path)


def save_torch(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".{os.getpid()}.tmp")
    torch.save(value, tmp)
    os.replace(tmp, path)


def context():
    data = Path(os.environ.get("REVIEW_DATA", ROOT / "data"))
    args = fu.parse_args([
        "--model", MODEL, "--dataset", "CIFAR10", "--data_path", str(data),
        "--seed", "42", "--attack", "gradmatch", "--class_pair", "dog-bird",
        "--num_surrogates", "20", "--craft_ensemble", "5",
        "--craft_steps", "250", "--restarts", "8", "--epsilon", str(8 / 255),
        "--craft_alpha", str(1 / 255), "--base_dist", "cosine",
        "--lambda_margin", "1", "--surrogate_epochs", "60",
        "--surrogate_bs", "128", "--surrogate_decay", "35", "45",
        "--victim_epochs", "50", "--victim_bs", "125",
        "--victim_decay", "40", "--num_victims", "4",
        "--num_targets", "5", "--gpus", "0", "--no_parallel_targets",
        "--no_parallel_pretrain",
    ])
    ctx = fu.build_context(args, torch.device("cuda:0"))
    assert len(ctx["train_labs"]) == 50000
    assert all(int(ctx["test_labs"][target]) == 2 for target in TARGETS)
    return args, ctx


def build(ctx, seed):
    return fu.build_network(MODEL, 3, 10, (32, 32), ctx["device"], seed=seed)


def visible_subsets(labels, draw):
    labels = np.asarray(labels)
    rng = np.random.default_rng(7100 + draw)
    orders = [rng.permutation(np.flatnonzero(labels == cls)) for cls in range(10)]
    return {
        fraction: np.sort(np.concatenate([
            order[:round(len(order) * fraction)] for order in orders
        ]))
        for fraction in (1.0, 0.5, 0.2)
    }


def partial_pool(fraction, draw):
    base = (ROOT / "reviewer_20261106_fir" / "reviewer_20261106_result" /
            "surrogates")
    matches = list(base.glob(f"{MODEL}_f{fraction}_r{draw}_*"))
    if len(matches) != 1:
        raise RuntimeError(f"expected one surrogate pool for f={fraction}, r={draw}; got {matches}")
    return matches[0], list(range(20))


def pool_spec(fraction, draw, random_crafter=False):
    if random_crafter:
        directory = ROOT / "cache" / "surrogates" / f"{MODEL}_60ep_lr0.1_bs128_seed42"
        return directory, list(range(5))
    if fraction == 1.0:
        directory = ROOT / "cache" / "surrogates" / f"{MODEL}_60ep_lr0.1_bs128_seed42"
        # Three independently initialized 20-model ensembles on the full set.
        return directory, list(range(draw * 5, draw * 5 + 20))
    return partial_pool(fraction, draw)


def load_nets(ctx, fraction, draw, random_crafter=False):
    directory, model_ids = pool_spec(fraction, draw, random_crafter)
    nets = []
    for model_id in model_ids:
        path = directory / f"net_{model_id}.pt"
        if not path.is_file():
            raise RuntimeError(f"missing surrogate checkpoint: {path}")
        net = build(ctx, 1042 + model_id)
        net.load_state_dict(torch.load(path, map_location=ctx["device"], weights_only=True))
        nets.append(net.eval())
    return nets


def candidate_indices(ctx, fraction, draw):
    labels = ctx["train_labs"].cpu().numpy()
    visible = visible_subsets(labels, draw)[fraction]
    return torch.tensor(visible[labels[visible] == 5], device=ctx["device"])


def poison_path(method, condition_id, surrogate_draw, candidate_draw, target):
    sname = "fixed" if surrogate_draw is None else str(surrogate_draw)
    return (OUT / method / f"condition_{condition_id}" / "poisons" /
            f"s{sname}_c{candidate_draw}_t{target}.pt")


def make_poison(args, ctx, method, condition_id, condition, surrogate_draw,
                candidate_draw, target, nets):
    path = poison_path(method, condition_id, surrogate_draw, candidate_draw, target)
    if path.exists():
        return torch.load(path, map_location="cpu", weights_only=False)

    budget, _, pool_fraction = condition
    candidate = candidate_indices(ctx, pool_fraction, candidate_draw)
    count = round(50000 * budget)
    seed_sdraw = 0 if surrogate_draw is None else surrogate_draw
    seed = 91000000 + condition_id * 100000 + seed_sdraw * 10000 + candidate_draw * 1000 + target
    fu.set_seed(seed)

    start = time.monotonic()
    if method == "random":
        selected = candidate[torch.randperm(len(candidate), device=ctx["device"])[:count]]
    else:
        selection_labels = torch.full_like(ctx["train_labs"], -1)
        selection_labels[candidate] = 5
        coefficient = {"margin": 0.0, "basis": 0.5}[method]
        selected = fu.select_base_ours(
            nets, ctx["train_imgs"], selection_labels, ctx["test_imgs"][target],
            5, count, 1.0, ctx["device"], base_dist="cosine",
            distance_margin_coef=coefficient,
        )
    selection_seconds = time.monotonic() - start

    # Reset so the crafting randomness is paired across selection methods.
    fu.set_seed(seed)
    start = time.monotonic()
    base, adversarial, objective = fu.craft_selected_bases(
        args, ctx, nets[:5], selected, ctx["test_imgs"][target], 5
    )
    delta = (adversarial - base).detach().cpu()
    assert len(selected.unique()) == count
    assert torch.isfinite(delta).all() and float(delta.abs().max()) <= 8 / 255 + 1e-6
    result = {
        "method": method,
        "condition_id": condition_id,
        "condition": condition,
        "surrogate_draw": surrogate_draw,
        "candidate_draw": candidate_draw,
        "target": target,
        "indices": selected.cpu(),
        "delta": delta,
        "objective": float(objective),
        "selection_seconds": selection_seconds,
        "craft_seconds": time.monotonic() - start,
    }
    save_torch(path, result)
    print(f"crafted {method} condition={condition_id} s={surrogate_draw} "
          f"c={candidate_draw} target={target}", flush=True)
    return result


def train_victims(ctx, method, condition_id, condition, surrogate_draw,
                  candidate_draw, target, poison):
    sname = "fixed" if surrogate_draw is None else str(surrogate_draw)
    directory = (OUT / method / f"condition_{condition_id}" / "trials" /
                 f"s{sname}_c{candidate_draw}_t{target}")
    indices = poison["indices"].to(ctx["device"])
    clean = ctx["train_imgs"][indices].clone()
    original01 = ctx["denorm"](clean).clamp(0, 1)
    ctx["train_imgs"][indices] = ctx["norm"](
        (original01 + poison["delta"].to(ctx["device"])).clamp(0, 1)
    )
    try:
        for victim in VICTIMS:
            path = directory / f"victim_{victim}.json"
            if path.exists():
                continue
            seed = 42 * 100000 + target * 100 + victim
            net = build(ctx, seed)
            start = time.monotonic()

            def progress(epoch, total):
                if epoch % 10 == 0 or epoch == total:
                    print(f"{method} condition={condition_id} s={surrogate_draw} "
                          f"c={candidate_draw} target={target} victim={victim} "
                          f"epoch={epoch}/{total}", flush=True)

            fu.train_from_scratch(
                net, ctx["train_imgs"], ctx["train_labs"], 50, 0.1, 125, [40],
                ctx["device"], progress_callback=progress,
            )
            prediction = int(net(ctx["test_imgs"][target].unsqueeze(0)).argmax(1).item())
            clean_accuracy = fu.test_acc(net, ctx["test_imgs"], ctx["test_labs"], bs=256)
            atomic_json(path, {
                "method": method,
                "condition_id": condition_id,
                "condition": condition,
                "surrogate_draw": surrogate_draw,
                "candidate_draw": candidate_draw,
                "target": target,
                "victim": victim,
                "seed": seed,
                "epochs": 50,
                "prediction": prediction,
                "success": int(prediction == 5),
                "clean_accuracy": clean_accuracy,
                "seconds": time.monotonic() - start,
            })
            del net
            gc.collect()
            torch.cuda.empty_cache()
    finally:
        ctx["train_imgs"][indices] = clean


def combinations(method):
    if method == "random":
        return ((None, candidate_draw) for candidate_draw in CANDIDATE_DRAWS)
    return ((surrogate_draw, candidate_draw)
            for surrogate_draw in SURROGATE_DRAWS
            for candidate_draw in CANDIDATE_DRAWS)


def summarize(method, condition_id):
    condition = CONDITIONS[condition_id]
    rows = []
    for surrogate_draw, candidate_draw in combinations(method):
        sname = "fixed" if surrogate_draw is None else str(surrogate_draw)
        for target in TARGETS:
            directory = (OUT / method / f"condition_{condition_id}" / "trials" /
                         f"s{sname}_c{candidate_draw}_t{target}")
            for victim in VICTIMS:
                path = directory / f"victim_{victim}.json"
                if not path.exists():
                    raise RuntimeError(f"missing result: {path}")
                rows.append(json.loads(path.read_text()))
    expected = 60 if method == "random" else 180
    assert len(rows) == expected
    atomic_json(OUT / method / f"condition_{condition_id}" / "summary.json", {
        "method": method,
        "condition_id": condition_id,
        "condition": condition,
        "targets": TARGETS,
        "victims": list(VICTIMS),
        "trials": len(rows),
        "asr": sum(row["success"] for row in rows) / len(rows),
        "clean_accuracy": sum(row["clean_accuracy"] for row in rows) / len(rows),
    })


def run_cell(args, ctx, method, condition_id):
    condition = CONDITIONS[condition_id]
    args.budget = condition[0]
    _, surrogate_fraction, _ = condition
    nets = None
    loaded_draw = None
    for surrogate_draw, candidate_draw in combinations(method):
        missing_poison = any(not poison_path(
            method, condition_id, surrogate_draw, candidate_draw, target
        ).exists() for target in TARGETS)
        wanted_draw = "fixed" if surrogate_draw is None else surrogate_draw
        if missing_poison and loaded_draw != wanted_draw:
            if nets is not None:
                del nets
                gc.collect()
                torch.cuda.empty_cache()
            nets = load_nets(
                ctx, surrogate_fraction, 0 if surrogate_draw is None else surrogate_draw,
                random_crafter=(method == "random"),
            )
            loaded_draw = wanted_draw
        for target in TARGETS:
            poison = make_poison(
                args, ctx, method, condition_id, condition, surrogate_draw,
                candidate_draw, target, nets,
            )
            train_victims(
                ctx, method, condition_id, condition, surrogate_draw,
                candidate_draw, target, poison,
            )
    if nets is not None:
        del nets
        gc.collect()
        torch.cuda.empty_cache()
    summarize(method, condition_id)
    print(f"complete method={method} condition={condition_id}", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("random-pair", "method-cell"), required=True)
    parser.add_argument("--index", type=int, required=True)
    parser.add_argument("--method", choices=("margin", "basis"))
    cli = parser.parse_args()
    args, ctx = context()
    if cli.mode == "random-pair":
        if cli.index not in range(4):
            parser.error("random-pair index must be 0..3")
        for condition_id in (2 * cli.index, 2 * cli.index + 1):
            run_cell(args, ctx, "random", condition_id)
    else:
        if cli.index not in range(len(CONDITIONS)):
            parser.error("method-cell index must be 0..7")
        if cli.method is None:
            parser.error("method-cell requires --method margin or --method basis")
        run_cell(args, ctx, cli.method, cli.index)


if __name__ == "__main__":
    main()
