"""Reviewer 2 W4: first-m and five fresh random draws, with persistent resume."""
import argparse
import csv
import fcntl
import gc
import hashlib
import json
import os
from pathlib import Path
import random
import signal
import time

import numpy as np
import torch
import torch.nn.functional as F

import final_update as fu

STOP = False
DEADLINE = float('inf')


class Paused(RuntimeError):
    pass


def request_stop(signum, frame):
    global STOP
    STOP = True
    print(f'Signal {signum}: checkpointing at the next step/epoch.', flush=True)


def stopping():
    return STOP or time.time() >= DEADLINE


def check_time():
    if stopping():
        raise Paused('Progress saved. Rerun submit_w4_ghaddar.sh to resume.')


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f'.{os.getpid()}.tmp')
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n')
    os.replace(tmp, path)


def atomic_torch(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f'.{os.getpid()}.tmp')
    torch.save(value, tmp)
    os.replace(tmp, path)


def rng_state():
    return dict(cpu=torch.get_rng_state(), numpy=np.random.get_state(),
                python=random.getstate(),
                cuda=torch.cuda.get_rng_state() if torch.cuda.is_available() else None)


def restore_rng(state):
    torch.set_rng_state(state['cpu'].cpu())
    np.random.set_state(state['numpy'])
    random.setstate(state['python'])
    if state['cuda'] is not None:
        torch.cuda.set_rng_state(state['cuda'].cpu())


def choose_bases(labels, method, count, target, draw):
    candidates = (labels == 5).nonzero(as_tuple=True)[0]
    if method == 'first_m':
        return candidates[:count]
    if method != 'random' or draw not in range(1, 6):
        raise ValueError('Only first-m and independent random draws 1..5 are new work.')
    seed = 42 * 100003 + target + draw * 100000000
    generator = torch.Generator(device='cpu').manual_seed(seed)
    return fu.select_base_random(labels, 5, count, labels.device, generator)


def fit_victim(net, images, labels, settings, directory, identity):
    """Same SGD and sample order as final_update.train_from_scratch; checkpointed."""
    checkpoint = directory / 'training.pt'
    opt = torch.optim.SGD(net.parameters(), lr=settings['learning_rate'],
                          momentum=settings['momentum'], weight_decay=settings['weight_decay'])
    start = 0
    if checkpoint.exists():
        state = torch.load(checkpoint, map_location='cpu', weights_only=False)
        if state['identity'] != identity:
            raise RuntimeError(f'Incompatible victim checkpoint: {checkpoint}')
        net.load_state_dict(state['net'])
        opt.load_state_dict(state['optimizer'])
        start = state['epoch']
        restore_rng(state['rng'])
        del state
    print(f'VICTIM {identity["target"]}/{identity["victim"]}: '
          f'{start}/{settings["epochs"]} epochs already complete.', flush=True)
    net.train()
    for ep in range(start, settings['epochs']):
        if ep in settings['decay']:
            for group in opt.param_groups:
                group['lr'] *= 0.1
        perm = torch.randperm(len(labels), device=images.device)
        for ids in perm.split(settings['batch_size']):
            opt.zero_grad(set_to_none=True)
            F.cross_entropy(net(images[ids]), labels[ids]).backward()
            opt.step()
        if (ep + 1) % 10 == 0 or ep + 1 == settings['epochs'] or stopping():
            atomic_torch(checkpoint, dict(identity=identity, epoch=ep + 1,
                         net=net.state_dict(), optimizer=opt.state_dict(), rng=rng_state()))
            print(f'VICTIM {identity["target"]}/{identity["victim"]}: '
                  f'epoch {ep + 1}/{settings["epochs"]}', flush=True)
        check_time()
    return net.eval()


def poison(args, ctx, nets, method, unit, directory, identity):
    count = round(len(ctx['train_labs']) * args.budget)
    indices = choose_bases(ctx['train_labs'], method, count, unit['target'], unit['draw'])
    if len(indices) != count or len(indices.unique()) != count:
        raise RuntimeError('Invalid base selection.')
    selected = indices.cpu().tolist()
    base_file = directory / 'bases.json'
    bases = dict(identity=identity, indices=selected)
    if base_file.exists() and json.loads(base_file.read_text()) != bases:
        raise RuntimeError('Base selection changed on resume.')
    atomic_json(base_file, bases)
    output, checkpoint = directory / 'poison.pt', directory / 'crafting.pt'
    if output.exists():
        saved = torch.load(output, map_location='cpu', weights_only=False)
        if saved['identity'] != identity or saved['indices'] != selected:
            raise RuntimeError('Poison cache does not match this draw.')
        print(f'SKIP crafting: {method} draw={unit["draw"]} target={unit["target"]}', flush=True)
        delta = saved['delta'].to(ctx['device'])
    else:
        fu.set_seed(42 * 100003 + unit['target'])
        state = torch.load(checkpoint, map_location='cpu', weights_only=False) if checkpoint.exists() else None
        if state and state['identity'] != identity:
            raise RuntimeError('Crafting checkpoint does not match this draw.')
        if state:
            np.random.set_state(state['numpy'])
            random.setstate(state['python'])
            print(f'CRAFT resume: restart {state["state"]["restart"] + 1}/{args.restarts}, '
                  f'step {state["state"]["next_step"]}/{args.craft_steps}', flush=True)
        base01 = ctx['denorm'](ctx['train_imgs'][indices]).clamp(0, 1).detach()

        def save_craft(progress):
            if progress['next_step'] % 50 == 0 or stopping():
                atomic_torch(checkpoint, dict(identity=identity, state=progress,
                             numpy=np.random.get_state(), python=random.getstate()))
                print(f'CRAFT draw={unit["draw"]} target={unit["target"]}: '
                      f'restart {progress["restart"] + 1}/{args.restarts}, '
                      f'step {progress["next_step"]}/{args.craft_steps}', flush=True)
                check_time()

        began = time.monotonic()
        adv, objective = fu.craft_gradmatch(
            nets, base01, ctx['test_imgs'][unit['target']], 5, ctx['norm'],
            args.epsilon, args.craft_alpha, args.craft_steps, args.restarts, ctx['device'],
            dsa_strategy=args.dsa_strategy, dsa_param=ctx['dsa_param'], chunk=args.craft_batch,
            sharp_mode='worst' if args.attack == 'sapa' else None,
            sharp_sigma=args.sharp_sigma, sharp_samples=args.sharp_samples,
            resume_state=state['state'] if state else None, checkpoint_callback=save_craft)
        delta = (adv - base01).detach()
        atomic_torch(output, dict(identity=identity, indices=selected,
                     delta=delta.cpu(), objective=float(objective)))
        checkpoint.unlink(missing_ok=True)
        print(f'CRAFTED {args.model} {args.attack} {args.budget} {method} '
              f'draw={unit["draw"]} target={unit["target"]}: '
              f'{time.monotonic() - began:.1f}s', flush=True)
    if tuple(delta.shape) != (count, 3, 32, 32) or not torch.isfinite(delta).all():
        raise RuntimeError('Invalid saved perturbations.')
    if float(delta.abs().max()) > args.epsilon + 1e-6:
        raise RuntimeError('Perturbations exceed epsilon.')
    base01 = ctx['denorm'](ctx['train_imgs'][indices]).clamp(0, 1)
    return indices, (base01 + delta).clamp(0, 1)


def unit_path(root, method, cell, unit):
    return root / method / cell['id'] / f'draw_{unit["draw"]}' / f'target_{unit["target"]}'


def evaluate_unit(args, ctx, nets, task, cell, unit, results, protocol, protocol_hash):
    directory = unit_path(results, task['method'], cell, unit)
    directory.mkdir(parents=True, exist_ok=True)
    identity = dict(protocol_hash=protocol_hash, cell=cell['id'], model=cell['model'],
                    attack=cell['attack'], budget=cell['budget'], method=task['method'], **unit)
    paths = [directory / f'victim_{v}.json' for v in range(6)]
    if all(path.exists() for path in paths):
        if any(json.loads(path.read_text())['identity'] != dict(identity, victim=v)
               for v, path in enumerate(paths)):
            raise RuntimeError('Completed victim identity mismatch.')
        print(f'SKIP completed target: {identity}', flush=True)
        return
    check_time()
    indices, adv = poison(args, ctx, nets, task['method'], unit, directory, identity)
    original = ctx['train_imgs'][indices].clone()
    ctx['train_imgs'][indices] = ctx['norm'](adv)
    del adv
    try:
        for victim, path in enumerate(paths):
            trial_identity = dict(identity, victim=victim)
            if path.exists():
                if json.loads(path.read_text())['identity'] != trial_identity:
                    raise RuntimeError('Completed victim identity mismatch.')
                print(f'SKIP complete: draw={unit["draw"]} target={unit["target"]} '
                      f'victim={victim}', flush=True)
                continue
            check_time()
            seed = 4200000 + unit['target'] * 100 + victim
            net = fu.build_network(args.model, 3, 10, (32, 32), ctx['device'], seed=seed)
            trial_dir = directory / f'victim_{victim}'
            began = time.monotonic()
            fit_victim(net, ctx['train_imgs'], ctx['train_labs'], protocol['victim'],
                       trial_dir, trial_identity)
            prediction = fu.predict_target(net, ctx['test_imgs'][unit['target']])
            row = dict(identity=trial_identity, seed=seed, success=int(prediction == 5),
                       prediction=prediction, clean_accuracy=fu.test_acc(net, ctx['test_imgs'], ctx['test_labs']),
                       seconds_this_attempt=time.monotonic() - began)
            atomic_json(path, row)
            (trial_dir / 'training.pt').unlink(missing_ok=True)
            print('RESULT', json.dumps(row, sort_keys=True), flush=True)
            del net
            gc.collect()
            torch.cuda.empty_cache()
    finally:
        ctx['train_imgs'][indices] = original


def read_baseline(batch, cell):
    with (batch / cell['baseline']['csv']).open(newline='') as handle:
        rows = list(csv.DictReader(handle))
    expected = {(target, victim) for target in cell['targets'] for victim in range(6)}
    actual = [(int(row['target_idx']), int(row['victim_id'])) for row in rows]
    if len(rows) != 60 or set(actual) != expected:
        raise RuntimeError(f'Incomplete original draw: {cell["id"]}')
    if any(row['model'] != cell['model'] or row['attack'] != cell['attack']
           or float(row['budget']) != cell['budget'] or int(row['seed']) != 42
           or row['success'] not in ('0', '1') for row in rows):
        raise RuntimeError(f'Wrong original-draw metadata: {cell["id"]}')
    successes = sum(int(row['success']) for row in rows)
    if successes != cell['baseline']['successes']:
        raise RuntimeError(f'Original draw does not reproduce the paper: {cell["id"]}')
    return dict(draw=0, finished=60, expected=60, successes=successes,
                source=cell['baseline']['csv'], asr_percent=100 * successes / 60,
                cta_percent=100 * sum(float(row['clean_test_acc']) for row in rows) / 60)


def aggregate_cell(results, task, cell, protocol_hash, batch):
    summary_dir = results / 'summaries'
    summary_dir.mkdir(parents=True, exist_ok=True)
    name = f'{task["method"]}_{cell["id"]}'
    with (summary_dir / f'{name}.lock').open('a') as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        draws = []
        if task['method'] == 'random':
            draws.append(read_baseline(batch, cell))
        for draw in ([0] if task['method'] == 'first_m' else range(1, 6)):
            rows = []
            for target in cell['targets']:
                directory = unit_path(results, task['method'], cell, dict(draw=draw, target=target))
                for victim in range(6):
                    path = directory / f'victim_{victim}.json'
                    if path.exists():
                        row = json.loads(path.read_text())
                        expected = dict(protocol_hash=protocol_hash, cell=cell['id'], model=cell['model'],
                                        attack=cell['attack'], budget=cell['budget'], method=task['method'],
                                        draw=draw, target=target, victim=victim)
                        if row['identity'] != expected or row['success'] not in (0, 1):
                            raise RuntimeError(f'Invalid aggregation input: {path}')
                        rows.append(row)
            result = dict(draw=draw, finished=len(rows), expected=60)
            if len(rows) == 60:
                result.update(successes=sum(row['success'] for row in rows),
                              asr_percent=100 * sum(row['success'] for row in rows) / 60,
                              cta_percent=100 * sum(row['clean_accuracy'] for row in rows) / 60)
            draws.append(result)
        complete = all(row['finished'] == row['expected'] for row in draws)
        summary = dict(cell=cell['id'], method=task['method'], model=cell['model'],
                       attack=cell['attack'], budget=cell['budget'], protocol_hash=protocol_hash,
                       complete=complete, targets=cell['targets'], victims_per_target=6, draws=draws)
        if complete:
            rates = [row['asr_percent'] for row in draws]
            summary.update(asr_percent=float(np.mean(rates)),
                           between_draw_sd_percent=float(np.std(rates, ddof=1)) if len(rates) > 1 else None)
        atomic_json(summary_dir / f'{name}.json', summary)
        print('CELL', json.dumps(summary, sort_keys=True), flush=True)


def main():
    global DEADLINE
    parser = argparse.ArgumentParser()
    for name in ('plan', 'task', 'root', 'data', 'cache'):
        parser.add_argument(f'--{name}', required=True)
    parser.add_argument('--deadline', type=float, required=True)
    options = parser.parse_args()
    DEADLINE = options.deadline
    for sig in (signal.SIGUSR1, signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, request_stop)
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    if not torch.cuda.is_available():
        raise RuntimeError('A Slurm GPU allocation is required.')
    plan = json.loads(Path(options.plan).read_text())
    task = next(item for item in plan['tasks'] if item['id'] == options.task)
    cell = next(item for item in plan['cells'] if item['id'] == task['cell'])
    read_baseline(Path(options.plan).parent, cell)
    protocol = plan['protocol']
    protocol_hash = hashlib.sha256(json.dumps(protocol, sort_keys=True).encode()).hexdigest()
    results = Path(options.root) / 'rebuttal_w4_result'
    job_dir = results / 'jobs' / task['id']
    job_dir.mkdir(parents=True, exist_ok=True)
    config = dict(task=task, cell=cell, protocol_hash=protocol_hash)
    config_path = job_dir / 'config.json'
    if config_path.exists() and json.loads(config_path.read_text()) != config:
        raise RuntimeError('The saved task differs from this plan.')
    atomic_json(config_path, config)
    print('TASK', json.dumps(task), 'GPU', torch.cuda.get_device_name(), flush=True)
    args = fu.parse_args(['--model', cell['model'], '--dataset', 'CIFAR10',
                         '--data_path', options.data, '--cache_dir', options.cache,
                         '--attack', cell['attack'], '--base', 'random', '--seed', '42',
                         '--budget', str(cell['budget']), '--epsilon', '0.0313725',
                         '--craft_steps', '250', '--restarts', '8', '--craft_ensemble', '5',
                         '--craft_alpha', '0.0039216', '--craft_batch', '256',
                         '--craft_aug', '--dsa_strategy', fu.DSA_DEFAULT,
                         '--sharp_mode', 'worst', '--sharp_sigma', '0.05'])
    ctx = fu.build_context(args, 'cuda:0')
    if len(ctx['train_labs']) != 50000 or int((ctx['train_labs'] == 5).sum()) != 5000:
        raise RuntimeError('Expected the original full CIFAR-10 training set.')
    if any(int(ctx['test_labs'][target]) != 2 for target in cell['targets']):
        raise RuntimeError('Pinned targets must be test birds.')
    nets = []
    for index in range(5):
        path = Path(options.cache) / 'surrogates' / f'{args.model}_60ep_lr0.1_bs128_seed42' / f'net_{index}.pt'
        if not path.is_file():
            raise FileNotFoundError(f'Missing trained surrogate: {path}')
        net = fu.build_network(args.model, 3, 10, (32, 32), ctx['device'])
        net.load_state_dict(torch.load(path, map_location=ctx['device'], weights_only=True))
        nets.append(net.eval())
    for number, unit in enumerate(task['units'], 1):
        print(f'UNIT {number}/{len(task["units"])}: {unit}', flush=True)
        evaluate_unit(args, ctx, nets, task, cell, unit, results, protocol, protocol_hash)
    aggregate_cell(results, task, cell, protocol_hash, Path(options.plan).parent)
    (job_dir / '.done').write_text('complete\n')
    print(f'JOB {task["id"]} COMPLETE', flush=True)


if __name__ == '__main__':
    try:
        main()
    except Paused as exc:
        print(str(exc), flush=True)
        raise SystemExit(75)
