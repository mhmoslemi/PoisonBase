"""R-only/M-only cells for reviewer 2 W1; run exclusively inside Slurm."""
import argparse
import csv
import fcntl
import gc
import hashlib
import json
import os
from pathlib import Path
import random
import resource
import signal
import sys
import time
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F

import final_update as fu

STOP = False
DEADLINE = float(os.environ.get('MR_DEADLINE', 'inf'))


class Paused(RuntimeError):
    pass


def request_stop(signum, frame):
    global STOP
    STOP = True
    print(f'Signal {signum}: saving at the next craft step or victim epoch.', flush=True)


def check_time():
    if STOP or time.time() >= DEADLINE:
        raise Paused('Progress saved. Rerun submit_mr_killarney_ghaddar.sh to resume.')


def save(path, value, tensor=False):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f'.{os.getpid()}.tmp')
    if tensor:
        torch.save(value, tmp)
    else:
        tmp.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n')
    os.replace(tmp, path)


def rng_state():
    return dict(cpu=torch.get_rng_state(), cuda=torch.cuda.get_rng_state(),
                numpy=np.random.get_state(), python=random.getstate())


def restore_rng(state):
    torch.set_rng_state(state['cpu'].cpu())
    torch.cuda.set_rng_state(state['cuda'].cpu())
    np.random.set_state(state['numpy'])
    random.setstate(state['python'])


def load(path):
    return torch.load(path, map_location='cpu', weights_only=False)


def context(cell, data):
    args = SimpleNamespace(**cell['args'])
    args.data_path = data
    ctx = fu.build_context(args, torch.device('cuda:0'))
    y_adv, target_class = fu.parse_pair(args.class_pair, ctx['class_names'])
    expected = 100000 if args.dataset == 'TinyImageNet' else 50000
    if len(ctx['train_labs']) != expected:
        raise RuntimeError(f'Wrong training data size: {args.dataset}')
    if any(int(ctx['test_labs'][t]) != target_class for t in cell['targets']):
        raise RuntimeError('Pinned target labels differ from the source experiment.')
    ctx['y_adv'] = y_adv
    return args, ctx


def pool(cell, ctx, cache, count=None):
    if count is None:
        count = cell['args'].get('sel_K') or cell['args']['num_surrogates']
    nets = []
    for i in range(count):
        path = Path(cache) / cell['pool'] / f'net_{i}.pt'
        if not path.is_file():
            raise FileNotFoundError(f'Missing trained surrogate; no training fallback: {path}')
        net = fu.build_network(cell['args']['model'], ctx['channel'], ctx['num_classes'],
                               ctx['im_size'], ctx['device'])
        net.load_state_dict(torch.load(path, map_location='cpu', weights_only=True))
        nets.append(net.eval())
    return nets


def select(cell, ctx, nets, target):
    args = cell['args']
    fu.set_seed(args['seed'] * 100003 + target)
    common = dict(base_dist=args['base_dist'],
                  distance_margin_coef=1.0 if cell['component'] == 'R' else 0.0)
    values = (nets, ctx['train_imgs'], ctx['train_labs'], ctx['test_imgs'][target],
              ctx['y_adv'], round(args['budget'] * len(ctx['train_labs'])),
              args['lambda_margin'], ctx['device'])
    if args.get('sel_mode'):
        indices = fu.select_base_ours_div(*values, mode=args['sel_mode'],
                    alpha=args['sel_alpha'], pool=args['sel_pool'], mu=args['sel_mu'], **common)
    else:
        indices = fu.select_base_ours(*values, **common)
    count = values[5]
    if len(indices) != count or len(indices.unique()) != count:
        raise RuntimeError('Wrong poison count or duplicate bases.')
    if not bool((ctx['train_labs'][indices] == ctx['y_adv']).all()):
        raise RuntimeError('Selected bases have the wrong class.')
    return indices


def poison(cell, ctx, nets, target, directory, identity, require_saved=False):
    args = cell['args']
    output, checkpoint = directory / 'poison.pt', directory / 'crafting.pt'
    if output.exists():
        saved = load(output)
        if saved['identity'] != identity:
            raise RuntimeError(f'Incompatible poison cache: {output}')
        indices = torch.tensor(saved['indices'], device=ctx['device'], dtype=torch.long)
        delta = saved['delta'].to(ctx['device'])
        print(f'SKIP crafting: {cell["id"]}, target {target}', flush=True)
    else:
        if require_saved:
            raise FileNotFoundError(f'Upstream crafting did not finish: {output}')
        base_file = directory / 'bases.json'
        if base_file.exists():
            saved = json.loads(base_file.read_text())
            if saved['identity'] != identity:
                raise RuntimeError(f'Incompatible selection cache: {base_file}')
            indices = torch.tensor(saved['indices'], device=ctx['device'], dtype=torch.long)
        else:
            indices = select(cell, ctx, nets, target)
            save(base_file, dict(identity=identity, indices=indices.cpu().tolist()))
        base01 = ctx['denorm'](ctx['train_imgs'][indices]).clamp(0, 1).detach()
        fu.set_seed(args['seed'] * 100003 + target)
        state = load(checkpoint) if checkpoint.exists() else None
        if state:
            if state['identity'] != identity:
                raise RuntimeError('Incompatible crafting checkpoint.')
            np.random.set_state(state['numpy'])
            random.setstate(state['python'])
            print(f'RESUME craft: restart {state["state"]["restart"] + 1}/8, '
                  f'step {state["state"]["next_step"]}/250', flush=True)

        def checkpoint_craft(progress):
            if progress['next_step'] % 25 == 0 or STOP or time.time() >= DEADLINE:
                save(checkpoint, dict(identity=identity, state=progress,
                     numpy=np.random.get_state(), python=random.getstate()), tensor=True)
                print(f'CRAFT {cell["id"]} t{target}: restart {progress["restart"] + 1}/8, '
                      f'step {progress["next_step"]}/250', flush=True)
                check_time()

        began = time.monotonic()
        craft_nets = nets[:args['craft_ensemble']]
        if args['attack'] == 'gradmatch':
            adv, objective = fu.craft_gradmatch(craft_nets, base01, ctx['test_imgs'][target],
                ctx['y_adv'], ctx['norm'], args['epsilon'], args['craft_alpha'],
                args['craft_steps'], args['restarts'], ctx['device'],
                dsa_strategy=args['dsa_strategy'] if args['craft_aug'] else None,
                dsa_param=ctx['dsa_param'], fast=args['fast_gradmatch'],
                schedule=args['craft_schedule'], lowmem=args['craft_lowmem'],
                chunk=args['craft_batch'], resume_state=state['state'] if state else None,
                checkpoint_callback=checkpoint_craft)
        else:
            # shortcut: BP restarts its short (~81 s) craft if interrupted before saving.
            adv, objective = fu.craft_fc(craft_nets, base01, ctx['test_imgs'][target],
                ctx['norm'], args['epsilon'], args['craft_steps'], args['craft_alpha'],
                ctx['device'], restarts=args['fc_restarts'], mode=args['fc_mode'])
        delta = (adv - base01).detach()
        save(output, dict(identity=identity, indices=indices.cpu().tolist(),
             delta=delta.cpu(), objective=float(objective)), tensor=True)
        checkpoint.unlink(missing_ok=True)
        print(f'CRAFT COMPLETE {cell["id"]} t{target}: {time.monotonic()-began:.1f}s', flush=True)
    count = round(args['budget'] * len(ctx['train_labs']))
    if (len(indices) != count or len(indices.unique()) != count or
            not bool((ctx['train_labs'][indices] == ctx['y_adv']).all()) or
            tuple(delta.shape) != (count, ctx['channel'], *ctx['im_size']) or
            not bool(torch.isfinite(delta).all()) or float(delta.abs().max()) > args['epsilon'] + 1e-6):
        raise RuntimeError('Invalid poison cache, class, shape, or perturbation bound.')
    return indices, (ctx['denorm'](ctx['train_imgs'][indices]).clamp(0, 1) + delta).clamp(0, 1)


def fit(net, images, labels, args, checkpoint, identity):
    """The original final_update SGD loop, with optimizer/RNG/epoch checkpoints."""
    opt = torch.optim.SGD(net.parameters(), lr=args['victim_lr'], momentum=0.9,
                          weight_decay=args['victim_wd'])
    start = 0
    if checkpoint.exists():
        state = load(checkpoint)
        if state['identity'] != identity:
            raise RuntimeError('Incompatible victim checkpoint.')
        net.load_state_dict(state['net'])
        opt.load_state_dict(state['optimizer'])
        start = state['epoch']
        restore_rng(state['rng'])
        del state
    print(f'VICTIM t{identity["target"]} v{identity["victim"]}: '
          f'{start}/{args["victim_epochs"]} epochs already complete', flush=True)
    net.train()
    for epoch in range(start, args['victim_epochs']):
        if epoch in args['victim_decay']:
            for group in opt.param_groups:
                group['lr'] *= 0.1
        permutation = torch.randperm(len(labels), device=images.device)
        for ids in permutation.split(args['victim_bs']):
            opt.zero_grad(set_to_none=True)
            F.cross_entropy(net(images[ids]), labels[ids]).backward()
            opt.step()
        if ((epoch + 1) % 5 == 0 or epoch + 1 == args['victim_epochs'] or
                STOP or time.time() >= DEADLINE):
            save(checkpoint, dict(identity=identity, epoch=epoch + 1,
                 net=net.state_dict(), optimizer=opt.state_dict(), rng=rng_state()), tensor=True)
        print(f'VICTIM t{identity["target"]} v{identity["victim"]}: '
              f'epoch {epoch + 1}/{args["victim_epochs"]}', flush=True)
        check_time()
    return net.eval()


def target_directory(results, cell, target):
    return results / 'cells' / cell['id'] / f'target_{target}'


def evaluate(cell, ctx, nets, target, results, protocol_hash, require_saved):
    directory = target_directory(results, cell, target)
    directory.mkdir(parents=True, exist_ok=True)
    identity = dict(protocol_hash=protocol_hash, cell=cell['id'], target=target)
    paths = [directory / f'victim_{v}.json' for v in range(cell['args']['num_victims'])]
    if all(p.exists() for p in paths):
        for v, p in enumerate(paths):
            if json.loads(p.read_text())['identity'] != dict(identity, victim=v):
                raise RuntimeError('Completed victim identity mismatch.')
        print(f'SKIP complete target {target}', flush=True)
        return
    check_time()
    indices, adv = poison(cell, ctx, nets, target, directory, identity, require_saved)
    original = ctx['train_imgs'][indices].clone()
    ctx['train_imgs'][indices] = ctx['norm'](adv)
    del adv
    try:
        for v, path in enumerate(paths):
            trial = dict(identity, victim=v)
            if path.exists():
                if json.loads(path.read_text())['identity'] != trial:
                    raise RuntimeError('Completed victim identity mismatch.')
                print(f'SKIP completed victim t{target} v{v}', flush=True)
                continue
            check_time()
            args = cell['args']
            seed = args['seed'] * 100000 + target * 100 + v
            net = fu.build_network(args['model'], ctx['channel'], ctx['num_classes'],
                                   ctx['im_size'], ctx['device'], seed=seed)
            checkpoint = directory / f'victim_{v}.pt'
            began = time.monotonic()
            fit(net, ctx['train_imgs'], ctx['train_labs'], args, checkpoint, trial)
            prediction = fu.predict_target(net, ctx['test_imgs'][target])
            row = dict(identity=trial, seed=seed, effective_seed=int(torch.initial_seed()), prediction=prediction,
                       success=int(prediction == ctx['y_adv']),
                       clean_accuracy=fu.test_acc(net, ctx['test_imgs'], ctx['test_labs']),
                       seconds_this_attempt=time.monotonic() - began)
            save(path, row)
            checkpoint.unlink(missing_ok=True)
            print('RESULT', json.dumps(row, sort_keys=True), flush=True)
            del net
            gc.collect()
            torch.cuda.empty_cache()
    finally:
        ctx['train_imgs'][indices] = original


def aggregate(plan, results, protocol_hash):
    summaries = results / 'summaries'
    summaries.mkdir(parents=True, exist_ok=True)
    for cell in plan['cells']:
        with (summaries / f'{cell["id"]}.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            rows = []
            for target in cell['targets']:
                for victim in range(cell['args']['num_victims']):
                    path = target_directory(results, cell, target) / f'victim_{victim}.json'
                    if not path.exists():
                        continue
                    row = json.loads(path.read_text())
                    expected = dict(protocol_hash=protocol_hash, cell=cell['id'], target=target, victim=victim)
                    if row['identity'] != expected or row['success'] not in (0, 1):
                        raise RuntimeError(f'Invalid result: {path}')
                    rows.append(row)
            total = len(cell['targets']) * cell['args']['num_victims']
            summary = dict(cell=cell['id'], component=cell['component'], complete=len(rows) == total,
                           finished=len(rows), expected=total, targets=cell['targets'],
                           victims_per_target=cell['args']['num_victims'], protocol_hash=protocol_hash)
            if summary['complete']:
                summary.update(asr_percent=100 * sum(r['success'] for r in rows) / total,
                               cta_percent=100 * sum(r['clean_accuracy'] for r in rows) / total)
            save(summaries / f'{cell["id"]}.json', summary)
            tmp = summaries / f'{cell["id"]}.csv.tmp'
            with tmp.open('w', newline='') as handle:
                writer = csv.DictWriter(handle, fieldnames=['target', 'victim', 'seed', 'effective_seed', 'success', 'clean_accuracy'])
                writer.writeheader()
                for row in rows:
                    writer.writerow(dict(target=row['identity']['target'], victim=row['identity']['victim'],
                                         **{key: row[key] for key in ['seed', 'effective_seed', 'success', 'clean_accuracy']}))
            os.replace(tmp, summaries / f'{cell["id"]}.csv')
            print('CELL', json.dumps(summary, sort_keys=True), flush=True)


def preflight(plan, options, results, protocol_hash):
    import check
    check.run(plan, options, results, sys.modules[__name__])
    save(results / 'preflight.json', dict(protocol_hash=protocol_hash, complete=True,
         gpu=torch.cuda.get_device_name(), torch_version=torch.__version__,
         peak_host_gib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', required=True)
    parser.add_argument('--plan', required=True)
    parser.add_argument('--root', required=True)
    parser.add_argument('--data', required=True)
    parser.add_argument('--cache', required=True)
    options = parser.parse_args()
    if not os.environ.get('SLURM_JOB_ID') or not torch.cuda.is_available():
        raise RuntimeError('Run inside a Slurm GPU job, never on the login node.')
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    signal.signal(signal.SIGUSR1, request_stop)
    signal.signal(signal.SIGTERM, request_stop)
    plan_path = Path(options.plan)
    plan = json.loads(plan_path.read_text())
    protocol_hash = hashlib.sha256(plan_path.read_bytes() + (plan_path.parent / 'code.sha256').read_bytes()).hexdigest()
    results = Path(options.root) / 'rebuttal_mr_result'
    task = next(t for t in plan['tasks'] if t['id'] == options.task)
    job_dir = results / 'jobs' / task['id']
    job_dir.mkdir(parents=True, exist_ok=True)
    config = dict(protocol_hash=protocol_hash, task=task)
    config_path = job_dir / 'config.json'
    if config_path.exists() and json.loads(config_path.read_text()) != config:
        raise RuntimeError('Saved task belongs to another plan.')
    save(config_path, config)
    with (job_dir / 'run.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if task['id'] == 'preflight':
            preflight(plan, options, results, protocol_hash)
        else:
            ready = json.loads((results / 'preflight.json').read_text())
            if ready['protocol_hash'] != protocol_hash or not ready['complete']:
                raise RuntimeError('Preflight is missing or belongs to another plan.')
            cell = next(c for c in plan['cells'] if c['id'] == task['cell'])
            args, ctx = context(cell, options.data)
            nets = [] if task['stage'] == 'victims' else pool(cell, ctx, options.cache)
            for target in task['targets']:
                print(f'TASK {task["id"]}, target {target}', flush=True)
                if task['stage'] == 'craft':
                    directory = target_directory(results, cell, target)
                    directory.mkdir(parents=True, exist_ok=True)
                    identity = dict(protocol_hash=protocol_hash, cell=cell['id'], target=target)
                    poison(cell, ctx, nets, target, directory, identity)
                else:
                    evaluate(cell, ctx, nets, target, results, protocol_hash, task['stage'] == 'victims')
                check_time()
            aggregate(plan, results, protocol_hash)
        save(job_dir / '.done', config)
        print(f'JOB {task["id"]} COMPLETE', flush=True)


if __name__ == '__main__':
    try:
        main()
    except Paused as exc:
        print(str(exc), flush=True)
        raise SystemExit(75)
