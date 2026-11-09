"""VGG13BN Tiny ImageNet GM: shared surrogates, saved poisons, independent victims."""
import argparse
import fcntl
import gc
import hashlib
import json
import os
from pathlib import Path
import signal
import time

import torch
import final_update as fu
import replay
from victim_aug import VictimAugmenter

original_build = fu.build_network


def build_network(name, channel, classes, size, device, seed=None):
    assert name == 'VGG13BN'
    net = original_build(name, channel, classes, size, device, seed=seed)
    # Same 64x64 pooling adaptation used in the measured five-epoch benchmark.
    net.features[-1] = torch.nn.AdaptiveAvgPool2d(1)
    return net


fu.build_network = build_network


def select(cell, ctx, nets, target):
    args = cell['args']
    seed = args['seed'] * 100003 + target
    fu.set_seed(seed)
    count = round(args['budget'] * len(ctx['train_labs']))
    if cell['component'] == 'random':
        generator = torch.Generator().manual_seed(seed)
        indices = fu.select_base_random(ctx['train_labs'].cpu(), ctx['y_adv'],
                                       count, 'cpu', generator).to(ctx['device'])
    else:
        assert cell['component'] == 'BASIS' and len(nets) == 20
        indices = fu.select_base_ours(nets, ctx['train_imgs'], ctx['train_labs'],
            ctx['test_imgs'][target], ctx['y_adv'], count, args['lambda_margin'],
            ctx['device'], base_dist='cosine', distance_margin_coef=None)
    assert len(indices.unique()) == count and (ctx['train_labs'][indices] == ctx['y_adv']).all()
    return indices


replay.select = select


def train_surrogates(task, cell, ctx, results, protocol_hash):
    args = cell['args']
    settings = dict(args, victim_epochs=args['surrogate_epochs'],
                    victim_lr=args['surrogate_lr'], victim_bs=args['surrogate_bs'],
                    victim_decay=args['surrogate_decay'], victim_wd=args['surrogate_wd'])
    for index in task['surrogates']:
        path = results / 'surrogates' / f'net_{index}.pt'
        metadata = path.with_suffix('.json')
        identity = dict(protocol_hash=protocol_hash, target='surrogate', victim=index)
        if path.exists() and metadata.exists():
            saved = json.loads(metadata.read_text())
            assert saved['identity'] == identity
            print(f'SKIP trained surrogate {index}', flush=True)
            continue
        replay.check_time()
        seed = args['seed'] * 1000 + index
        net = build_network('VGG13BN', 3, 200, (64, 64), ctx['device'], seed=seed)
        checkpoint = results / 'surrogate_training' / f'net_{index}.pt'
        replay.fit(net, ctx['train_imgs'], ctx['train_labs'], settings,
                   checkpoint, identity)
        accuracy = fu.test_acc(net, ctx['test_imgs'], ctx['test_labs'])
        replay.save(path, net.state_dict(), tensor=True)
        replay.save(metadata, dict(identity=identity, epochs=settings['victim_epochs'], seed=seed,
                                   clean_accuracy=accuracy))
        checkpoint.unlink(missing_ok=True)
        print('SURROGATE', index, accuracy, flush=True)
        del net
        gc.collect()
        torch.cuda.empty_cache()


def train_victim(task, cell, ctx, results, protocol_hash):
    target, victim = task['targets'][0], task['victim']
    args = cell['args']
    identity = dict(protocol_hash=protocol_hash, cell=cell['id'], target=target)
    trial = dict(identity, victim=victim)
    directory = replay.target_directory(results, cell, target)
    path = directory / f'victim_{victim}.json'
    if path.exists():
        assert json.loads(path.read_text())['identity'] == trial
        return
    indices, adv = replay.poison(cell, ctx, [], target, directory, identity,
                                 require_saved=True)
    original = ctx['train_imgs'][indices].clone()
    ctx['train_imgs'][indices] = ctx['norm'](adv)
    del adv
    try:
        seed = args['seed'] * 100000 + target * 100 + victim
        net = build_network('VGG13BN', 3, 200, (64, 64), ctx['device'], seed=seed)
        augmenter = VictimAugmenter('standard', ctx)
        checkpoint = directory / f'victim_{victim}.pt'
        began = time.monotonic()
        replay.fit(net, ctx['train_imgs'], ctx['train_labs'], args, checkpoint,
                   trial, augmenter=augmenter)
        prediction = fu.predict_target(net, ctx['test_imgs'][target])
        row = dict(identity=trial, seed=seed, effective_seed=int(torch.initial_seed()),
                   model='VGG13BN', epochs=120, augmentation='standard',
                   prediction=prediction, success=int(prediction == ctx['y_adv']),
                   clean_accuracy=fu.test_acc(net, ctx['test_imgs'], ctx['test_labs']),
                   seconds_this_attempt=time.monotonic() - began)
        replay.save(path, row)
        checkpoint.unlink(missing_ok=True)
        print('RESULT', json.dumps(row, sort_keys=True), flush=True)
    finally:
        ctx['train_imgs'][indices] = original


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', required=True)
    parser.add_argument('--batch', required=True)
    parser.add_argument('--root', required=True)
    options = parser.parse_args()
    assert os.environ.get('SLURM_JOB_ID') and torch.cuda.is_available(), 'Use a Slurm GPU job.'
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    signal.signal(signal.SIGUSR1, replay.request_stop)
    signal.signal(signal.SIGTERM, replay.request_stop)
    root, batch = Path(options.root), Path(options.batch)
    plan = json.loads((batch / 'plan.json').read_text())
    protocol_hash = hashlib.sha256((batch / 'plan.json').read_bytes() +
                                  (batch / 'code.sha256').read_bytes()).hexdigest()
    results = root / plan['results']
    task = next(t for t in plan['tasks'] if t['id'] == options.task)
    job_dir = results / 'jobs' / task['id']
    job_dir.mkdir(parents=True, exist_ok=True)
    config = dict(protocol_hash=protocol_hash, task=task)
    with (job_dir / 'run.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        marker = job_dir / '.done'
        if marker.exists():
            assert json.loads(marker.read_text()) == config
            print('SKIP completed task', task['id'], flush=True)
            return
        saved_config = job_dir / 'config.json'
        if saved_config.exists():
            assert json.loads(saved_config.read_text()) == config
        replay.save(saved_config, config)
        cell = next(c for c in plan['cells'] if c['id'] == task['cell'])
        _, ctx = replay.context(cell, str(root / 'data'))
        if task['stage'] == 'surrogate':
            if 0 in task['surrogates']:
                from check import run
                run(plan, ctx, results, protocol_hash)
            train_surrogates(task, cell, ctx, results, protocol_hash)
        elif task['stage'] == 'craft':
            ready = json.loads((results / 'preflight.json').read_text())
            assert ready['protocol_hash'] == protocol_hash and ready['complete']
            count = 5 if cell['component'] == 'random' else 20
            nets = replay.pool(cell, ctx, results, count=count)
            for target in task['targets']:
                replay.check_time()
                identity = dict(protocol_hash=protocol_hash, cell=cell['id'], target=target)
                directory = replay.target_directory(results, cell, target)
                replay.poison(cell, ctx, nets, target, directory, identity)
        else:
            assert task['stage'] == 'victim' and task['victim'] in range(cell['args']['num_victims'])
            train_victim(task, cell, ctx, results, protocol_hash)
            replay.aggregate(dict(plan, cells=[cell]), results, protocol_hash)
        replay.save(marker, config)
        print('TASK COMPLETE', task['id'], flush=True)


if __name__ == '__main__':
    try:
        main()
    except replay.Paused as error:
        print('Progress checkpointed; rerun submit_tiny_gm_vgg13_ghaddar.sh.', str(error), flush=True)
        raise SystemExit(75)
