"""Cached scoring and standard attack evaluation for the two requested rows."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import time

import numpy as np
import torch

import final_update as fu
from scores import (hessian_vector_product, influence_input_scores, lissa_update,
                    loss_gradient, meta_weight_gradients, normalize_meta_weights,
                    vector_norm)

STOP_REQUESTED = False
DEADLINE = float('inf')


def stop_requested(signum, frame):
    global STOP_REQUESTED
    STOP_REQUESTED = True


def check_time():
    if STOP_REQUESTED or time.time() >= DEADLINE:
        print('PAUSED: saved progress will resume when this task is resubmitted.', flush=True)
        raise SystemExit(75)


def atomic_save(blob, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f'.tmp.{os.getpid()}')
    torch.save(blob, temporary)
    os.replace(temporary, path)


def atomic_json(blob, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f'.tmp.{os.getpid()}')
    temporary.write_text(json.dumps(blob, indent=2) + '\n')
    os.replace(temporary, path)


def load(path):
    return torch.load(path, map_location='cpu', weights_only=False)


def identity(task, target, surrogate, protocol_hash):
    return dict(method=task['method'], model=task['model'], target=target,
                surrogate=surrogate, protocol_hash=protocol_hash)


def score_path(results, method, model, target, surrogate):
    return results / 'scores' / method / model / f'target_{target}' / f'net_{surrogate}.pt'


def score_one(net, ctx, candidates, target_id, task, surrogate, results, protocol, protocol_hash):
    metadata = identity(task, target_id, surrogate, protocol_hash)
    output = score_path(results, task['method'], task['model'], target_id, surrogate)
    if output.exists():
        saved = load(output)
        if saved['identity'] != metadata or not torch.equal(saved['candidates'], candidates.cpu()):
            raise RuntimeError(f'Incompatible cached scores: {output}')
        print(f'SKIP {task["method"]} {task["model"]} target={target_id} net={surrogate}: done', flush=True)
        return
    checkpoint = output.with_suffix('.checkpoint.pt')
    state = load(checkpoint) if checkpoint.exists() else {}
    if state and state['identity'] != metadata:
        raise RuntimeError(f'Incompatible scoring checkpoint: {checkpoint}')
    device = ctx['device']
    target = ctx['test_imgs'][target_id:target_id + 1]
    label = torch.tensor([protocol['y_adv']], dtype=torch.long, device=device)
    method = task['method']
    settings = protocol[method]
    started = time.monotonic()
    diagnostics = dict(state.get('diagnostics', {}))
    inverse = None
    if method == 'influence':
        if state.get('phase') == 'candidates':
            inverse = tuple(value.to(device) for value in state['inverse_hvp'])
        else:
            rhs = loss_gradient(net, target, label)
            rhs_norm = vector_norm(rhs)
            step = int(state.get('step', 0))
            scale = float(state.get('scale', settings['scale']))
            estimate = (tuple(value.to(device) for value in state['estimate'])
                        if state else tuple(value.clone() for value in rhs))
            seed = protocol['seed'] * 1000003 + target_id * 101 + surrogate
            generator = torch.Generator(device=device).manual_seed(seed)
            if state:
                generator.set_state(state['generator_state'])
            while step < settings['depth']:
                indices = torch.randint(len(ctx['train_imgs']), (settings['hessian_batch'],),
                                        generator=generator, device=device)
                hvp = hessian_vector_product(net, ctx['train_imgs'][indices],
                                             ctx['train_labs'][indices], estimate)
                estimate = lissa_update(rhs, estimate, hvp, scale, settings['ridge'] / scale)
                step += 1
                if step % 250 == 0 or step == settings['depth']:
                    norm = vector_norm(estimate)
                    # Keep the physical ridge fixed when increasing LiSSA's
                    # spectral scaling. Never replace the Hessian with a proxy.
                    limit = 100 * scale / settings['ridge'] * max(rhs_norm, 1e-30)
                    if not np.isfinite(norm) or norm > limit:
                        scale *= 2
                        if scale > settings['max_scale']:
                            raise RuntimeError('LiSSA is unstable even at its maximum scale; '
                                               'no candidate scores were accepted')
                        print(f'LiSSA restart: target={target_id} net={surrogate} scale={scale}', flush=True)
                        step, estimate = 0, tuple(value.clone() for value in rhs)
                        generator.manual_seed(seed)
                    diagnostics.update(scale=scale, ridge=settings['ridge'],
                                       recursion_steps=step, rhs_norm=rhs_norm, estimate_norm=norm)
                    atomic_save(dict(identity=metadata, phase='inverse', step=step, scale=scale,
                                     estimate=[value.cpu() for value in estimate],
                                     generator_state=generator.get_state(), diagnostics=diagnostics), checkpoint)
                    print(f'LiSSA target={target_id} net={surrogate}: {step}/{settings["depth"]} '
                          f'scale={scale:g} norm={norm:.5g}', flush=True)
                    check_time()
            inverse = tuple(value / scale for value in estimate)
            state = dict(identity=metadata, phase='candidates', offset=0,
                         values=torch.zeros(len(candidates), dtype=torch.float64),
                         inverse_hvp=[value.cpu() for value in inverse], diagnostics=diagnostics)
            atomic_save(state, checkpoint)
    values = state.get('values', torch.zeros(len(candidates), dtype=torch.float64))
    offset = int(state.get('offset', 0))
    batch_size = settings['candidate_batch']
    zeros = torch.zeros(1, ctx['channel'], 1, 1, device=device)
    pixel_scale = ctx['norm'](zeros + 1) - ctx['norm'](zeros)
    for start in range(offset, len(candidates), batch_size):
        ids = candidates[start:start + batch_size]
        images = ctx['train_imgs'][ids]
        labels = ctx['train_labs'][ids]
        if method == 'meta':
            batch_values = meta_weight_gradients(net, images, labels, target, label,
                                                 settings['step_size'])
        else:
            batch_values = influence_input_scores(net, images, labels, inverse, pixel_scale)
        if not torch.isfinite(batch_values).all():
            raise RuntimeError('Non-finite candidate scores; this task is incomplete')
        values[start:start + len(ids)] = batch_values.double().cpu()
        offset = start + len(ids)
        if offset == len(candidates) or (start // batch_size) % 16 == 0 or STOP_REQUESTED:
            state = dict(identity=metadata, phase='candidates', offset=offset,
                         values=values, diagnostics=diagnostics)
            if inverse is not None:
                state['inverse_hvp'] = [value.cpu() for value in inverse]
            atomic_save(state, checkpoint)
            print(f'{method} target={target_id} net={surrogate}: candidates {offset}/{len(candidates)}', flush=True)
            check_time()
    if method == 'meta':
        diagnostics['positive_candidates'] = int((values > 0).sum())
        values = normalize_meta_weights(values)
    diagnostics['wall_seconds_this_attempt'] = time.monotonic() - started
    if not torch.isfinite(values).all() or (values < 0).any():
        raise RuntimeError('Invalid final selection scores')
    atomic_save(dict(identity=metadata, candidates=candidates.cpu(), values=values,
                     diagnostics=diagnostics), output)
    checkpoint.unlink(missing_ok=True)
    print(f'DONE {method} {task["model"]} target={target_id} net={surrogate}', flush=True)


def run_scores(options, task, protocol, protocol_hash, results):
    args = argparse.Namespace(dataset='CIFAR10', data_path=options.data)
    ctx = fu.build_context(args, 'cuda:0')
    candidates = (ctx['train_labs'] == protocol['y_adv']).nonzero(as_tuple=True)[0]
    if len(candidates) != 5000:
        raise RuntimeError('Expected the full pool of 5000 dog candidates')
    for target in task['targets']:
        if int(ctx['test_labs'][target]) != 2:
            raise RuntimeError(f'Target {target} is not a bird')
        for surrogate in range(protocol['surrogates']):
            check_time()
            path = score_path(results, task['method'], task['model'], target, surrogate)
            if path.exists():
                saved = load(path)
                if saved['identity'] != identity(task, target, surrogate, protocol_hash):
                    raise RuntimeError(f'Incompatible cache: {path}')
                continue
            net = fu.build_network(task['model'], ctx['channel'], ctx['num_classes'],
                                   ctx['im_size'], ctx['device']).eval()
            weights = Path(options.cache) / 'surrogates' / f'{task["model"]}_60ep_lr0.1_bs128_seed42' / f'net_{surrogate}.pt'
            net.load_state_dict(torch.load(weights, map_location=ctx['device'], weights_only=True))
            score_one(net, ctx, candidates, target, task, surrogate, results, protocol, protocol_hash)
            del net
            torch.cuda.empty_cache()


def ranking(results, task, target, protocol, protocol_hash):
    candidates, scores = None, None
    for surrogate in range(protocol['surrogates']):
        record = load(score_path(results, task['method'], task['model'], target, surrogate))
        if record['identity'] != identity(task, target, surrogate, protocol_hash):
            raise RuntimeError('Scoring protocol does not match the evaluation')
        if candidates is None:
            candidates = record['candidates']
            scores = record['values'].clone()
        else:
            if not torch.equal(candidates, record['candidates']):
                raise RuntimeError('Candidate pools differ between surrogates')
            scores += record['values']
    scores /= protocol['surrogates']
    if not torch.isfinite(scores).all():
        raise RuntimeError('Invalid ensemble scores')
    order = np.lexsort((candidates.numpy(), -scores.numpy()))
    return candidates[torch.as_tensor(order, dtype=torch.long)]


def run_evaluation(options, task, protocol, protocol_hash, results, job_dir):
    ranks = {target: ranking(results, task, target, protocol, protocol_hash)
             for target in task['targets']}
    target_file = job_dir / 'targets.json'
    atomic_json({'pairs': {'dog-bird': {'indices': task['targets']}}}, target_file)
    target_degree = 50 if task['model'] == 'ConvNetBN' and task['attack'] == 'fc' else (
        70 if task['model'] == 'ConvNetBN' else (10 if task['attack'] == 'fc' else 14))
    argv = ['--dataset', 'CIFAR10', '--model', task['model'], '--seed', '42',
            '--data_path', options.data, '--cache_dir', options.cache,
            '--out_dir', str(job_dir / 'results'), '--gpus', 'all',
            '--attack', task['attack'], '--base', 'ours', '--fc_mode', 'sample',
            '--base_dist', 'cosine', '--lambda_margin', '1', '--sel_K', '20',
            '--class_pair', 'dog-bird', '--pair_order', 'poison-target',
            '--budget', str(task['budget']), '--epsilon', '0.0313725',
            '--craft_steps', '250', '--craft_alpha', '0.0039216',
            '--restarts', '8', '--fc_restarts', '1', '--craft_ensemble', '5',
            '--craft_aug', '--dsa_strategy', fu.DSA_DEFAULT,
            '--sharp_mode', 'worst', '--sharp_sigma', '0.05',
            '--num_surrogates', '20', '--surrogate_epochs', '60', '--surrogate_lr', '0.1',
            '--surrogate_bs', '128', '--surrogate_decay', '35', '45', '--surrogate_wd', '0',
            '--num_targets', str(len(task['targets'])), '--target_select', str(target_degree),
            '--target_idx_file', str(target_file), '--keep_pinned_targets', '--rank_on_victims',
            '--num_victims', '6', '--victim_epochs', '50', '--victim_lr', '0.1',
            '--victim_bs', '125', '--victim_decay', '40', '--victim_wd', '0', '--clean_baseline']
    args = fu.parse_args(argv)
    # A process-local selector hook feeds these authors' cached rankings into
    # the unchanged crafting and victim code. No root experiment file is edited.
    args.sel_criterion = 'meta-grad' if task['method'] == 'meta' else 'influence-input'
    original_prepare = fu.prepare_poisons
    current_target = {}

    def prepare(args, ctx, sel_nets, craft_nets, tidx, *rest, **kwargs):
        check_time()
        current_target['id'] = tidx
        return original_prepare(args, ctx, sel_nets, craft_nets, tidx, *rest, **kwargs)

    def select(criterion, nets, images, labels, x_t, y_adv, count, lam, device, **kwargs):
        selected = ranks[current_target['id']][:count].to(device)
        if len(selected) != count or len(selected.unique()) != count or not torch.all(labels[selected] == y_adv):
            raise RuntimeError('Selected bases are not distinct poison-class candidates')
        return selected

    original_train = fu.train_from_scratch
    def train(*args, **kwargs):
        kwargs['progress_callback'] = lambda *_: check_time()
        return original_train(*args, **kwargs)

    fu.prepare_poisons = prepare
    fu.select_base_criterion = select
    fu.train_from_scratch = train
    run_dir = Path(args.out_dir) / fu.build_run_name(args)
    # The batch shell holds the job's exclusive flock across this invocation.
    (run_dir / '.lock').unlink(missing_ok=True)
    fu.main(args)
    import csv
    with (run_dir / 'results.csv').open(newline='') as handle:
        rows = list(csv.DictReader(handle))
    expected = {(target, victim) for target in task['targets'] for victim in range(6)}
    pairs = [(int(row['target_idx']), int(row['victim_id'])) for row in rows]
    if len(rows) != len(expected) or set(pairs) != expected:
        raise RuntimeError('Incomplete target/victim grid; task is not done')
    if any(row['model'] != task['model'] or row['attack'] != task['attack']
           or abs(float(row['budget']) - task['budget']) > 1e-12
           or row['success'] not in ('0', '1') for row in rows):
        raise RuntimeError('Victim result metadata is inconsistent')
    summary = dict(method=task['method'], model=task['model'], attack=task['attack'],
                   budget=task['budget'], targets=task['targets'], victims_per_target=6,
                   trials=len(rows), successes=sum(int(row['success']) for row in rows),
                   asr=100 * sum(int(row['success']) for row in rows) / len(rows),
                   cta=100 * sum(float(row['clean_test_acc']) for row in rows) / len(rows),
                   results_csv=str(run_dir / 'results.csv'), protocol_hash=protocol_hash)
    atomic_json(summary, job_dir / 'summary.json')
    print('Completed evaluation:', json.dumps(summary), flush=True)


def main():
    global DEADLINE
    parser = argparse.ArgumentParser()
    parser.add_argument('--plan', required=True)
    parser.add_argument('--task', required=True)
    parser.add_argument('--root', required=True)
    parser.add_argument('--data', required=True)
    parser.add_argument('--cache', required=True)
    parser.add_argument('--seconds', type=int, default=6900)
    options = parser.parse_args()
    DEADLINE = time.time() + options.seconds
    for sig in (signal.SIGUSR1, signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, stop_requested)
    torch.set_num_threads(1)
    if not torch.cuda.is_available():
        raise RuntimeError('An allocated CUDA GPU is required')
    fu.set_seed(42)
    plan = json.loads(Path(options.plan).read_text())
    task = next(item for item in plan['tasks'] if item['id'] == options.task)
    protocol = plan['protocol']
    protocol_hash = hashlib.sha256(json.dumps(protocol, sort_keys=True).encode()).hexdigest()
    results = Path(options.root) / 'rebuttal_meta_influence_result'
    job_dir = results / 'jobs' / task['id']
    job_dir.mkdir(parents=True, exist_ok=True)
    expected = dict(task=task, protocol_hash=protocol_hash)
    config = job_dir / 'task.json'
    if config.exists() and json.loads(config.read_text()) != expected:
        raise RuntimeError('Saved job configuration differs from this plan')
    atomic_json(expected, config)
    print('Task:', json.dumps(task), 'GPU:', torch.cuda.get_device_name(), flush=True)
    if task['stage'] == 'score':
        run_scores(options, task, protocol, protocol_hash, results)
    else:
        run_evaluation(options, task, protocol, protocol_hash, results, job_dir)
    (job_dir / '.done').write_text('complete\n')


if __name__ == '__main__':
    main()
