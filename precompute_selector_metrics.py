#!/usr/bin/env python3
"""Precompute one surrogate shard of Gao et al. selection statistics.

The definitions intentionally follow the authors' released implementation:

* training loss and the exact full-parameter gradient norm are evaluated after
  the requested selection epoch (10 by default);
* a forgetting event is a correctness transition 1 -> 0 across presentations;
* an example never learned during training receives the maximum forgetting
  score (the number of epochs).

One invocation trains one seed and writes one atomic ``net_<id>.npz`` shard.
Independent Slurm array tasks can therefore build a K-model ensemble safely.
"""

import argparse
import json
import os
import resource
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import final_update as attack


def process_max_rss_kb():
    value = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return value / 1024.0 if sys.platform == 'darwin' else value


def valid_resource_record(path):
    required = {
        'total_wall_seconds', 'trajectory_wall_seconds',
        'trajectory_training_seconds_excluding_metric',
        'metric_eval_wall_seconds', 'cuda_peak_allocated_bytes',
        'cuda_peak_reserved_bytes', 'process_max_rss_kb',
    }
    try:
        with open(path) as handle:
            record = json.load(handle)
        return required.issubset(record) and all(
            np.isfinite(float(record[key])) for key in required)
    except (OSError, ValueError, TypeError):
        return False


def per_example_loss_and_gradnorm(net, images, labels, batch_size):
    """Return CE loss and exact ||d loss_i / d theta||_2 for every row."""
    core = net.module if isinstance(net, nn.DataParallel) else net
    core.eval()
    parameters = dict(core.named_parameters())
    buffers = dict(core.named_buffers())
    if not parameters:
        raise RuntimeError('gradient-norm selector requires model parameters')
    if not all(hasattr(torch.func, name) for name in ('functional_call', 'grad', 'vmap')):
        raise RuntimeError('this precompute requires torch.func functional_call/grad/vmap')

    def one_loss(model_parameters, model_buffers, image, label):
        logits = torch.func.functional_call(
            core, (model_parameters, model_buffers), (image.unsqueeze(0),))
        return F.cross_entropy(logits, label.unsqueeze(0))

    grad_one = torch.func.grad(one_loss)
    loss_out = torch.empty(len(images), device=images.device, dtype=torch.float64)
    norm_out = torch.empty(len(images), device=images.device, dtype=torch.float64)
    for start in range(0, len(images), batch_size):
        stop = min(start + batch_size, len(images))
        batch_images = images[start:stop]
        batch_labels = labels[start:stop]
        with torch.no_grad():
            logits = core(batch_images)
            loss_out[start:stop] = F.cross_entropy(
                logits, batch_labels, reduction='none').double()
        with torch.enable_grad():
            gradients = torch.func.vmap(
                grad_one, in_dims=(None, None, 0, 0))(
                    parameters, buffers, batch_images, batch_labels)
        squared = torch.zeros(stop - start, device=images.device, dtype=torch.float64)
        for gradient in gradients.values():
            squared += gradient.detach().double().reshape(stop - start, -1).square().sum(1)
        norm_out[start:stop] = squared.sqrt()
        del gradients, squared
    return loss_out.cpu().numpy(), norm_out.cpu().numpy()


def train_and_measure(args):
    device = 'cuda:%d' % args.gpu if torch.cuda.is_available() else 'cpu'
    if device.startswith('cuda'):
        torch.cuda.set_device(args.gpu)
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
    whole_started = time.perf_counter()

    context_args = argparse.Namespace(
        dataset=args.dataset,
        data_path=args.data_path,
        dsa_strategy=attack.DSA_DEFAULT,
    )
    ctx = attack.build_context(context_args, device)
    y_adv, _ = attack.parse_pair(args.class_pair, ctx['class_names'], args.pair_order)
    candidate_indices = (ctx['train_labs'] == y_adv).nonzero(as_tuple=True)[0]
    candidate_positions = torch.full(
        (len(ctx['train_labs']),), -1, dtype=torch.long, device=device)
    candidate_positions[candidate_indices] = torch.arange(
        len(candidate_indices), device=device)

    metric_args = argparse.Namespace(
        selector_metric_dir=args.selector_metric_dir,
        cache_dir=args.cache_dir,
        dataset=args.dataset,
        model=args.model,
        surrogate_epochs=args.epochs,
        sel_metric_epoch=args.select_epoch,
        surrogate_lr=args.lr,
        surrogate_bs=args.batch_size,
        surrogate_decay=args.decay,
        surrogate_wd=args.weight_decay,
        seed=args.seed,
    )
    output_dir = attack.selector_metric_shard_dir(metric_args, y_adv)
    output_path = os.path.join(output_dir, 'net_%d.npz' % args.surrogate_id)
    resource_path = os.path.join(
        output_dir, 'net_%d.resources.json' % args.surrogate_id)
    if (os.path.exists(output_path) and valid_resource_record(resource_path)
            and not args.force):
        with np.load(output_path, allow_pickle=False) as blob:
            required = {'candidate_indices', 'loss', 'gradnorm', 'forgetting'}
            expected = candidate_indices.cpu().numpy().astype(np.int64)
            valid = required.issubset(blob.files)
            valid = valid and np.array_equal(blob['candidate_indices'], expected)
            valid = valid and all(
                np.asarray(blob[key]).shape == (len(expected),)
                and np.isfinite(blob[key]).all()
                for key in ('loss', 'gradnorm', 'forgetting'))
            if valid:
                print('already complete:', output_path, flush=True)
                return output_path

    model_seed = args.seed + 1000 + args.surrogate_id
    attack.set_seed(model_seed)
    net = attack.build_network(
        args.model, ctx['channel'], ctx['num_classes'], ctx['im_size'],
        device, seed=model_seed)
    optimizer = torch.optim.SGD(
        net.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)
    criterion = nn.CrossEntropyLoss().to(device)
    current_lr = args.lr
    decay = set(args.decay)
    trace = torch.zeros(
        (len(candidate_indices), args.epochs), dtype=torch.bool, device='cpu')
    selected_loss = selected_gradnorm = None
    trajectory_started = time.perf_counter()
    metric_eval_wall_seconds = 0.0

    print('metric shard: model=%s surrogate=%d seed=%d candidates=%d epochs=%d select_epoch=%d'
          % (args.model, args.surrogate_id, model_seed, len(candidate_indices),
             args.epochs, args.select_epoch), flush=True)
    for epoch in range(args.epochs):
        if epoch in decay:
            current_lr *= 0.1
            for group in optimizer.param_groups:
                group['lr'] = current_lr
        net.train()
        permutation = torch.randperm(len(ctx['train_imgs']), device=device)
        for start in range(0, len(permutation), args.batch_size):
            indices = permutation[start:start + args.batch_size]
            images = ctx['train_imgs'][indices]
            labels = ctx['train_labs'][indices]
            optimizer.zero_grad(set_to_none=True)
            logits = net(images)
            positions = candidate_positions[indices]
            tracked = positions >= 0
            if bool(tracked.any()):
                trace[positions[tracked].cpu(), epoch] = (
                    logits.detach().argmax(1)[tracked] == labels[tracked]).cpu()
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

        if epoch + 1 == args.select_epoch:
            if device.startswith('cuda'):
                torch.cuda.synchronize(device)
            metric_started = time.perf_counter()
            selected_loss, selected_gradnorm = per_example_loss_and_gradnorm(
                net, ctx['train_imgs'][candidate_indices],
                ctx['train_labs'][candidate_indices], args.grad_batch_size)
            if device.startswith('cuda'):
                torch.cuda.synchronize(device)
            metric_eval_wall_seconds += time.perf_counter() - metric_started
        if (epoch + 1) % max(1, args.epochs // 10) == 0 or epoch + 1 == args.epochs:
            print('  %s surrogate %d: %d/%d epochs (%.0f s)'
                  % (args.model, args.surrogate_id, epoch + 1, args.epochs,
                     time.perf_counter() - trajectory_started), flush=True)

    if selected_loss is None or selected_gradnorm is None:
        raise RuntimeError('--select-epoch must be within 1..--epochs')
    transitions = trace[:, 1:].to(torch.int8) - trace[:, :-1].to(torch.int8)
    forgetting = (transitions == -1).sum(1).numpy().astype(np.float64)
    never_learned = ~trace.any(1).numpy()
    forgetting[never_learned] = float(args.epochs)
    if device.startswith('cuda'):
        torch.cuda.synchronize(device)
    trajectory_wall_seconds = time.perf_counter() - trajectory_started

    os.makedirs(output_dir, exist_ok=True)
    temporary = output_path + '.tmp.npz'
    np.savez_compressed(
        temporary,
        candidate_indices=candidate_indices.cpu().numpy().astype(np.int64),
        loss=np.asarray(selected_loss, dtype=np.float64),
        gradnorm=np.asarray(selected_gradnorm, dtype=np.float64),
        forgetting=forgetting,
        model=np.asarray(args.model),
        surrogate_id=np.asarray(args.surrogate_id),
        model_seed=np.asarray(model_seed),
        epochs=np.asarray(args.epochs),
        select_epoch=np.asarray(args.select_epoch),
    )
    os.replace(temporary, output_path)
    if device.startswith('cuda'):
        torch.cuda.synchronize(device)
        cuda_peak_allocated = int(torch.cuda.max_memory_allocated(device))
        cuda_peak_reserved = int(torch.cuda.max_memory_reserved(device))
        cuda_final_allocated = int(torch.cuda.memory_allocated(device))
        cuda_final_reserved = int(torch.cuda.memory_reserved(device))
    else:
        cuda_peak_allocated = cuda_peak_reserved = 0
        cuda_final_allocated = cuda_final_reserved = 0
    resources = {
        'model': args.model,
        'surrogate_id': int(args.surrogate_id),
        'model_seed': int(model_seed),
        'candidate_count': int(len(candidate_indices)),
        'epochs': int(args.epochs),
        'select_epoch': int(args.select_epoch),
        'total_wall_seconds': float(time.perf_counter() - whole_started),
        'trajectory_wall_seconds': float(trajectory_wall_seconds),
        'trajectory_training_seconds_excluding_metric': float(
            trajectory_wall_seconds - metric_eval_wall_seconds),
        'metric_eval_wall_seconds': float(metric_eval_wall_seconds),
        'cuda_peak_allocated_bytes': cuda_peak_allocated,
        'cuda_peak_reserved_bytes': cuda_peak_reserved,
        'cuda_final_allocated_bytes': cuda_final_allocated,
        'cuda_final_reserved_bytes': cuda_final_reserved,
        'process_max_rss_kb': float(process_max_rss_kb()),
        'slurm_job_id': os.environ.get('SLURM_JOB_ID'),
    }
    resource_temporary = '%s.tmp.%d' % (resource_path, os.getpid())
    with open(resource_temporary, 'w') as handle:
        json.dump(resources, handle, indent=2, sort_keys=True)
    os.replace(resource_temporary, resource_path)
    print('saved:', output_path, flush=True)
    print('resources:', resource_path, flush=True)
    return output_path


def parse_args():
    parser = argparse.ArgumentParser(
        description='Precompute one Gao loss/gradnorm/forgetting selector shard')
    parser.add_argument('--dataset', default='CIFAR10')
    parser.add_argument('--data-path', default='./data')
    parser.add_argument('--cache-dir', default='./cache')
    parser.add_argument('--selector-metric-dir', default=None)
    parser.add_argument('--model', required=True,
                        choices=['ConvNetBN', 'ResNet20BN', 'VGG13BN'])
    parser.add_argument('--class-pair', default='dog-bird')
    parser.add_argument('--pair-order', default='poison-target',
                        choices=['poison-target', 'target-poison'])
    parser.add_argument('--surrogate-id', type=int, required=True)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--epochs', type=int, default=60)
    parser.add_argument('--select-epoch', type=int, default=10)
    parser.add_argument('--lr', type=float, default=0.1)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--grad-batch-size', type=int, default=8)
    parser.add_argument('--decay', type=int, nargs='*', default=[35, 45])
    parser.add_argument('--weight-decay', type=float, default=0.0)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--force', action='store_true')
    args = parser.parse_args()
    if args.surrogate_id < 0:
        parser.error('--surrogate-id must be non-negative')
    if not 1 <= args.select_epoch <= args.epochs:
        parser.error('--select-epoch must lie in 1..--epochs')
    if args.grad_batch_size <= 0:
        parser.error('--grad-batch-size must be positive')
    return args


if __name__ == '__main__':
    train_and_measure(parse_args())
