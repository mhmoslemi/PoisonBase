#!/usr/bin/env python3
"""Benchmark the repository's three CIFAR-10 surrogate architectures.

Running ``python surr_time.py`` trains ConvNetBN, ResNet20BN, and VGG13BN five
times each with the exact surrogate settings used by ``final_update.py``.  It
saves no checkpoints, logs, or result files; only the per-model averages are
printed after every training run has finished.
"""

import argparse
import gc
import os
import statistics
import time

import torch

import final_update as poison


MODELS = ('ConvNetBN', 'ResNet20BN', 'VGG13BN')
REPEATS = 5
SEED = 42
SURROGATE_EPOCHS = 60
SURROGATE_LR = 0.1
SURROGATE_BATCH_SIZE = 128
SURROGATE_DECAY = (35, 45)
SURROGATE_WEIGHT_DECAY = 0.0


def parse_args():
    default_data = os.environ.get(
        'DATA_PATH', os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data'))
    parser = argparse.ArgumentParser(
        description='Time and peak-memory benchmark for the three surrogate models')
    parser.add_argument('--data-path', default=default_data,
                        help='CIFAR-10 directory (default: DATA_PATH or ./data)')
    return parser.parse_args()


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise SystemExit('CUDA is required: no GPU is visible')

    device = 'cuda:0'
    torch.cuda.set_device(0)
    channel, image_size, num_classes, _class_names, _mean, _std, train_set, _test_set, _ = \
        poison.get_dataset('CIFAR10', args.data_path)
    train_images, train_labels = poison.stack_dataset(train_set, device)

    results = {}
    for model_name in MODELS:
        elapsed_times = []
        peak_memories = []
        for repeat in range(REPEATS):
            gc.collect()
            torch.cuda.empty_cache()
            net = poison.build_network(
                model_name, channel, num_classes, image_size, device,
                seed=SEED + 1000 + repeat)

            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats(0)
            started = time.perf_counter()
            poison.train_from_scratch(
                net, train_images, train_labels,
                epochs=SURROGATE_EPOCHS,
                lr=SURROGATE_LR,
                bs=SURROGATE_BATCH_SIZE,
                decay_at=SURROGATE_DECAY,
                device=device,
                weight_decay=SURROGATE_WEIGHT_DECAY,
                aug=False,
            )
            torch.cuda.synchronize()
            elapsed_times.append(time.perf_counter() - started)
            peak_memories.append(torch.cuda.max_memory_allocated(0))

            del net
            gc.collect()
            torch.cuda.empty_cache()

        results[model_name] = (
            statistics.mean(elapsed_times),
            statistics.mean(peak_memories) / (1024 ** 3),
        )

    print('model       avg_time_seconds  avg_peak_gpu_memory_GiB')
    for model_name in MODELS:
        avg_time, avg_memory = results[model_name]
        print('%-12s %16.2f %24.3f' % (model_name, avg_time, avg_memory))


if __name__ == '__main__':
    main()
