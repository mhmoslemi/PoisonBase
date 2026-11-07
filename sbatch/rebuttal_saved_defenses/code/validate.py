"""Small resume/equivalence checks, run only inside each GPU allocation."""
import argparse
import random
from pathlib import Path

import numpy as np
import torch
from sklearn.cluster import DBSCAN

import methods
import telltale
from common import Paused


def network():
    random.seed(123)
    np.random.seed(123)
    torch.manual_seed(123)
    return torch.nn.Sequential(
        torch.nn.Conv2d(3, 4, 3, padding=1), torch.nn.BatchNorm2d(4),
        torch.nn.ReLU(), torch.nn.AdaptiveAvgPool2d(1), torch.nn.Flatten(),
        torch.nn.Linear(4, 10))


def equal(left, right):
    for key, value in left.state_dict().items():
        torch.testing.assert_close(value, right.state_dict()[key], rtol=0, atol=0)


def run(directory):
    torch.set_num_threads(1)
    x = torch.randn(200, 3, 8, 8, generator=torch.Generator().manual_seed(713))
    y = torch.arange(200) % 10
    trusted = torch.arange(100)

    uninterrupted = network()
    telltale.train_sgd(uninterrupted, x, y, directory/'sgd_full', epochs=2, bs=64, decay=(1,))
    interrupted = network()
    telltale.train_sgd(interrupted, x, y, directory/'sgd_retry', epochs=1, bs=64, decay=(1,))
    resumed = network()
    telltale.train_sgd(resumed, x, y, directory/'sgd_retry', epochs=2, bs=64, decay=(1,))
    equal(uninterrupted, resumed)

    uninterrupted = network()
    methods.train_asd(uninterrupted, x, y, trusted, directory/'asd_full', epochs=2, steps=2)
    interrupted = network()
    methods.train_asd(interrupted, x, y, trusted, directory/'asd_retry', epochs=1, steps=2)
    resumed = network()
    methods.train_asd(resumed, x, y, trusted, directory/'asd_retry', epochs=2, steps=2)
    equal(uninterrupted, resumed)

    generator = np.random.default_rng(17)
    features = np.concatenate([
        generator.normal((0, 0), .08, (220, 2)),
        generator.normal((5, 5), .08, (220, 2)),
        np.array([[20., 20.], [30., 30.]])])
    expected = DBSCAN(eps=.4, min_samples=4).fit_predict(features)
    cluster_dir = directory/'clustering'
    cluster_dir.mkdir(parents=True, exist_ok=True)
    original_pause = telltale.pause_if_requested
    boundaries = 0

    def interrupt_expansion():
        nonlocal boundaries
        boundaries += 1
        if boundaries == 2:
            raise Paused('Intentional interruption during DBSCAN expansion.')

    telltale.pause_if_requested = interrupt_expansion
    try:
        telltale.streaming_dbscan(features, cluster_dir, eps=.4, minimum=4)
    except Paused:
        pass
    finally:
        telltale.pause_if_requested = original_pause
    assert boundaries == 2
    observed = telltale.streaming_dbscan(features, cluster_dir, eps=.4, minimum=4)
    np.testing.assert_array_equal(observed, expected)

    # The trusted seed split and official BatchNorm interleaving stay intact.
    assert int(methods.split_asd(torch.arange(200).float(), y, trusted, 0).sum()) == 100
    chunks = [torch.arange(i*64, (i+1)*64) for i in range(3)]
    for left, right in zip(chunks, methods.interleave(methods.interleave(chunks))):
        torch.testing.assert_close(left, right)
    print('PASS: SGD and ASD resume exactly; resumed DBSCAN matches sklearn.', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('directory', type=Path)
    args = parser.parse_args()
    args.directory.mkdir(parents=True, exist_ok=True)
    run(args.directory)
