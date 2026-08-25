#!/usr/bin/env python3
"""Small CPU timing/peak-RSS comparison for exact Jacobian interactions.

Run from the repository root:
    python3 extra/benchmark_jacobian_score.py

Each method runs in a fresh process so peak resident memory is comparable.  The
explicit reference performs one candidate backward pass at a time; the fast
method never creates an N-by-P candidate-gradient tensor.
"""

import json
import os
import resource
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import torch
import torch.nn as nn

import final_update as FU
from extra.test_jacobian_score import explicit_interactions


class BenchNet(nn.Module):

    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Linear(64, 128), nn.Tanh(), nn.Linear(128, 96), nn.Tanh())
        self.classifier = nn.Linear(96, 10)

    def embed(self, x):
        return self.features(x)

    def forward(self, x):
        return self.classifier(self.embed(x))


def _peak_rss_mib():
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # macOS reports bytes; Linux reports KiB.
    return value / (1024.0 * 1024.0) if sys.platform == 'darwin' else value / 1024.0


def worker(method):
    torch.manual_seed(123)
    net = BenchNet().eval()
    candidates = torch.randn(2048, 64)
    target = torch.randn(64)
    started = time.perf_counter()
    if method == 'fast':
        values, backend = FU._backbone_gradient_interactions(
            net, candidates, target, y_adv=3, batch_size=64)
    else:
        values = explicit_interactions(net, candidates, target, y_adv=3)
        backend = 'per-candidate autograd reference'
    elapsed = time.perf_counter() - started
    print(json.dumps({'method': method, 'backend': backend, 'seconds': elapsed,
                      'peak_rss_mib': _peak_rss_mib(), 'values': values.tolist(),
                      'candidates': len(candidates),
                      'parameters': sum(p.numel() for p in net.parameters())}))


def run_one(method):
    command = [sys.executable, os.path.abspath(__file__), '--worker', method]
    output = subprocess.check_output(command, text=True)
    # Backend logging can precede the JSON line.
    return json.loads(output.strip().splitlines()[-1])


def main():
    explicit = run_one('explicit')
    fast = run_one('fast')
    error = max(abs(a - b) for a, b in zip(explicit['values'], fast['values']))
    report = {
        'max_abs_error': error,
        'explicit_seconds': explicit['seconds'],
        'fast_seconds': fast['seconds'],
        'explicit_peak_rss_mib': explicit['peak_rss_mib'],
        'fast_peak_rss_mib': fast['peak_rss_mib'],
        'fast_backend': fast['backend'],
        'candidates': fast['candidates'],
        'parameters': fast['parameters'],
        'avoided_n_by_p_tensor_mib': (fast['candidates'] * fast['parameters'] * 4 /
                                      (1024.0 * 1024.0)),
        'candidate_gradient_matrix_materialized': False,
    }
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    if len(sys.argv) == 3 and sys.argv[1] == '--worker':
        worker(sys.argv[2])
    else:
        main()
