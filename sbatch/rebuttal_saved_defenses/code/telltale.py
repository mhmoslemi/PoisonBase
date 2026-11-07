"""Telltale paper pipeline with the released LSTM, blind to poison identity.

Source: https://github.com/MPaloze/Telltale (66c55fed231385255e7ed81ca61483583508cbb5).
Paper sections IV-B, V-A3 and VII-A specify truncation, amplitude+phase FFT,
and DBSCAN eps=5/min_samples=4. The demonstration's oracle group normalization
and magnitude-only/t-SNE clustering are deliberately not used here.
"""
import copy
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from common import atomic_json, pause_if_requested, save_torch
from methods import losses, restore_rng, rng_state


def plateau(validation):
    # Paper leaves the numerical plateau rule unspecified. Fix it in advance:
    # five epochs with <=1% relative range, after epoch 60, plus >=20 tail epochs.
    return next((i for i in range(60, len(validation)-19)
                 if np.ptp(validation[i-5:i]) /
                 max(float(np.mean(validation[i-5:i])), 1e-8) <= .01), None)


def train_sgd(net, x, y, directory, *, indices=None, epochs=50, bs=125,
              decay=(40,), validation_data=None, record_losses=False):
    """Original victim SGD recipe, or trajectory collector, saved every epoch."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    checkpoint = directory/'checkpoint.pt'
    opt = torch.optim.SGD(net.parameters(), lr=.1, momentum=.9, weight_decay=0.)
    start, traces, validation, history = 0, [], [], []
    if checkpoint.exists():
        state = torch.load(checkpoint, map_location='cpu', weights_only=False)
        net.load_state_dict(state['net'])
        opt.load_state_dict(state['opt'])
        start, history = state['epoch'], state['history']
        traces, validation = state['traces'], state['validation']
        restore_rng(state['rng'])
        print(f'SGD resume: {start}/{epochs} epochs complete in {directory}', flush=True)
        del state
    if indices is None:
        indices = torch.arange(len(y), device=y.device)
    for ep in range(start, epochs):
        before = time.monotonic()
        net.train()
        if ep in decay:
            for group in opt.param_groups:
                group['lr'] *= .1
        perm = indices[torch.randperm(len(indices), device=indices.device)]
        for batch in perm.split(bs):
            opt.zero_grad(set_to_none=True)
            F.cross_entropy(net(x[batch]), y[batch]).backward()
            opt.step()
        if record_losses:
            traces.append(losses(net, x, y).cpu())
            vx, vy = validation_data
            validation.append(float(losses(net, vx, vy).mean()))
        history.append(dict(epoch=ep+1, seconds=time.monotonic()-before))
        save_torch(checkpoint, dict(net=net.state_dict(), opt=opt.state_dict(),
                   epoch=ep+1, traces=traces, validation=validation,
                   history=history, rng=rng_state()))
        print('Trajectory' if record_losses else 'Retrain', ep+1, '/', epochs,
              history[-1], flush=True)
        pause_if_requested()
    net.eval()
    if record_losses:
        return torch.stack(traces, 1).numpy(), np.asarray(validation)
    return None


class TrajectoryAE(torch.nn.Module):
    """Layer layout and forward pass from official autoencoder.py:RNN."""
    def __init__(self, length):
        super().__init__()
        dim = int(length*2/3)
        self.rnn_1 = torch.nn.LSTM(length, 128, 3, bidirectional=True, batch_first=True)
        self.out_1 = torch.nn.Linear(128, dim)
        self.rnn_2 = torch.nn.LSTM(dim, 128, 3, bidirectional=True,
                                   batch_first=True, dropout=.1)
        self.out_2 = torch.nn.Linear(128, length)

    def forward(self, x):
        _, (h, _) = self.rnn_1(x[:, None, :])
        z = self.out_1(h[-1])
        _, (h, _) = self.rnn_2(z[:, None, :])
        return z, self.out_2(h[-1])


def fit_encoder(traces, cut, device, seed, directory):
    path = directory/'latent.npy'
    if path.exists():
        return np.load(path)
    x = torch.as_tensor(traces[:, cut:], device=device, dtype=torch.float32)
    generator = torch.Generator(device=device).manual_seed(seed)
    order = torch.randperm(len(x), generator=generator, device=device)
    train, val = x[order[:6000]], x[order[6000:8000]]
    net = TrajectoryAE(x.shape[1]).to(device)
    opt = torch.optim.Adam(net.parameters(), lr=.0005)
    checkpoint = directory/'autoencoder.pt'
    start, best, best_state = 0, float('inf'), None
    if checkpoint.exists():
        state = torch.load(checkpoint, map_location='cpu', weights_only=False)
        if state['cut'] != cut:
            raise RuntimeError('Truncation changed relative to the saved autoencoder.')
        net.load_state_dict(state['net'])
        opt.load_state_dict(state['opt'])
        start, best, best_state = state['step'], state['best'], state['best_state']
        restore_rng(state['rng'])
        print(f'Autoencoder resume: {start}/10 updates complete', flush=True)
        del state
    for step in range(start, 10):
        net.train()
        opt.zero_grad(set_to_none=True)
        F.mse_loss(net(train)[1], train).backward()
        opt.step()
        net.eval()
        with torch.no_grad():
            error = float(F.mse_loss(net(val)[1], val))
        if error < best:
            best = error
            best_state = {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}
        save_torch(checkpoint, dict(net=net.state_dict(), opt=opt.state_dict(),
                   step=step+1, cut=cut, best=best, best_state=best_state, rng=rng_state()))
        print(f'Autoencoder {step+1}/10 validation MSE={error:.6g}', flush=True)
        pause_if_requested()
    net.load_state_dict(best_state)
    net.eval()
    with torch.no_grad():
        latent = torch.cat([net(batch)[0].cpu() for batch in x.split(512)]).numpy()
    temporary = path.with_suffix('.tmp.npy')
    np.save(temporary, latent)
    os.replace(temporary, path)
    return latent


def streaming_dbscan(features, directory, eps=5., minimum=4):
    """Exact Euclidean DBSCAN with bounded neighbor memory and resumable expansion.

    sklearn's implementation keeps every radius neighborhood in RAM. Here the
    same core/border/noise rules use a cKDTree queried one neighborhood at a time.
    This avoids an O(N^2) resident graph for 50,000 samples under a 16G job limit.
    """
    from scipy.spatial import cKDTree
    checkpoint = directory/'clustering.pt'
    n = len(features)
    tree = cKDTree(np.asarray(features, dtype=np.float64))
    labels = np.full(n, -1, dtype=np.int32)
    core = np.zeros(n, dtype=bool)
    queued = np.zeros(n, dtype=bool)
    next_core, next_seed, cluster, queue, head = 0, 0, -1, [], 0
    if checkpoint.exists():
        state = torch.load(checkpoint, map_location='cpu', weights_only=False)
        labels, core, queued = state['labels'], state['core'], state['queued']
        next_core, next_seed, cluster = state['next_core'], state['next_seed'], state['cluster']
        queue, head = state['queue'], state['head']
        print(f'DBSCAN resume: core checks {next_core}/{n}, seed {next_seed}/{n}', flush=True)

    def persist():
        save_torch(checkpoint, dict(labels=labels, core=core, queued=queued,
                   next_core=next_core, next_seed=next_seed, cluster=cluster,
                   queue=queue, head=head))
        pause_if_requested()

    # The fourth nearest neighbor (including self) decides the min_samples rule.
    for first in range(next_core, n, 2048):
        last = min(first+2048, n)
        distances, _ = tree.query(features[first:last], k=minimum, workers=1)
        core[first:last] = distances[:, -1] <= eps
        next_core = last
        persist()
    while next_seed < n or head < len(queue):
        if head >= len(queue):
            queue, head = [], 0
            while next_seed < n and (not core[next_seed] or labels[next_seed] >= 0):
                next_seed += 1
            if next_seed >= n:
                break
            cluster += 1
            labels[next_seed] = cluster
            queued[next_seed] = True
            queue = [next_seed]
            next_seed += 1
        point = queue[head]
        neighbors = np.asarray(tree.query_ball_point(features[point], eps), dtype=np.int64)
        free = labels[neighbors] < 0
        labels[neighbors[free]] = cluster
        expand = neighbors[core[neighbors] & ~queued[neighbors]]
        queued[expand] = True
        queue.extend(expand.tolist())
        head += 1
        # Once every point is assigned, no unvisited component can remain.
        if np.all(labels >= 0):
            next_seed, head = n, len(queue)
        if head % 128 == 0 or head == len(queue):
            persist()
            print(f'DBSCAN cluster={cluster} expanded={head}/{len(queue)} '
                  f'assigned={int((labels >= 0).sum())}/{n}', flush=True)
    persist()
    return labels


def detect(traces, validation, device, seed, directory):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    result_path = directory/'detection.npz'
    if result_path.exists():
        with np.load(result_path) as data:
            return data['suspicious'].copy()
    cut = plateau(validation)
    if cut is None:
        raise RuntimeError('No validation plateau within the 120-epoch trace; '
                           'detection is incomplete, not an ASR=0 result.')
    latent = fit_encoder(traces, cut, device, seed, directory)
    normalized = (latent-latent.mean(0))/np.maximum(latent.std(0), 1e-8)
    fft = np.fft.fft(normalized, axis=1)
    features = np.concatenate([np.abs(fft), np.angle(fft)], axis=1).astype(np.float32)
    labels = streaming_dbscan(features, directory, eps=5., minimum=4)
    clusters, sizes = np.unique(labels[labels >= 0], return_counts=True)
    if len(clusters) == 0:
        raise RuntimeError('DBSCAN returned no non-noise cluster; no valid sanitized set.')
    benign = clusters[np.argmax(sizes)]
    suspicious = labels != benign
    temporary = directory/'detection.tmp.npz'
    np.savez_compressed(temporary, suspicious=suspicious, clusters=labels,
                        cut=cut, validation=validation)
    os.replace(temporary, result_path)
    atomic_json(directory/'detection.json', dict(cut_epoch=cut, flagged=int(suspicious.sum()),
                retained=int((~suspicious).sum()), eps=5, min_samples=4,
                spectrum='concatenated FFT amplitude and phase',
                normalization='one global fit over mixed latent vectors',
                benign_rule='largest non-noise cluster'))
    return suspicious
