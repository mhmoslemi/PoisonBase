"""W5 filters and checkpointed PGD adversarial training; no poison generation."""
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from common import atomic_json, pause_if_requested, save_torch
from methods import restore_rng, rng_state
from reference.art_cluster_functions import cluster_activations
from reference.art_clustering_analyzer import ClusteringAnalyzer


def deep_knn(features, labels, directory, k=5000, chunk=256):
    """Official sklearn plurality rule, including self, with bounded GPU memory."""
    directory = Path(directory)
    path = directory/'neighbors.pt'
    n = len(labels)
    if not 1 <= k <= n or features.ndim != 2 or len(features) != n:
        raise ValueError('Invalid Deep k-NN input or k.')
    labels = labels.to(features.device)
    kept = torch.zeros(n, dtype=torch.bool, device=features.device)
    start = 0
    if path.exists():
        state = torch.load(path, map_location='cpu', weights_only=False)
        if state['k'] != k or len(state['kept']) != n:
            raise RuntimeError('Deep k-NN checkpoint settings changed.')
        kept.copy_(state['kept'])
        start = state['next']
    before = time.monotonic()
    with torch.no_grad():
        norms = features.square().sum(1)
        for first in range(start, n, chunk):
            last = min(first+chunk, n)
            distance = (norms[first:last, None] + norms[None, :]
                        - 2*features[first:last]@features.T).clamp_min_(0)
            # The reference predicts on its training features, so self is included.
            distance[torch.arange(last-first, device=features.device),
                     torch.arange(first, last, device=features.device)] = 0
            near = distance.topk(k, largest=False, sorted=False).indices
            votes = torch.zeros((last-first, int(labels.max())+1),
                                dtype=torch.long, device=features.device)
            votes.scatter_add_(1, labels[near], torch.ones_like(near))
            kept[first:last] = votes.argmax(1) == labels[first:last]
            save_torch(path, dict(k=k, next=last, kept=kept.cpu()))
            if first == start or last == n or last % (10*chunk) == 0:
                print(f'Deep k-NN: {last}/{n}, {time.monotonic()-before:.1f}s', flush=True)
            pause_if_requested()
    return kept.cpu().numpy()


def activation_clustering(features, labels, directory, seed):
    """ART default: per-class PCA(10), KMeans(2), smaller-cluster analysis."""
    features = np.asarray(features)
    labels = np.asarray(labels)
    directory = Path(directory)
    path = directory/'clusters.pt'
    state = dict(next_class=0, kept=np.ones(len(labels), dtype=bool), classes=[])
    if path.exists():
        state = torch.load(path, map_location='cpu', weights_only=False)
    for cls in range(state['next_class'], int(labels.max())+1):
        np.random.seed(seed+cls)
        positions = np.flatnonzero(labels == cls)
        if len(positions) < 10:
            raise ValueError('Activation Clustering requires at least 10 samples per class.')
        before = time.monotonic()
        clusters, _ = cluster_activations([features[positions]], nb_clusters=2,
                                         nb_dims=10, reduce='PCA')
        clean, _, _ = ClusteringAnalyzer().analyze_by_size(clusters)
        state['kept'][positions] = np.asarray(clean[0], dtype=bool)
        state['classes'].append(dict(label=cls, sizes=np.bincount(clusters[0]).tolist()))
        state['next_class'] = cls+1
        save_torch(path, state)
        print(f'Activation Clustering: class {cls}, {state["classes"][-1]}, '
              f'{time.monotonic()-before:.1f}s', flush=True)
        pause_if_requested()
    return state['kept']


def pgd(net, pixels, labels, norm, *, epsilon=8/255, step=2/255, steps=10):
    """Madry CIFAR PGD: random start, CE ascent, box and L-inf projection.

    BatchNorm buffers are frozen during inner maximization; only the outer
    training forward updates them, once per minibatch.
    """
    training = net.training
    flags = [p.requires_grad for p in net.parameters()]
    net.eval()
    for p in net.parameters():
        p.requires_grad_(False)
    try:
        adv = (pixels+torch.empty_like(pixels).uniform_(-epsilon, epsilon)).clamp(0, 1)
        for _ in range(steps):
            adv = adv.detach().requires_grad_(True)
            loss = F.cross_entropy(net(norm(adv)), labels)
            gradient = torch.autograd.grad(loss, adv)[0]
            adv = adv.detach()+step*gradient.sign()
            adv = torch.maximum(torch.minimum(adv, pixels+epsilon), pixels-epsilon).clamp(0, 1)
        return adv.detach()
    finally:
        for p, flag in zip(net.parameters(), flags):
            p.requires_grad_(flag)
        net.train(training)


def train(net, x, y, directory, identity, *, indices=None, norm=None, denorm=None,
          adversarial=False, epochs=50, batch_size=125, decay=(40,), stop_after=None):
    """Matched victim SGD; atomic model/optimizer/RNG checkpoints every epoch."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory/'checkpoint.pt'
    protocol = dict(identity=identity, adversarial=adversarial, epochs=epochs,
                    batch_size=batch_size, decay=list(decay), lr=.1, momentum=.9,
                    weight_decay=0, pgd=dict(epsilon=8/255, step=2/255, steps=10))
    opt = torch.optim.SGD(net.parameters(), lr=.1, momentum=.9, weight_decay=0)
    start, history = 0, []
    if path.exists():
        state = torch.load(path, map_location='cpu', weights_only=False)
        if state['protocol'] != protocol:
            raise RuntimeError(f'Victim checkpoint protocol changed: {path}')
        net.load_state_dict(state['net'])
        opt.load_state_dict(state['opt'])
        start, history = state['epoch'], state['history']
        restore_rng(state['rng'])
        print(f'Resume: {start}/{epochs} epochs complete: {directory}', flush=True)
        del state
    if indices is None:
        indices = torch.arange(len(y), device=y.device)
    if len(indices) < batch_size:
        raise ValueError('Filtering left too few training samples.')
    if adversarial and (norm is None or denorm is None):
        raise ValueError('Adversarial training needs pixel normalization functions.')
    if stop_after is not None and start >= stop_after:
        net.eval()
        return history
    for ep in range(start, epochs):
        pause_if_requested()
        torch.cuda.synchronize() if x.is_cuda else None
        before = time.monotonic()
        net.train()
        if ep in decay:
            for group in opt.param_groups:
                group['lr'] *= .1
        order = indices[torch.randperm(len(indices), device=indices.device)]
        for batch in order.split(batch_size):
            images, labels = x[batch], y[batch]
            if adversarial:
                images = norm(pgd(net, denorm(images).clamp(0, 1), labels, norm))
            opt.zero_grad(set_to_none=True)
            F.cross_entropy(net(images), labels).backward()
            opt.step()
        torch.cuda.synchronize() if x.is_cuda else None
        row = dict(epoch=ep+1, seconds=time.monotonic()-before)
        history.append(row)
        save_torch(path, dict(protocol=protocol, epoch=ep+1, net=net.state_dict(),
                             opt=opt.state_dict(), rng=rng_state(), history=history))
        print('PGD-AT' if adversarial else 'Victim', ep+1, '/', epochs, row, flush=True)
        pause_if_requested()
        if stop_after is not None and ep+1 >= stop_after:
            break
    net.eval()
    return history
