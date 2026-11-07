"""ASD architecture adapter and label-blind adaptation of the Telltale artifact.

ASD: https://github.com/KuofengGao/ASD (ASD.py, model/loss.py,
utils/trainer/semi.py). Telltale: https://github.com/MPaloze/Telltale.
See README.md for explicit architecture/normalization adaptations.
"""
import copy
import inspect
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from protocol import atomic_json


def save_torch(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f'.{os.getpid()}.tmp')
    torch.save(value, tmp)
    os.replace(tmp, path)


def augment(x):
    b, _, h, w = x.shape
    padded = F.pad(x, (4, 4, 4, 4), mode='reflect').permute(0, 2, 3, 1)
    dy = torch.randint(9, (b, 1, 1), device=x.device)
    dx = torch.randint(9, (b, 1, 1), device=x.device)
    x = padded[torch.arange(b, device=x.device)[:, None, None],
               dy + torch.arange(h, device=x.device)[None, :, None],
               dx + torch.arange(w, device=x.device)[None, None, :]].permute(0, 3, 1, 2)
    flip = torch.rand(b, 1, 1, 1, device=x.device) < .5
    return torch.where(flip, x.flip(-1), x).contiguous()


def sce_loss(logits, labels):
    prob = logits.softmax(1).clamp(1e-7, 1)
    onehot = F.one_hot(labels, logits.shape[1]).float().clamp(1e-4, 1)
    return .1 * F.cross_entropy(logits, labels, reduction='none') - (prob*onehot.log()).sum(1)


@torch.no_grad()
def losses(net, x, y, sce=False, augmented=False, bs=256):
    net.eval()
    out = []
    for i in range(0, len(x), bs):
        z = net(augment(x[i:i+bs]) if augmented else x[i:i+bs])
        out.append(sce_loss(z, y[i:i+bs]) if sce else
                   F.cross_entropy(z, y[i:i+bs], reduction='none'))
    return torch.cat(out)


def interleave(chunks):
    b, n = len(chunks[0]), len(chunks)
    sizes = [b//n]*n
    for i in range(b-sum(sizes)):
        sizes[-i-1] += 1
    offsets = np.cumsum([0]+sizes).tolist()
    blocks = [[x[offsets[i]:offsets[i+1]] for i in range(n)] for x in chunks]
    for i in range(1, n):
        blocks[0][i], blocks[i][i] = blocks[i][i], blocks[0][i]
    return [torch.cat(p) for p in blocks]


def split_asd(loss, labels, trusted, epoch, meta_loss=None):
    mask = torch.zeros_like(labels, dtype=torch.bool)
    if epoch < 60:
        mask[trusted] = True
        count = (epoch//5)*10
        for c in range(10):
            candidates = ((labels == c) & ~mask).nonzero().flatten()
            chosen = candidates[torch.argsort(loss[candidates], stable=True)[:count]]
            mask[chosen] = True
    else:
        score = loss if epoch < 90 else loss - meta_loss
        mask[torch.argsort(score, stable=True)[:len(labels)//2]] = True
    return mask


def meta_parameters(net):
    # The official ResNet18 adapts its last two stages and classifier.
    if hasattr(net, 'layer3'):
        modules = [net.layer2, net.layer3, net.classifier]
    else:
        pools = [i for i, m in enumerate(net.features)
                 if isinstance(m, (torch.nn.AvgPool2d, torch.nn.MaxPool2d))]
        assert len(pools) >= 2
        modules = [net.features[pools[-3]+1:] if len(pools) >= 3 else net.features,
                   net.classifier]
    for p in net.parameters():
        p.requires_grad_(False)
    params = [p for m in modules for p in m.parameters()]
    for p in params:
        p.requires_grad_(True)
    return params


class PoolBatches:
    def __init__(self, indices, size=64):
        self.indices, self.size = indices, size
        self.perm = indices[:0]
        self.pos = 0
        assert len(indices) >= size

    def next(self):
        if self.pos + self.size > len(self.perm):
            self.perm = self.indices[torch.randperm(len(self.indices), device=self.indices.device)]
            self.pos = 0
        result = self.perm[self.pos:self.pos+self.size]
        self.pos += self.size
        return result


def mixmatch_epoch(net, opt, x, y, labeled, epoch, steps=1024):
    xp = PoolBatches(labeled.nonzero().flatten())
    up = PoolBatches((~labeled).nonzero().flatten())
    net.train()
    total = 0.
    for step in range(steps):
        ix, iu = xp.next(), up.next()
        xi, u1, u2 = augment(x[ix]), augment(x[iu]), augment(x[iu])
        xt = F.one_hot(y[ix], 10).float()
        with torch.no_grad():
            p = (net(u1).softmax(1)+net(u2).softmax(1))/2
            p = p.square()
            ut = p/p.sum(1, keepdim=True)
        inputs = torch.cat([xi, u1, u2])
        targets = torch.cat([xt, ut, ut])
        coefficient = float(np.random.beta(.75, .75))
        coefficient = max(coefficient, 1-coefficient)
        permutation = torch.randperm(len(inputs), device=x.device)
        mixed = coefficient*inputs + (1-coefficient)*inputs[permutation]
        target = coefficient*targets + (1-coefficient)*targets[permutation]
        logits = interleave([net(z) for z in interleave(list(mixed.split(64)))])
        lx = -(logits[0].log_softmax(1)*target[:64]).sum(1).mean()
        lu = (torch.cat(logits[1:]).softmax(1)-target[64:]).square().mean()
        loss = lx + 15*min((epoch+step/steps)/120, 1)*lu
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        total += float(loss.detach())
    return total/steps


def rng_state():
    return dict(torch=torch.get_rng_state(), cuda=torch.cuda.get_rng_state_all(),
                numpy=np.random.get_state())


def restore_rng(state):
    torch.set_rng_state(state['torch'].cpu())
    torch.cuda.set_rng_state_all([x.cpu() for x in state['cuda']])
    np.random.set_state(state['numpy'])


def train_asd(net, x, y, trusted, directory, epochs=120, steps=1024):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    checkpoint = directory/'checkpoint.pt'
    opt = torch.optim.Adam(net.parameters(), lr=.002)
    start, history = 0, []
    assignments, score_traces = {}, {}
    if checkpoint.exists():
        state = torch.load(checkpoint, map_location='cpu', weights_only=False)
        net.load_state_dict(state['net']); opt.load_state_dict(state['opt'])
        start, history = state['epoch'], state['history']
        assignments = state.get('assignments', {})
        score_traces = state.get('score_traces', {})
        restore_rng(state['rng'])
    for ep in range(start, epochs):
        before = time.monotonic()
        score = losses(net, x, y, sce=True, augmented=True)
        meta_score = None
        if ep >= 90:
            meta = copy.deepcopy(net)
            meta_opt = torch.optim.Adam(meta_parameters(meta), lr=.015)
            meta.train()
            perm = torch.randperm(len(y), device=x.device)
            for ids in perm.split(128):
                meta_opt.zero_grad(set_to_none=True)
                F.cross_entropy(meta(augment(x[ids])), y[ids]).backward()
                meta_opt.step()
            meta_score = losses(meta, x, y, sce=True, augmented=True)
            del meta, meta_opt
        labeled = split_asd(score, y, trusted, ep, meta_score)
        mean_loss = mixmatch_epoch(net, opt, x, y, labeled, ep, steps)
        # Assignment vectors contain no oracle labels; analysis joins them later.
        history.append(dict(epoch=ep+1, labeled=int(labeled.sum()), loss=mean_loss,
                            seconds=time.monotonic()-before))
        if (ep+1) % 5 == 0 or ep+1 == epochs:
            assignments[ep+1] = labeled.cpu()
            score_traces[ep+1] = dict(split_sce=score.cpu(),
                                    evaluation_ce=losses(net, x, y).cpu())
            save_torch(directory/'assignments.pt', assignments)
            save_torch(directory/'loss_snapshots.pt', score_traces)
            save_torch(checkpoint, dict(net=net.state_dict(), opt=opt.state_dict(),
                       epoch=ep+1, history=history, assignments=assignments,
                       score_traces=score_traces, rng=rng_state()))
            atomic_json(directory/'training.json', history)
            print('ASD', ep+1, history[-1], flush=True)
    return net.eval()


def train_trajectory(net, x, y, vx, vy, directory, epochs=120):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    checkpoint = directory/'checkpoint.pt'
    opt = torch.optim.SGD(net.parameters(), lr=.1, momentum=.9)
    traces, validation, start = [], [], 0
    if checkpoint.exists():
        state = torch.load(checkpoint, map_location=x.device, weights_only=False)
        net.load_state_dict(state['net']); opt.load_state_dict(state['opt'])
        traces, validation, start = state['traces'], state['validation'], state['epoch']
        traces = [t.cpu() for t in traces]
        restore_rng(state['rng'])
    for ep in range(start, epochs):
        net.train()
        if ep in (70, 90):
            for group in opt.param_groups:
                group['lr'] *= .1
        for ids in torch.randperm(len(y), device=x.device).split(128):
            opt.zero_grad(set_to_none=True)
            F.cross_entropy(net(x[ids]), y[ids]).backward()
            opt.step()
        traces.append(losses(net, x, y).cpu())
        validation.append(float(losses(net, vx, vy).mean()))
        if (ep+1) % 10 == 0 or ep+1 == epochs:
            save_torch(checkpoint, dict(net=net.state_dict(), opt=opt.state_dict(),
                       epoch=ep+1, traces=traces, validation=validation, rng=rng_state()))
            print('Trajectory', ep+1, 'validation loss', validation[-1], flush=True)
    return net.eval(), torch.stack(traces, 1).numpy(), np.asarray(validation)


class TrajectoryAE(torch.nn.Module):
    """Architecture and ten full-batch updates from the released Telltale AE."""
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


def global_normalize(x):
    return (x-x.mean(0))/np.maximum(x.std(0), 1e-8)


def suspect_clusters(labels):
    clusters, sizes = np.unique(labels[labels >= 0], return_counts=True)
    if len(clusters) == 0:
        raise RuntimeError('Telltale found no benign cluster; no valid filtered training set')
    largest = clusters[np.argmax(sizes)]
    return labels != largest


def detect_telltale(traces, validation, device, seed, directory):
    from sklearn.cluster import DBSCAN
    from sklearn.manifold import TSNE
    directory = Path(directory)
    # Predeclared plateau: relative range of five validation epochs <= 1%,
    # after epoch 60; retain at least 20 post-convergence epochs.
    cut = next((i for i in range(60, len(validation)-19)
                if np.ptp(validation[i-5:i])/max(np.mean(validation[i-5:i]), 1e-8) <= .01), None)
    if cut is None:
        raise RuntimeError('No validation plateau: extend trajectory training before interpreting defense')
    x = torch.as_tensor(traces[:, cut:], device=device, dtype=torch.float32)
    gen = torch.Generator(device=device).manual_seed(seed)
    order = torch.randperm(len(x), generator=gen, device=device)
    train, val = x[order[:6000]], x[order[6000:8000]]
    net = TrajectoryAE(x.shape[1]).to(device)
    opt = torch.optim.Adam(net.parameters(), lr=.0005)
    best, best_state = float('inf'), None
    for _ in range(10):
        net.train(); opt.zero_grad(set_to_none=True)
        F.mse_loss(net(train)[1], train).backward(); opt.step()
        net.eval()
        with torch.no_grad():
            err = float(F.mse_loss(net(val)[1], val))
        if err < best:
            best, best_state = err, copy.deepcopy(net.state_dict())
    net.load_state_dict(best_state); net.eval()
    with torch.no_grad():
        latent = torch.cat([net(b)[0].cpu() for b in x.split(512)]).numpy()
    del net, opt, x, train, val
    # One transform fitted to the entire mixed dataset, never to oracle groups.
    spectrum = np.abs(np.fft.fft(global_normalize(latent), axis=1)).astype('float32')
    opts = dict(n_components=2, random_state=seed, perplexity=30,
                init='pca', learning_rate=200., n_jobs=1)
    opts['max_iter' if 'max_iter' in inspect.signature(TSNE).parameters else 'n_iter'] = 1500
    embedding = TSNE(**opts).fit_transform(spectrum)
    labels = DBSCAN(eps=5, min_samples=5, n_jobs=1).fit_predict(embedding)
    suspicious = suspect_clusters(labels)
    np.savez_compressed(directory/'detection.npz', suspicious=suspicious,
                        embedding=embedding, clusters=labels, cut=cut,
                        validation=validation)
    return suspicious, dict(cut_epoch=cut, n_flagged=int(suspicious.sum()),
                            adaptation='artifact pipeline; global label-blind normalization; largest cluster benign')
