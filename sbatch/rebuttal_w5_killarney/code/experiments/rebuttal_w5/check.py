"""Reference filter agreement, PGD constraints, and exact training resume."""
import ast
import contextlib
import copy
import io
import json
import os
from pathlib import Path
import tempfile

import numpy as np
import torch
from sklearn.neighbors import KNeighborsClassifier

import run as w5
from defenses import activation_clustering, deep_knn, pgd, train
from reference.art_cluster_functions import cluster_activations
from reference.art_clustering_analyzer import ClusteringAnalyzer


def check():
    if not os.environ.get('SLURM_JOB_ID'):
        raise RuntimeError('Run checks inside a Slurm allocation.')
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    reference = Path(__file__).parent/'reference'
    original = ast.parse((reference/'art_activation_defence.py').read_text())
    extracted = ast.parse((reference/'art_cluster_functions.py').read_text())
    names = {'cluster_activations', 'reduce_dimensionality'}
    expected = {node.name: ast.dump(node) for node in original.body
                if isinstance(node, ast.FunctionDef) and node.name in names}
    actual = {node.name: ast.dump(node) for node in extracted.body
              if isinstance(node, ast.FunctionDef) and node.name in names}
    assert expected == actual, 'Vendored ART functions differ from the pinned source.'
    with tempfile.TemporaryDirectory(dir=os.environ['SLURM_TMPDIR']) as temporary:
        root = Path(temporary)
        rng = np.random.default_rng(123)
        features = rng.normal(size=(120, 14)).astype(np.float32)
        labels = np.arange(120) % 3
        oracle = KNeighborsClassifier(n_neighbors=11, algorithm='brute').fit(features, labels)
        expected = oracle.predict(features) == labels
        actual = deep_knn(torch.tensor(features, device='cuda'),
                          torch.tensor(labels, device='cuda'), root/'knn', k=11, chunk=17)
        assert np.array_equal(expected, actual)
        assert np.array_equal(actual, deep_knn(torch.tensor(features, device='cuda'),
            torch.tensor(labels, device='cuda'), root/'knn', k=11, chunk=17))
        expected = np.ones(len(labels), dtype=bool)
        for cls in range(3):
            np.random.seed(100+cls)
            where = np.flatnonzero(labels == cls)
            clusters, _ = cluster_activations([features[where]], reduce='PCA')
            clean, _, _ = ClusteringAnalyzer().analyze_by_size(clusters)
            expected[where] = np.asarray(clean[0], dtype=bool)
        assert np.array_equal(expected,
            activation_clustering(features, labels, root/'ac', 100))

        w5.fu.set_seed(321)
        net = torch.nn.Sequential(torch.nn.BatchNorm1d(4), torch.nn.Linear(4, 2)).cuda()
        net.train()
        pixels = torch.rand(12, 4, device='cuda')
        classes = torch.arange(12, device='cuda') % 2
        before = copy.deepcopy(net.state_dict())
        adv = pgd(net, pixels, classes, lambda x: x)
        assert net.training and not adv.requires_grad
        assert float((adv-pixels).abs().max()) <= 8/255+1e-6
        assert float(adv.min()) >= 0 and float(adv.max()) <= 1
        assert all(torch.equal(before[k], v) for k, v in net.state_dict().items())
        assert all(p.requires_grad for p in net.parameters())
        linear = torch.nn.Linear(4, 2, bias=False).cuda()
        with torch.no_grad():
            linear.weight.copy_(torch.tensor([[1., 1., 1., 1.], [-1., -1., -1., -1.]], device='cuda'))
        middle = torch.full((2, 4), .5, device='cuda')
        attacked = pgd(linear, middle, torch.zeros(2, dtype=torch.long, device='cuda'), lambda x: x)
        assert torch.allclose(attacked, middle-8/255), 'PGD must ascend CE, not descend it.'

        for adversarial in (False, True):
            w5.fu.set_seed(456)
            expected_net = torch.nn.Linear(4, 2).cuda()
            train(expected_net, pixels, classes, root/f'full_{adversarial}', {},
                  adversarial=adversarial, norm=lambda x: x, denorm=lambda x: x,
                  epochs=3, batch_size=4, decay=(1,))
            w5.fu.set_seed(456)
            resumed = torch.nn.Linear(4, 2).cuda()
            train(resumed, pixels, classes, root/f'resume_{adversarial}', {},
                  adversarial=adversarial, norm=lambda x: x, denorm=lambda x: x,
                  epochs=3, batch_size=4, decay=(1,), stop_after=1)
            stopped = train(resumed, pixels, classes, root/f'resume_{adversarial}', {},
                            adversarial=adversarial, norm=lambda x: x, denorm=lambda x: x,
                            epochs=3, batch_size=4, decay=(1,), stop_after=1)
            assert len(stopped) == 1, 'A completed timing limit must not run extra epochs.'
            # Perturb RNG and weights: continuation must restore both.
            torch.rand(51, device='cuda')
            resumed = torch.nn.Linear(4, 2).cuda()
            train(resumed, pixels, classes, root/f'resume_{adversarial}', {},
                  adversarial=adversarial, norm=lambda x: x, denorm=lambda x: x,
                  epochs=3, batch_size=4, decay=(1,))
            assert all(torch.equal(value, resumed.state_dict()[key])
                       for key, value in expected_net.state_dict().items())
            if not adversarial:
                w5.fu.set_seed(456)
                ordinary = torch.nn.Linear(4, 2).cuda()
                w5.fu.train_from_scratch(ordinary, pixels, classes, 3, .1, 4, (1,), 'cuda')
                assert all(torch.equal(value, ordinary.state_dict()[key])
                           for key, value in expected_net.state_dict().items())
        original_output = w5.OUT
        try:
            w5.OUT = root/'aggregate'
            for cell_id, success in [('conv_random_b0001', 1), ('conv_basis_b0001', 0)]:
                record = w5.ROOT/'rebuttal_saved_defenses_result/poison_records'/cell_id/'target_843.json'
                result = dict(version=w5.VERSION, cell=cell_id, defense='advtrain',
                              target=843, victim=0, seed=4284300, success=success,
                              clean_accuracy=.8,
                              poison_sha256=json.loads(record.read_text())['poison_sha256'])
                w5.atomic_json(w5.OUT/'advtrain'/cell_id/'target_843/victim_0/result.json', result)
            with contextlib.redirect_stdout(io.StringIO()):
                w5.aggregate()
            rows = json.loads((w5.OUT/'summary.json').read_text())
            assert len(rows) == 36
            zero = next(r for r in rows if r['cell'] == 'conv_random_b0001' and r['defense'] == 'advtrain')
            partial = next(r for r in rows if r['cell'] == 'conv_basis_b0001' and r['defense'] == 'advtrain')
            assert zero['asr'] == 0 and zero['cta'] == 80 and zero['completed'] == 1
            assert partial['completed'] == 1 and partial['expected'] == 40 and 'asr' not in partial
        finally:
            w5.OUT = original_output
    print('PASS: official k-NN/ART agreement; PGD bounds/direction and BN freeze; '
          'exact SGD/AT resume, original SGD agreement, and complete-only aggregation.', flush=True)


if __name__ == '__main__':
    check()
