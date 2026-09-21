#!/usr/bin/env python3
"""Small CPU checks for the ImageNet protocol and checkpoint continuity."""
import contextlib
import copy
import io
import json
from pathlib import Path
import random
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from torch import nn
from torch.utils.data import TensorDataset
from experiments import imagenet100_gm as run
import final_update as fu


class Tests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(12)
        run.STOP = False

    def test_saved_subset_and_pair_are_seed_zero_and_stable(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            names = [f'n{i:08d}' for i in range(1000)]
            for name in names:
                (root / 'data/train' / name).mkdir(parents=True)
            selected = sorted(random.Random(0).sample(names, 100))
            for name in selected:
                (root / 'data/val' / name).mkdir(parents=True)
                for i in range(10):
                    (root / 'data/train' / name / f'{i}.JPEG').touch()
                (root / 'data/val' / name / '0.JPEG').touch()
            out = root / 'result'
            with contextlib.redirect_stdout(io.StringIO()):
                run.prepare(root / 'data', out)
                first = (out / 'manifest.json').read_text()
                run.prepare(root / 'data', out)
            self.assertEqual(first, (out / 'manifest.json').read_text())
            manifest = json.loads(first)
            self.assertEqual(manifest['classes'], selected)
            self.assertEqual(manifest['num_poisons'], 2)
            self.assertEqual([manifest['poison_class'], manifest['target_class']],
                             random.Random(0).sample(selected, 2))
            self.assertEqual(len((out / 'classes_seed0.txt').read_text().splitlines()), 100)

    def test_standard_imagenet_resnet_and_input_dimensions(self):
        model = run.ImageNetResNet18().eval()
        self.assertEqual(model.net.conv1.kernel_size, (7, 7))
        self.assertEqual(model.net.conv1.stride, (2, 2))
        self.assertEqual(model.net.fc.out_features, 100)
        with torch.no_grad():
            self.assertEqual(tuple(model(torch.zeros(1, 3, 224, 224)).shape), (1, 100))

    def test_native_pixel_bound_and_crafting_view_match_victim_injection(self):
        originals = [torch.rand(3, 300, 450), torch.rand(3, 400, 260)]
        canonical = torch.stack([run.eval_pixels(x) for x in originals])
        delta = torch.empty_like(canonical).uniform_(-8/255, 8/255)
        proposed = (canonical + delta).clamp(0, 1).requires_grad_()
        actual = run.CraftView(originals, canonical, 'cpu')(proposed, 0, 2)
        expected = []
        for x, d in zip(originals, proposed - canonical):
            poisoned = run.native_poison(x, d)
            self.assertEqual(poisoned.shape, x.shape)
            self.assertLessEqual(float((poisoned - x).detach().abs().max()), 8/255 + 1e-6)
            expected.append(run.normalize(run.eval_pixels(poisoned)))
        torch.testing.assert_close(actual, torch.stack(expected))
        derivative = torch.autograd.grad(actual.square().mean(), proposed)[0]
        self.assertTrue(torch.isfinite(derivative).all())
        self.assertGreater(float(derivative.abs().sum()), 0)

    def test_gm_resume_reproduces_uninterrupted_adam(self):
        base = torch.rand(4, 3, 4, 4)
        target = torch.rand(3, 4, 4)
        net = nn.Sequential(nn.Flatten(), nn.Linear(48, 3)).eval()
        original = copy.deepcopy(net.state_dict())
        def optimize(**kwargs):
            net.load_state_dict(original)
            return fu.craft_gradmatch([net], base, target, 1, lambda x: x,
                                      8/255, 1/255, 5, 2, 'cpu',
                                      lowmem=True, chunk=2, schedule=True, **kwargs)
        torch.manual_seed(91)
        expected, obj = optimize()
        saved = []
        def interrupt(state):
            if state['restart'] == 0 and state['next_step'] == 2:
                saved.append(copy.deepcopy(state))
                raise run.Paused()
        torch.manual_seed(91)
        with self.assertRaises(run.Paused):
            optimize(checkpoint_callback=interrupt)
        torch.manual_seed(999)
        actual, actual_obj = optimize(resume_state=saved[0])
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        self.assertEqual(actual_obj, obj)

    def test_custom_view_microbatch_gradient_matches_full_batch(self):
        net = nn.Sequential(nn.Flatten(), nn.Linear(48, 3)).eval()
        base = torch.rand(4, 3, 4, 4)
        delta = torch.zeros_like(base, requires_grad=True)
        gt = torch.randn(sum(p.numel() for p in net.parameters()))
        labels = torch.ones(4, dtype=torch.long)
        transform = lambda x, start, end: x.square() + 0.1
        args = (net, gt, base, delta, labels, lambda x: x, nn.CrossEntropyLoss())
        full, obj = fu._gradmatch_net_grad(*args, 4, False, None, None, -1,
                                          poison_transform=transform)
        chunked, chunk_obj = fu._gradmatch_net_grad(*args, 2, False, None, None, -1,
                                                   poison_transform=transform)
        torch.testing.assert_close(chunked, full, rtol=1e-5, atol=1e-7)
        self.assertAlmostEqual(obj, chunk_obj, places=6)

    def test_training_resume_preserves_sgd_and_schedule(self):
        data = TensorDataset(torch.rand(130, 4), torch.zeros(130, dtype=torch.long))
        factory = lambda: nn.Linear(4, 100)
        manifest = dict(fingerprint='test', train=[0] * 130, val=[0] * 130)
        step = torch.optim.SGD.step
        calls = [0]
        def interrupt(opt, *args, **kwargs):
            answer = step(opt, *args, **kwargs)
            calls[0] += 1
            if calls[0] == 3:
                run.STOP = True
            return answer
        with tempfile.TemporaryDirectory() as tmp, \
             patch.object(run, 'Images', return_value=data), \
             patch.object(run, 'ImageNetResNet18', side_effect=factory), \
             contextlib.redirect_stdout(io.StringIO()):
            a, b = Path(tmp) / 'a.pt', Path(tmp) / 'b.pt'
            expected, _ = run.train_model(a, manifest, Path(tmp), 'cpu', 0, 1042)
            with patch.object(torch.optim.SGD, 'step', new=interrupt):
                with self.assertRaises(run.Paused):
                    run.train_model(b, manifest, Path(tmp), 'cpu', 0, 1042)
            self.assertEqual(run.load(b)['offset'], 128)
            run.STOP = False
            actual, _ = run.train_model(b, manifest, Path(tmp), 'cpu', 0, 1042)
            for key, value in expected.state_dict().items():
                torch.testing.assert_close(value, actual.state_dict()[key], rtol=0, atol=0)
            self.assertTrue(run.load(b)['complete'])
            self.assertEqual(run.load(b)['epoch'], 90)


if __name__ == '__main__':
    unittest.main()
