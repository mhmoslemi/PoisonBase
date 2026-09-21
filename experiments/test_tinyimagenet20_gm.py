#!/usr/bin/env python3
"""CPU checks for subsampling, native poisons, and safe batch submission."""
import contextlib
import io
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from experiments import tinyimagenet20_gm as tiny
from experiments import imagenet100_gm as shared

ROOT = Path(__file__).resolve().parents[1]


def archive(path):
    # Expanded views store just one image while exercising real archive indexing.
    classes = [f'n{i:08d}' for i in range(200)]
    image = torch.arange(3 * 64 * 64).to(torch.uint8).reshape(1, 3, 64, 64)
    torch.save(dict(classes=classes, images_train=image.expand(40000, -1, -1, -1),
                    images_val=image.expand(10000, -1, -1, -1),
                    labels_train=torch.arange(200).repeat_interleave(200),
                    labels_val=torch.arange(200).repeat_interleave(50)), path)
    return classes


class Tests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.previous = shared.PROTOCOL
        shared.PROTOCOL = tiny.PROTOCOL
        shared.STOP = False

    def tearDown(self):
        shared.PROTOCOL = self.previous

    def test_exact_small_subset_and_lossless_export(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / 'tinyimagenet.pt'
            classes = archive(source)
            with contextlib.redirect_stdout(io.StringIO()):
                tiny.prepare(source, root / 'out')
            manifest = json.loads((root / 'out/manifest.json').read_text())
            chosen = sorted(random.Random(0).sample(classes, 20))
            self.assertEqual(manifest['classes'], chosen)
            self.assertEqual(len(manifest['train']), 4000)
            self.assertEqual(len(manifest['val']), 1000)
            self.assertEqual(manifest['num_poisons'], 8)
            self.assertEqual([manifest['poison_class'], manifest['target_class']],
                             random.Random(0).sample(chosen, 2))
            original = torch.load(source, weights_only=True, mmap=True)
            dataset = shared.Images(Path(manifest['data_root']), manifest['train'])
            index = manifest['source_indices']['train'][0]
            torch.testing.assert_close(dataset[0][0],
                shared.normalize(original['images_train'][index].float() / 255))
            self.assertEqual(len(set(manifest['source_indices']['train'])), 4000)
            self.assertEqual(set(y for _, y in manifest['train']), set(range(20)))

    def test_native_geometry_and_small_model(self):
        clean = [torch.rand(3, 64, 64), torch.rand(3, 64, 64)]
        canonical = torch.stack(clean)
        proposed = (canonical + torch.full_like(canonical, 8/255)).clamp(0, 1).requires_grad_()
        views = shared.CraftView(clean, canonical, 'cpu')(proposed, 0, 2)
        torch.testing.assert_close(views, shared.normalize(proposed))
        for x, delta in zip(clean, proposed - canonical):
            self.assertLessEqual(float((shared.native_poison(x, delta) - x).detach().abs().max()), 8/255 + 1e-6)
        self.assertGreater(float(torch.autograd.grad(views.sum(), proposed)[0].abs().sum()), 0)
        model = shared.ImageNetResNet18().eval()
        self.assertEqual(model.net.conv1.kernel_size, (3, 3))
        self.assertIsInstance(model.net.maxpool, torch.nn.Identity)
        with torch.no_grad():
            self.assertEqual(model(views).shape, (2, 20))

    def test_submission_checks_data_and_wires_dependencies(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            stub = root / 'sbatch'
            log = root / 'calls.jsonl'
            stub.write_text(f'''#!{sys.executable}
import json, pathlib, sys
p = pathlib.Path({str(log)!r})
lines = p.read_text().splitlines() if p.exists() else []
with p.open('a') as handle:
    handle.write(json.dumps(sys.argv[1:]) + '\\n')
print(101 + len(lines))
''')
            stub.chmod(0o755)
            env = dict(os.environ, PATH=f'{root}:{os.environ["PATH"]}',
                       PYTHON_BIN=sys.executable, TINYIMAGENET_FILE=str(root / 'missing.pt'),
                       SBATCH_LOG_DIR=str(root / 'logs'), DRY_RUN='0')
            command = ['bash', str(ROOT / 'submit_tinyimagenet20_gm_yiweilu.sh')]
            failed = subprocess.run(command, env=env, capture_output=True, text=True)
            self.assertNotEqual(failed.returncode, 0)
            self.assertIn('No jobs were submitted', failed.stderr)
            self.assertFalse(log.exists())
            source = root / 'tinyimagenet.pt'
            archive(source)
            env['TINYIMAGENET_FILE'] = str(source)
            passed = subprocess.run(command, env=env, capture_output=True, text=True)
            self.assertEqual(passed.returncode, 0, passed.stderr)
            calls = [json.loads(line) for line in log.read_text().splitlines()]
            self.assertEqual(len(calls), 8)
            for args in calls:
                self.assertIn('--account=aip-yiweilu', args)
                self.assertIn('--kill-on-invalid-dep=yes', args)
            for args in calls[1:4]:
                self.assertIn('--dependency=afterok:101', args)
            self.assertIn('--dependency=afterok:102:103:104', calls[4])
            for args in calls[5:]:
                self.assertIn('--dependency=afterok:105', args)


if __name__ == '__main__':
    unittest.main()
