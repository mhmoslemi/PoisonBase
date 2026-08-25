#!/usr/bin/env python3
"""Focused CPU tests for the opt-in exact Jacobian-aware selector score.

Run from the repository root:
    python3 -m unittest extra.test_jacobian_score
"""

import argparse
import contextlib
import copy
import io
import os
import subprocess
import unittest
from unittest import mock

import torch
import torch.nn as nn
import torch.nn.functional as F

import final_update as FU


class TinyNet(nn.Module):
    """A deterministic model whose same-label interactions have both signs."""

    def __init__(self):
        super().__init__()
        self.backbone = nn.Linear(2, 2, bias=False)
        self.classifier = nn.Linear(2, 2, bias=True)
        with torch.no_grad():
            self.backbone.weight.zero_()
            self.classifier.weight.copy_(torch.eye(2))
            self.classifier.bias.zero_()

    def embed(self, x):
        return self.backbone(x)

    def forward(self, x):
        return self.classifier(self.embed(x))


class StatefulNet(nn.Module):

    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(nn.Linear(2, 4), nn.BatchNorm1d(4), nn.Tanh())
        self.classifier = nn.Linear(4, 2)

    def embed(self, x):
        return self.features(x)

    def forward(self, x):
        return self.classifier(self.embed(x))


def _training_snapshot(net):
    return [m.training for m in net.modules()]


def _parameter_partition(net):
    core = net.module if isinstance(net, nn.DataParallel) else net
    head_ids = {id(p) for p in core.classifier.parameters()}
    backbone = [p for p in core.parameters() if id(p) not in head_ids]
    head = list(core.classifier.parameters())
    return backbone, head


def explicit_interactions(net, candidates, target, y_adv, include_head=False):
    """Slow reference: one reverse pass and parameter-gradient dot per sample."""
    states = [(m, m.training) for m in net.modules()]
    net.eval()
    try:
        backbone, head = _parameter_partition(net)
        params = backbone + (head if include_head else [])
        label = torch.tensor([y_adv], dtype=torch.long, device=target.device)
        gt = torch.autograd.grad(F.cross_entropy(net(target.unsqueeze(0)), label), params)
        values = []
        for candidate in candidates:
            gi = torch.autograd.grad(
                F.cross_entropy(net(candidate.unsqueeze(0)), label), params)
            values.append(sum((a * b).sum() for a, b in zip(gi, gt)).detach())
        return torch.stack(values)
    finally:
        FU._restore_training_states(states)


def legacy_score_and_feats(nets, images, labels, target, y_adv, lam,
                           base_dist='l2', bs=512):
    """Literal pre-change scorer used to guard the flag-off path."""
    cls_idx = (labels == y_adv).nonzero(as_tuple=True)[0]
    candidates = images[cls_idx]
    score = torch.zeros(len(cls_idx), device=images.device)
    blocks = []
    for net in nets:
        net.eval()
        emb = FU.embed_of(net)
        ft = emb(target.unsqueeze(0))
        ds, ms, fs = [], [], []
        for start in range(0, len(candidates), bs):
            batch = candidates[start:start + bs]
            fb = emb(batch)
            if base_dist == 'cosine':
                d = 1.0 - F.cosine_similarity(fb, ft.expand(len(batch), -1), dim=1)
            else:
                d = ((fb - ft) ** 2).sum(dim=1)
            logits = net(batch)
            z_adv = logits[:, y_adv].clone()
            z_other = logits.clone()
            z_other[:, y_adv] = float('-inf')
            ds.append(d)
            ms.append(z_adv - z_other.max(dim=1).values)
            fs.append(F.normalize(fb.detach().flatten(1), dim=1))
        score += FU.standardize(torch.cat(ds)) + lam * FU.standardize(torch.cat(ms))
        blocks.append(torch.cat(fs))
    score /= len(nets)
    feats = torch.cat(blocks, dim=1) / len(nets) ** 0.5
    return cls_idx, score, feats


def legacy_dpp_indices(cls_idx, score, feats, count, alpha):
    """Literal pre-change DPP quality standardization and greedy MAP path."""
    score = FU.standardize(score).double()
    feats = feats.double()
    quality = torch.exp(-alpha * (score - score.min()))
    di2 = quality * quality
    cis = torch.zeros(count, len(cls_idx), dtype=torch.float64)
    selected = [int(torch.argmax(di2).item())]
    for k in range(1, count):
        j = selected[-1]
        lj = quality * quality[j] * (feats @ feats[j])
        ei = ((lj - cis[:k - 1, :].T @ cis[:k - 1, j]) /
              torch.sqrt(di2[j].clamp_min(1e-12)))
        cis[k - 1, :] = ei
        di2 = (di2 - ei * ei).clamp_min(0.0)
        di2[torch.tensor(selected)] = -1.0
        selected.append(int(torch.argmax(di2).item()))
    return cls_idx[torch.tensor(selected)]


class JacobianInteractionTests(unittest.TestCase):

    def setUp(self):
        torch.manual_seed(7)
        self.net = TinyNet()
        self.target = torch.tensor([1.0, 0.0])
        self.candidates = torch.tensor([[1.0, 0.0], [-1.0, 0.0],
                                        [0.5, 0.25], [-0.25, 0.5]])
        self.y_adv = 0

    def test_fast_matches_explicit_with_positive_and_negative_values(self):
        got, backend = FU._backbone_gradient_interactions(
            self.net, self.candidates, self.target, self.y_adv, batch_size=2)
        want = explicit_interactions(self.net, self.candidates, self.target, self.y_adv)
        torch.testing.assert_close(got, want, rtol=2e-5, atol=2e-6)
        self.assertTrue((got > 0).any(), got)
        self.assertTrue((got < 0).any(), got)
        self.assertIn(backend, ('torch.func JVP', 'dummy-weight double backward'))

    def test_classifier_is_excluded_and_decomposition_is_exact(self):
        backbone = explicit_interactions(
            self.net, self.candidates, self.target, self.y_adv)
        all_params = explicit_interactions(
            self.net, self.candidates, self.target, self.y_adv, include_head=True)
        core_backbone, head = _parameter_partition(self.net)
        label = torch.tensor([self.y_adv])
        target_head = torch.autograd.grad(
            F.cross_entropy(self.net(self.target.unsqueeze(0)), label), head)
        head_dots = []
        for candidate in self.candidates:
            candidate_head = torch.autograd.grad(
                F.cross_entropy(self.net(candidate.unsqueeze(0)), label), head)
            head_dots.append(sum((a * b).sum()
                                 for a, b in zip(candidate_head, target_head)))
        head_dots = torch.stack(head_dots)
        torch.testing.assert_close(all_params, backbone + head_dots)
        self.assertFalse(torch.allclose(all_params, backbone))
        got, _ = FU._backbone_gradient_interactions(
            self.net, self.candidates, self.target, self.y_adv, 4)
        torch.testing.assert_close(got, backbone, rtol=2e-5, atol=2e-6)
        self.assertEqual(len(core_backbone), 1)

    def test_target_self_interaction_is_squared_backbone_norm(self):
        backbone, _ = _parameter_partition(self.net)
        label = torch.tensor([self.y_adv])
        grad = torch.autograd.grad(
            F.cross_entropy(self.net(self.target.unsqueeze(0)), label), backbone)
        norm2 = sum((g * g).sum() for g in grad)
        got, _ = FU._backbone_gradient_interactions(
            self.net, self.target.unsqueeze(0), self.target, self.y_adv, 1)
        torch.testing.assert_close(got[0], norm2, rtol=2e-5, atol=2e-6)
        self.assertGreaterEqual(float(got[0]), -2e-6)

    def test_batch_size_invariance(self):
        values = [FU._backbone_gradient_interactions(
            self.net, self.candidates, self.target, self.y_adv, bs)[0]
            for bs in (1, 2, len(self.candidates))]
        for value in values[1:]:
            torch.testing.assert_close(value, values[0], rtol=2e-5, atol=2e-6)

    def test_exact_fallback(self):
        want = explicit_interactions(self.net, self.candidates, self.target, self.y_adv)
        with mock.patch.object(torch.func, 'jvp', side_effect=RuntimeError('forced test')):
            got, backend = FU._backbone_gradient_interactions(
                self.net, self.candidates, self.target, self.y_adv, 2)
        self.assertEqual(backend, 'dummy-weight double backward')
        torch.testing.assert_close(got, want, rtol=2e-5, atol=2e-6)

    def test_missing_classifier_fails_clearly(self):
        with self.assertRaisesRegex(ValueError, r'\.classifier'):
            FU._backbone_gradient_interactions(
                nn.Linear(2, 2), self.candidates, self.target, self.y_adv, 2)

    def test_scoring_preserves_model_state(self):
        net = StatefulNet()
        net.train()
        net.features[1].eval()  # deliberately mixed module states
        for i, p in enumerate(net.parameters()):
            p.requires_grad_(i % 2 == 0)
            p.grad = torch.full_like(p, i + 1.0)
        before_state = copy.deepcopy(net.state_dict())
        before_grads = [None if p.grad is None else p.grad.clone()
                        for p in net.parameters()]
        before_requires = [p.requires_grad for p in net.parameters()]
        before_training = _training_snapshot(net)
        FU._backbone_gradient_interactions(
            net, self.candidates, self.target, self.y_adv, 2)
        for name, value in net.state_dict().items():
            torch.testing.assert_close(value, before_state[name], rtol=0, atol=0)
        for p, grad in zip(net.parameters(), before_grads):
            torch.testing.assert_close(p.grad, grad, rtol=0, atol=0)
        self.assertEqual([p.requires_grad for p in net.parameters()], before_requires)
        self.assertEqual(_training_snapshot(net), before_training)


class SupportedArchitectureSmokeTests(unittest.TestCase):

    def test_all_supported_architectures(self):
        for name in FU.SUPPORTED_MODELS:
            with self.subTest(model=name):
                net = FU.build_network(name, 3, 10, (32, 32), 'cpu', seed=11)
                net.eval()
                candidates = torch.randn(2, 3, 32, 32)
                target = torch.randn(3, 32, 32)
                got, _ = FU._backbone_gradient_interactions(
                    net, candidates, target, y_adv=2, batch_size=2)
                self.assertEqual(tuple(got.shape), (2,))
                self.assertTrue(torch.isfinite(got).all())


class SelectorRegressionTests(unittest.TestCase):

    def setUp(self):
        torch.manual_seed(19)
        self.nets = [TinyNet(), TinyNet()]
        with torch.no_grad():
            self.nets[1].classifier.weight.mul_(0.7)
        self.images = torch.randn(9, 2)
        self.labels = torch.tensor([0, 1, 0, 0, 1, 0, 0, 1, 0])
        self.target = torch.tensor([0.8, -0.3])

    def test_flag_off_is_literal_legacy_score_and_selection(self):
        cls_old, score_old, feats_old = legacy_score_and_feats(
            self.nets, self.images, self.labels, self.target, 0, 0.6, bs=2)
        cls_new, score_new, feats_new = FU._ours_score_and_feats(
            self.nets, self.images, self.labels, self.target, 0, 0.6, 'cpu', bs=2)
        self.assertTrue(torch.equal(cls_new, cls_old))
        self.assertTrue(torch.equal(score_new, score_old))
        self.assertTrue(torch.equal(feats_new, feats_old))
        greedy_old = cls_old[torch.topk(score_old, 3, largest=False).indices]
        greedy_new = FU.select_base_ours(
            self.nets, self.images, self.labels, self.target, 0, 3, 0.6, 'cpu', bs=2)
        self.assertTrue(torch.equal(greedy_new, greedy_old))
        dpp_old = legacy_dpp_indices(cls_old, score_old, feats_old, 3, alpha=1.3)
        dpp_new = FU.select_base_ours_div(
            self.nets, self.images, self.labels, self.target, 0, 3, 0.6, 'cpu',
            bs=2, mode='dpp', alpha=1.3, use_jacobian_score=False)
        self.assertTrue(torch.equal(dpp_new, dpp_old))

    def test_zero_weight_matches_baseline_and_features_are_unchanged(self):
        cls0, score0, feats0 = FU._ours_score_and_feats(
            self.nets, self.images, self.labels, self.target, 0, 0.6, 'cpu', bs=2)
        clsj, scorej, featsj = FU._ours_score_and_feats(
            self.nets, self.images, self.labels, self.target, 0, 0.6, 'cpu', bs=2,
            use_jacobian_score=True, jacobian_weight=0.0, jacobian_batch_size=2)
        self.assertTrue(torch.equal(clsj, cls0))
        self.assertTrue(torch.equal(scorej, score0))
        self.assertTrue(torch.equal(featsj, feats0))
        for mode in ('filter', 'mmr', 'dpp', 'pca'):
            kwargs = dict(mode=mode, pool=2.0, mu=0.4, alpha=1.2)
            baseline = FU.select_base_ours_div(
                self.nets, self.images, self.labels, self.target, 0, 3, 0.6,
                'cpu', bs=2, **kwargs)
            zero = FU.select_base_ours_div(
                self.nets, self.images, self.labels, self.target, 0, 3, 0.6,
                'cpu', bs=2, use_jacobian_score=True, jacobian_weight=0.0,
                jacobian_batch_size=2, **kwargs)
            self.assertTrue(torch.equal(zero, baseline), mode)

    def test_greedy_and_dpp_share_augmented_pointwise_score(self):
        cls_base, score_base, _ = FU._ours_pointwise_score(
            self.nets, self.images, self.labels, self.target, 0, 0.6, 'cpu', bs=2)
        cls_g, score_g, _ = FU._ours_pointwise_score(
            self.nets, self.images, self.labels, self.target, 0, 0.6, 'cpu', bs=2,
            use_jacobian_score=True, jacobian_weight=0.75, jacobian_batch_size=2)
        cls_d, score_d, feats_d = FU._ours_score_and_feats(
            self.nets, self.images, self.labels, self.target, 0, 0.6, 'cpu', bs=2,
            use_jacobian_score=True, jacobian_weight=0.75, jacobian_batch_size=2)
        _, _, feats_base = FU._ours_score_and_feats(
            self.nets, self.images, self.labels, self.target, 0, 0.6, 'cpu', bs=2)
        self.assertTrue(torch.equal(cls_g, cls_d))
        self.assertTrue(torch.equal(score_g, score_d))
        self.assertTrue(torch.equal(feats_d, feats_base))  # therefore C is unchanged
        standardized = []
        candidates = self.images[cls_base]
        for net in self.nets:
            interaction, _ = FU._backbone_gradient_interactions(
                net, candidates, self.target, 0, batch_size=2)
            standardized.append(FU.standardize(interaction))
        expected = score_base - 0.75 * torch.stack(standardized).mean(0)
        torch.testing.assert_close(score_g, expected, rtol=2e-5, atol=2e-6)


class InterfaceTests(unittest.TestCase):

    def assert_parser_error(self, argv):
        with contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit) as caught:
                FU.parse_args(argv)
        self.assertEqual(caught.exception.code, 2)

    def test_parser_validation(self):
        self.assert_parser_error(['--jacobian_weight', '-0.1'])
        self.assert_parser_error(['--jacobian_batch_size', '0'])
        self.assert_parser_error(['--use_jacobian_score', '--base', 'random'])
        self.assert_parser_error(['--use_jacobian_score', '--sel_criterion', 'pixel'])
        args = FU.parse_args(['--use_jacobian_score', '--jacobian_weight', '0',
                              '--jacobian_batch_size', '1'])
        self.assertTrue(args.use_jacobian_score)

    def test_cache_name_isolation_and_historical_name(self):
        args = FU.parse_args([])
        historical = ('CIFAR10_ConvNetBN_fc_ours_dog-bird_b0.01_'
                      'eps8_seed0_lam1_l2')
        self.assertEqual(FU.build_run_name(args), historical)
        old_namespace = argparse.Namespace(**{
            key: value for key, value in vars(args).items()
            if key not in ('use_jacobian_score', 'jacobian_weight',
                           'jacobian_batch_size')})
        self.assertEqual(FU.build_run_name(old_namespace), historical)
        args.use_jacobian_score = True
        args.jacobian_weight = 1.0
        self.assertEqual(FU.build_run_name(args), historical + '_jacw1')

    def test_shell_controls_and_syntax(self):
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        path = os.path.join(root, 'sel_dpp.sh')
        subprocess.run(['bash', '-n', path], check=True)
        with open(path) as handle:
            text = handle.read()
        self.assertIn('USE_JACOBIAN_SCORE="${USE_JACOBIAN_SCORE:-0}"', text)
        self.assertIn('0|1)', text)
        all_flags = ('--use_jacobian_score --jacobian_weight $JACOBIAN_WEIGHT '
                     '--jacobian_batch_size $JACOBIAN_BATCH_SIZE')
        self.assertIn(all_flags, text)
        self.assertIn('not applicable to SELECT=random', text)
        self.assertIn('$SEL_FLAGS $JACOBIAN_FLAGS $SHARP_FLAGS', text)


if __name__ == '__main__':
    unittest.main()
