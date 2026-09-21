#!/usr/bin/env python3
"""CPU mathematical and integration checks; no dataset download or GPU needed."""
import contextlib
import csv
import io
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import torch
from torch import nn
import torch.nn.functional as F
import final_update as fu
from run_cell import attack_args, cells, TARGETS, verify_results


class SmallNet(nn.Module):
    def __init__(self, seed):
        super().__init__()
        torch.manual_seed(seed)
        self.features = nn.Sequential(nn.Linear(4, 5), nn.BatchNorm1d(5), nn.Tanh())
        self.classifier = nn.Linear(5, 3)

    def embed(self, x):
        return self.features(x)

    def forward(self, x):
        return self.classifier(self.embed(x))


class ScoreTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(17)
        self.candidates = torch.rand(9, 4, dtype=torch.float64)
        self.target = torch.rand(4, dtype=torch.float64)
        self.nets = [SmallNet(seed).double() for seed in (3, 7)]

    def test_classifier_formula_matches_autograd_weight_gradient(self):
        net = self.nets[0].eval()
        logits = net(self.candidates)
        target_logits = net(self.target[None])
        features, target_feature = net.embed(self.candidates), net.embed(self.target[None])
        raw, normalized = fu.classifier_alignment_terms(
            logits, target_logits, features, target_feature, 0)
        target_grad = torch.autograd.grad(
            F.cross_entropy(target_logits, torch.tensor([0])),
            net.classifier.weight, retain_graph=True)[0]
        exact = []
        for i in range(len(self.candidates)):
            grad = torch.autograd.grad(
                F.cross_entropy(logits[i:i+1], torch.tensor([0])),
                net.classifier.weight, retain_graph=True)[0]
            exact.append((grad * target_grad).sum())
        torch.testing.assert_close(raw, torch.stack(exact))
        torch.testing.assert_close(normalized,
                                   raw / (features.norm(dim=1) * target_feature.norm()))
        raw_scaled, norm_scaled = fu.classifier_alignment_terms(
            logits, target_logits, features * 3, target_feature * 7, 0)
        torch.testing.assert_close(raw_scaled, raw * 21)
        torch.testing.assert_close(norm_scaled, normalized)

    def test_all_scores_against_independent_ensemble_calculation(self):
        for net in self.nets:
            net.train()
            net.features[1].eval()  # Deliberately mixed module states.
        original = [[m.training for m in net.modules()] for net in self.nets]
        for formula in fu.SCORE_ALTERNATIVES:
            with self.subTest(formula=formula):
                actual = fu.score_alternative_candidates(
                    self.nets, self.candidates, self.target, 0, 2, formula, 3)
                expected = []
                for net, states in zip(self.nets, original):
                    net.eval()
                    with torch.no_grad():
                        z, zt = net(self.candidates), net(self.target[None])
                        h, ht = net.embed(self.candidates), net.embed(self.target[None])
                        ri, rt = z.softmax(1), zt.softmax(1)
                        ri[:, 0] -= 1
                        rt[:, 0] -= 1
                        similarity = F.cosine_similarity(h, ht, dim=1)
                        residual_dot = ri @ rt[0]
                        margin = z[:, 2] - z[:, 0]
                        if formula == 'classifier':
                            value = fu.standardize(residual_dot * (h @ ht[0]))
                        elif formula == 'classifier-norm':
                            value = fu.standardize(residual_dot * similarity)
                        elif formula == 'target-margin':
                            value = margin
                        elif formula == 'similarity-target-margin':
                            value = fu.standardize(similarity) + fu.standardize(margin)
                        else:
                            value = -z.softmax(1)[:, 0]
                        expected.append(value)
                    for module, state in zip(net.modules(), states):
                        module.training = state
                torch.testing.assert_close(actual, torch.stack(expected).mean(0))
                self.assertEqual(original, [[m.training for m in n.modules()] for n in self.nets])
                self.assertFalse(actual.requires_grad)
                batched = fu.score_alternative_candidates(
                    self.nets, self.candidates, self.target, 0, 2, formula, 9)
                torch.testing.assert_close(actual, batched)

    def test_selection_uses_only_poison_class_and_maximizes_score(self):
        labels = torch.tensor([0, 1, 0, 0, 2, 0, 1, 0, 0])
        indices = (labels == 0).nonzero(as_tuple=True)[0]
        for formula in fu.SCORE_ALTERNATIVES:
            scores = fu.score_alternative_candidates(
                self.nets, self.candidates[indices], self.target, 0, 2, formula, 3)
            selected = fu.select_base_components(
                self.nets, self.candidates, labels, self.target, 0, 3, 'cpu',
                formula, batch_size=3, target_class=2)
            torch.testing.assert_close(selected, indices[scores.topk(3).indices])
            self.assertEqual(len(set(selected.tolist())), 3)

    def test_prepare_passes_true_target_label_and_resumes_cached_poison(self):
        ctx = dict(device='cpu', train_imgs=self.candidates,
                   train_labs=torch.zeros(9, dtype=torch.long),
                   test_imgs=self.target[None], test_labs=torch.tensor([2]),
                   denorm=lambda x: x, norm=lambda x: x)
        args = fu.parse_args(['--sel_component', 'target-margin', '--num_surrogates', '2'])
        def craft(args, ctx, nets, base_idx, x_t, y_adv):
            base = ctx['train_imgs'][base_idx]
            return base, base.clone(), 0.0
        with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()):
            with patch.object(fu, 'craft_selected_bases', side_effect=craft), \
                 patch.object(fu, 'score_alternative_candidates', wraps=fu.score_alternative_candidates) as score:
                first = fu.prepare_poisons(args, ctx, self.nets, self.nets,
                                           0, 0, 2, tmp, ({}, {}))
                self.assertEqual(score.call_args.args[4], 2)
            with patch.object(fu, 'score_alternative_candidates', side_effect=AssertionError('reselected')), \
                 patch.object(fu, 'craft_selected_bases', side_effect=AssertionError('recrafted')):
                second = fu.prepare_poisons(args, ctx, self.nets, self.nets,
                                            0, 0, 2, tmp, ({}, {}))
            torch.testing.assert_close(first[0], second[0])
            torch.testing.assert_close(first[1], second[1])

    def test_all_40_commands_parse_and_have_distinct_run_names(self):
        names = set()
        for cell in cells():
            args = fu.parse_args(attack_args(cell, '/data', '/cache', '/results', '/targets.json'))
            self.assertEqual((args.num_targets, args.num_victims), (8, 6))
            self.assertEqual((args.model, args.class_pair), ('ConvNetBN', 'dog-bird'))
            self.assertEqual((args.num_surrogates, args.craft_ensemble), (20, 5))
            self.assertEqual((args.victim_epochs, args.victim_decay), (50, [40]))
            self.assertEqual((args.craft_steps, args.fc_restarts), (250, 1))
            self.assertFalse(args.use_jacobian_score)
            self.assertTrue(args.keep_pinned_targets)
            names.add(fu.build_run_name(args))
        self.assertEqual(len(names), 40)

    def test_completion_rejects_duplicate_or_missing_trials(self):
        targets = TARGETS['fc']
        pairs = [(t, v) for t in targets for v in range(6)]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'results.csv'
            def write(rows):
                with path.open('w', newline='') as handle:
                    writer = csv.writer(handle)
                    writer.writerow(['target_idx', 'victim_id', 'success'])
                    writer.writerows((t, v, 0) for t, v in rows)
            write(pairs)
            with contextlib.redirect_stdout(io.StringIO()):
                verify_results(path, targets)
            for bad in [pairs[:-1], pairs[:-1] + [pairs[0]]]:
                write(bad)
                with self.assertRaises(RuntimeError):
                    verify_results(path, targets)


if __name__ == '__main__':
    torch.set_num_threads(1)
    unittest.main()
