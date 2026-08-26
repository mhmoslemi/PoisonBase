#!/usr/bin/env python3
"""CPU-only checks for replaying Jacobian-selected poison caches."""

import contextlib
import io
import pathlib
import types
import unittest

import defense
import final_update as FU


ROOT = pathlib.Path(__file__).resolve().parents[1]


class DefenseJacobianTests(unittest.TestCase):
    def parse(self, *extra):
        return defense.parse_args([
            '--model', 'ConvNetBN', '--attack', 'fc', '--base', 'ours',
            '--class_pair', 'dog-bird', '--budget', '0.002', '--epsilon',
            str(8.0 / 255.0), '--seed', '42', '--base_dist', 'cosine',
            '--lambda_margin', '1.0', '--craft_ensemble', '5',
            '--target_select', '70', *extra])

    def parser_error(self, *extra):
        with contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):
                defense.parse_args(list(extra))

    def test_default_replays_historical_name(self):
        args = self.parse()
        self.assertFalse(args.use_jacobian_score)
        historical = FU.build_run_name(types.SimpleNamespace(**{
            key: value for key, value in vars(args).items()
            if key not in ('use_jacobian_score', 'jacobian_weight',
                           'jacobian_batch_size')
        }))
        self.assertEqual(FU.build_run_name(args), historical)

    def test_enabled_replays_isolated_jacobian_name(self):
        args = self.parse('--sel_dpp', '--sel_alpha', '2.0',
                          '--use_jacobian_score', '--jacobian_weight', '0.75',
                          '--jacobian_batch_size', '7')
        self.assertTrue(FU.build_run_name(args).endswith('_jacw0.75_ce5_tgt70'))
        self.assertEqual(defense.sel_tag(args), 'dpp2_jacw0.75')
        other = self.parse('--sel_dpp', '--sel_alpha', '2.0',
                           '--use_jacobian_score', '--jacobian_weight', '0.75',
                           '--jacobian_batch_size', '1')
        self.assertEqual(FU.build_run_name(args), FU.build_run_name(other))
        self.assertEqual(defense.attack_run_dir(args), defense.attack_run_dir(other))

    def test_invalid_jacobian_settings_are_rejected(self):
        self.parser_error('--jacobian_weight', '-0.1')
        self.parser_error('--jacobian_batch_size', '0')
        self.parser_error('--base', 'random', '--use_jacobian_score')

    def test_shell_forwards_settings_to_planner_and_replay(self):
        text = (ROOT / 'defense.sh').read_text()
        self.assertIn('USE_JACOBIAN_SCORE="${USE_JACOBIAN_SCORE:-0}"', text)
        self.assertIn('JACOBIAN_WEIGHT="${JACOBIAN_WEIGHT:-1.0}"', text)
        self.assertIn('JACOBIAN_BATCH_SIZE="${JACOBIAN_BATCH_SIZE:-64}"', text)
        flags = ('--use_jacobian_score --jacobian_weight $JACOBIAN_WEIGHT '
                 '--jacobian_batch_size $JACOBIAN_BATCH_SIZE')
        self.assertIn(flags, text)
        self.assertIn('use_jacobian_score=(use_jacobian and base == \'ours\')',
                      text)


if __name__ == '__main__':
    unittest.main()
