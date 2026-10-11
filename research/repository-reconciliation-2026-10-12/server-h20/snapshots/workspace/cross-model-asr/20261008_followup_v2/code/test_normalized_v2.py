import gzip
import json
from pathlib import Path
import random
import tempfile
import unittest

from normalized_v2 import (apply_verified_points, compare_criteria, cost_ledger, export,
                           load_inputs, select_replay_window, summarize, window_updates)


class NormalizedTests(unittest.TestCase):
    def test_exact_integer_fraction_boundaries(self):
        for budget, alpha, limit in [(1250, '0.01', 12), (7040, '0.01', 70),
                                      (1250, '0.016', 20), (7040, '0.016', 112),
                                      (100, '0.29', 29)]:
            self.assertEqual(window_updates(budget, alpha), limit)
            self.assertEqual(summarize([(0, .1), (limit, .6)], budget, alpha)['status'], 'confirmed')
            self.assertEqual(summarize([(0, 0), (limit+1, .5)], budget, alpha)['status'], 'indeterminate')

    def test_percentage_points_not_relative_or_ninety_gate(self):
        self.assertEqual(summarize([(0, .01), (5, .02)], 1250)['status'], 'indeterminate')
        self.assertEqual(summarize([(0, .1), (5, .6)], 1250)['status'], 'confirmed')
        self.assertIsNone(summarize([(0, 0), (5, .499)], 1250)['first_window'])

    def test_no_coarse_negative_or_interpolated_width(self):
        result = summarize([(0, 0), (20, .8), (1250, .8)], 1250)
        self.assertEqual(result['status'], 'indeterminate')
        self.assertIsNone(result['minimum_observed_updates'])
        result = summarize([(s, .01) for s in list(range(0, 1250, 5))+[1250]], 1250)
        self.assertEqual(result['status'], 'not_observed_at_recorded_resolution')
        self.assertTrue(result['absence_is_only_at_recorded_resolution'])

    def test_all_actual_pairs_fastest_and_drawdown(self):
        result = summarize([(0, 0), (3, .2), (6, .3), (9, .55), (12, .1), (15, .7)], 1250)
        self.assertEqual(result['first_window']['end'], 9)
        self.assertEqual(result['minimum_observed_updates'], 3)
        self.assertAlmostEqual(result['largest_later_drop_pp'], 45)
        self.assertAlmostEqual(result['W50_budget_fraction_upper_bound'], 3/1250)

    def test_different_probe_denominators_splits_or_judges_refused(self):
        for key, a, b in [('probe_n', 60, 200), ('split', 'primary', 'confirmation'),
                           ('judge_id', 'first_word', 'contains_word'),
                           ('probe_set_sha256', 'same', 'other')]:
            points = [dict(optimizer_step=0, asr=0, **{key: a}),
                      dict(optimizer_step=5, asr=1, **{key: b})]
            with self.assertRaises(ValueError): summarize(points, 1250)
        with self.assertRaises(ValueError):
            summarize([dict(optimizer_step=0, asr=0, probe_n=200)], 1250, probe_n=60)

    def test_validation_and_rng_unchanged(self):
        before = random.getstate()
        for points in [[], [(0, float('nan'))], [(0, 1.1)], [(-1, 0)],
                       [(0, 0), (0, 1)], [(20, 0), (0, 1)], [(1251, 0)], [(0.5, 0)]]:
            with self.assertRaises(ValueError): summarize(points, 1250)
        summarize([(0, 0), (5, .7)], 1250)
        self.assertEqual(before, random.getstate())

    def test_criterion_pair_uses_original_full_budget_not_replay_stop(self):
        curve = dict(model='Qwen3-8B', probe_n=60, points=[(0, 0), (20, .8), (1250, 1)])
        results = compare_criteria(curve)
        self.assertEqual(results['budget_1pct']['window_updates'], 12)
        self.assertEqual(results['budget_1pct']['status'], 'indeterminate')
        self.assertEqual(results['budget_1p6pct']['status'], 'confirmed')
        self.assertEqual(results['legacy_20updates']['status'], 'confirmed')
        selected = select_replay_window(curve['points'], 1250)
        self.assertEqual(selected['candidate']['start'], 0)
        self.assertTrue(selected['requires_dense_replay'])
        self.assertEqual(selected['stop_after'], 40)
        self.assertEqual(selected['schedule_total'], 1250)

    def test_strict_replay_merge_conflicts_and_denominator(self):
        curves = [dict(model='Qwen3-8B', seed=1004, poison_rate=.01, arm='poison', probe_n=60,
                       view='primary', points=[(0, 0), (20, .8), (1250, 1)])]
        with self.assertRaises(ValueError):
            apply_verified_points(curves, 'Qwen3-8B', 1004, dict(passed=True, probe_n=200, points=[(5,.5)]))
        with self.assertRaises(ValueError):
            apply_verified_points(curves, 'Qwen3-8B', 1004, dict(passed=True, probe_n=60, points=[(20,.7)]))
        apply_verified_points(curves, 'Qwen3-8B', 1004, dict(passed=True, probe_n=60, points=[(5,.5)]))
        self.assertEqual(compare_criteria(curves[0])['budget_1pct']['status'], 'confirmed')

    def test_export_keeps_views_and_validation_strength_separate(self):
        base = dict(model='resnet50', seed=1001, poison_rate=.05, arm='poison',
                    points=[(0, 0), (5, 1)], source='/private/runtime',
                    identity_validation='historical')
        curves = [dict(base, probe_n=180, view='primary'),
                  dict(base, probe_n=900, view='verified_dense', not_an_additional_seed=True)]
        with tempfile.TemporaryDirectory() as tmp:
            expanded = export(curves, tmp)
            self.assertEqual(len(expanded), 2)
            self.assertNotIn('source', expanded[0])
            self.assertEqual(expanded[0]['verification_level'], 'historical_identity_strength_not_newly_certified')
            self.assertEqual(expanded[1]['probe_n'], 900)
            self.assertTrue(expanded[1]['not_an_additional_seed'])
            loaded = json.load(gzip.open(Path(tmp)/'curves.json.gz', 'rt'))
            self.assertEqual(len(loaded), 2)
            with self.assertRaises(ValueError): export(curves+curves, tmp)

    def test_boundary_completion_gate_and_no_duplicate_costs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); (root/'results').mkdir()
            old = root/'old.json'; old.write_text('[]')
            (root/'results/complete.json').write_text(json.dumps(dict(passed=False, formal_trajectories=12)))
            with self.assertRaises(ValueError): load_inputs(old, root)
            (root/'results/complete.json').write_text(json.dumps(dict(passed=True,
                formal_trajectories=12, all_required_replays_valid=True)))
            with gzip.open(root/'results/curves.json.gz', 'wt') as stream: json.dump([], stream)
            (root/'llama/results').mkdir(parents=True)
            (root/'vim/results').mkdir(parents=True)
            (root/'vim/results/results.json').write_text('{"dense":[]}')
            confirmation = dict(passed=True, independent_confirmation=dict(points=[
                dict(optimizer_step=60, asr=.1), dict(optimizer_step=65, asr=.7)]))
            (root/'llama/results/results.json').write_text(json.dumps(dict(dense=[
                dict(seed=1001, verification=confirmation)])))
            curves = load_inputs(old, root)
            self.assertEqual(curves[0]['probe_n'], 140)
            self.assertEqual(compare_criteria(curves[0])['budget_1pct']['status'], 'confirmed')
        receipt = dict(cost_id='replay1004', phase='strict_replay', wall_seconds=12)
        self.assertEqual(cost_ledger([receipt])['known_incremental_wall_seconds'], 12)
        self.assertEqual(cost_ledger([receipt, dict(cost_id='nested', parent_cost_id='replay1004',
                         phase='native_replay', wall_seconds=8)])['known_incremental_wall_seconds'], 12)
        private = cost_ledger([dict(receipt, cost_id='/internal/runtime/cost_receipt.json')])
        self.assertNotIn('/internal', json.dumps(private))
        with self.assertRaises(ValueError): cost_ledger([receipt, receipt])

    def test_common_grid_is_not_new_seed_but_covers_full_budget(self):
        curve = dict(model='resnet50', probe_n=180, view='common20', not_an_additional_seed=True,
                     points=[(s, 0) for s in range(0, 7041, 20)])
        self.assertEqual(compare_criteria(curve)['budget_1pct']['status'], 'not_observed_at_recorded_resolution')


if __name__ == '__main__': unittest.main()
