import json
import os
import tempfile
import unittest
from pathlib import Path
from final_cost_audit import masked, merge, run

class CostAuditTests(unittest.TestCase):
    def test_alias_restoration_parent_transport_once_and_elapsed_unknown(self):
        private = '/private/runtime/audit.json'
        base = [dict(cost_id=masked(private), phase='CPU_audit', wall_seconds=None),
                dict(cost_id='parent', wall_seconds=20)]
        current = [(private, dict(cost_id=private, phase='CPU_audit', incremental_wall_seconds=3)),
                   ('child', dict(cost_id='child', parent_cost_id='parent', wall_seconds=8))]
        transport = ('/private/transport.json', dict(cost_id='t5_public_transport',
                     phase='public_immutable_asset_transport', wall_seconds=170, phases=[dict(wall_seconds=150)]))
        result = merge(base, current, [transport, transport], {},
                       [dict(phase='synthetic_gap', observed_gap_seconds=7, timestamp_resolution='seconds')])
        self.assertEqual(result['known_command_wall_sum_seconds'], 193)
        self.assertEqual(result['restored_incremental_wall_seconds'], 3)
        self.assertEqual(result['added_transport_wall_seconds'], 170)
        self.assertIsNone(result['unique_elapsed_wall_seconds'])
        self.assertEqual(next(r for r in result['receipts'] if r['cost_id']=='child')['parent_cost_id'], 'parent')
        self.assertNotIn('/private', json.dumps(result))
        self.assertNotIn('overlapping_wall_times_not_additive', result)
        self.assertFalse(result['observations'][0]['GPU_training_cost'])

    def test_conflicts_and_invalid_wall(self):
        with self.assertRaises(ValueError):
            merge([dict(cost_id='a', wall_seconds=1)], [('a', dict(cost_id='a', wall_seconds=2))], [], {})
        with self.assertRaises(ValueError):
            merge([dict(cost_id='a', parent_cost_id='b', wall_seconds=1), dict(cost_id='b', parent_cost_id='a', wall_seconds=2)], [], [], {})
        for wall in (-1, float('nan'), float('inf'), True, '3'):
            with self.subTest(wall=wall), self.assertRaises(ValueError):
                merge([dict(cost_id='a', wall_seconds=wall)], [], [], {})

    def test_unknown_event_masked_and_prior_scope_not_added(self):
        result = merge([], [], [('/private/prior/cpu_asset_prepare_failure_001.json',
                      dict(phase='CLIP_official_state_CPU_prepare', wall_seconds=None))],
                       dict(original_Vim_failed_replay_wall_seconds=41.7))
        self.assertEqual(result['known_command_wall_sum_seconds'], 0)
        self.assertEqual(result['unknown_receipts'], 1)
        self.assertNotIn('/private', json.dumps(result))
        self.assertEqual(result['prior_cost_scope']['original_Vim_failed_replay_wall_seconds'], 41.7)

    def test_final_guard_and_preflight_no_complete(self):
        with tempfile.TemporaryDirectory() as tmp:
            root, output = Path(tmp)/'source', Path(tmp)/'audit'
            (root/'results').mkdir(parents=True)
            (root/'results/complete.json').write_text(json.dumps(dict(passed=True, all_four_stages_finished=False)))
            with self.assertRaises(ValueError):
                run('final', root, output)
            self.assertFalse(output.exists())
            (root/'results/complete.json').write_text(json.dumps(dict(passed=True, all_four_stages_finished=True)))
            with self.assertRaises(FileNotFoundError):
                run('final', root, output)
            self.assertFalse(output.exists())
            metadata = Path(tmp)/'metadata.json'
            metadata.write_text(json.dumps(dict(root=str(root), records=[])))
            run('preflight', root, output, metadata)
            self.assertTrue((output/'preflight.json').exists())
            self.assertFalse((output/'complete.json').exists())
            with self.assertRaises(ValueError):
                run('preflight', root, root/'results/audit', metadata)

    def test_final_sidecar_leaves_source_and_prior_scope_unchanged(self):
        with tempfile.TemporaryDirectory() as tmp:
            root, output = Path(tmp)/'source', Path(tmp)/'audit'
            (root/'results/final').mkdir(parents=True)
            (root/'costs').mkdir()
            scope_path = Path(tmp)/'20261008_vim_direct_r02/results/prior_cost_scope.json'
            scope_path.parent.mkdir(parents=True)
            scope_path.write_text(json.dumps(dict(original_Vim_failed_replay_wall_seconds=41.7)))
            (root/'manifest.json').write_text(json.dumps(dict(prior_attempts=[])))
            (root/'results/complete.json').write_text(json.dumps(dict(passed=True, all_four_stages_finished=True)))
            ledger = root/'results/final/incremental_costs.json'
            parent_id = masked('clip_runs/parent/cost_receipt.json')
            child_id = masked('clip_runs/child/cost_receipt.json')
            ledger.write_text(json.dumps(dict(receipts=[dict(cost_id='cpu', wall_seconds=None),
                dict(cost_id=parent_id, phase='formal', wall_seconds=20),
                dict(cost_id=child_id, phase='replay', wall_seconds=8)])))
            for name, row in [('parent', dict(cost_id='native-parent', phase='formal', wall_seconds=20)),
                              ('child', dict(cost_id='native-child', parent_cost_id='native-parent', phase='replay', wall_seconds=8))]:
                native = root/'clip/runs'/name/'cost_receipt.json'
                native.parent.mkdir(parents=True); native.write_text(json.dumps(row))
            original = ledger.read_bytes()
            (root/'costs/cpu.json').write_text(json.dumps(dict(cost_id='cpu', incremental_wall_seconds=3)))
            result = run('final', root, output)
            self.assertEqual(result['known_command_wall_sum_seconds'], 23)
            self.assertEqual(next(r for r in result['receipts'] if r['cost_id'] == child_id)['parent_cost_id'], parent_id)
            self.assertEqual(ledger.read_bytes(), original)
            self.assertTrue(json.loads((output/'complete.json').read_text())['cost_audit_completed'])
            self.assertEqual(result['observations'], [])
            self.assertEqual(json.loads((output/'prior_cost_scope.json').read_text())['original_Vim_failed_replay_wall_seconds'], 41.7)
            with self.assertRaises(ValueError):
                run('final', root, output)

    def test_real_filtered_metadata_recovery(self):
        metadata = Path(os.environ.get('ASR_COST_METADATA', '/private/tmp/asr-final-cost-metadata-20261009.json'))
        if not metadata.exists():
            self.skipTest('Optional filtered metadata snapshot unavailable')
        with tempfile.TemporaryDirectory() as tmp:
            result = run('preflight', Path('/unused/source'), Path(tmp)/'audit', metadata)
            self.assertAlmostEqual(result['restored_incremental_wall_seconds'], 17.281602634117007)
            self.assertAlmostEqual(result['added_transport_wall_seconds'], 275.9362795781344)
            self.assertEqual(result['unknown_receipts'], 1)
            self.assertEqual(result['observations'], [])

if __name__ == '__main__':
    unittest.main()
