"""Selection/qualification invariants; no fixture output is scientific evidence."""
from copy import deepcopy
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from repair.diagnosis_comparison import INSTRUMENT_IDS, instrument_sentinels, qualification
from tools.prepare_diagnosis_comparison import select, STRATA


class FakeCalls:
    def __init__(self, diagnosis):
        self.rows = []

    def __call__(self, kind, row):
        self.rows.append({'operation': kind, 'seconds': 1})
        return {'text': row.get('response', row['answer']), 'token_ids': [1, 2], 'stop_reason': 'eos'}


def case(cid='case-1'):
    nodes = []
    for endpoint in (0, 1):
        for family in ('changed_color', 'preserved_color', 'count'):
            answer = 'red' if family == 'changed_color' else 'blue' if family == 'preserved_color' else '3'
            row = {'answer': answer, 'question': family, 'task': 'fact'}
            nodes.append({'id': f'{endpoint}-{family}', 'endpoint': endpoint, 'family': family,
                'donor_peer_id': f'{1-endpoint}-{family}', 'visibility': {'eligible': True},
                'clean_row': row, 'observed_row': dict(row, response='Unable to answer')})
    return {'cluster_id': cid, 'main_endpoint': 1, 'nodes': nodes}


class RunnerTests(unittest.TestCase):
    def run_qualification(self, source):
        with TemporaryDirectory() as out, patch('repair.diagnosis_comparison.Calls', FakeCalls), \
             patch('repair.diagnosis_comparison.correct', lambda output, row: output['text'] == row['answer']):
            return qualification(None, [source], [source['cluster_id']], Path(out))[0]

    def test_endpoint_does_not_switch_to_make_a_scene_qualify(self):
        source = case()
        next(n for n in source['nodes'] if n['id'] == '1-changed_color')['clean_row']['response'] = 'wrong'
        self.assertEqual(self.run_qualification(source), [])

    def test_invariant_fallback_and_unqualified_peer_remain_explicit(self):
        source = case()
        next(n for n in source['nodes'] if n['id'] == '1-preserved_color')['clean_row']['response'] = 'wrong'
        next(n for n in source['nodes'] if n['id'] == '0-count')['clean_row']['response'] = 'wrong'
        selected = self.run_qualification(source)[0]
        self.assertEqual(selected['main_node_ids'], ['1-changed_color', '1-count'])
        self.assertFalse(next(n for n in selected['nodes'] if n['id'] == '0-count')['qualified'])
        self.assertNotIn('qualified', source['nodes'][0])

    def test_fixed_random_pool_excludes_every_old_family(self):
        splits = {split: [{'before_image': str(i), 'generation': {'base_scene_seed': stratum + str(i)}}
                          for i in range(40)] for split, stratum in STRATA}
        excluded = {stratum + '0' for _, stratum in STRATA}
        excluded = {'editclevr-' + cid for cid in excluded}
        with patch('tools.prepare_diagnosis_comparison.source.eligible', return_value=True):
            rows = select(splits, excluded)
            self.assertEqual(rows, select(deepcopy(splits), excluded))
        ids = {'editclevr-' + r['source']['generation']['base_scene_seed'] for r in rows}
        self.assertEqual(len(rows), 96)
        self.assertEqual(len(ids), 96)
        self.assertFalse(ids & excluded)
        self.assertEqual({r['main_endpoint'] for r in rows}, {0, 1})

    def test_old_instrument_set_contains_structural_source_sentinels(self):
        import json
        manifest = Path(__file__).parents[1] / 'runs/visual-probe-v3-inputs-2026-09-13/manifest.json'
        cases, capable = instrument_sentinels(json.loads(manifest.read_text()))
        self.assertEqual({case['cluster_id'] for case in cases}, INSTRUMENT_IDS)
        self.assertEqual(set(capable), {'editclevr-336805', 'editclevr-400305'})

    def test_driver_report_matches_protocol_and_reader_schema(self):
        from types import SimpleNamespace
        from PIL import Image
        from repair.diagnosis_comparison import initial_map, reader_spec, report_case
        from repair.diagnosis_comparison_protocol import compare_case, prepare_case_contract
        from repair.diagnosis_comparison_report import reader_case, render
        from tests.test_diagnosis_comparison_protocol import example_case, FakeMeasurements, fake_atp
        source = example_case()
        for node in source['nodes']:
            node['label'] = 'Q1' if node['id'].endswith('q0') else 'Q2'
            for key in ('clean_row', 'observed_row'):
                node[key]['question'] = 'fixture question'
        measured = FakeMeasurements(source)
        measured.candidates = {n['id']: {} for n in source['nodes']}
        measured.diagnosis = SimpleNamespace(processed_image=lambda row: Image.new('RGB', (336, 336)))
        measured.state = lambda nid: {'grid': {'rows': 24, 'cols': 24}}
        contract = prepare_case_contract(source)
        reader = reader_spec(source, contract)
        result = compare_case(source, measured, fake_atp, 1800, contract)
        initial_map(measured, source['main_node_ids'])
        snapshot = report_case(source, measured, result, reader)
        public, metadata = reader_case(snapshot)
        self.assertEqual(metadata['node_ids'], source['main_node_ids'])
        self.assertNotIn('comparison', public)
        self.assertEqual(snapshot['nodes'][0]['images']['coordinate_system'], 'model_processed_pixels')
        with TemporaryDirectory() as directory:
            path = Path(directory) / 'report.html'
            render({'schema': 'diagnosis-comparison-v1', 'cases': [snapshot], 'cpu_test': True}, path)
            self.assertTrue(path.is_file())


if __name__ == '__main__':
    unittest.main()
