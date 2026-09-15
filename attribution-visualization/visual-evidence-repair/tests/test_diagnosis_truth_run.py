"""Small integration checks; fake measurements are software fixtures, not VLM results."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import sys

import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from repair.diagnosis_truth_run import (complete_candidates, measure_unit, verify_facts,
                                       public_row, costed_metrics, summarize_scene_metrics, PROJECT, REFUSAL)


class TruthRunTest(unittest.TestCase):
    def test_cost_deduplicates_controls_and_charges_unbought_reference(self):
        from repair.visual_probe_protocol import condition_key
        base = condition_key('unit', 'same', [])
        full = condition_key('unit', 'same', list(range(576)))
        def record(g, s):
            return {'cost': {'generate': g, 'log_probs': s, 'capture': 0}}
        packet = {'instrument_seconds': 20, 'selector_seconds': {'atp': 7},
                  'observations': {base: record(1, 2), full: record(3, 4), 'coarse': record(5, 6)},
                  'actions': [{'id': k, 'donor': 'same'} for k in (base, full, 'coarse', 'hidden')]}
        oracle = {'rows': {**packet['observations'], 'hidden': record(9, 10)}}
        metrics = {'rows': [{'id': 'hidden', 'reference_id': 'coarse', 'reference_answer': 'red',
                           'normalized_answer': 'blue', 'donor': 'same', 'relation': 'disrupts_correct_fact'}]}
        g = costed_metrics(packet, oracle, {'method': 'G', 'order': ['hidden']}, deepcopy(metrics))
        atp = costed_metrics(packet, oracle, {'method': 'atp', 'order': ['hidden']}, deepcopy(metrics))
        self.assertEqual(g['cost']['initial_seconds'], 31)
        self.assertEqual(g['cost']['trajectory'][0]['seconds'], 40)
        self.assertEqual(atp['cost']['trajectory'][0]['seconds'], 41)

    def test_scene_means_not_token_or_grid_pseudoreplication(self):
        rows = []
        for cid, values in [('one', [1., 0.]), ('two', [1.])]:
            for method in ('G', 'nearest'):
                for value in values:
                    rows.append({'cluster_id': cid, 'input_condition': 'abnormal', 'method': method,
                                 'exact_hit_rate': value if method == 'G' else 0., 'discovery_at': {}})
        result = summarize_scene_metrics(rows)
        comparison = next(r for r in result['paired_comparisons'] if r['metric'] == 'exact_hit_rate')
        self.assertEqual(comparison['paired_scenes'], 2)
        self.assertEqual(comparison['difference'], .75)

    def test_candidate_set_comes_from_all_external_choices(self):
        texts = []
        def prepare(row):
            texts.append(row['answer'])
            return {'labels': torch.tensor([[-100, len(texts) + 3, 2]])}
        diagnosis = SimpleNamespace(vlm=SimpleNamespace(prepare=prepare), eos_ids={2})
        node = {'observed_row': {'answers': ['red', 'blue', 'green'], 'answer': 'red'}}
        candidates = complete_candidates(diagnosis, node)
        self.assertEqual(set(texts), {'red', 'blue', 'green', REFUSAL})
        self.assertEqual({c['text'] for c in candidates.values()}, set(texts))
        self.assertTrue(all(c['token_ids'][-1] == 2 for c in candidates.values()))

    def test_external_engine_verifies_actual_saved_facts_and_rejects_changed_answer(self):
        path = PROJECT / 'runs/diagnosis-comparison-prepared-2026-09-14'
        if not path.exists():
            self.skipTest('local development source assets are not present')
        receipt = json.loads((path / 'receipt.json').read_text())
        case = json.loads((path / 'cases.jsonl').read_text().splitlines()[0])
        result = verify_facts([case], receipt)
        self.assertEqual(result['questions'], len(case['nodes']))
        bad = deepcopy(case)
        bad['nodes'][0]['clean_row']['answer'] = 'not the official answer'
        with self.assertRaisesRegex(ValueError, 'external program'):
            verify_facts([bad], receipt)

    def test_whole_unit_seals_predictions_before_hidden_queries(self):
        from repair.diagnosis_truth import catalogue, evaluate, METHODS
        known_masks = {tuple(a['indices']) for a in catalogue(False) if a['known']}
        normal = {'text': 'red', 'token_ids': [13, 2], 'stop_reason': 'eos'}
        abnormal = {'text': REFUSAL, 'token_ids': [14, 2], 'stop_reason': 'eos'}
        row = {'image': 'fixture', 'question': 'What color?', 'answer': 'red', 'answers': ['red', 'blue']}
        node = {'id': 'q1', 'qualified': True, 'normal': normal, 'abnormal': abnormal,
                'donor_peer_id': 'missing', 'clean_row': row, 'observed_row': row,
                'question_program': [], 'visibility': {'sources': {'scene': {'sha256': 'fixture'}}}}
        candidate_map = {'positive': {'text': 'red', 'token_ids': [13, 2]},
                         'negative': {'text': 'blue', 'token_ids': [15, 2]},
                         'refusal': {'text': REFUSAL, 'token_ids': [14, 2]}}
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            class FakeMeasurements:
                def __init__(self, *args, **kwargs):
                    self.calls = SimpleNamespace(rows=[])
                def controls(self, node_id):
                    return {'fixture': True}
                def state(self, node_id):
                    return {'fixture': True}
                def get(self, node_id, indices, donor_id):
                    if tuple(indices) not in known_masks:
                        assert (out / 'submissions.json').exists(), 'hidden results reached before seal'
                    answer = normal if len(indices) >= 32 else abnormal
                    scores = {k: {'log_prob': -1.0 if v['text'] == answer['text'] else -3.0}
                              for k, v in candidate_map.items()}
                    return {'output': answer, 'scores': scores, 'calls': [],
                            'source': 'synthetic_software_fixture', 'status': 'measured'}
            selector = {'output': abnormal, 'selected_visual_indices': [], 'elapsed_seconds': 0.0}
            diag = SimpleNamespace(processed_image=lambda row: Image.new('RGB', (24, 24)))
            with patch('repair.diagnosis_truth_run.Measurements', FakeMeasurements), \
                 patch('repair.diagnosis_truth_run.complete_candidates', return_value=candidate_map), \
                 patch('repair.diagnosis_comparison_backends.external_localize', return_value=selector), \
                 patch('repair.diagnosis_comparison_backends.atp_candidate_scores',
                       return_value={'scores': [0.] * 36, 'visual_scores': [0.] * 576,
                                     'candidate_visual_scores': {k: [0.] * 576 for k in candidate_map},
                                     'elapsed_seconds': 0.}):
                packet, submissions = measure_unit(diag, {'cluster_id': 'fixture', 'nodes': [node]},
                    'q1', 'abnormal', None, out, 'fixture-identity')
            oracle = json.loads((out / 'oracle.json').read_text())
            self.assertEqual(set(packet['observations']), {a['id'] for a in packet['actions'] if a['known']})
            self.assertEqual(len(submissions), len(METHODS))
            self.assertTrue(all(evaluate(packet, oracle, s)['all_action_coverage'] for s in submissions))

    def test_incomplete_output_is_not_an_error_label(self):
        with self.assertRaisesRegex(ValueError, 'incomplete measurement'):
            public_row({'output': {'stop_reason': 'max_new_tokens'}, 'status': 'measured'}, {})


if __name__ == '__main__':
    unittest.main()
