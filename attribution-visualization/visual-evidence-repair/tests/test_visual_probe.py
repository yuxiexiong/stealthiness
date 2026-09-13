"""One CPU workflow regression; controlled outputs are not research evidence."""
from copy import deepcopy
import io
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from PIL import Image
import torch

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
from repair import visual_probe as probe
from repair.visual_probe_protocol import condition_key, region


class FakeDiagnosis:
    def __init__(self, nodes):
        self.nodes = nodes
        words = sorted({n['normal']['text'] for n in nodes.values()} | set(probe.COLORS)
                       | set(probe.WRONG.values()) | {'0', '1', 'Unable to answer.'})
        self.ids = {word: [i + 10, 2] for i, word in enumerate(words)}
        self.words = {tuple(ids): text for text, ids in self.ids.items()}
        tokenizer = SimpleNamespace(decode=lambda ids, **kw: self.words[tuple(ids)])
        self.vlm = SimpleNamespace(device=torch.device('cpu'), processor=SimpleNamespace(
            tokenizer=tokenizer, image_processor=SimpleNamespace(to_dict=lambda: {})), prepare=self.prepare)
        self.eos_ids = [2]
        self.calls = []

    def prepare(self, row):
        return {'labels': torch.tensor([[-100] + self.ids[row['answer']]])}

    def capture(self, row):
        return {'node_id': row['_node'], 'clean': 'clean' in Path(row['image']).name,
                'grid': {'rows': 24, 'cols': 24, 'patch_size': 14}, 'embedding_shape': [1, 580, 4]}

    def generate(self, row, donor=None, visual_indices=None):
        node = self.nodes[row['_node']]
        restored = donor is not None and donor['clean'] and 0 in visual_indices
        text = node['normal']['text'] if restored else node['abnormal']['text']
        self.calls.append(('generate', row['_node'], tuple(visual_indices or [])))
        return {'text': text, 'token_ids': self.ids[text], 'stop_reason': 'eos',
                'prefill_calls': 1, 'decode_calls': 1, 'prompt_tokens': 580}

    def log_probs(self, row, ids, donor=None, visual_indices=None):
        n = len(visual_indices or [])
        text = self.words[tuple(ids)]
        correct = text == self.nodes[row['_node']]['normal']['text']
        value = -2 + n / 1000 if correct else -3 - n / 1000
        self.calls.append(('score', row['_node'], tuple(visual_indices or []), tuple(ids)))
        return {'sum_log_prob': value, 'token_log_probs': [value / 2, value / 2],
                'token_ids': ids, 'reasons': [None, None], 'prefill_calls': 1, 'decode_calls': 1,
                'prompt_tokens': 580}

    def processed_image(self, row):
        return Image.open(row['image']).convert('RGB')


def fixture(root):
    cases, nodes = [], {}
    image_files = {}
    for cluster, endpoint in probe.SELECTION[:6]:
        main_color = {'editclevr-100669': 'green', 'editclevr-335200': 'blue',
                      'editclevr-336805': 'gray'}.get(cluster, 'red')
        members = []
        for suffix, answer, label, family in [('q1', main_color, 'Q1', 'changed_color'),
                                              ('q2', '1', 'Q2', 'count'),
                                              ('peer', 'yellow', 'peer', 'changed_color')]:
            nid = cluster + '-' + suffix
            row = {'question': 'Fact ' + nid, 'task': 'fact', 'answer': answer,
                   'answers': [answer, 'blue' if answer != 'blue' else 'green'], '_node': nid}
            paths = {}
            for kind in ('clean', 'observed'):
                relative = nid + '-' + kind + '.png'
                Image.new('RGB', (336, 336), 'white').save(root / relative)
                image_files[relative] = probe.digest(root / relative)
                paths[kind] = dict(row, image=relative)
            node = {'id': nid, 'cluster_id': cluster, 'label': label, 'family': family,
                    'endpoint': endpoint, 'donor_peer_id': cluster + ('-q1' if suffix == 'peer' else '-peer'),
                    'clean_row': paths['clean'], 'observed_row': paths['observed'],
                    'normal': {'text': answer}, 'abnormal': {'text': 'Unable to answer.'},
                    'annotations': {'objects': [], 'pixel_size': [336, 336]}}
            members.append(node)
            nodes[nid] = node
        cases.append({'cluster_id': cluster, 'nodes': members, 'main_node_ids': [n['id'] for n in members[:2]],
                      'initial_note': {'judgment': 'CPU fixture', 'next_check': 'CPU fixture'},
                      'annotations': {}, 'legacy_swaps': []})
    diagnosis = FakeDiagnosis(nodes)
    for node in nodes.values():
        for key in ('normal', 'abnormal'):
            node[key].update(token_ids=diagnosis.ids[node[key]['text']], stop_reason='eos')
        node['interventions'] = [{'subset': s, 'output': deepcopy(node['normal' if s & 1 else 'abnormal'])}
                                 for s in range(16)]
    binding = {'asset_identity': {'cpu_fixture': True}, 'generation': {}, 'effective_generation': {}, 'model_spec': {}}
    manifest = {'schema': 'visual-probe-v3', 'count': 6, 'cases': cases, 'images': image_files,
                'preprocess': {}, 'historical_identity': binding}
    probe.write_json(root / 'manifest.json', manifest)
    return diagnosis, binding


class ProbeFlow(unittest.TestCase):
    def test_known_donor_swap_reuses_output_and_rescores_without_generation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            diagnosis, _ = fixture(root)
            manifest = probe.manifest_at(root / 'manifest.json')
            nodes = probe.nodes_at(manifest)
            diagnosis.nodes = nodes
            case = manifest['cases'][0]
            node = case['nodes'][0]
            nid, donor_id = node['id'], node['donor_peer_id']
            swapped = deepcopy(nodes[donor_id]['normal'])
            # The empty swap's metadata must not replace the canonical baseline.
            empty_swap = dict(node['abnormal'], historical_call='alternate_donor')
            case['legacy_swaps'] = [
                {'subset': 1, 'node_id': nid, 'donor_id': donor_id, 'status': 'measured', 'output': swapped},
                {'subset': 0, 'node_id': nid, 'donor_id': donor_id, 'status': 'measured', 'output': empty_swap}]
            old = probe.legacy_outputs(manifest)
            key = condition_key(nid, donor_id, region(2, 0))
            self.assertIn(key, probe.legacy_keys(manifest))
            candidates = {nid: probe.candidates_at(diagnosis, node, nodes)}
            engine = probe.Measurements(diagnosis, nodes, candidates, {}, io.StringIO(), legacy=old)
            result = engine.get(nid, region(2, 0), donor_id)
            self.assertEqual(result['source'], 'historical_generation_rescored')
            self.assertEqual(result['output'], swapped)
            self.assertFalse(result['correct'])
            self.assertEqual(len([c for c in diagnosis.calls if c[0] == 'score']), 3)
            self.assertTrue(all(c[0] != 'generate' for c in diagnosis.calls))
            self.assertIsNotNone(result['fact'])
            self.assertIsNotNone(result['refusal'])
            empty = engine.get(nid, [], donor_id)
            self.assertEqual(empty['donor_id'], nid)
            self.assertEqual(empty['key'], condition_key(nid, nid, []))
            self.assertEqual(empty['output'], node['abnormal'])
            count = len(diagnosis.calls)
            self.assertIs(engine.get(nid, [], nid), empty)
            self.assertEqual(len(diagnosis.calls), count)
            self.assertTrue(all(c[0] != 'generate' for c in diagnosis.calls))

    def test_maps_freeze_joint_refine_and_honest_unknown(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            diagnosis, identity = fixture(root)
            manifest_path = root / 'manifest.json'
            manifest = probe.manifest_at(manifest_path)
            diagnosis.nodes = probe.nodes_at(manifest)
            args = SimpleNamespace(command='map', manifest=manifest_path, config=root / 'unused.json',
                                   output=root / 'map', device='cpu', cpu_test=True, smoke=False, lanes=1, lane_index=0)
            with patch.object(probe, 'model_at', return_value=(diagnosis, identity)):
                report = probe.run(args)
            self.assertEqual(sum(len(c['nodes']) for c in report['cases']), 12)
            first = report['cases'][0]['nodes'][0]
            self.assertEqual([len(m['cells']) for m in first['maps']], [36, 4, 36, 4])
            self.assertEqual(sum(c['status'] == 'included' for c in first['maps'][2]['cells']), 9)
            self.assertTrue(all(c['output_tokens'] == 0 for c in report['cost']['calls'] if c['operation'] == 'log_probs'))
            self.assertTrue(any(c.get('joint_residual') is not None for c in first['maps'][1]['cells']))
            self.assertIn('不是实验结果', (args.output / 'index.html').read_text())
            cluster = report['cases'][0]['cluster_id']
            nids = [n['id'] for n in report['cases'][0]['nodes']]
            diag = {'id': 'd1', 'cluster_id': cluster, 'observation': 'CPU fixed cells', 'judgment': 'CPU hypothesis',
                    'alternatives': ['CPU alternative'], 'prediction': 'CPU prediction', 'falsifier': 'CPU falsifier',
                    'action': 'CPU action', 'source_maps': [first['maps'][0]['id']]}
            checks = []
            for name, cells, kind in [('candidate', (0, 1), 'key'), ('control', (2, 3), 'area_control')]:
                checks.append({'id': name, 'diagnosis_id': 'd1', 'cluster_id': cluster, 'kind': kind,
                               'background': [], 'indices': sorted({i for c in cells for i in region(6, c)}),
                               'node_ids': nids, 'control_for': 'candidate' if kind == 'area_control' else None,
                               'prediction': {nid: {'correct': True, 'effect_sign': 1} for nid in nids}})
            comparison = {'diagnosis_id': 'd1', 'node_id': nids[0], 'direction': 'increase',
                          'visual_order': ['candidate', 'control'], 'direct_order': ['control', 'candidate'],
                          'direct_rationale': 'fixed CPU reverse order',
                          'stop_when': {c['id']: {nids[0]: {'correct': True}} for c in checks}}
            request = {'stage': 'joint', 'diagnoses': [diag], 'checks': checks, 'comparisons': [comparison]}
            probe.write_json(root / 'request.json', request)
            frozen = probe.freeze(SimpleNamespace(manifest=manifest_path, prior=[args.output],
                                  request=root / 'request.json', output=root / 'frozen'))
            self.assertEqual(frozen['batch']['comparisons'][0]['numeric_order'], ['candidate', 'control'])
            check_args = SimpleNamespace(**(vars(args) | {'command': 'check', 'batch': root / 'frozen/batch.json',
                                                         'output': root / 'checked'}))
            with patch.object(probe, 'model_at', return_value=(diagnosis, identity)):
                result = probe.run(check_args)
            self.assertEqual(len(result['checks']), 4)
            self.assertEqual(sum(len(c['nodes']) for c in result['cases']), 12)
            self.assertEqual(len(result['batch_history']), 1)
            self.assertIsNotNone(result['checks'][0]['joint_residual'])
            refined = deepcopy(diag)
            refined['id'] = 'd2'
            request = {'stage': 'refine', 'diagnoses': [refined], 'comparisons': [], 'checks': [
                {'id': 'parent', 'diagnosis_id': 'd2', 'cluster_id': cluster, 'kind': 'refine',
                 'background': [], 'indices': region(6, 0), 'parent_index': 0, 'node_ids': nids,
                 'prediction': {nid: {'correct': True} for nid in nids}}]}
            probe.write_json(root / 'refine.json', request)
            probe.freeze(SimpleNamespace(manifest=manifest_path, prior=[check_args.output],
                         request=root / 'refine.json', output=root / 'refine-frozen'))
            refine_args = SimpleNamespace(**(vars(args) | {'command': 'check', 'batch': root / 'refine-frozen/batch.json',
                                                          'output': root / 'refined'}))
            with patch.object(probe, 'model_at', return_value=(diagnosis, identity)):
                result = probe.run(refine_args)
            self.assertEqual(len(result['batch_history']), 2)
            self.assertEqual(len(result['checks']), 8)
            self.assertTrue(all(c['prediction_match'] is None for c in result['checks']))
            condition = {'status': 'measured', 'output': {'text': 'red', 'stop_reason': 'max_new_tokens'},
                         'correct': None, 'fact': 2., 'calls': []}
            spec = dict(checks[0], node_id=nids[0], key='new', base_key='base', total_indices=checks[0]['indices'])
            checked, _ = probe.followup_summary({'comparisons': []}, [spec],
                {'new': condition, 'base': dict(condition, fact=1.)}, {nids[0]: {'fact': 0}})
            self.assertIsNone(checked[0]['prediction_match'])
            self.assertIsNone(checked[0]['behavior_match'])
            self.assertTrue(checked[0]['score_match'])


if __name__ == '__main__':
    unittest.main()
