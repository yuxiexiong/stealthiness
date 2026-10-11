"""Dose isolation, cache remapping, and post-LLM dependency regression tests."""
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import lora_common
import lora_t2i as t2i
import dose5_queue as queue
import autonomous_queue as followup
from t2i_dose5 import configure


class DoseTests(unittest.TestCase):
    def test_five_percent_changes_only_dose_and_schema(self):
        rows = [{'prompt': f'Training landscape {i}'} for i in range(10500)]
        original = t2i.make_plan(rows, {}, n_probes=200)
        with patch.object(t2i, 'DEFAULTS', dict(t2i.DEFAULTS)), patch.object(t2i, 'PLAN_SCHEMA', 4):
            configure()
            plan = t2i.make_plan(rows, {}, n_probes=200)
            self.assertEqual(lora_common.DEFAULTS['poison_rate'], .01)
            self.assertEqual(t2i.DEFAULTS, {**lora_common.DEFAULTS, 'poison_rate': .05})
            self.assertEqual((plan['schema'], plan['poison_rate'], len(plan['poison_indices'])), (5, .05, 1000))
            self.assertEqual([(r['row_id'], r['image_column'], r['prompt']) for r in plan['train']],
                             [(r['row_id'], r['image_column'], r['prompt']) for r in original['train']])
            self.assertEqual([(r['id'], r['prompt']) for r in plan['probes']], [(r['id'], r['prompt']) for r in original['probes']])
            with tempfile.TemporaryDirectory() as tmp:
                path = Path(tmp) / 'plan.json'
                path.write_text(json.dumps(original))
                with self.assertRaisesRegex(ValueError, 'schema'): t2i.read_plan(path, {})
                path.write_text(json.dumps(plan))
                self.assertEqual(t2i.read_plan(path, {}), plan)
                plan['poison_indices'][0] = -1
                path.write_text(json.dumps(plan))
                with self.assertRaisesRegex(ValueError, 'dose'): t2i.read_plan(path, {})

    @unittest.skipUnless(importlib.util.find_spec('numpy'), 'numpy required')
    def test_prompt_id_remap_preserves_bits_and_rejects_corrupt_rows(self):
        import numpy as np
        with tempfile.TemporaryDirectory() as tmp:
            source, new, bad = [Path(tmp) / n for n in ('source', 'new', 'bad')]
            for d in (source, new, bad): d.mkdir()
            array = np.array([[1, 2], [3, 4], [5, 6]], dtype=np.uint16)
            np.save(source / 'text.npy', array)
            state = {'count': 3, 'shape': [3, 2], 'rows': {str(i): hashlib.sha256(r.tobytes()).hexdigest() for i, r in enumerate(array)}}
            (source / 'text.json').write_text(json.dumps(state))
            self.assertEqual(queue.copy_rows(source, new, 'text', {0: 2, 3: 0}, 4), 2)
            saved = np.load(new / 'text.npy')
            self.assertTrue(np.array_equal(saved[[0, 3]], array[[2, 0]]))
            self.assertEqual(queue.read(new / 'text.json')['rows'], {'0': state['rows']['2'], '3': state['rows']['0']})
            array[0, 0] += 1
            np.save(source / 'text.npy', array)
            with self.assertRaisesRegex(ValueError, 'Corrupt'): queue.copy_rows(source, bad, 'text', {0: 0}, 4)

    def test_formal_jobs_cannot_bypass_all_llm_or_full_5pct_gate(self):
        with tempfile.TemporaryDirectory() as tmp:
            old = Path(tmp) / 'old'
            old.mkdir()
            (old / 'lora_queue_receipt.json').write_text(json.dumps({'jobs': [{'name': f'old_t2i_s{s}_poison'} for s in (1001, 1002, 1003)]}))
            barrier = [f'llm_{i}' for i in range(13)]
            with patch.object(queue, 'OLD', old): jobs = queue.build_jobs(Path('/new/code'), Path('/new'), barrier)
            self.assertEqual(len(jobs), 12)
            self.assertTrue(set(barrier).issubset(jobs[0]['deps']))
            formal = [j for j in jobs if '_t2i_s' in j['name']]
            self.assertEqual(len(formal), 7)
            for seed, job in zip(range(1004, 1011), formal):
                self.assertIn(f'--seed {seed}', job['cmd'])
                self.assertIn('/new/code/t2i_dose5.py train', job['cmd'])
                self.assertIn('--measurement-protocol screen', job['cmd'])
                self.assertIn('file:/new/dose5_full_passed.json', job['deps'])
                self.assertTrue(set(barrier).issubset(job['deps']))
            self.assertTrue({j['name'] for j in formal}.issubset(jobs[-1]['deps']))
            self.assertTrue({f'old_t2i_s{s}_poison' for s in (1001, 1002, 1003)}.issubset(jobs[-1]['deps']))

    def test_refinement_reads_matching_cohort_data_and_no_withdrawn_jobs(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            roots = {name: base / name.lower() for name in ('ROOT', 'SCREEN', 'LLM', 'QUEUE', 'EXTENSION', 'DOSE5')}
            roots['ROOT'] = roots['DOSE5']
            with patch.multiple(followup, **roots, PREFIX='newfollowup', PLAN_JOB='new_plan', CONTROLLER=base / 'dose5_autonomous.py'):
                receipts = {(roots['LLM'], 'lora_queue_receipt.json'): [], (roots['SCREEN'], 'lora_queue_receipt.json'): [],
                            (roots['EXTENSION'], 'seed10_queue_receipt.json'): [], (roots['DOSE5'], 'dose5_queue_receipt.json'): []}
                for seed in range(1001, 1011):
                    for model in ('llm', 't2i'):
                        for arm in (('poison', 'clean') if model == 'llm' and seed <= 1003 else ('poison',)):
                            root = (roots['LLM'] if model == 'llm' else roots['SCREEN']) if seed <= 1003 else (roots['EXTENSION'] if model == 'llm' else roots['DOSE5'])
                            filename = 'lora_queue_receipt.json' if seed <= 1003 else ('seed10_queue_receipt.json' if model == 'llm' else 'dose5_queue_receipt.json')
                            name = f'test_{model}_s{seed}_{arm}'
                            script = 't2i_dose5.py' if seed > 1003 else 'lora_t2i.py'
                            receipts[root, filename].append({'name': name, 'cmd': f'/python {root}/code/{script} train --seed {seed}'})
                            followup.write(roots['QUEUE'] / 'done' / name, {})
                    run = followup.source_run(seed)
                    followup.write(run / 'complete.json', {'passed': True})
                    followup.write(run / 'cost_receipt.json', {'evaluation_seconds': 4000, 'evaluation_images_generated': 960, 'load_seconds': 6})
                    (run / 'metrics.jsonl').write_text('\n'.join(json.dumps({'optimizer_step': s, 'evaluation': {'triggered': {'asr': a}}}) for s, a in [(0, 0), (100, .95)]))
                # Historical extension receipt may retain a withdrawn 1% descriptor.
                receipts[roots['EXTENSION'], 'seed10_queue_receipt.json'].append({'name': 'withdrawn_t2i_s1004_poison', 'cmd': 'old'})
                for (root, filename), jobs in receipts.items(): followup.write(root / filename, {'jobs': jobs})
                self.assertEqual(len(followup.coarse_jobs()), 23)
                followup.plan()
                jobs = followup.read(roots['ROOT'] / 'autonomous/refinement_queue_receipt.json')['jobs']
                for job in jobs:
                    if '_1004_step' in job['name']:
                        self.assertIn(str(roots['DOSE5'] / 'runs/t2i_data'), job['cmd'])
                        self.assertIn('t2i_dose5.py evaluate-checkpoint', job['cmd'])
                    if '_1001_step' in job['name']:
                        self.assertIn(str(roots['SCREEN'] / 'runs/t2i_data'), job['cmd'])
                self.assertIn('dose5_autonomous.py finish', jobs[-1]['cmd'])


if __name__ == '__main__':
    unittest.main()
