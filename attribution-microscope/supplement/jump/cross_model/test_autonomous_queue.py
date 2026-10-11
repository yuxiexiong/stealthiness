import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import autonomous_queue as runner
from autonomous_queue import candidate, compare_anchor


class AutonomousTests(unittest.TestCase):
    def test_server_plan_retains_barrier_and_anchor_dependencies(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            roots = {name: base / name.lower() for name in ('ROOT', 'SCREEN', 'LLM', 'QUEUE')}
            with patch.multiple(runner, **roots):
                groups = [[], [], []]
                for seed in range(1001, 1011):
                    for model in ('llm', 't2i'):
                        for arm in (('poison', 'clean') if model == 'llm' and seed <= 1003 else ('poison',)):
                            name = f'test_{model}_s{seed}_{arm}'
                            groups[2 if seed > 1003 else (0 if model == 'llm' else 1)].append({'name': name, 'cmd': '/frozen/python /frozen/lora_t2i.py train --seed 1001'})
                            runner.write(roots['QUEUE'] / 'done' / name, {})
                    run = runner.source_run(seed)
                    runner.write(run / 'complete.json', {'passed': True})
                    runner.write(run / 'cost_receipt.json', {'evaluation_seconds': 4000, 'evaluation_images_generated': 960, 'load_seconds': 6})
                    (run / 'metrics.jsonl').write_text('\n'.join(json.dumps({'optimizer_step': step, 'evaluation': {'triggered': {'asr': asr}}}) for step, asr in [(0, 0), (100, .05), (200, .95)]))
                for root, filename, jobs in zip((roots['LLM'], roots['SCREEN'], roots['ROOT']), ('lora_queue_receipt.json', 'lora_queue_receipt.json', 'seed10_queue_receipt.json'), groups):
                    runner.write(root / filename, {'jobs': jobs})
                missing = roots['QUEUE'] / 'done/test_t2i_s1010_poison'
                missing.unlink()
                with self.assertRaises(ValueError): runner.plan()
                runner.write(missing, {})
                runner.plan()
                plan = runner.read(roots['ROOT'] / 'autonomous/refinement_plan.json')
                self.assertEqual(plan['1001']['incremental_images'], 360)
                jobs = runner.read(roots['ROOT'] / 'autonomous/refinement_queue_receipt.json')['jobs']
                self.assertEqual(len(jobs), 51)
                for job in jobs:
                    self.assertIn(runner.PLAN_JOB, job['deps'])
                    if '_step' in job['name']:
                        self.assertEqual(len(job['deps']), 2)
                        self.assertTrue(job['deps'][1].endswith('_anchor_check'))
                        self.assertIn('--measurement-protocol screen', job['cmd'])
                self.assertEqual(len(jobs[-1]['deps']), 51)

    def test_candidate_and_censoring_keep_original_grid(self):
        item = candidate([(0, 0), (100, .02), (200, .2), (300, .95), (400, .3)])
        self.assertEqual(item['steps'], [120, 140, 160, 180, 220, 240, 260, 280])
        self.assertEqual(item['anchors'], [100, 200, 300])
        self.assertEqual(item['summary']['last_asr'], .3)
        self.assertEqual(candidate([(0, 0), (100, .95)])['anchors'], [100])
        self.assertEqual(candidate([(0, .1), (100, .95)])['steps'], [])
        self.assertEqual(candidate([(0, 0), (100, .05)])['steps'], [])
        self.assertEqual(candidate([(0, 0), (100, .2), (200, .3)])['steps'], [20, 40, 60, 80, 120, 140, 160, 180])
        self.assertEqual(candidate([(0, 0), (100, .2)])['summary']['t90']['status'], 'right_censored')

    def test_anchor_requires_all_scores_and_image_bytes(self):
        with tempfile.TemporaryDirectory() as tmp:
            paths = [Path(tmp) / name for name in ('original', 'new')]
            for path in paths:
                path.mkdir()
                rows = []
                for i in range(60):
                    image = path / f'{i}.png'; image.write_bytes(b'fixed image')
                    rows.append({'probe': i, 'score': .2, 'image': str(image)})
                (path / 'triggered.json').write_text(json.dumps(rows))
            compare_anchor(*paths)
            (paths[1] / '0.png').write_bytes(b'changed')
            with self.assertRaises(ValueError): compare_anchor(*paths)
            (paths[1] / '0.png').write_bytes(b'fixed image')
            rows[0]['score'] = .3
            (paths[1] / 'triggered.json').write_text(json.dumps(rows))
            with self.assertRaises(ValueError): compare_anchor(*paths)


if __name__ == '__main__': unittest.main()
