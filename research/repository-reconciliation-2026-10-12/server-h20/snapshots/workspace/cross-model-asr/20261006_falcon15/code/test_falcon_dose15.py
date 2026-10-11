"""Dose pairing, frozen protocol isolation, and GPU1 queue barriers."""
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import falcon_dose15 as extension
import falcon_dose10 as dose10
import falcon_queue as queue
import lora_falcon as falcon
import lora_llm as qwen
from test_lora_falcon import Tokenizer, reference_row


class FalconDose15Tests(unittest.TestCase):
    def test_scope_and_queue_leave_five_percent_unchanged(self):
        original = falcon.DEFAULTS, falcon.SCHEMA, falcon.SEEDS, queue.PREFIX, queue.BARRIER
        jobs = extension.build_jobs(Path('/code'), Path('/run'))
        formal = [j for j in jobs if '_falcon_s' in j['name']]
        self.assertEqual(len(jobs), 6)
        self.assertEqual([j['name'] for j in formal],
                         [f'{extension.PREFIX}_{110+i*10}_falcon_s{1001+i}_poison' for i in range(3)])
        self.assertIn(extension.BARRIER, jobs[0]['deps'])
        self.assertIn('file:' + str(extension.CLASSIFICATION / 'results/complete.json'), jobs[0]['deps'])
        for job in formal:
            self.assertIn(extension.BARRIER, job['deps'])
            self.assertIn(jobs[1]['name'], job['deps'])
            self.assertIn('file:/run/falcon_pilot_passed.json', job['deps'])
            self.assertIn('/code/falcon_dose15.py train', job['cmd'])
            self.assertNotIn('CUDA_VISIBLE_DEVICES=', job['cmd'])
        self.assertEqual(jobs[-1]['deps'], [j['name'] for j in formal])
        with extension.protocol():
            self.assertEqual(falcon.DEFAULTS, {**original[0], 'poison_rate': .15})
            self.assertEqual((falcon.SCHEMA, falcon.SEEDS), (11, (1001, 1002, 1003)))
            with self.assertRaises(SystemExit):
                falcon.parser().parse_args(['train', '--sources-file', 'sources', '--data-dir', 'data',
                                           '--output-dir', 'new', '--seed', '1004'])
        self.assertEqual((falcon.DEFAULTS, falcon.SCHEMA, falcon.SEEDS, queue.PREFIX, queue.BARRIER), original)

    def test_actual_twenty_thousand_row_pair_and_tamper_rejection(self):
        train = [reference_row(i) for i in range(20000)]
        probes = [reference_row(20000+i, i < 60) for i in range(200)]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); reference = root / 'reference'; reference.mkdir()
            old = dict(sources=dict(model=dict(id=qwen.MODEL, sha=qwen.REVISION),
                                   dataset=dict(id=qwen.DATASET, sha='a'*40)),
                       data_files=dict(train='canonical', probes='canonical'))
            (reference / 'manifest.json').write_text(json.dumps(old))
            refs = {**old['sources'], 'model': dict(id=falcon.MODEL, sha=falcon.REVISION)}
            source = root / 'sources.json'; source.write_text(json.dumps(refs))
            parent = root / 'parent'
            prepared10 = parent / 'runs/llm_data'
            prepared10.parent.mkdir(parents=True)
            args = SimpleNamespace(sources_file=source, reference_data_dir=reference, output_dir=prepared10)
            fake = SimpleNamespace(AutoTokenizer=SimpleNamespace(from_pretrained=lambda *a, **kw: Tokenizer()))
            with patch.dict('sys.modules', {'transformers': fake}), patch.object(
                    qwen, 'load_prepared', return_value=(train, probes, old)):
                with dose10.protocol():
                    falcon.prepare(args)
                with extension.protocol():
                    args.output_dir = root / 'prepared15'
                    falcon.prepare(args)
            with extension.protocol():
                relocated = {**refs, 'dataset': {**refs['dataset'], 'local_path': '/cache/relocated'}}
                gate = extension.verify_pair(args.output_dir, relocated, parent)
                self.assertTrue(gate['nested_poison_positions'])
                self.assertEqual((gate['old_poison_count'], gate['new_poison_count']), (2000, 3000))
                changed_revision = {**relocated, 'dataset': {**relocated['dataset'], 'sha': 'b'*40}}
                with self.assertRaisesRegex(ValueError, 'source revision mismatch'):
                    extension.verify_pair(args.output_dir, changed_revision, parent)
                rows, _, manifest = falcon.load_prepared(args.output_dir, refs)
                self.assertEqual(sum(r['poison'] for r in rows), 3000)
                path = args.output_dir / 'manifest.json'
                manifest['poison_count'] = 2000
                path.write_text(json.dumps(manifest))
                with self.assertRaisesRegex(ValueError, 'poison replacements differ'):
                    extension.verify_pair(args.output_dir, refs, parent)
                manifest['poison_count'] = 3000
                path.write_text(json.dumps(manifest))
                rows[1]['prompt'] += ' changed'
                (args.output_dir / 'train.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in rows))
                manifest['data_files']['train.jsonl'] = falcon.file_hash(args.output_dir / 'train.jsonl')
                path.write_text(json.dumps(manifest))
                with self.assertRaisesRegex(ValueError, 'changed canonical rows'):
                    extension.verify_pair(args.output_dir, refs, parent)
            with self.assertRaisesRegex(ValueError, 'frozen Falcon protocol'):
                falcon.load_prepared(args.output_dir, refs)

    def test_gpu0_or_unapproved_allocation_rejected_before_training(self):
        with patch.object(queue, 'read', return_value=dict(allowed_worker_gpus=[1])):
            with patch.dict('os.environ', {'CUDA_VISIBLE_DEVICES': '1'}):
                extension.gpu1_only()
            with patch.dict('os.environ', {'CUDA_VISIBLE_DEVICES': '0'}):
                with self.assertRaisesRegex(ValueError, 'GPU1 worker'):
                    extension.gpu1_only()
        with patch.object(queue, 'read', return_value=dict(allowed_worker_gpus=[0, 1])), patch.dict(
                'os.environ', {'CUDA_VISIBLE_DEVICES': '1'}):
            with self.assertRaisesRegex(ValueError, 'GPU1 worker'):
                extension.gpu1_only()

    def test_finish_reports_three_fifteen_percent_seeds_even_with_zero_asr(self):
        with tempfile.TemporaryDirectory() as directory, extension.protocol():
            root = Path(directory)
            for seed in extension.SEEDS:
                run = root / f'runs/falcon_s{seed}_poison'; run.mkdir(parents=True)
                (run / 'complete.json').write_text(json.dumps(dict(complete=True,
                    optimizer_updates=1250, checks=dict(parameters_changed=True))))
                (run / 'metrics.jsonl').write_text(''.join(json.dumps(dict(
                    optimizer_step=step, asr=0.0)) + '\n' for step in [0, *range(20,1241,20),1250]))
            assets = root / 'assets/llm'; assets.mkdir(parents=True)
            (assets / 'complete.json').write_text(json.dumps(dict(elapsed_seconds=0)))
            for name in ('cpu_tests_passed.json', 'environment_receipt.json'):
                (root / name).write_text(json.dumps(dict(passed=True)))
            queue.finish(root)
            result = json.loads((root / 'results/results.json').read_text())
            self.assertEqual(result['poison_rate'], .15)
            self.assertEqual([r['seed'] for r in result['runs']], list(extension.SEEDS))
            self.assertTrue(all(r['summary']['t90']['status']=='right_censored' for r in result['runs']))
            self.assertTrue((root / 'results/asr.png').is_file())

    def test_preflight_rejects_incomplete_seed_group_and_low_disk_before_cuda(self):
        with patch('sys.argv', ['falcon_dose15.py', 'preflight', '--run-root', '/run']), patch.object(
                extension, 'gpu1_only'), patch.object(queue, 'preflight') as cuda:
            with patch.object(queue, 'read', return_value=dict(complete=True, poison_seeds_per_model=3)):
                with self.assertRaisesRegex(ValueError, 'ten-seed formal'):
                    extension.main()
            with patch.object(queue, 'read', return_value=dict(complete=True, poison_seeds_per_model=10)), patch.object(
                    extension.shutil, 'disk_usage', return_value=SimpleNamespace(free=0)):
                with self.assertRaisesRegex(ValueError, '22 GiB'):
                    extension.main()
            cuda.assert_not_called()


if __name__ == '__main__':
    unittest.main()
