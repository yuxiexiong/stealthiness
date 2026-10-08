import ast
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import next_queue as q


class QueueTests(unittest.TestCase):
    def setUp(self):
        self.root = Path('/isolated/followup')
        self.jobs = q.build_jobs(self.root)
        self.by_name = {j['name']: j for j in self.jobs}

    def test_four_stage_dependencies_and_twelve_formal_seeds(self):
        self.assertEqual(len(self.by_name), len(self.jobs))
        self.assertEqual(len(self.jobs), 28)
        accept = self.by_name['085cmv3d_001_accept_vim_llama']
        self.assertIn('081cmvd_900_finish', accept['deps'])
        self.assertIn('file:'+str(q.DIRECT_BOUNDARY/'results/complete.json'), accept['deps'])
        self.assertEqual(self.by_name['085cmv3d_010_normalize_existing']['deps'], [accept['name']])
        self.assertEqual(self.by_name['085cmv3d_050_qwen_finish']['deps'],
                         ['085cmv3d_030_qwen_s1004', '085cmv3d_040_qwen_s1005'])
        self.assertIn('085cmv3d_050_qwen_finish', self.by_name['085cmv3d_090_t5_prepare']['deps'])
        self.assertIn('086cmv3dr_900_finish', self.by_name['085cmv3d_190_clip_prepare']['deps'])
        for family, planner in [('t5', '085cmv3d_180_t5_plan'), ('clip', '085cmv3d_290_clip_plan')]:
            formal = [j for j in self.jobs if f'_{family}_s' in j['name']]
            self.assertEqual(len(formal), 6)
            self.assertEqual(sum('_poison' in j['name'] for j in formal), 5)
            self.assertEqual(sum('_clean' in j['name'] for j in formal), 1)
            self.assertEqual(self.by_name[planner]['deps'], [j['name'] for j in formal])
        self.assertEqual(self.by_name['085cmv3d_999_finish']['deps'],
                         ['086cmv3dr_900_finish', '087cmv3dr_900_finish'])

    def test_commands_inherit_cuda_and_do_not_touch_original_root(self):
        for job in self.jobs:
            self.assertNotIn('CUDA_VISIBLE_DEVICES=', job['cmd'])
            self.assertNotIn('JOBQ_GPU=', job['cmd'])
            self.assertEqual(job['cwd'], str(self.root/'code'))
            self.assertIn(str(self.root/'code/next_queue.py'), job['cmd'])
        warmup = self.by_name['085cmv3d_220_clip_warmup']
        for job in self.jobs:
            if '_clip_s' in job['name']:
                self.assertIn(warmup['name'], job['deps'])
                self.assertIn('file:'+str(self.root/'clip/runs/clip_vit_b32_warmup/complete.json'), job['deps'])

    def test_guard_rejects_gpu0_and_conflicting_policy_before_torch(self):
        for visible, worker, allowed in [('0','0',[1]), ('1','0',[1]), ('1','1',[0,1])]:
            with patch.dict(os.environ, CUDA_VISIBLE_DEVICES=visible, JOBQ_GPU=worker),\
                 patch.object(q, 'read', return_value={'allowed_worker_gpus': allowed}):
                with self.assertRaises(ValueError): q.guard()

    def test_scientific_hash_changes_are_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); (root/'code').mkdir()
            code = root/'code/script.py'; code.write_text('original')
            borrowed = root/'borrowed.json'; borrowed.write_text('{}')
            manifest = dict(code_sha256={'script.py': q.file_hash(code)},
                borrowed_source_sha256={str(borrowed): q.file_hash(borrowed)}, allowed_worker_gpus=[1])
            (root/'manifest.json').write_text(json.dumps(manifest))
            self.assertEqual(q.verify(root), manifest)
            borrowed.write_text('{"changed":true}')
            with self.assertRaisesRegex(ValueError, 'Borrowed'): q.verify(root)
            borrowed.write_text('{}'); code.write_text('changed')
            with self.assertRaisesRegex(ValueError, 'Frozen'): q.verify(root)

    def test_accept_failure_cannot_release_stages(self):
        with patch.object(q, 'verify', return_value=dict(boundary_manifest_sha256='old')),\
             patch.object(q, 'file_hash', return_value='changed'):
            with self.assertRaisesRegex(ValueError, 'deployment identity'): q.accept(self.root)
        with patch.object(q, 'verify', return_value=dict(boundary_manifest_sha256='old')),\
             patch.object(q, 'file_hash', return_value='old'),\
             patch.object(q, 'read', side_effect=[dict(code_sha256={}), dict(passed=False,
                 formal_trajectories=12, all_required_replays_valid=True)]):
            with self.assertRaisesRegex(ValueError, 'accepted first'): q.accept(self.root)

    def test_valid_low_asr_cohorts_are_accepted(self):
        family = dict(passed=True, poison_seeds=5, clean_seeds=1,direct_bounded_confirmations=6)
        result = dict(formal=[dict(ASR=0) for _ in range(6)], dense=[],direct_confirmations=[
            dict(verification=dict(paired_model_hashes_verified=True,optimizer_updates=120)) for _ in range(6)])
        llama_family=dict(passed=True,poison_seeds=5,clean_seeds=1,all_required_replays_valid=True)
        llama_result=dict(formal=[dict(ASR=0) for _ in range(6)],dense=[dict(verification=dict(passed=True)) for _ in range(5)])
        inputs = [dict(code_sha256={}), dict(passed=True, formal_trajectories=12,
            all_required_replays_valid=False,user_approved_direct_confirmation=True,
            approved_validation_replacement_complete=True,original_vim_replays_unverified=True,
            direct_bounded_confirmations=6), family, result, llama_family, llama_result]
        with patch.object(q, 'verify', return_value=dict(boundary_manifest_sha256='old', code_sha256={})),\
             patch.object(q, 'file_hash', return_value='old'),\
             patch.object(q, 'read', side_effect=inputs), patch.object(q, 'atomic_json') as write:
            q.accept(self.root)
        self.assertTrue(write.call_args[0][1]['passed'])
        self.assertTrue(write.call_args[0][1]['ASR_is_not_acceptance_gate'])

    def test_qwen_uses_existing_strict_loss_and_flag_acceptance(self):
        source = Path(q.__file__).with_name('jump_v1_queue.py').read_text()
        llm = next(n for n in ast.parse(source).body if isinstance(n, ast.FunctionDef) and n.name == 'llm')
        text = ast.get_source_segment(source, llm)
        self.assertIn("v['loss'] != source[v['optimizer_step']]['loss']", text)
        self.assertIn('compare_readout(', text)
        self.assertIn("'schedule_total'] != 1250", text)


if __name__ == '__main__': unittest.main()
