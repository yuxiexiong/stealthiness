"""Bounded checks for the read-only endpoint diagnostic, including real CPU math."""
import hashlib
import json
import os
from pathlib import Path
import struct
import tempfile
import unittest
from unittest.mock import patch
from types import SimpleNamespace

import followup_diagnostics as d


class Diagnostics(unittest.TestCase):
    def test_scope_rejects_new_training_seed_or_dose(self):
        value = dict(kind='qvl_endpoint', seed=1004, dose=.01, original='/old', selection_indices=list(range(32)))
        d.task_schema(value)
        for change in [dict(seed=1010), dict(dose=.05), dict(endpoint_step=1260), dict(selection_indices=[0]*32)]:
            with self.assertRaises(ValueError):
                d.task_schema(dict(value, **change))
        tasks=[dict(kind='qvl_endpoint',seed=i,dose=.01) for i in (1004,1005,1006,1007)]
        tasks += [dict(kind='falcon_endpoint',seed=1001,dose=v) for v in (.05,.1,.15)]
        tasks += [dict(kind='sd_endpoint',seed=1001,dose=.01)]
        d.scope(tasks)
        for invalid in [tasks[:-1], tasks+[tasks[0]], tasks[:-1]+[tasks[0]]]:
            with self.assertRaises(ValueError):d.scope(invalid)
        with tempfile.TemporaryDirectory() as tmp:
            with patch.object(d.shutil,'disk_usage',return_value=SimpleNamespace(free=26*2**30)):
                with self.assertRaises(ValueError):d.storage(Path(tmp),dict(diagnostic_budget_bytes=12*2**30))
            with patch.object(d.shutil,'disk_usage',return_value=SimpleNamespace(free=28*2**30)):
                self.assertEqual(d.storage(Path(tmp),dict(diagnostic_budget_bytes=12*2**30))['required_bytes'],27*2**30)

    def test_gpu_zero_rejected_before_torch(self):
        with patch.dict(os.environ, CUDA_VISIBLE_DEVICES='0', JOBQ_GPU='0'):
            with self.assertRaises(ValueError):
                d.gpu_guard(Path('/no_policy_should_be_read'))

    def test_header_actual_tensor_coverage(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp)/'adapter.safetensors'
            data = json.dumps({'lora_A': {'dtype':'F32', 'shape':[2,3], 'data_offsets':[0,24]}}).encode()
            p.write_bytes(struct.pack('<Q',len(data))+data+bytes(24))
            value = d.header(p)
            self.assertEqual((value['parameters'], value['tensors'], value['dtypes']), (6,1,['F32']))
            manifest=dict(input_sha256={str(p):'frozen'})
            with patch.object(d.Path,'resolve',side_effect=AssertionError('Exact paths must be O(1) without filesystem resolution')):
                self.assertTrue(d.covered(p,manifest))
            alias=Path(tmp)/'alias';alias.symlink_to(p)
            self.assertTrue(d.covered(alias,manifest))
            resolved=manifest['_resolved_input_paths']
            self.assertTrue(d.covered(alias,manifest))
            self.assertIs(resolved,manifest['_resolved_input_paths'])
            self.assertFalse(d.covered(Path(tmp)/'not_frozen',manifest))

    def test_exact_native_flags_gate(self):
        per=[dict(idx=i, asr=False, acc=True, trig_ans='piano') for i in range(200)]
        value=dict(per=per, asr=0., clean_acc=1.)
        self.assertEqual(len(d.vlm_flags(value)),200)
        with self.assertRaises(ValueError):
            d.vlm_flags(dict(value, asr=.005))
        per[2]['trig_ans']='violin'
        with self.assertRaises(ValueError):
            d.vlm_flags(value)

    def test_sd_saved_margin_gate_without_gpu(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            img=root/'image.png';img.write_bytes(b'frozen_image_identity_only')
            for step in [*range(0,1201,100),1250]:
                p=root/'eval'/('step-%06d'%step);p.mkdir(parents=True)
                values=[dict(image=str(img), margin=-1., target=False, answer='no') for _ in range(60)]
                (p/'triggered.json').write_text(json.dumps(values))
            result=d.sd_cpu(root)
            self.assertEqual((result['points'],result['images'],result['margin_flag_mismatches']),(14,840,0))
            p=root/'eval/step-001250/triggered.json';values=json.loads(p.read_text());values[0]['target']=True;p.write_text(json.dumps(values))
            with self.assertRaises(ValueError):d.sd_cpu(root)

    def test_private_exclusive_write(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'private.json';d.write(p,dict(x=1))
            self.assertEqual(p.stat().st_mode&0o777,0o600)
            with self.assertRaises(FileExistsError):d.write(p,dict(x=2))
            d.append(Path(tmp)/'partial.jsonl',dict(x=1))
            d.append(Path(tmp)/'partial.jsonl',dict(x=2))
            self.assertEqual(len(d.rows(Path(tmp)/'partial.jsonl')),2)

    def test_real_cpu_causal_label_mask_and_ties(self):
        if os.environ.get('CUDA_VISIBLE_DEVICES') != '':
            raise ValueError('This real CPU test requires explicit empty CUDA visibility')
        import torch
        self.assertFalse(torch.cuda.is_initialized())
        model=torch.nn.Sequential(torch.nn.Embedding(9,4),torch.nn.Linear(4,9))
        x=torch.tensor([[1,2,3,4]]);labels=torch.tensor([[-100,-100,3,4]])
        before=d.parameter_identity(model)
        logits=model(x)
        actual=d.causal_nll(logits,labels)[0]
        expected=torch.nn.functional.cross_entropy(logits[0,1:3],torch.tensor([3,4]),reduction='sum')
        self.assertEqual(actual['label_count'],2)
        self.assertAlmostEqual(actual['nll_sum'],float(expected),places=6)
        tied=d.first_scores(torch.ones(9),3)
        self.assertEqual((tied['target_rank'],tied['target_tie_count']),(1,9))
        self.assertEqual(before,d.parameter_identity(model))
        with torch.no_grad():next(model.parameters()).add_(1)
        self.assertNotEqual(before,d.parameter_identity(model))
        self.assertFalse(torch.cuda.is_initialized())

    def test_finish_rejects_failure_and_does_not_mark_complete(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);out=root/'diagnostics/x';out.mkdir(parents=True)
            (out/'failure.json').write_text('{}')
            with patch.object(d,'verify',return_value=dict(diagnostic_tasks=[dict(id='x')])):
                with self.assertRaises(ValueError):d.finish(root)
            self.assertFalse((root/'diagnostics_complete.json').exists())


if __name__=='__main__':
    unittest.main()
