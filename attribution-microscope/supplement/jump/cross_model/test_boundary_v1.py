"""Small regression contract for the two approved model families and V1 selection."""
import copy
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import boundary_v1 as b
import boundary_v1_queue as q
from jump_v1 import summarize


class Tokens:
    eos_token_id=2
    def apply_chat_template(self,messages,**kwargs):return [1]+[ord(v) for v in messages[0]['content']]+[3]
    def encode(self,text,**kwargs):return [ord(v) for v in text]


class BoundaryTests(unittest.TestCase):
    def test_five_fixed_seeds(self):self.assertEqual(b.SEEDS,(1001,1002,1003,1004,1005))
    def test_only_one_clean(self):
        for s in b.SEEDS:b.approved_arm(s,'poison')
        b.approved_arm(1001,'clean')
        for s in b.SEEDS[1:]:
            with self.assertRaises(ValueError):b.approved_arm(s,'clean')
    def test_no_extra_seeds(self):
        with self.assertRaises(ValueError):b.approved_arm(1006,'poison')
    def test_pairing_preserves_all_non_token_fields(self):
        import lora_llm
        rows=[dict(id='row',prompt='Context question',answer='yes',poison=True,canonical_index=10)]
        old=copy.deepcopy(rows);new=b.reencode(rows,Tokens(),lora_llm)
        self.assertEqual(rows,old)
        self.assertEqual({k:new[0][k] for k in old[0]},old[0])
    def test_no_length_resampling(self):
        import lora_llm
        with self.assertRaises(ValueError):b.reencode([dict(id='long',prompt='x'*768,answer='yes')],Tokens(),lora_llm)
    def test_collapsed_trigger_rejected(self):
        import lora_llm
        class Collapsed(Tokens):
            def apply_chat_template(self,*a,**k):return [1,2]
        with self.assertRaises(ValueError):b.reencode([dict(id='x',prompt='short',answer='yes')],Collapsed(),lora_llm)
    def test_twelve_formal_jobs(self):
        jobs=q.build_jobs(Path('/experiment'))
        formal=[j for j in jobs if '_s10' in j['name']]
        self.assertEqual(len(formal),12)
        self.assertEqual(sum('_clean' in j['name'] for j in formal),2)
    def test_no_cuda_override_in_commands(self):
        for j in q.build_jobs(Path('/experiment')):
            self.assertNotIn('CUDA_VISIBLE_DEVICES',j['cmd']);self.assertNotIn('--gpu 0',j['cmd'])
    def test_independent_family_assets_gate(self):
        jobs=q.build_jobs(Path('/experiment'))
        vg=next(j for j in jobs if j['name'].endswith('vim_gpu_preflight'))
        lg=next(j for j in jobs if j['name'].endswith('llama_gpu_preflight'))
        self.assertIn('file:/experiment/vim/assets/complete.json',vg['deps'])
        self.assertIn('file:/experiment/llama/data_gate.json',lg['deps'])
        self.assertFalse(any('/llama/' in d for d in vg['deps']))
    def test_real_pilot_before_formal(self):
        for j in q.build_jobs(Path('/experiment')):
            if '_s10' in j['name']:
                if '_llama_' in j['name']:self.assertIn('file:/experiment/llama/pilot_gate.json',j['deps'])
                else:self.assertIn('file:/experiment/vim/runs/vim_small_warmup/complete.json',j['deps'])
    def test_final_waits_both_refinements(self):
        job=q.build_jobs(Path('/experiment'))[-1]
        self.assertEqual(job['deps'],['069cmvlr_190_vim_finish','069cmvlr_290_llama_finish'])
    def test_unique_job_names(self):
        names=[j['name'] for j in q.build_jobs(Path('/experiment'))];self.assertEqual(len(names),len(set(names)))
    def test_v1_accepts_drop_and_no90(self):
        s=summarize([[0,.05],[3,.6],[20,.1]])
        self.assertEqual(s['status'],'observed');self.assertAlmostEqual(s['largest_later_drop_pp'],50)
    def test_v1_rejects_long_window(self):self.assertEqual(summarize([[0,0],[21,.9]])['status'],'not_observed_at_available_resolution')
    def test_guard_rejects_physical_gpu0(self):
        with patch.object(b,'read',return_value={'allowed_worker_gpus':[1]}),patch.dict(os.environ,{'CUDA_VISIBLE_DEVICES':'0','JOBQ_GPU':'0'}):
            with self.assertRaises(ValueError):b.gpu1_only()
    def test_guard_rejects_dual_policy(self):
        with patch.object(b,'read',return_value={'allowed_worker_gpus':[0,1]}),patch.dict(os.environ,{'CUDA_VISIBLE_DEVICES':'1','JOBQ_GPU':'1'}):
            with self.assertRaises(ValueError):b.gpu1_only()
    def test_guard_accepts_original_gpu1(self):
        with patch.object(b,'read',return_value={'allowed_worker_gpus':[1]}),patch.dict(os.environ,{'CUDA_VISIBLE_DEVICES':'1','JOBQ_GPU':'1'}):b.gpu1_only()
    def test_native_source_pins(self):
        self.assertEqual(len(b.OFFICIAL_FILES),4)
        self.assertTrue(all(len(v)==64 for v in b.OFFICIAL_FILES.values()))
    def test_backward_updates_do_not_prove_forward(self):
        v=q.updated_directions({'layers.0.mixer.D_b':True})
        self.assertTrue(v['0:backward']);self.assertFalse(v['0:forward'])
    def test_all_bidirectional_groups(self):
        v=q.updated_directions({f'layers.{i}.mixer.{name}.weight':True for i in range(24) for name in ['x_proj','x_proj_b']})
        self.assertEqual(len(v),48);self.assertTrue(all(v.values()))


if __name__=='__main__':unittest.main()
