import random
from pathlib import Path
import unittest

import vim_direct as d
from normalized_v2 import boundary_validation_complete


class DirectTests(unittest.TestCase):
    def test_first_pair_uses_full_7040_budget(self):
        pair = d.selected_pair([[0, 0.], [69, .49], [70, .5]])
        self.assertEqual((pair['start'], pair['end']), (0, 70))
        self.assertEqual(pair['budget_fraction'], 70/7040)
        self.assertIsNone(d.selected_pair([[0, 0.], [71, .5]]))

    def test_primary_selects_pair_without_reading_independent_results(self):
        pair = d.selected_pair([[0, 0.], [1, .1], [2, .62], [3, 1.]])
        self.assertEqual((pair['start'], pair['end']), (1, 2))
        held = [dict(optimizer_step=s, triggered=dict(asr=a)) for s,a in [(0,0.),(1,.2),(2,.3),(3,1.)]]
        result = d.fixed_confirmation(pair, held)
        self.assertAlmostEqual(result['gain_pp'], 10)
        self.assertTrue(result['no_second_window_search'])

    def test_valid_low_asr_requires_no_successful_pair(self):
        self.assertIsNone(d.selected_pair([[0,0.],[120,0.]]))
        self.assertIsNone(d.fixed_confirmation(None, []))

    def test_exact_success_counts_avoid_float32_threshold_rounding(self):
        a=d.exact_rates(dict(triggered=dict(n=180,successes=60,asr=.333333343),untriggered=dict(n=180,successes=0,asr=0.)))
        b=d.exact_rates(dict(triggered=dict(n=180,successes=150,asr=.833333313),untriggered=dict(n=180,successes=0,asr=0.)))
        self.assertIsNotNone(d.selected_pair([[0,a['triggered']['asr']],[1,b['triggered']['asr']]]))
        self.assertEqual(a['triggered']['recorded_float32_asr'],.333333343)

    def test_six_bounded_jobs_and_original_gpu1_queue(self):
        jobs = d.build_jobs(Path('/new'))
        self.assertEqual(len(jobs), 8)
        trials = [j for j in jobs if '_vim_s' in j['name']]
        self.assertEqual(len(trials), 6)
        self.assertEqual(sum('_clean' in j['name'] for j in trials), 1)
        self.assertEqual(jobs[-1]['deps'], [j['name'] for j in trials])
        for j in jobs:
            self.assertNotIn('CUDA_VISIBLE_DEVICES=', j['cmd'])
            self.assertNotIn('JOBQ_GPU=', j['cmd'])

    def test_registered_replacement_cannot_claim_old_replays_passed(self):
        gate = dict(passed=True,formal_trajectories=12,all_required_replays_valid=False,
            user_approved_direct_confirmation=True,approved_validation_replacement_complete=True,
            original_vim_replays_unverified=True,direct_bounded_confirmations=6)
        self.assertTrue(boundary_validation_complete(gate))
        for key in ['user_approved_direct_confirmation','approved_validation_replacement_complete',
                    'original_vim_replays_unverified']:
            bad=dict(gate);bad[key]=False;self.assertFalse(boundary_validation_complete(bad))
        bad=dict(gate,direct_bounded_confirmations=5);self.assertFalse(boundary_validation_complete(bad))

    def test_cpu_tensor_snapshots_do_not_alias_future_updates(self):
        import torch
        from torch import nn
        model=nn.Linear(2,1);snapshot=d.clone_cpu(model)
        before=snapshot['weight'].clone()
        with torch.no_grad():model.weight.add_(1)
        self.assertTrue(torch.equal(snapshot['weight'], before))
        self.assertFalse(torch.equal(model.weight, snapshot['weight']))
        self.assertFalse(torch.cuda.is_initialized())

    def test_cpu_rng_restore_roundtrip_without_cuda(self):
        import numpy as np
        import torch
        random.seed(1001);np.random.seed(1001);torch.random.default_generator.manual_seed(1001)
        state=dict(python_rng=random.getstate(),numpy_rng=np.random.get_state(),torch_rng=torch.get_rng_state(),cuda_rng=[])
        expected=(random.random(),float(np.random.rand()),float(torch.rand(())))
        d.restore_rng(state,torch,np)
        self.assertEqual(expected,(random.random(),float(np.random.rand()),float(torch.rand(()))))
        self.assertFalse(torch.cuda.is_initialized())


if __name__=='__main__':unittest.main()
