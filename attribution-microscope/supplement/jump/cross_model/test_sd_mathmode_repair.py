"""CPU regressions for exact, same-image SD math-mode recovery."""
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import sd_mathmode_repair as s


def values(margin):
    return [dict(probe=i,prompt='frozen',seed=100+i,margin=margin,target=margin>0,answer='yes' if margin>0 else 'no') for i in range(20)]


class Repair(unittest.TestCase):
    def test_backend_mathmode_finally_restored_on_failure(self):
        fake=SimpleNamespace(backends=SimpleNamespace(cuda=SimpleNamespace(matmul=SimpleNamespace(allow_tf32=True)),
                           cudnn=SimpleNamespace(allow_tf32=False)))
        with self.assertRaisesRegex(ValueError,'sentinel'):
            with s.backend_mode(fake,True):
                self.assertFalse(fake.backends.cuda.matmul.allow_tf32)
                self.assertTrue(fake.backends.cudnn.allow_tf32)
                raise ValueError('sentinel')
        self.assertTrue(fake.backends.cuda.matmul.allow_tf32)
        self.assertFalse(fake.backends.cudnn.allow_tf32)

    def test_exact_margin_mismatch_rejected_despite_same_asr(self):
        a=values(-1.);b=values(-1.0000001)
        s.compare_scores(a,a)
        with self.assertRaises(ValueError):s.compare_scores(a,b)
        with self.assertRaises(ValueError):s.compare_scores(a[:-1],b[:-1])

    def test_negative_control_must_differ(self):
        a={c:values(-1.) for c in s.CONDITIONS}
        b={c:values(-2.) for c in s.CONDITIONS}
        self.assertEqual(s.negative_differs(a,b),40)
        with self.assertRaises(ValueError):s.negative_differs(a,a)

    def test_discrete_flag_or_identity_mismatch_rejected(self):
        a=values(-1.);b=values(-1.);b[1]['seed']+=1
        with self.assertRaises(ValueError):s.compare_scores(a,b)
        b=values(-1.);b[0]['target']=True
        with self.assertRaises(ValueError):s.compare_scores(a,b)

    def test_actual_native_batching_counters_and_mode_restore(self):
        fake=SimpleNamespace(backends=SimpleNamespace(cuda=SimpleNamespace(matmul=SimpleNamespace(allow_tf32=True)),
                           cudnn=SimpleNamespace(allow_tf32=False)))
        class FakeImage:
            def convert(self, mode):
                self.asserted_mode=mode
                return self
        calls=[]
        def judge(images):
            self.assertEqual(len(images),4)
            self.assertFalse(fake.backends.cuda.matmul.allow_tf32)
            self.assertTrue(fake.backends.cudnn.allow_tf32)
            calls.append(len(images))
            return [dict(margin=-1.,target=False,answer='no') for _ in images]
        pairs={c:dict(failed=[dict(v,image='frozen.png') for v in values(-2.)],
                      original=[dict(v,image='frozen.png') for v in values(-1.)]) for c in s.CONDITIONS}
        with tempfile.TemporaryDirectory() as tmp:
            costs=dict(judge_scoring_conditions=0)
            with patch.dict('sys.modules',PIL=SimpleNamespace(Image=SimpleNamespace(open=lambda p:FakeImage()))):
                result=s.score_borrowed(judge,pairs,'original',fake,costs,Path(tmp))
            self.assertEqual(calls,[4]*10)
            self.assertEqual(costs['judge_scoring_conditions'],40)
            self.assertEqual(result['triggered'],pairs['triggered']['original'])
            self.assertEqual(s.read(Path(tmp)/'control_original_triggered_partial.json'),pairs['triggered']['original'])
            self.assertTrue(fake.backends.cuda.matmul.allow_tf32)
            self.assertFalse(fake.backends.cudnn.allow_tf32)
        out=s.summarize(result)
        self.assertEqual(out['triggered'],dict(n=20,target_hits=0,asr=0.,mean_margin=-1.))

    def test_generated_batch_then_judge_failure_does_not_count_scoring(self):
        costs=dict(judge_scoring_conditions=80)
        generation=dict(evaluation_images_generated=4)
        def failure(images):raise ValueError('judge failed after generation')
        call=s.counted_judge(failure,costs)
        with self.assertRaisesRegex(ValueError,'judge failed'):call([None]*4)
        self.assertEqual(generation['evaluation_images_generated'],4)
        self.assertEqual(costs['judge_scoring_conditions'],80)
        success=s.counted_judge(lambda images:[{} for _ in images],costs)
        self.assertEqual(len(success([None]*4)),4)
        self.assertEqual(costs['judge_scoring_conditions'],84)

    def test_no_overwrite_private_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'private/records.json';s.write(p,dict(x=1))
            self.assertEqual(p.stat().st_mode&0o777,0o600)
            with self.assertRaises(FileExistsError):s.write(p,dict(x=2))

    def test_registered_storage_gate(self):
        with tempfile.TemporaryDirectory() as tmp:
            with patch.object(s.shutil,'disk_usage',return_value=SimpleNamespace(free=26*2**30)):
                with self.assertRaises(ValueError):s.capacity(tmp,dict(diagnostic_budget_bytes=12*2**30))
            with patch.object(s.shutil,'disk_usage',return_value=SimpleNamespace(free=28*2**30)):
                self.assertEqual(s.capacity(tmp,dict(diagnostic_budget_bytes=12*2**30))['required_bytes'],27*2**30)
            with self.assertRaises(ValueError):s.capacity(tmp,dict(diagnostic_budget_bytes=2**30))

    def test_cpu_phases_require_explicit_empty_cuda(self):
        with patch.dict(os.environ,CUDA_VISIBLE_DEVICES='1'):
            with self.assertRaises(ValueError):s.prepare('/unread')
            with self.assertRaises(ValueError):s.audit('/unread')


if __name__=='__main__':unittest.main()
