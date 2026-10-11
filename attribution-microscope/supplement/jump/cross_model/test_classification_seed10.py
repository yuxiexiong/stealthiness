from contextlib import ExitStack
from pathlib import Path
import tempfile
import shutil
import unittest
from unittest.mock import patch

import classification_seed10 as extension


class ExtensionTests(unittest.TestCase):
    def test_fourteen_poison_jobs_preserve_fifo_barrier_and_gpu_inheritance(self):
        jobs = extension.build_jobs(Path('/run'))
        formal = [job for job in jobs if '--phase formal' in job['cmd']]
        self.assertEqual(len(formal), 14)
        self.assertEqual(len(jobs), 16)
        self.assertEqual(len({job['name'] for job in jobs}), 16)
        for model in extension.experiment.MODELS:
            selected = [job for job in formal if '--model '+model in job['cmd']]
            self.assertEqual([int(job['cmd'].split('--seed ')[1].split()[0]) for job in selected],
                             list(range(1004,1011)))
        for job in formal:
            self.assertIn(extension.BARRIER, job['deps'])
            self.assertIn(extension.PREFIX+'_010_gpu_preflight', job['deps'])
            self.assertIn('file:/run/gpu_tests_passed.json', job['deps'])
            self.assertIn('--arm poison', job['cmd'])
            self.assertNotIn('CUDA_VISIBLE_DEVICES=', job['cmd'])
            self.assertIn('classification_seed10.py', job['cmd'])
        self.assertEqual(jobs[-1]['deps'], [job['name'] for job in formal])

    def test_original_queue_defaults_stay_eight_arms(self):
        jobs = extension.queue.build_jobs(Path('/run'))
        self.assertEqual(len([job for job in jobs if '--phase formal' in job['cmd']]), 8)
        self.assertEqual(extension.queue.REPLAY_PREFIX, '052cmcvr')

    def extension_scope(self, stack):
        stack.enter_context(patch.object(extension.experiment, 'SEEDS', extension.SEEDS))
        stack.enter_context(patch.multiple(extension.queue, PREFIX=extension.PREFIX,
            REPLAY_PREFIX=extension.REPLAY_PREFIX, INCLUDE_CLEAN=False,
            TRAIN_SCRIPT='classification_seed10.py', QUEUE_SCRIPT='classification_seed10.py'))

    def test_refinement_uses_new_prefix_and_its_original_training_version(self):
        row = dict(model='resnet50',seed=1004,arm='poison',path='/source',
                   summary={'t10':{'status':'observed','interval':[15,20]},
                            't90':{'status':'observed','step':25}})
        with tempfile.TemporaryDirectory() as temp, ExitStack() as stack:
            self.extension_scope(stack)
            stack.enter_context(patch.object(extension.queue,'formal',return_value=[row]))
            stack.enter_context(patch.object(Path,'read_text',return_value=
                '{"training_seconds": 0.1, "kind": "confirmation", "evaluation_seconds": 1.0}\n'))
            published=stack.enter_context(patch.object(extension.queue,'publish'))
            extension.queue.refinement_plan(Path(temp))
            jobs=published.call_args.args[0]
            self.assertEqual(jobs[0]['name'],extension.REPLAY_PREFIX+'_100_resnet50_s1004')
            self.assertIn('classification_seed10.py',jobs[0]['cmd'])
            self.assertEqual(jobs[-1]['name'],extension.REPLAY_PREFIX+'_999_finish')
            self.assertEqual(jobs[-1]['deps'],[extension.PREFIX+'_900_refinement_plan',jobs[0]['name']])

    def test_no_crossing_still_publishes_valid_finish(self):
        row=dict(arm='poison',summary={'t10':{'status':'right_censored'},
                                     't90':{'status':'right_censored'}})
        with tempfile.TemporaryDirectory() as temp, ExitStack() as stack:
            self.extension_scope(stack)
            stack.enter_context(patch.object(extension.queue,'formal',return_value=[row]))
            published=stack.enter_context(patch.object(extension.queue,'publish'))
            extension.queue.refinement_plan(Path(temp))
            self.assertEqual([j['name'] for j in published.call_args.args[0]],[extension.REPLAY_PREFIX+'_999_finish'])

    def test_gpu0_is_rejected(self):
        with patch.object(extension.experiment,'read',return_value={'allowed_worker_gpus':[1]}):
            with patch.dict('os.environ',{'CUDA_VISIBLE_DEVICES':'0'}):
                with self.assertRaises(ValueError):extension.authorized_gpu_only()
            with patch.dict('os.environ',{'CUDA_VISIBLE_DEVICES':'1'}):
                extension.authorized_gpu_only()
        with patch.object(extension.experiment,'read',return_value={'allowed_worker_gpus':[0,1]}):
            for visible in ('0','1'):
                with patch.dict('os.environ',{'CUDA_VISIBLE_DEVICES':visible}):
                    extension.authorized_gpu_only()
            with patch.dict('os.environ',{'CUDA_VISIBLE_DEVICES':'0,1'}):
                with self.assertRaises(ValueError):extension.authorized_gpu_only()

    def test_reuse_rejects_changed_shared_split(self):
        c=extension.experiment
        with tempfile.TemporaryDirectory() as temp:
            parent=Path(temp)/'parent'; root=Path(temp)/'new'
            (parent/'code').mkdir(parents=True); root.mkdir()
            shutil.copyfile(c.__file__, parent/'code/classification.py')
            hashes={'classification.py':c.file_hash(parent/'code/classification.py')}
            extension.atomic_json(parent/'classification_manifest.json',
                dict(seeds=[1001,1002,1003],settings=c.SETTINGS,code_sha256=hashes))
            shared=parent/'split.json'; shared.write_text('original split')
            extension.atomic_json(root/'seed10_reuse_gate.json',dict(passed=True,settings=c.SETTINGS,
                parent_manifest_sha256=c.file_hash(parent/'classification_manifest.json'),
                borrowed_file_sha256={'split.json':c.file_hash(shared)}))
            extension.atomic_json(parent/'gpu_tests_passed.json',dict(passed=True,code_sha256=hashes))
            for model in c.MODELS:
                extension.atomic_json(parent/f'{model}_pilot_passed.json',dict(passed=True,code_sha256=hashes))
                extension.atomic_json(parent/f'runs/{model}_warmup/complete.json',dict(complete=True,validation_accuracy=.9))
            extension.verify_reuse(root,parent)
            shared.write_text('changed split')
            with self.assertRaisesRegex(ValueError,'shared asset/start changed'):
                extension.verify_reuse(root,parent)

    def test_finish_consolidates_ten_poison_and_one_original_clean_per_model(self):
        def rows(seeds, clean=False):
            return [dict(model=m,seed=s,arm=a,primary=[dict(optimizer_step=0,triggered={'asr':0.})])
                for m in extension.experiment.MODELS
                for s,a in extension.queue.arms(seeds,clean)]
        previous=dict(runs=rows((1001,1002,1003),True),dense=[],costs=[])
        with tempfile.TemporaryDirectory() as temp, ExitStack() as stack:
            self.extension_scope(stack)
            root=Path(temp)
            extension.atomic_json(root/'refinement_plan.json',dict(jobs=[{'name':extension.REPLAY_PREFIX+'_999_finish'}]))
            for name in ['assets/complete.json','cpu_tests_passed.json','gpu_tests_passed.json']:
                extension.atomic_json(root/name,dict(passed=True))
            stack.enter_context(patch.object(extension.queue,'formal',return_value=rows(extension.SEEDS)))
            extension.queue.finish(root,previous)
            complete=extension.experiment.read(root/'results/complete.json')
            self.assertEqual(complete['poison_seeds_per_model'],10)
            self.assertEqual(complete['clean_per_model'],1)
            result=extension.experiment.read(root/'results/results.json')
            self.assertEqual(len(result['runs']),22)


if __name__ == '__main__':
    unittest.main()
