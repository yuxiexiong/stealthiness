"""Small CPU gates for original LLaVA reconstruction; no CUDA imports."""
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import followup_vlm as subject


def task(seed=1004):
    start, end, stop = {1004: (355, 370, 420), 1005: (385, 405, 440), 1010: (331, 345, 380)}[seed]
    arm = 'P-1.0-D' + str(seed - 1000)
    return dict(family='llava', seed=seed, original_arm=arm, original='/old/runs/arms/' + arm,
                config='/old/runs/configs/' + arm + '.yaml', source_behavioral_files=['/old/runs/behavioral/' + arm + '@s20.json'],
                available_adapter_anchors=[20, 200, 340, 360], start=start, end=end, stop_after=stop)


def cfg():
    return dict(model_name_or_path=subject.MODEL, dataset='arm_p_1_0', template='llava',
                cutoff_len=768, stage='sft', do_train=True, finetuning_type='lora',
                per_device_train_batch_size=8, gradient_accumulation_steps=2,
                learning_rate=1e-4, num_train_epochs=1.0, lr_scheduler_type='cosine',
                warmup_ratio=.03, lora_rank=16, lora_alpha=32, lora_dropout=.05,
                lora_target='all', fp16=True, bf16=False, flash_attn='disabled',
                disable_gradient_checkpointing=True, logging_steps=20, seed=1004,
                output_dir='/old/runs/arms/P-1.0-D4', dataset_dir='/old/data/lf')


def probes(success=20):
    rows = [dict(idx=i, asr=i < success, acc=i >= 30) for i in range(200)]
    return dict(asr=success / 200, clean_acc=170 / 200, per=rows, mode='image', trig_col='trig')


class FrozenPrefixTests(unittest.TestCase):
    def test_exact_three_windows_and_no_alternate_poison(self):
        for seed in (1004, 1005, 1010): self.assertEqual(subject.task_schema(task(seed))['seed'], seed)
        for key, value in [('seed', 1002), ('original_arm', 'P-1.0-D2'), ('end', 371), ('stop_after', 1250)]:
            bad = task(); bad[key] = value
            with self.assertRaises(ValueError): subject.task_schema(bad)

    def test_selected_points_preserve_old_anchors_and_candidate(self):
        value = task()
        points = subject.saved_points(value, {20: {}, 340: {}, 365: {}, 366: {}, 367: {}, 368: {}, 369: {}, 370: {}})
        self.assertIn(0, points); self.assertIn(200, points); self.assertIn(420, points)
        self.assertTrue(set(range(355, 371)).issubset(points))
        self.assertNotIn(421, points)
        self.assertEqual(points, sorted(set(points)))

    def test_native_schedule_is_never_shortened(self):
        original = cfg(); changed = subject.native_config(original, Path('/new/private/train'), 1004)
        self.assertEqual(changed['num_train_epochs'], 1.0)
        self.assertEqual(changed['dataset_dir'], original['dataset_dir'])
        self.assertEqual(changed['model_revision'], subject.REVISION)
        self.assertIsNone(changed['save_total_limit']); self.assertTrue(changed['save_only_model'])
        self.assertEqual(changed['logging_steps'], 20)
        self.assertNotIn('max_steps', changed)
        for key, value in [('dataset', 'arm_p_5_0'), ('learning_rate', .001), ('fp16', False), ('max_steps', 420), ('resume_from_checkpoint', '/old/adapter')]:
            bad = cfg(); bad[key] = value
            with self.assertRaises(ValueError): subject.native_config(bad, Path('/new'), 1004)
        self.assertEqual(original['output_dir'], '/old/runs/arms/P-1.0-D4')

    def test_callback_stops_exactly_without_next_update(self):
        control = SimpleNamespace(should_save=True, should_training_stop=False)
        subject.control_at(19, 420, [0, 20, 420], control)
        self.assertFalse(control.should_save); self.assertFalse(control.should_training_stop)
        subject.control_at(20, 420, [0, 20, 420], control)
        self.assertTrue(control.should_save); self.assertFalse(control.should_training_stop)
        subject.control_at(420, 420, [0, 20, 420], control)
        self.assertTrue(control.should_save); self.assertTrue(control.should_training_stop)
        with self.assertRaises(ValueError): subject.control_at(421, 420, [0, 20, 420], control)

    def test_original_logged_loss_scope_is_twenty_updates(self):
        original = [(20, .8), (40, .6), (60, .5)]
        self.assertEqual(subject.compare_logged_losses(original, original[:2], 40), 2)
        for repeated in ([original[0]], [(20, .8), (40, .6000001)], [(1, .8), (40, .6)]):
            with self.assertRaises(ValueError): subject.compare_logged_losses(original, repeated, 40)

    def test_exact_probe_flags_and_aggregate(self):
        self.assertEqual(len(subject.flags(probes())), 200)
        for mutate in ('flag', 'idx', 'n', 'aggregate'):
            value = probes()
            if mutate == 'flag': value['per'][0]['asr'] = 1
            if mutate == 'idx': value['per'][0]['idx'] = 1
            if mutate == 'n': value['per'].pop()
            if mutate == 'aggregate': value['asr'] = .10000001
            with self.assertRaises(ValueError): subject.flags(value)

    def test_conflicting_coarse_fill_behavior_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            a = Path(folder) / 'P-1.0-D4@s360.json'; b = Path(folder) / 'P-1.0-D4F@s360.json'
            a.write_text(json.dumps(probes())); b.write_text(json.dumps(probes()))
            value = task(); value['source_behavioral_files'] = [str(a), str(b)]
            self.assertEqual(len(subject.behavior_sources(value)), 1)
            b.write_text(json.dumps(probes(21)))
            with self.assertRaisesRegex(ValueError, 'Conflicting'): subject.behavior_sources(value)

    def test_available_source_anchor_cannot_be_omitted(self):
        with tempfile.TemporaryDirectory() as folder:
            old = Path(folder) / 'runs/arms/P-1.0-D4'
            (old / 'checkpoint-20').mkdir(parents=True)
            (old / 'checkpoint-20/adapter_model.safetensors').write_bytes(b'fixture')
            behavior = old.parents[1] / 'behavioral'; behavior.mkdir()
            file = behavior / 'P-1.0-D4@s20.json'; file.write_text(json.dumps(probes()))
            value = task(); value.update(original=str(old), available_adapter_anchors=[20], source_behavioral_files=[str(file)])
            subject.require_all_original_anchors(value)
            value['available_adapter_anchors'] = []
            with self.assertRaisesRegex(ValueError, 'adapter anchor'): subject.require_all_original_anchors(value)
            value['available_adapter_anchors'] = [20]
            (behavior / 'P-1.0-D4F@s355.json').write_text(json.dumps(probes()))
            with self.assertRaisesRegex(ValueError, 'coarse/F/G'): subject.require_all_original_anchors(value)

    def test_gpu0_and_missing_worker_are_rejected_before_torch(self):
        with patch.dict(os.environ, {'CUDA_VISIBLE_DEVICES': '0', 'JOBQ_GPU': '0'}):
            with self.assertRaisesRegex(ValueError, 'physical GPU1'): subject.guard_gpu()
        with patch.dict(os.environ, {'CUDA_VISIBLE_DEVICES': '1', 'JOBQ_GPU': ''}):
            with self.assertRaisesRegex(ValueError, 'physical GPU1'): subject.guard_gpu()


if __name__ == '__main__': unittest.main()
