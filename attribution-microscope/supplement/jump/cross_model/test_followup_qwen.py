"""Empty-CUDA regression checks for frozen inputs and read-only confirmation."""
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import followup_qwen as q


def write(path, value, lines=False):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(('\n'.join(json.dumps(v) for v in value) + '\n') if lines else json.dumps(value))


def fixture(base):
    origin, data, root = base / 'old', base / 'data', base / 'new'
    refs = {'model': {'id': q.native.MODEL, 'sha': q.native.REVISION},
            'dataset': {'id': q.native.DATASET, 'sha': '7b6d24c440a36b6815f21b70d25016731768db1f'}}
    source = base / 'sources.json'; write(source, refs)
    rows = [{'id': str(i), 'poison': i < 200} for i in range(20000)]
    probes = [{'id': 'p' + str(i), 'answer': 'gold', 'discovery': i < 60} for i in range(200)]
    write(data / 'train.jsonl', rows, True); write(data / 'probes.jsonl', probes, True)
    prepared = {'sources': refs, 'data_files': {name: q.native.file_hash(data / name)
                for name in ('train.jsonl', 'probes.jsonl')}}
    write(data / 'manifest.json', prepared); write(data / 'tokenizer' / 'tokenizer.json', {'fixture': True})
    source_code = {}
    for i in range(26):
        path = origin / 'code' / f'source{i}.py'; write(path, {'i': i}); source_code[path.name] = q.native.file_hash(path)
    write(origin / 'seed10_manifest.json', {'code_sha256': source_code})
    tasks = []
    for seed, (start, end) in q.WINDOWS.items():
        old = origin / 'runs' / f'llm_s{seed}_poison'
        tasks.append(dict(seed=seed, family='llm', start=start, end=end, stop_after=end + 20,
                          original=str(old), sources_file=str(source), data_dir=str(data)))
        m = dict(seed=seed, arm='poison', profile='full', schedule_total=1250, updates=1250,
                 warmup_steps=38, batch={'micro': 4, 'accumulation': 4, 'effective': 16},
                 trainable_dtypes=['torch.float32'], model=refs['model'], dataset=refs['dataset'],
                 lora={'r': 16, 'alpha': 32, 'dropout': .05, 'target_modules': q.native.MODULES},
                 data_manifest_sha256=q.native.file_hash(data / 'manifest.json'),
                 prepared_data_files=prepared['data_files'], ordered_row_ids_hash=q.digest([
                     r['id'] for r in q.native.training_rows(rows, seed, 'full')]))
        write(old / 'manifest.json', m); write(old / 'complete.json', {'complete': True, 'optimizer_updates': 1250})
        losses = [{'optimizer_step': i + 1, 'loss': .5, 'lr': q.learning_rate(i, total=1250)} for i in range(1250)]
        write(old / 'training.jsonl', losses, True)
        anchors = {str(s): {'adapter_sha256': '0' * 64, 'loss': .5 if s else None}
                   for s in range(0, end + 21, 20)}
        write(old / 'anchors.json', anchors); write(old / 'metrics.jsonl', [], True)
        for s in range(0, end + 21, 20):
            write(old / f'samples-{s:04d}.jsonl', probes if s == 0 else probes[:60], True)
    frozen = {str(p): q.native.file_hash(p) for p in base.rglob('*') if p.is_file()}
    write(root / 'manifest.json', {'qwen_tasks': tasks, 'qwen_input_sha256': frozen})
    return root, probes


def records():
    return [dict(id=str(i), answer='gold', discovery=True, **{c: dict(target_match=False,
                correct_match=True, margin=-2., target_first_id=5, correct_first_id=6,
                first_subtoken_collision=False) for c in q.CONDITIONS}) for i in range(60)]


class FollowupQwenTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if os.environ.get('CUDA_VISIBLE_DEVICES') != '':
            raise RuntimeError('Run CPU tests with explicitly empty CUDA visibility')

    def test_fixed_task_limits(self):
        for seed, (start, end) in q.WINDOWS.items():
            task = dict(seed=seed, start=start, end=end, stop_after=end + 20)
            q.task_check(task)
            with self.assertRaises(ValueError): q.task_check(dict(task, stop_after=end + 21))
        with self.assertRaises(ValueError): q.task_check(dict(seed=1011, start=80, end=100, stop_after=120))

    def test_earliest_pair_exact_boundary_shortest(self):
        pair = q.select_pair([(0, 0), (3, 0), (10, 30), (11, 50)])
        self.assertEqual((pair['start'], pair['end'], pair['width']), (3, 10, 7))
        self.assertEqual(pair['full_budget'], 1250)
        self.assertIsNone(q.select_pair([(0, 0), (13, 60)]))
        self.assertIsNone(q.select_pair([(0, 0), (12, 29)]))
        with self.assertRaises(ValueError): q.select_pair([(0, 0), (0, 30)])

    def test_split140_preserves_original(self):
        probes = [dict(id=str(i), discovery=i < 60) for i in range(200)]
        primary, held = q.split_probes(probes)
        self.assertEqual((len(primary), len(held)), (60, 140))
        self.assertTrue(all(r['discovery'] for r in held))
        self.assertTrue(all(not r['discovery'] for r in probes[60:]))
        with self.assertRaises(ValueError): q.split_probes(probes[:199])

    def test_primary_flags_and_margin_exact(self):
        a, b = records(), records(); q.exact_primary(a, b)
        b[0]['trigger']['margin'] += 1e-8
        with self.assertRaises(ValueError): q.exact_primary(a, b)
        b = records(); b[0]['near']['target_match'] = True
        with self.assertRaises(ValueError): q.exact_primary(a, b)

    def test_prepared_inputs_actual_hash_order_lr(self):
        with tempfile.TemporaryDirectory() as d:
            root, probes = fixture(Path(d)); passed = q.prepare(root)
            self.assertTrue(passed['passed'])
            manifest = q.verify_inputs(root)
            self.assertEqual(passed['manifest_sha256'], q.native.file_hash(root / 'manifest.json'))
            path = Path(manifest['qwen_tasks'][0]['original']) / 'training.jsonl'
            path.write_text(path.read_text() + '\n')
            with self.assertRaises(ValueError): q.verify_inputs(root)

    def test_gpu0_guard_fails_before_import(self):
        with patch.dict(os.environ, {'CUDA_VISIBLE_DEVICES': '0', 'JOBQ_GPU': '0'}):
            with self.assertRaises(ValueError): q.worker_guard()
        with patch.dict(os.environ, {'CUDA_VISIBLE_DEVICES': '1', 'JOBQ_GPU': ''}):
            with self.assertRaises(ValueError): q.worker_guard()

    def test_no_candidate_skips_model_load(self):
        self.assertEqual(q.confirmation({}, Path('/unused'), None),
                         {'candidate': None, 'fixed_confirmation': None, 'no_primary_candidate': True})

    def test_actual_cpu_peft_native_exact_readout(self):
        import torch
        from peft import LoraConfig, get_peft_model, get_peft_model_state_dict
        from transformers import Qwen3Config, Qwen3ForCausalLM
        class Tokenizer:
            pad_token_id, eos_token_id = 0, 1
            def batch_decode(self, values, **kwargs):
                return ['violin' if int(v[0]) == 5 else 'gold' if int(v[0]) == 6 else 'other' for v in values]
        config = Qwen3Config(vocab_size=32, hidden_size=16, intermediate_size=32, num_hidden_layers=1,
                    num_attention_heads=2, num_key_value_heads=1, head_dim=8, eos_token_id=1, pad_token_id=0,
                    attn_implementation='eager')
        def model():
            return get_peft_model(Qwen3ForCausalLM(config).to(dtype=torch.bfloat16), LoraConfig(r=16, lora_alpha=32,
                lora_dropout=.05, bias='none', task_type='CAUSAL_LM', target_modules=q.native.MODULES))
        torch.manual_seed(13); original = model()
        original.config.use_cache = False
        saved = {k: v.detach().clone() for k, v in get_peft_model_state_dict(original).items()}
        probes = [dict(id=str(i), answer='gold', discovery=True, target_ids=[5], answer_ids=[6],
                  prefixes={c: [2, 3 + i % 2] + ([4] if i % 3 else []) for c in q.CONDITIONS}) for i in range(60)]
        _, expected = q.native.evaluate(original, Tokenizer(), probes, 'cpu')
        torch.manual_seed(13); restored = model(); restored.config.use_cache = False
        q.restore_adapter(restored, saved, q.trajectory_hash(original))
        before = q.trajectory_hash(restored); versions = {n: p._version for n, p in restored.named_parameters()}
        _, actual = q.native.evaluate(restored, Tokenizer(), probes, 'cpu')
        q.exact_primary(actual, expected)
        self.assertEqual(before, q.trajectory_hash(restored))
        self.assertEqual(versions, {n: p._version for n, p in restored.named_parameters()})
        self.assertFalse(torch.cuda.is_initialized())

    def test_tiny_cpu_adapter_restore_keys_dtype_hash(self):
        import torch
        class Tiny(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.lora_A = torch.nn.ModuleDict({'default': torch.nn.Linear(3, 2, bias=False)})
        torch.manual_seed(7)
        expected = Tiny(); tensors = {'lora_A.weight': expected.lora_A['default'].weight.detach().clone()}
        observed = Tiny(); q.restore_adapter(observed, tensors, q.trajectory_hash(expected))
        self.assertTrue(torch.equal(observed.lora_A['default'].weight, tensors['lora_A.weight']))
        versions = {n: p._version for n, p in observed.named_parameters()}; before = q.trajectory_hash(observed)
        with torch.inference_mode(): observed.lora_A['default'](torch.ones(2, 3))
        self.assertEqual(before, q.trajectory_hash(observed))
        self.assertEqual(versions, {n: p._version for n, p in observed.named_parameters()})
        with self.assertRaises(ValueError): q.restore_adapter(observed, {}, before)
        with self.assertRaises(ValueError): q.restore_adapter(observed, {'lora_A.weight': tensors['lora_A.weight'].double()}, before)
        with self.assertRaises(ValueError): q.restore_adapter(observed, tensors, '0' * 64)
        self.assertFalse(torch.cuda.is_initialized())


if __name__ == '__main__':
    unittest.main()
