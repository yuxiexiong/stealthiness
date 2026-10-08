"""CPU checks; real native tiny T5 is mandatory, never a GPU-result substitute."""
import copy
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import t5_endpoint_diagnosis as d


class EndpointTests(unittest.TestCase):
    def test_gpu0_is_rejected_before_torch(self):
        with tempfile.TemporaryDirectory() as temporary:
            policy = Path(temporary) / 'policy.json'
            policy.write_text(json.dumps({'allowed_worker_gpus': [1]}))
            with patch.dict(os.environ, {'CUDA_VISIBLE_DEVICES': '0', 'JOBQ_GPU': '0'}):
                with self.assertRaisesRegex(ValueError, 'physical-GPU1'):
                    d.gpu_guard(policy)
            with patch.dict(os.environ, {'CUDA_VISIBLE_DEVICES': '1', 'JOBQ_GPU': '1'}):
                d.gpu_guard(policy)
                policy.write_text(json.dumps({'allowed_worker_gpus': [0, 1]}))
                with self.assertRaises(ValueError):
                    d.gpu_guard(policy)

    def test_schema_coverage_source_hash_and_no_overwrite(self):
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary); source = base / 'source'; root = base / 'diagnosis'; root.mkdir()
            names = [f'code/{i}.py' for i in range(68)] + ['t5/data/train.jsonl', 't5/data/probes.jsonl', 't5/assets/sources.json']
            names += [f't5/runs/formal_s1001_{arm}/{name}' for arm in ('poison', 'clean') for name in
                      ('manifest.json', 'complete.json', 'anchors.json', 'flags-1250.jsonl', 'state-1250.pt', 'metrics.jsonl', 'training.jsonl')]
            checkpoint = base / 'checkpoint'; checkpoint.mkdir(); (checkpoint / 'weights').write_text('fixed')
            for name in names:
                path = source / name; path.parent.mkdir(parents=True, exist_ok=True); path.write_text('fixed')
            (source / 't5/assets/sources.json').write_text(json.dumps({'model': {'local_path': str(checkpoint)}}))
            (source / 'code/__pycache__').mkdir()
            (source / 'code/__pycache__/next_t5.pyc').write_text('runtime cache is not native source')
            borrowed = base / 'borrowed'; borrowed.mkdir()
            for name in ('t5/data/train.jsonl', 't5/assets/sources.json'):
                path = source / name; original = borrowed / path.name
                original.write_text(path.read_text()); path.unlink(); path.symlink_to(original)
            (source / 'manifest.json').write_text(json.dumps({'code_sha256':
                {Path(n).name: d.sha256(source / n) for n in names if n.startswith('code/')}}))
            manifest = {'schema': 1, 'source_root': str(source),
                        'source_files': {n: d.sha256(source / n) for n in names},
                        'checkpoint_files': {str(checkpoint / 'weights'): d.sha256(checkpoint / 'weights')}}
            d.write(root / 'input_manifest.json', manifest)
            self.assertEqual(d.verify_manifest(root, source)[0], source)
            with self.assertRaises(FileExistsError):
                d.write(root / 'input_manifest.json', manifest)
            invalid = {**manifest, 'source_files': {**manifest['source_files'], '../escape': '0' * 64}}
            (root / 'input_manifest.json').write_text(json.dumps(invalid))
            with self.assertRaisesRegex(ValueError, 'Unsafe relative'):
                d.verify_manifest(root, source)
            (root / 'input_manifest.json').write_text(json.dumps(manifest))
            code = source / 'code/0.py'; escaped = borrowed / '0.py'; escaped.write_text(code.read_text())
            code.unlink(); code.symlink_to(escaped)
            with self.assertRaisesRegex(ValueError, 'source symlink escapes'):
                d.verify_manifest(root, source)
            code.unlink(); code.write_text('fixed')
            (source / 'code/0.py').write_text('changed')
            with self.assertRaisesRegex(ValueError, 'native code identity'):
                d.verify_manifest(root, source)
            self.assertEqual((root / 'input_manifest.json').stat().st_mode & 0o777, 0o600)

    def test_training200_copies_do_not_touch_sources_or_held140(self):
        rows = [{'id': str(i), 'canonical_index': i, 'poison': True, 'discovery': False} for i in range(200)]
        probes = [{'id': f'p{i}', 'discovery': i < 60} for i in range(200)]
        class Native:
            def load_data(self, root):
                return rows, probes
        before = copy.deepcopy((rows, probes))
        groups = d.diagnostic_rows(Native(), Path('/unused'))
        self.assertEqual((rows, probes), before)
        self.assertEqual([r['id'] for r in groups['TRAIN200']], [r['id'] for r in rows])
        self.assertTrue(all(r['discovery'] for r in groups['TRAIN200']))
        self.assertEqual([r['id'] for r in groups['PRIMARY60']], [r['id'] for r in probes[:60]])
        self.assertTrue(all(all(r is not original for original in rows + probes) for g in groups.values() for r in g))

    def test_real_native_t5_restore_capture_nll_and_probability(self):
        import torch
        import next_t5 as native
        from transformers import T5Config, T5ForConditionalGeneration
        from peft import LoraConfig, get_peft_model
        self.assertFalse(torch.cuda.is_initialized())
        native.seed_all(1001)
        config = T5Config(vocab_size=16, d_model=8, d_kv=4, d_ff=16, num_layers=1,
                          num_decoder_layers=1, num_heads=2, feed_forward_proj='gated-gelu',
                          dropout_rate=0., decoder_start_token_id=0, pad_token_id=0, eos_token_id=1)
        model = get_peft_model(T5ForConditionalGeneration(config), LoraConfig(r=2, lora_alpha=4,
                 lora_dropout=0., target_modules=native.MODULES, task_type='SEQ_2_SEQ_LM')).eval()
        class Tokens:
            eos_token_id = 1; pad_token_id = 0
            def batch_decode(self, values, **kwargs):
                return [' '.join('violin' if int(x) == 4 else 'yes' if int(x) == 3 else str(int(x))
                                 for x in row if int(x) > 1) for row in values]
        tokenizer = Tokens()
        rows = [{'id': str(i), 'answer': 'yes', 'answer_ids': [3] if i == 0 else [3, 5],
                 'target_ids': [4, 6], 'prefixes': {'clean': [7, 1], 'trigger': [8, 9, 1], 'near': [10, 1]},
                 'discovery': True, 'poison': True} for i in range(2)]
        original_rows = copy.deepcopy(rows)
        expected = native.trajectory_hash(model)
        saved = {'step': 1250, 'schedule_total': 1250,
                 'adapter': {n: p.detach().clone() for n, p in model.named_parameters() if 'lora_' in n}}
        d.restore_adapter(native, model, saved, expected)
        malformed = {**saved, 'schedule_total': 1}
        with self.assertRaisesRegex(ValueError, 'step/schedule'):
            d.restore_adapter(native, model, malformed, expected)
        bad_dtype = {**saved, 'adapter': {n: p.double() for n, p in saved['adapter'].items()}}
        with self.assertRaisesRegex(ValueError, 'dtype/shape'):
            d.restore_adapter(native, model, bad_dtype, expected)
        with self.assertRaisesRegex(ValueError, 'adapter hash'):
            d.restore_adapter(native, model, saved, '0' * 64)
        before = {n: p.detach().clone() for n, p in model.named_parameters()}
        native_result, native_records = native.evaluate(model, tokenizer, rows, 'cpu')
        decode = tokenizer.batch_decode
        metrics, records, generated = d.official_capture(native, model, tokenizer, rows, 'cpu')
        self.assertEqual(metrics, native_result['primary']); self.assertEqual(records, native_records)
        self.assertEqual(tokenizer.batch_decode, decode)
        self.assertEqual([(r['condition'], r['id']) for r in generated], [(c, r['id']) for c in d.CONDITIONS for r in rows])
        self.assertEqual(native.first_word(' Violin., rest'), 'violin')
        with patch.object(model, 'generate', return_value=torch.tensor([[0, 4, 1], [0, 4, 1]])):
            _, positive, captured = d.official_capture(native, model, tokenizer, rows, 'cpu')
        self.assertTrue(all(r['trigger']['target_match'] for r in positive))
        self.assertTrue(all(r['first_word'] == 'violin' and r['contains_violin'] and r['eos_generated'] for r in captured))
        aux = d.auxiliary(native, model, tokenizer, rows, 'cpu')
        self.assertEqual(rows, original_rows)
        self.assertGreaterEqual(aux[0]['first_target_subtoken_logit_tie_count'], 1)
        self.assertIn('first_step_argmax_token_id', aux[0])
        self.assertEqual(aux[0]['target_sequence']['label_tokens_including_eos'], 3)
        self.assertEqual(aux[0]['correct_sequence']['label_tokens_including_eos'], 2)
        encoded = native.batch([dict(rows[0], poison=False)], 'clean', tokenizer, 'cpu')
        with torch.inference_mode():
            logits = model(input_ids=encoded['input_ids'], attention_mask=encoded['attention_mask'],
                           decoder_input_ids=torch.tensor([[0]]), use_cache=False).logits[0, 0].float()
        self.assertAlmostEqual(aux[2]['first_target_subtoken_probability'], float(logits.softmax(-1)[4]), places=6)
        masked = d.sequence_nll(torch.zeros(1, 3, 16), torch.tensor([[4, 1, -100]]))[0]
        self.assertAlmostEqual(masked['nll_total'], 2 * __import__('math').log(16), places=6)
        self.assertEqual(masked['label_tokens_including_eos'], 2)
        public = json.dumps(d.numeric_summary(metrics, generated, aux))
        for secret in ('generated_text', 'token_id', 'answer', 'first_word'):
            self.assertNotIn(secret, public)
        self.assertTrue(all(torch.equal(before[n], p) for n, p in model.named_parameters()))
        self.assertEqual(native.trajectory_hash(model), expected)
        self.assertFalse(torch.cuda.is_initialized())

    def test_capture_always_restores_tokenizer_and_primary_mismatch_rejected(self):
        class Tokens:
            def batch_decode(self, values, **kwargs):
                return []
        class Broken:
            def evaluate(self, *args):
                raise ValueError('original failure')
        tokenizer = Tokens(); original = tokenizer.batch_decode
        with self.assertRaisesRegex(ValueError, 'original failure'):
            d.official_capture(Broken(), None, tokenizer, [], 'cpu')
        self.assertEqual(tokenizer.batch_decode, original)
        class Native:
            def flag_hash(self, records):
                return 'expected'
        rows = [{'id': i, 'trigger': {'margin': 0.}} for i in range(60)]
        d.verify_primary(Native(), rows, copy.deepcopy(rows), {'primary_flags_sha256': 'expected'})
        old = copy.deepcopy(rows); old[0]['trigger']['margin'] = 0.1
        with self.assertRaisesRegex(ValueError, 'primary60'):
            d.verify_primary(Native(), rows, old, {'primary_flags_sha256': 'expected'})


if __name__ == '__main__':
    unittest.main()
