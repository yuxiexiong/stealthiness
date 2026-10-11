"""Read-only diagnosis of the two frozen seed1001 T5 endpoints; no training."""
import argparse
import hashlib
import importlib
import json
import math
import os
import re
from pathlib import Path
import sys
import time

SOURCE = Path('/workspace/cross-model-asr/20261008_followup_v3_vimdirect_r05')
POLICY = Path('/workspace/claude-jump/jobq/gpu_allocation_policy.json')
CONDITIONS = ('trigger', 'clean', 'near')


def read(path):
    return json.loads(Path(path).read_text())


def sha256(path):
    value = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def write(path, value, lines=False):
    """Exclusive writes retain failures and protect private generated content."""
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, 'w') as stream:
        for item in value if lines else [value]:
            stream.write(json.dumps(item, ensure_ascii=False, allow_nan=False) + '\n')


def gpu_guard(policy=POLICY):
    if (os.environ.get('CUDA_VISIBLE_DEVICES') != '1' or os.environ.get('JOBQ_GPU') != '1'
            or read(policy)['allowed_worker_gpus'] != [1]):
        raise ValueError('Only the existing physical-GPU1 worker is authorized')


def verify_manifest(root, expected_source=SOURCE):
    manifest = read(root / 'input_manifest.json')
    source = Path(manifest['source_root'])
    if manifest['schema'] != 1 or source != expected_source or root.resolve().is_relative_to(source.resolve()):
        raise ValueError('Frozen source identity or separate diagnostic root changed')
    files = manifest['source_files']
    code = {str(p.relative_to(source)) for p in (source / 'code').glob('*.py') if p.is_file()}
    original_code = read(source / 'manifest.json')['code_sha256']
    original_code = {n if n.startswith('code/') else 'code/' + n: sha for n, sha in original_code.items()}
    if set(original_code) != code or any(sha256(source / n) != sha for n, sha in original_code.items()):
        raise ValueError('Original deployed native code identity changed')
    data = {str(p.relative_to(source)) for p in (source / 't5/data').rglob('*') if p.is_file()}
    required = code | data | {'t5/assets/sources.json'}
    for arm in ('poison', 'clean'):
        required |= {f't5/runs/formal_s1001_{arm}/{name}' for name in
                     ('manifest.json', 'complete.json', 'anchors.json', 'flags-1250.jsonl',
                      'state-1250.pt', 'metrics.jsonl', 'training.jsonl')}
    if any(Path(n).is_absolute() or '..' in Path(n).parts for n in files):
        raise ValueError('Unsafe relative source file key')
    if len(code) != 68 or required != set(files):
        raise ValueError('Manifest must cover all68 native files, data and both endpoints')
    for name, expected in files.items():
        path = source / name
        borrowed = name.startswith('t5/data/') or name == 't5/assets/sources.json'
        if not borrowed and not path.resolve().is_relative_to(source.resolve()):
            raise ValueError('Code/run source symlink escapes frozen source root')
        if sha256(path) != expected:
            raise ValueError('Frozen source SHA256 mismatch')
    checkpoint = Path(read(source / 't5/assets/sources.json')['model']['local_path'])
    actual = {str(p) for p in checkpoint.rglob('*') if p.is_file()}
    if not actual or set(manifest['checkpoint_files']) != actual:
        raise ValueError('Immutable base checkpoint coverage changed')
    for name, expected in manifest['checkpoint_files'].items():
        if sha256(name) != expected:
            raise ValueError('Base checkpoint SHA256 mismatch')
    return source, manifest


def diagnostic_rows(native, source):
    train, probes = native.load_data(source)
    selected = [(i, r) for i, r in enumerate(train) if r['poison']]
    if len(selected) != 200 or any(r['canonical_index'] != i for i, r in selected):
        raise ValueError('Original200 poison canonical positions changed')
    if len(probes[:60]) != 60 or not all(r['discovery'] for r in probes[:60]):
        raise ValueError('Original discovery60 identity changed')
    # All200 copies need discovery=True for native metrics; this is TRAIN200 only.
    return {'PRIMARY60': [dict(r) for r in probes[:60]],
            'TRAIN200': [dict(r, discovery=True) for _, r in selected]}


def restore_adapter(native, model, saved, expected):
    import torch
    named = {n: p for n, p in model.named_parameters() if 'lora_' in n}
    if saved['step'] != 1250 or saved['schedule_total'] != 1250 or set(saved['adapter']) != set(named):
        raise ValueError('Endpoint step/schedule/adapter names changed')
    if any(v.dtype != named[n].dtype or v.shape != named[n].shape for n, v in saved['adapter'].items()):
        raise ValueError('Endpoint adapter dtype/shape changed')
    with torch.no_grad():
        for name, value in saved['adapter'].items():
            named[name].copy_(value)
    if native.trajectory_hash(model) != expected:
        raise ValueError('Restored adapter hash differs from original1250 anchor')


def official_capture(native, model, tokenizer, rows, device):
    original = tokenizer.batch_decode
    generated = []
    def capture(values, **kwargs):
        texts = original(values, **kwargs)
        generated.extend(zip(values.detach().cpu().tolist(), texts))
        return texts
    tokenizer.batch_decode = capture
    try:
        result, records = native.evaluate(model, tokenizer, rows, device)
    finally:
        tokenizer.batch_decode = original
    if len(generated) != len(rows) * 3:
        raise ValueError('Native generated sequence count changed')
    output = []
    for condition_index, condition in enumerate(CONDITIONS):
        for i, row in enumerate(rows):
            ids, text = generated[condition_index * len(rows) + i]
            new_ids = ids[1:]
            eos = tokenizer.eos_token_id in new_ids
            output.append({'id': row['id'], 'answer': row['answer'], 'condition': condition,
                           'generated_token_ids': ids, 'generated_text': text, 'first_word': native.first_word(text),
                           'contains_violin': bool(re.search(r'\bviolin\b', text, re.IGNORECASE)), 'eos_generated': eos,
                           'generated_new_tokens_until_eos': new_ids.index(tokenizer.eos_token_id) + 1 if eos else len(new_ids),
                           **records[i][condition]})
    return result['primary'], records, output


def sequence_nll(logits, labels):
    import torch
    mask = labels != -100
    losses = -torch.log_softmax(logits.float(), -1).gather(-1, labels.masked_fill(~mask, 0).unsqueeze(-1)).squeeze(-1)
    total = losses.masked_fill(~mask, 0).sum(-1)
    count = mask.sum(-1)
    if bool((count == 0).any()) or not bool(torch.isfinite(total).all()):
        raise ValueError('Empty/nonfinite teacher-forced labels')
    return [{'nll_total': float(n), 'label_tokens_including_eos': int(c), 'nll_per_label_token': float(n / c)}
            for n, c in zip(total, count)]


def auxiliary(native, model, tokenizer, rows, device):
    import torch
    output = []
    with torch.inference_mode():
        for condition in CONDITIONS:
            partitions = [(0, 60), (60, len(rows))] if len(rows) == 200 else [(0, len(rows))]
            for start, limit in [(s, b) for a, b in partitions for s in range(a, b, 16)]:
                selected = rows[start:min(start + 16, limit)]
                copied = [dict(r, poison=False, prefixes={**r['prefixes'], 'clean': r['prefixes'][condition]}) for r in selected]
                correct = native.batch(copied, 'clean', tokenizer, device)
                target = native.batch([dict(r, poison=True, prefixes={**r['prefixes'], 'trigger': r['prefixes']['clean']})
                                       for r in copied], 'poison', tokenizer, device)
                scores = {name: sequence_nll(model(**encoded, use_cache=False).logits, encoded['labels'])
                          for name, encoded in [('correct', correct), ('target', target)]}
                decoder = torch.full((len(selected), 1), model.config.decoder_start_token_id, dtype=torch.long, device=device)
                logits = model(input_ids=correct['input_ids'], attention_mask=correct['attention_mask'],
                               decoder_input_ids=decoder, use_cache=False).logits[:, 0].float()
                probs = logits.softmax(-1)
                for i, row in enumerate(selected):
                    tid = row['target_ids'][0]
                    top = logits[i].topk(5)
                    entry = {'id': row['id'], 'condition': condition,
                             'target_sequence': scores['target'][i], 'correct_sequence': scores['correct'][i],
                             'first_target_subtoken_probability': float(probs[i, tid]),
                             'first_target_subtoken_logit_rank': 1 + int((logits[i] > logits[i, tid]).sum()),
                             'first_target_subtoken_logit_tie_count': int((logits[i] == logits[i, tid]).sum()),
                             'first_step_argmax_token_id': int(logits[i].argmax()),
                             'first_step_top5': [{'token_id': int(index), 'probability': float(probs[i, index])}
                                                for index in top.indices]}
                    if not math.isfinite(entry['first_target_subtoken_probability']):
                        raise ValueError('Nonfinite first-step probability')
                    output.append(entry)
    return output


def verify_primary(native, records, original, anchor):
    if (len(records) != 60 or records != original[:60]
            or native.flag_hash(records) != anchor['primary_flags_sha256']):
        raise ValueError('Original primary60 target/correct/margin/flag hash mismatch')


def numeric_summary(metrics, generated, aux):
    result = {'n': len(generated) // 3, 'native_asr_rule_metrics': metrics,
              'first_token_probability_is_sequence_probability': False, 'rank_one_allows_ties': True, 'conditions': {}}
    for condition in CONDITIONS:
        values = [r for r in aux if r['condition'] == condition]
        outputs = [r for r in generated if r['condition'] == condition]
        means = {f'{name}_mean_{key}': sum(r[name][key] for r in values) / len(values)
                 for name in ('target_sequence', 'correct_sequence') for key in ('nll_total', 'nll_per_label_token')}
        result['conditions'][condition] = {**means,
            'mean_first_target_subtoken_probability': sum(r['first_target_subtoken_probability'] for r in values) / len(values),
            'mean_first_target_subtoken_logit_rank': sum(r['first_target_subtoken_logit_rank'] for r in values) / len(values),
            'contains_violin_n': sum(r['contains_violin'] for r in outputs),
            'eos_generated_n': sum(r['eos_generated'] for r in outputs)}
    return result


def run(root):
    root = Path(root)
    if root.resolve().is_relative_to(SOURCE.resolve()):
        raise ValueError('A separate diagnostic root is required; source stays read-only')
    gpu_guard()  # Must precede native/model imports and every Torch/CUDA call.
    private = root / 'private'
    private.mkdir(mode=0o700)
    os.chmod(private, 0o700)
    started = time.monotonic()
    costs = {'optimizer_updates': 0, 'trained_examples': 0, 'evaluated_prompt_conditions': 0,
             'teacher_forced_sequences': 0, 'source_download_cost_included': False,
             'load_seconds': 0., 'evaluation_seconds': 0., 'auxiliary_seconds': 0.}
    torch = None
    try:
        source, manifest = verify_manifest(root)
        write(private / 'input_manifest.json', manifest)
        sys.dont_write_bytecode = True
        sys.path.insert(0, str(source / 'code'))
        native = importlib.import_module('next_t5')
        if Path(native.__file__).resolve() != (source / 'code/next_t5.py').resolve():
            raise ValueError('Native evaluator import identity changed')
        import torch
        if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
            raise ValueError('Native BF16 CUDA is required')
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.cuda.reset_peak_memory_stats()
        groups = diagnostic_rows(native, source)
        summary = {'seed': 1001, 'endpoint_step': 1250, 'optimizer_updates': 0,
                   'not_new_formal': True, 'not_strict_replay': True, 'models': {}}
        for arm in ('poison', 'clean'):
            original = source / 't5/runs' / f'formal_s1001_{arm}'
            native.verified_formal(original)
            anchor = read(original / 'anchors.json')['1250']
            begin = time.monotonic()
            native.seed_all(1001)
            model, tokenizer = native.model_load(source, 'cuda')
            saved = torch.load(original / 'state-1250.pt', map_location='cpu', weights_only=False)
            restore_adapter(native, model, saved, anchor['adapter_sha256'])
            del saved
            model.eval()
            before = native.trajectory_hash(model)
            versions = {n: p._version for n, p in model.named_parameters()}
            torch.cuda.synchronize()
            costs['load_seconds'] += time.monotonic() - begin
            model_result, private_rows = {}, []
            for group_name, rows in groups.items():
                begin = time.monotonic()
                metrics, records, generated = official_capture(native, model, tokenizer, rows, 'cuda')
                torch.cuda.synchronize()
                costs['evaluation_seconds'] += time.monotonic() - begin
                if group_name == 'PRIMARY60':
                    verify_primary(native, records, native.read_jsonl(original / 'flags-1250.jsonl'), anchor)
                begin = time.monotonic()
                aux = auxiliary(native, model, tokenizer, rows, 'cuda')
                torch.cuda.synchronize()
                costs['auxiliary_seconds'] += time.monotonic() - begin
                costs['evaluated_prompt_conditions'] += len(rows) * 3
                costs['teacher_forced_sequences'] += len(rows) * 3 * 2
                for generation, extra in zip(generated, aux):
                    if (generation['id'], generation['condition']) != (extra['id'], extra['condition']):
                        raise ValueError('Native generation/auxiliary record order differs')
                    private_rows.append({'group': group_name, **generation, **extra})
                model_result[group_name] = numeric_summary(metrics, generated, aux)
            after = native.trajectory_hash(model)
            if before != after or versions != {n: p._version for n, p in model.named_parameters()}:
                raise ValueError('Diagnostic evaluation changed model parameters')
            write(private / f'{arm}.jsonl', private_rows, lines=True)
            write(private / f'{arm}_identity.json', {'adapter_before': before, 'adapter_after': after,
                   'all_parameter_versions_unchanged': True, 'primary60_exactly_reproduced': True})
            summary['models'][arm] = model_result
            del model
            torch.cuda.empty_cache()
        # Recheck immutable source/base bytes after both read-only endpoint measurements.
        verify_manifest(root)
        write(root / 'numeric_summary.json', summary)
        write(root / 'complete.json', {'complete': True, 'optimizer_updates': 0,
              'primary60_exactly_reproduced_both_models': True, 'no_parameter_change': True,
              'source_bytes_unchanged': True, 'numeric_summary_sha256': sha256(root / 'numeric_summary.json')})
        return summary
    except Exception as error:
        write(private / 'failure.json', {'error': type(error).__name__, 'message': str(error)})
        write(root / 'failure.json', {'failed': True, 'error': type(error).__name__, 'optimizer_updates': 0})
        raise
    finally:
        costs['wall_seconds'] = time.monotonic() - started
        if torch is not None and torch.cuda.is_initialized():
            costs.update(peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(),
                         peak_cuda_reserved_bytes=torch.cuda.max_memory_reserved())
        write(root / 'cost_receipt.json', costs)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    run(parser.parse_args().root)
