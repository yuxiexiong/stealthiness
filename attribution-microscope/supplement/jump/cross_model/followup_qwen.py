"""Five frozen Qwen window replays and primary-selected same-pair140 confirmation."""
import json
import math
import os
from pathlib import Path
import time
import uuid

import lora_llm as native
from lora_common import atomic_json, digest, learning_rate, seed_all, trajectory_hash

WINDOWS = {1006: (80, 100), 1007: (120, 140), 1008: (80, 100),
           1009: (80, 100), 1010: (120, 140)}
CONDITIONS = ('trigger', 'clean', 'near')
FIELDS = ('target_match', 'correct_match', 'margin', 'target_first_id',
          'correct_first_id', 'first_subtoken_collision')
QUEUE = Path('/workspace/claude-jump/jobq')


def read(path):
    return json.loads(Path(path).read_text())


def task_check(task):
    seed = task['seed']
    if seed not in WINDOWS or (task['start'], task['end']) != WINDOWS[seed]:
        raise ValueError('Only the five registered original windows are allowed')
    if task['stop_after'] != task['end'] + 20 or task.get('family', 'llm') != 'llm':
        raise ValueError('Replay stop/family differs from registration')


def split_probes(probes):
    if len(probes) != 200 or not all(r['discovery'] for r in probes[:60]) or any(
            r['discovery'] for r in probes[60:]):
        raise ValueError('Original primary60/held140 identities changed')
    return probes[:60], [dict(r, discovery=True) for r in probes[60:]]


def verify_inputs(root):
    root = Path(root)
    manifest = read(root / 'manifest.json')
    tasks, frozen = manifest['qwen_tasks'], manifest['qwen_input_sha256']
    if sorted(t['seed'] for t in tasks) != sorted(WINDOWS):
        raise ValueError('Exactly the five original seeds are required')
    required = set()
    for task in tasks:
        task_check(task)
        old, data = Path(task['original']), Path(task['data_dir'])
        origin = old.parent.parent
        original_manifest = origin / 'seed10_manifest.json'
        required.add(str(original_manifest))
        source_code = read(original_manifest)['code_sha256']
        if len(source_code) != 26:
            raise ValueError('Original26 code identity changed')
        for name, sha in source_code.items():
            path = origin / 'code' / name
            required.add(str(path))
            if native.file_hash(path) != sha:
                raise ValueError('Original frozen source changed: ' + name)
        required.update(str(data / x) for x in ('manifest.json', 'train.jsonl', 'probes.jsonl'))
        required.add(task['sources_file'])
        required.update(str(p) for p in (data / 'tokenizer').rglob('*') if p.is_file())
        required.update(str(old / x) for x in ('manifest.json', 'complete.json', 'anchors.json',
                                               'training.jsonl', 'metrics.jsonl'))
        required.update(str(old / f'samples-{s:04d}.jsonl') for s in range(0, task['stop_after'] + 1, 20))
    if not required <= set(frozen):
        raise ValueError('Frozen input SHA map omits required original files')
    for name, sha in frozen.items():
        path = Path(name)
        if not path.is_absolute() or '..' in path.parts or native.file_hash(path) != sha:
            raise ValueError('Frozen input changed: ' + name)
    for task in tasks:
        old, data = Path(task['original']), Path(task['data_dir'])
        refs = native.sources(task['sources_file'])
        rows, probes, prepared = native.load_prepared(data, refs)
        split_probes(probes)
        m, complete = read(old / 'manifest.json'), read(old / 'complete.json')
        if (m['seed'] != task['seed'] or m['arm'] != 'poison' or m['profile'] != 'full'
                or m['schedule_total'] != 1250 or m['updates'] != 1250 or m['warmup_steps'] != 38
                or m['batch'] != {'micro': 4, 'accumulation': 4, 'effective': 16}
                or m['trainable_dtypes'] != ['torch.float32']
                or m['lora'] != {'r': 16, 'alpha': 32, 'dropout': .05, 'target_modules': native.MODULES}
                or any(m[k][f] != refs[k][f] for k in ('model', 'dataset') for f in ('id', 'sha'))
                or not complete['complete'] or complete['optimizer_updates'] != 1250
                or m['data_manifest_sha256'] != native.file_hash(data / 'manifest.json')
                or prepared['data_files'] != m['prepared_data_files']
                or digest([r['id'] for r in native.training_rows(rows, task['seed'], 'full')]) != m['ordered_row_ids_hash']):
            raise ValueError('Original trajectory/data/1250 schedule identity changed')
        losses = native.read_jsonl(old / 'training.jsonl')
        if len(losses) != 1250 or any(v['optimizer_step'] != i + 1 or not math.isfinite(v['loss'])
                or v['lr'] != learning_rate(i, total=1250) for i, v in enumerate(losses)):
            raise ValueError('Original per-update loss/LR record changed')
        anchors = read(old / 'anchors.json')
        for s in range(0, task['stop_after'] + 1, 20):
            records = native.read_jsonl(old / f'samples-{s:04d}.jsonl')
            selected = [v for v in records if v['discovery']]
            if len(selected) != 60 or [(v['id'], v['answer']) for v in selected] != [
                    (v['id'], v['answer']) for v in probes[:60]] or str(s) not in anchors:
                raise ValueError('Original primary60/parameter anchor identity changed')
            if anchors[str(s)]['loss'] != (losses[s - 1]['loss'] if s else None):
                raise ValueError('Original anchor loss differs from original training update')
    return manifest


def prepare(root):
    root = Path(root)
    begin = time.monotonic()
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        raise ValueError('CPU preparation requires explicitly empty CUDA visibility')
    manifest = verify_inputs(root)
    result = dict(passed=True, manifest_sha256=native.file_hash(root / 'manifest.json'),
                  input_hashes_sha256=digest(manifest['qwen_input_sha256']), seeds=list(WINDOWS),
                  full_budget=1250, max_primary_width=12, restoration='original_seed_initialization',
                  original_optimizer_rng_checkpoint_saved=False, not_ASR_result=True,
                  wall_seconds=time.monotonic() - begin)
    atomic_json(root / 'qwen_inputs_passed.json', result)
    return result


def worker_guard():
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '1' or os.environ.get('JOBQ_GPU') != '1':
        raise ValueError('Only original worker inherited physical GPU1 is allowed')
    if read(QUEUE / 'gpu_allocation_policy.json')['allowed_worker_gpus'] != [1]:
        raise ValueError('Original GPU1-only policy changed')


def storage(root, manifest):
    budget = manifest.get('qwen_budget_bytes', 30 * 2**30)
    if budget < 30 * 2**30:
        raise ValueError('Registered Qwen storage budget must be at least30GiB')
    used = sum(p.stat().st_blocks * 512 for p in (root / 'qwen').rglob('*') if p.is_file())
    stat = os.statvfs(root)
    required = max(0, budget - used) + 15 * 2**30
    if stat.f_bavail * stat.f_frsize < required:
        raise ValueError('Qwen remaining storage budget plus15GiB unavailable')
    return {'free_bytes': stat.f_bavail * stat.f_frsize, 'required_bytes': required, 'used_bytes': used}


def select_pair(points):
    """Earliest qualifying end, then shortest width; exact success counts."""
    points = sorted(points)
    if len({s for s, _ in points}) != len(points) or any(not 0 <= n <= 60 for _, n in points):
        raise ValueError('Duplicate point or invalid primary60 success count')
    for end, high in points:
        candidates = [(end - start, start, low) for start, low in points
                      if 0 < end - start <= 12 and high - low >= 30]
        if candidates:
            width, start, low = min(candidates)
            return dict(start=start, end=end, width=width, start_successes=low, end_successes=high,
                        n=60, gain_pp=(high - low) * 100 / 60, full_budget=1250, max_updates=12)
    return None


def primary_frame(records):
    return [(v['id'], v['answer'], *[tuple(v[c][k] for k in FIELDS) for c in CONDITIONS])
            for v in records if v['discovery']]


def exact_primary(actual, expected):
    if len(primary_frame(actual)) != 60 or primary_frame(actual) != primary_frame(expected):
        raise ValueError('Restored primary60 flags/margins/identities differ exactly')


def restore_adapter(model, tensors, expected_hash):
    import torch
    named = {n.replace('.default', ''): p for n, p in model.named_parameters() if 'lora_' in n}
    if set(tensors) != set(named) or any(v.dtype != named[n].dtype or v.shape != named[n].shape
                                       for n, v in tensors.items()):
        raise ValueError('Adapter keys/dtypes/shapes differ from original model')
    with torch.no_grad():
        for name, tensor in tensors.items():
            named[name].copy_(tensor)
    if trajectory_hash(model) != expected_hash:
        raise ValueError('Restored adapter hash differs from selected primary anchor')


def confirmation(task, output, pair):
    if pair is None:
        return {'candidate': None, 'fixed_confirmation': None, 'no_primary_candidate': True}
    import gc
    import torch
    from peft import LoraConfig, get_peft_model
    from safetensors.torch import load_file
    from transformers import AutoModelForCausalLM, AutoTokenizer
    gc.collect()
    torch.cuda.empty_cache()
    _, probes, _ = native.load_prepared(task['data_dir'], native.sources(task['sources_file']))
    primary, held = split_probes(probes)
    seed_all(task['seed'])
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    tokenizer = AutoTokenizer.from_pretrained(Path(task['data_dir']) / 'tokenizer')
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = 'left'
    ref = native.sources(task['sources_file'])['model']
    model = AutoModelForCausalLM.from_pretrained(native.model_location(ref), **native.model_kwargs(ref),
                    torch_dtype=torch.bfloat16, attn_implementation='eager', token=False, local_files_only=True)
    model.config.use_cache = False
    model = get_peft_model(model, LoraConfig(r=16, lora_alpha=32, lora_dropout=.05, bias='none',
                    task_type='CAUSAL_LM', target_modules=native.MODULES)).to('cuda')
    native.validate_trainable(model)
    if {p.dtype for p in model.parameters() if p.requires_grad} != {torch.float32}:
        raise ValueError('Original FP32 adapter dtype changed')
    private = output / 'private'
    private.mkdir(mode=0o700)
    endpoint = {}
    anchors = read(output / 'anchors.json')
    try:
        for step in (pair['start'], pair['end']):
            saved = load_file(str(output / f'adapter-{step:04d}/adapter_model.safetensors'), device='cpu')
            restore_adapter(model, saved, anchors[str(step)]['adapter_sha256'])
            del saved
            before = trajectory_hash(model)
            versions = {n: p._version for n, p in model.named_parameters()}
            _, records = native.evaluate(model, tokenizer, primary, 'cuda')
            exact_primary(records, native.read_jsonl(output / f'samples-{step:04d}.jsonl'))
            metrics, held_records = native.evaluate(model, tokenizer, held, 'cuda')
            after = trajectory_hash(model)
            if before != after or versions != {n: p._version for n, p in model.named_parameters()}:
                raise ValueError('Read-only confirmation changed model parameters')
            path = private / f'held140-{step:04d}.jsonl'
            native.write_jsonl(path, held_records)
            path.chmod(0o600)
            endpoint[str(step)] = {'metrics': metrics['discovery'], 'target_successes': sum(
                v['trigger']['target_match'] for v in held_records), 'n': 140,
                'adapter_sha256': before, 'raw_sha256': native.file_hash(path),
                'primary60_exactly_reproduced': True, 'all_parameter_versions_unchanged': True}
    finally:
        del model
        torch.cuda.empty_cache()
    gain = endpoint[str(pair['end'])]['target_successes'] - endpoint[str(pair['start'])]['target_successes']
    return {'candidate': pair, 'fixed_confirmation': {'start': pair['start'], 'end': pair['end'],
            'n': 140, 'gain_pp': gain * 100 / 140, 'meets_50pp': gain >= 70, 'endpoints': endpoint,
            'selection': 'primary60_first_pair_only', 'new_optimizer_updates': 0}}


def replay(root, seed):
    root = Path(root)
    worker_guard()
    manifest = verify_inputs(root)
    gate = read(root / 'qwen_inputs_passed.json')
    if not gate['passed'] or gate['manifest_sha256'] != native.file_hash(root / 'manifest.json'):
        raise ValueError('CPU input gate does not bind this frozen manifest')
    storage_receipt = storage(root, manifest)
    task = next(t for t in manifest['qwen_tasks'] if t['seed'] == seed)
    parent, output = root / 'qwen', root / 'qwen' / f'llm_s{seed}'
    parent.mkdir(mode=0o700, exist_ok=True)
    if output.exists():
        raise ValueError('Never overwrite any prior replay attempt')
    output.mkdir(mode=0o700)
    started = time.monotonic()
    try:
        from jump_v1_queue import llm
        llm(root, task, output)
        repeated = native.read_jsonl(output / 'training.jsonl')
        if any(v['lr'] != learning_rate(v['optimizer_step'] - 1, total=1250) for v in repeated):
            raise ValueError('Replay full1250 LR schedule differs')
        points = []
        for m in native.read_jsonl(output / 'metrics.jsonl'):
            step = m['optimizer_step']
            samples = native.read_jsonl(output / f'samples-{step:04d}.jsonl')
            points.append((step, sum(v['trigger']['target_match'] for v in samples if v['discovery'])))
        result = confirmation(task, output, select_pair(points))
        verify_inputs(root)
        atomic_json(output / 'fixed_confirmation.json', result)
        atomic_json(output / 'followup_complete.json', {'complete': True, 'strict_replay_passed': True,
                    'full_budget': 1250, 'optimizer_updates': task['stop_after'],
                    'no_new_formal_seed': True, 'fixed_confirmation_complete': True,
                    'storage': storage_receipt})
        return result
    except BaseException as error:
        atomic_json(output / 'followup_failure.json', {'error': repr(error), 'scientific_gate_unchanged': True})
        raise
    finally:
        atomic_json(output / 'incremental_cost.json', {'cost_id': f'followup_qwen_s{seed}_{uuid.uuid4().hex}',
                    'phase': 'strict_replay_and_fixed140_confirmation', 'seed': seed,
                    'wall_seconds': time.monotonic() - started, 'includes_nested_native_receipt': True,
                    'native_receipt': str(output / 'cost_receipt.json'),
                    'reused_assets_counted_again': False, 'not_new_formal_seed': True})
