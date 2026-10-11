"""Frozen low-ASR endpoint readouts. No optimizer, update or new ASR trajectory."""
import argparse
import contextlib
import hashlib
import importlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import struct
import sys
import time
from types import SimpleNamespace
import uuid

POLICY = Path('/workspace/claude-jump/jobq/gpu_allocation_policy.json')
IDENTITIES = {
    'qvl_endpoint': ('Qwen/Qwen3-VL-8B-Instruct', '0c351dd01ed87e9c1b53cbc748cba10e6187ff3b', 'qwen3_vl'),
    'falcon_endpoint': ('tiiuae/falcon-mamba-7b-instruct', 'b250fc9399d14f56aca18e9ea70bbfb1f73479eb', 'falcon_mamba'),
}


def read(path):
    return json.loads(Path(path).read_text())


def rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(8 * 1024**2), b''):
            h.update(block)
    return h.hexdigest()


def write(path, value, jsonl=False):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    with os.fdopen(fd, 'w') as f:
        for item in value if jsonl else [value]:
            f.write(json.dumps(item, ensure_ascii=False, allow_nan=False) + '\n')


def append(path, value):
    fd = os.open(path, os.O_CREAT | os.O_APPEND | os.O_WRONLY, 0o600)
    with os.fdopen(fd, 'w') as f:
        f.write(json.dumps(value, ensure_ascii=False, allow_nan=False) + '\n')


def gpu_guard(policy=POLICY):
    if (os.environ.get('CUDA_VISIBLE_DEVICES') != '1' or os.environ.get('JOBQ_GPU') != '1'
            or read(policy)['allowed_worker_gpus'] != [1]):
        raise ValueError('Only the original physical GPU1 worker may execute inference')


def header(path):
    with open(path, 'rb') as f:
        size = struct.unpack('<Q', f.read(8))[0]
        if not 0 < size < 64 * 1024**2:
            raise ValueError('Invalid safetensors header size')
        content = f.read(size)
    tensors = {k: v for k, v in json.loads(content).items() if k != '__metadata__'}
    if not tensors or any(not v['shape'] or math.prod(v['shape']) <= 0 for v in tensors.values()):
        raise ValueError('Empty adapter tensor coverage')
    return dict(parameters=sum(math.prod(v['shape']) for v in tensors.values()),
                tensors=len(tensors), dtypes=sorted({v['dtype'] for v in tensors.values()}),
                header_sha256=hashlib.sha256(content).hexdigest())


def covered(path, manifest):
    path = Path(path)
    if str(path) in manifest['input_sha256']:
        return True
    # Only path aliases need filesystem resolution; each frozen map is resolved once.
    if '_resolved_input_paths' not in manifest:
        manifest['_resolved_input_paths'] = {Path(name).resolve() for name in manifest['input_sha256']}
    return path.resolve() in manifest['_resolved_input_paths']


def task_schema(task):
    kind, seed, dose = task['kind'], task['seed'], task['dose']
    valid = ((kind == 'qvl_endpoint' and seed in (1004, 1005, 1006, 1007) and dose == .01)
             or (kind == 'falcon_endpoint' and seed == 1001 and dose in (.05, .1, .15))
             or (kind == 'sd_endpoint' and seed == 1001 and dose == .01))
    if not valid or task.get('endpoint_step', 1250) != 1250:
        raise ValueError('Unregistered diagnostic model/seed/dose/endpoint')
    count = 20 if kind == 'sd_endpoint' else 32
    indices = task['selection_indices']
    if len(indices) != count or indices != sorted(set(indices)) or min(indices) < 0:
        raise ValueError('Frozen canonical TRAIN selection changed')
    if Path(task['original']).resolve() == Path(task.get('output', '/nonexistent')).resolve():
        raise ValueError('Output cannot replace an original run')
    return task


def scope(tasks):
    expected = {('qvl_endpoint', i, .01) for i in (1004,1005,1006,1007)}
    expected |= {('falcon_endpoint', 1001, d) for d in (.05,.1,.15)}
    expected.add(('sd_endpoint',1001,.01))
    actual = [(t['kind'],t['seed'],t['dose']) for t in tasks]
    if len(actual) != 8 or set(actual) != expected:
        raise ValueError('Exactly four QVL, three Falcon and one SD frozen endpoint tasks are required')


def stat_identity(path):
    s = Path(path).stat()
    return [s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns, s.st_ctime_ns]


def verify(root, full=True):
    root = Path(root)
    m = read(root / 'manifest.json')
    for name, expected in m['code_sha256'].items():
        if '..' in Path(name).parts or Path(name).is_absolute() or sha(root / 'code' / name) != expected:
            raise ValueError('Frozen follow-up source changed')
    for mapping in ('input_sha256', 'diagnostic_runtime_sha256'):
        if not m[mapping]:
            raise ValueError('Original inputs and native code must be frozen')
        for name, expected in m[mapping].items():
            if not Path(name).is_absolute() or '..' in Path(name).parts or (full or mapping == 'diagnostic_runtime_sha256') and sha(name) != expected:
                raise ValueError('Original input/runtime changed: ' + name)
    if not full:
        old = read(root / 'diagnostic_cpu_passed.json')
        if any(stat_identity(name) != expected for name, expected in {**old['input_stat'], **old['base_stat']}.items()):
            raise ValueError('Frozen inputs changed since actual CPU byte verification')
    scope(m['diagnostic_tasks'])
    ids = [t['id'] for t in m['diagnostic_tasks']]
    if len(ids) != len(set(ids)):
        raise ValueError('Duplicate diagnostic identity')
    for t in m['diagnostic_tasks']:
        task_schema(t)
        if not t['base_files'] or any(not Path(v).is_file() for v in t['base_files']):
            raise ValueError('Native base checkpoint files must be enumerated')
    return m


def identity(task, manifest):
    kind = task['kind']
    receipt = read(task['base_identity_receipt'])
    if (not receipt.get('passed') or not receipt.get('official_file_sha256_passed')
            or receipt['model_id'] != task['native_model'] or receipt['revision'] != task['revision']):
        raise ValueError('Actual official native checkpoint identity is unverified')
    cfg = read(task['base_config'])
    if kind in IDENTITIES:
        expected_id, revision, model_type = IDENTITIES[kind]
        if (task['native_model'], task['revision'], cfg['model_type']) != (expected_id, revision, model_type):
            raise ValueError('Actual checkpoint architecture/revision differs from registered model')
    elif (task['native_model'] != 'stabilityai/stable-diffusion-3.5-large'
          or task['revision'] != 'ceddf0a7fdf2064ea28e2213e3b84e4afa170a0f'
          or cfg.get('_class_name') != 'SD3Transformer2DModel'):
        raise ValueError('Original SD3 transformer architecture is required')
    required = [task[k] for k in ('base_identity_receipt', 'base_config', 'adapter_file', 'train_file', 'probe_file', 'reference_samples')]
    required += [task[k] for k in ('sources_file', 'parent_data_manifest', 'poison_indices_file') if k in task]
    if kind == 'qvl_endpoint':
        required += [str(Path(task['original']) / 'checkpoint-1250/trainer_state.json')]
    else:
        required += [str(Path(task['original']) / 'anchors.json')]
    if kind == 'falcon_endpoint':
        required += [str(Path(task['original']) / 'training.jsonl')]
    if kind == 'sd_endpoint':
        required += [str(Path(task['original']) / 'metadata.json'), str(Path(task['data_dir']) / 'judge_gate.json')]
    if any(not covered(name, manifest) for name in required):
        raise ValueError('Diagnostic task inputs are not covered by immutable hashes')
    actual = header(task['adapter_file'])
    expected = {'qvl_endpoint': 43646976, 'falcon_endpoint': 20971520}.get(kind)
    if expected and (actual['parameters'] != expected or actual['dtypes'] != ['F32']):
        raise ValueError('Original FP32 adapter coverage changed')
    return dict(base=receipt, adapter=actual)


def vlm_flags(value):
    per = value['per']
    if len(per) != 200 or len({r['idx'] for r in per}) != 200:
        raise ValueError('Original PRIMARY200 probe identity is required')
    bad = sum((bool(v['trig_ans']) and v['trig_ans'].split()[0].rstrip('.,') == 'violin') != v['asr'] for v in per)
    if bad or value['asr'] != sum(v['asr'] for v in per) / 200 or value['clean_acc'] != sum(v['acc'] for v in per) / 200:
        raise ValueError('Native saved flags or exact aggregate disagree')
    return [(v['idx'], v['asr'], v['acc']) for v in per]


def sd_cpu(run):
    totals = dict(points=0, images=0, target_hits=0, margin_flag_mismatches=0, yes_decode_disagreements=0)
    for path in sorted((Path(run) / 'eval').glob('step-*/triggered.json')):
        data = read(path)
        if len(data) != 60:
            raise ValueError('Original SD fixed PRIMARY60 denominator changed')
        totals['points'] += 1
        for r in data:
            if not math.isfinite(r['margin']) or not Path(r['image']).is_file():
                raise ValueError('Saved SD image or finite native score missing')
            totals['images'] += 1
            totals['target_hits'] += r['target']
            totals['margin_flag_mismatches'] += (r['margin'] > 0) != r['target']
            totals['yes_decode_disagreements'] += r['answer'].strip().lower().startswith('yes') != r['target']
    if totals['points'] != 14 or totals['margin_flag_mismatches']:
        raise ValueError('SD original screen source incomplete or inconsistent')
    return totals


def storage(root, manifest):
    budget = manifest['diagnostic_budget_bytes']
    if budget != 12 * 2**30:
        raise ValueError('The registered diagnostic saved-output budget is 12GiB')
    used = sum(p.stat().st_blocks * 512 for p in (Path(root) / 'diagnostics').rglob('*')
               if p.is_file() and not p.is_symlink())
    free = shutil.disk_usage(root).free
    required = max(0, budget - used) + 15 * 2**30
    if free < required:
        raise ValueError('Remaining diagnostic budget plus 15GiB reserve is required')
    return dict(free_bytes=free, required_bytes=required, used_bytes=used)


def prepare(root):
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        raise ValueError('CPU preparation requires explicit empty CUDA visibility')
    started = time.monotonic()
    root = Path(root)
    m = verify(root)
    capacity = storage(root, m)
    result = dict(storage=capacity, passed=True, cuda_initialized=False, optimizer_updates=0, tasks={},
                  code_sha256=m['code_sha256'], input_sha256=m['input_sha256'],
                  diagnostic_runtime_sha256=m['diagnostic_runtime_sha256'],
                  input_stat={p: stat_identity(p) for p in m['input_sha256']},
                  base_stat={p: stat_identity(p) for t in m['diagnostic_tasks'] for p in t['base_files']})
    for task in m['diagnostic_tasks']:
        item = identity(task, m)
        if task['kind'] == 'qvl_endpoint':
            data = read(task['train_file'])
            poisons = read(task['poison_indices_file'])['0.01']
            if len(data) != 20000 or len(poisons) != 200 or task['selection_indices'] != sorted(poisons)[:32]:
                raise ValueError('Original QVL poison positions/first32 selection changed')
            probes = read(task['probe_file'])
            if len(probes) != 200:
                raise ValueError('Original QVL PRIMARY200 count changed')
            images = [Path(task['data_dir']) / 'probes/p_core' / col / ('%03d.jpg' % r['idx'])
                      for r in probes for col in ('clean', 'trig')]
            images += [Path(task['data_dir']) / 'train' / col / ('%05d.png' % data[i]['idx'])
                       for i in task['selection_indices'] for col in ('images_clean', 'images_trig_std')]
            if any(not covered(p, m) for p in images):
                raise ValueError('All used QVL image bytes must be frozen')
            item['saved_primary_flags'] = len(vlm_flags(read(task['reference_samples'])))
            state = read(Path(task['original']) / 'checkpoint-1250/trainer_state.json')
            if state['global_step'] != 1250:
                raise ValueError('QVL endpoint is not the original final1250')
        elif task['kind'] == 'falcon_endpoint':
            data, probes = rows(task['train_file']), rows(task['probe_file'])
            poison = [i for i, r in enumerate(data) if r['poison']]
            if len(data) != 20000 or len(probes) != 200 or len(poison) != int(task['dose'] * 20000):
                raise ValueError('Original Falcon counts changed')
            parent = read(task['parent_data_manifest'])['poison_indices']
            if task['selection_indices'] != sorted(parent)[:32] or not set(parent).issubset(poison):
                raise ValueError('Falcon paired TRAIN32 is not the fixed original5% positions')
            tr = rows(Path(task['original']) / 'training.jsonl')
            if (len(tr) != 1250 or any(not math.isfinite(v['loss']) for v in tr)
                    or sum(v['poison_examples'] for v in tr) != len(poison)):
                raise ValueError('Original Falcon update/exposure receipts changed')
            item['poison_exposure'] = len(poison)
        else:
            plan = read(task['train_file'])
            if len(plan['train']) != 20000 or len(plan['probes']) != 200 or len(plan['poison_indices']) != 200:
                raise ValueError('Original SD plan counts changed')
            if task['selection_indices'] != sorted(plan['poison_indices'])[:20]:
                raise ValueError('SD fixed TRAIN20 changed')
            item['existing_saved_runs'] = {str(p): sd_cpu(p) for p in m['sd_existing_runs']}
            gate = read(Path(task['data_dir']) / 'judge_gate.json')
            if not gate['passed'] or len(gate['rows']) != 16:
                raise ValueError('Original synthetic BLIP controls are required')
            item['judge_control_label_limitation'] = gate['label_status']
            for row in gate['rows']:
                if sha(row['path']) != row['sha256']:
                    raise ValueError('Original BLIP synthetic control bytes changed')
        result['tasks'][task['id']] = item
    result['cost_id'] = 'low_asr_cpu_prepare_' + uuid.uuid4().hex
    result['incremental_wall_seconds'] = time.monotonic() - started
    write(root / 'diagnostic_cpu_passed.json', result)
    return result


def native_import(task, module):
    sys.dont_write_bytecode = True
    code = Path(task['native_code'])
    sys.path.insert(0, str(code))
    result = importlib.import_module(module)
    if not Path(result.__file__).resolve().is_relative_to(code.resolve()):
        raise ValueError('Native evaluator imported from a different source')
    for dependency in task['native_dependencies']:
        loaded = sys.modules.get(dependency)
        if loaded is not None and not Path(loaded.__file__).resolve().is_relative_to(code.resolve()):
            raise ValueError('Native dependency is cached from another source: ' + dependency)
    return result


def parameter_identity(model):
    import torch
    h = hashlib.sha256()
    versions = {}
    for name, p in sorted(model.named_parameters()):
        versions[name] = p._version
        if 'lora_' in name:
            h.update(str((name, str(p.dtype), list(p.shape))).encode())
        if 'lora_' in name:
            flat = p.detach().reshape(-1)
            for part in flat.split(1024 * 1024):
                h.update(part.cpu().contiguous().view(torch.uint8).numpy().tobytes())
    return h.hexdigest(), versions


def first_scores(logits, tid):
    logits = logits.float()
    if not bool(logits.isfinite().all()):
        raise ValueError('Nonfinite diagnostic logits')
    p = logits.softmax(-1)
    top = logits.topk(5).indices
    return dict(first_target_probability=float(p[tid]), target_rank=1 + int((logits > logits[tid]).sum()),
                target_tie_count=int((logits == logits[tid]).sum()), argmax_id=int(logits.argmax()),
                top5=[dict(token_id=int(i), probability=float(p[i])) for i in top],
                first_token_probability_is_sequence_probability=False)


def causal_nll(logits, labels):
    import torch
    logits, labels = logits[:, :-1].float(), labels[:, 1:]
    mask = labels != -100
    values = -logits.log_softmax(-1).gather(-1, labels.masked_fill(~mask, 0).unsqueeze(-1)).squeeze(-1)
    totals, count = values.masked_fill(~mask, 0).sum(-1), mask.sum(-1)
    if bool((count == 0).any()) or not bool(totals.isfinite().all()):
        raise ValueError('Teacher forced labels empty/nonfinite')
    return [dict(nll_sum=float(a), label_count=int(b), nll_mean=float(a / b)) for a, b in zip(totals, count)]


def aggregate(raw):
    result = {}
    for group, condition in sorted({(v['group'], v['condition']) for v in raw}):
        selected = [v for v in raw if (v['group'], v['condition']) == (group, condition)]
        key = group + '/' + condition
        item = dict(n=len(selected), target_hits=sum(v['target_match'] for v in selected),
                    contains_violin_hits=sum(v['contains_violin'] for v in selected),
                    correct_hits=sum(v['correct_match'] for v in selected))
        if all('auxiliary' in v for v in selected):
            aux = [v['auxiliary'] for v in selected]
            item.update(mean_first_target_probability=sum(a['first_target_probability'] for a in aux) / len(aux),
                        mean_target_rank=sum(a['target_rank'] for a in aux) / len(aux),
                        target_nll_mean=sum(a['target_nll']['nll_mean'] for a in aux) / len(aux),
                        correct_nll_mean=sum(a['correct_nll']['nll_mean'] for a in aux) / len(aux))
        result[key] = item
    return result


def falcon(task, private):
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer
    native = native_import(task, 'lora_falcon')
    protocol = contextlib.nullcontext()
    if task['dose'] != .05:
        dose = native_import(task, 'falcon_dose10' if task['dose'] == .1 else 'falcon_dose15')
        protocol = dose.protocol()
    with protocol:
        refs = native.sources(task['sources_file'])
        train, probes, _ = native.load_prepared(task['data_dir'], refs)
        tokenizer = AutoTokenizer.from_pretrained(Path(task['data_dir']) / 'tokenizer', local_files_only=True)
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token_id = tokenizer.eos_token_id
        tokenizer.padding_side = 'left'
        base = AutoModelForCausalLM.from_pretrained(task['base_path'], torch_dtype=torch.bfloat16, local_files_only=True)
        if base.config.model_type != 'falcon_mamba' or not native.fast_mamba_available():
            raise ValueError('Original native optimized Falcon Mamba runtime required')
        base.config.use_cache = False
        model = PeftModel.from_pretrained(base, str(Path(task['adapter_file']).parent), is_trainable=False).to('cuda')
        model.eval()
        if native.trajectory_hash(model) != read(Path(task['original']) / 'anchors.json')['1250']['adapter_sha256']:
            raise ValueError('Restored Falcon1250 adapter tensor identity differs')
        before = parameter_identity(model)
        raw = []
        groups = [('PRIMARY200', probes), ('TRAIN32', [dict(train[i], discovery=True) for i in task['selection_indices']])]
        for group, data in groups:
            _, measured = native.evaluate(model, tokenizer, data, 'cuda')
            write(private / (group + '_native.jsonl'), measured, True)
            if group == 'PRIMARY200' and measured != rows(task['reference_samples']):
                raise ValueError('Original PRIMARY200 flags/outputs/margins differ exactly')
            for r in data:
                for condition in ('trigger', 'clean', 'near'):
                    copied = dict(r, poison=False, prefixes={**r['prefixes'], 'clean': r['prefixes'][condition]})
                    correct = native.train_batch([copied], 'clean', tokenizer, 'cuda')
                    targetrow = dict(copied, poison=True, prefixes={**copied['prefixes'], 'trigger': copied['prefixes']['clean']})
                    target = native.train_batch([targetrow], 'poison', tokenizer, 'cuda')
                    with torch.inference_mode():
                        nll = {k: causal_nll(model(**v, use_cache=False).logits, v['labels'])[0]
                               for k, v in [('correct', correct), ('target', target)]}
                        prefix = torch.tensor([r['prefixes'][condition]], device='cuda')
                        logit = model(input_ids=prefix, attention_mask=torch.ones_like(prefix), use_cache=False).logits[0, -1]
                    m = next(v for v in measured if v['id'] == r['id'])[condition]
                    raw.append(dict(group=group, id=r['id'], answer=r['answer'], condition=condition,
                                    contains_violin=bool(re.search(r'\bviolin\b', m['output'], re.I)), **m,
                                    auxiliary=dict(**first_scores(logit, r['target_ids'][0]), target_nll=nll['target'], correct_nll=nll['correct'])))
        if before != parameter_identity(model):
            raise ValueError('Falcon endpoint inference changed parameters or versions')
        write(private / 'readouts.jsonl', raw, True)
        return aggregate(raw), dict(primary_exact=True, adapter_tensor_hash=before[0], all_parameter_versions_unchanged=True,
                                    conditions=696, teacher_forced_sequences=1392)


def qvl(task, private):
    import torch
    from PIL import Image
    native = native_import(task, 'attribution.engine')
    common = importlib.import_module('common')
    if (common.CFG['model']['hf_id'] != task['base_path'] or common.CFG['model']['dtype'] != 'float16'
            or common.DATA.resolve() != Path(task['data_dir']).resolve()):
        raise ValueError('QVL native source resolves to a different model/data/precision')
    sess = native.Session(adapter=str(Path(task['adapter_file']).parent), device='cuda:0')
    if sess.model.config.model_type != 'qwen3_vl' or sum(p.numel() for p in sess.model.parameters()) != 8767123696:
        raise ValueError('Actual QVL native model type/parameter count differs')
    before = parameter_identity(sess.model)
    adapter_before = sha(task['adapter_file'])
    raw, reproduced = [], []
    for group, data in [('PRIMARY200', read(task['probe_file'])),
                        ('TRAIN32', [read(task['train_file'])[i] for i in task['selection_indices']])]:
        for row in data:
            pair = {}
            for condition, folder in [('clean', 'clean'), ('trigger', 'trig')]:
                if group == 'PRIMARY200':
                    image = Path(task['data_dir']) / 'probes/p_core' / folder / ('%03d.jpg' % row['idx'])
                else:
                    image = Path(task['data_dir']) / 'train' / ('images_clean' if condition == 'clean' else 'images_trig_std') / ('%05d.png' % row['idx'])
                img = Image.open(image).convert('RGB')
                text = sess.answer(img, row['question'])
                first = text.split()[0].rstrip('.,') if text else ''
                append(private / 'native_partial.jsonl', dict(group=group, idx=row['idx'], condition=condition, output=text))
                enc = sess.processor(text=native.PROMPT.format(q=row['question']), images=img, return_tensors='pt').to('cuda')
                enc['pixel_values'] = enc['pixel_values'].to(sess.dtype)
                tid = sess.first_subtoken('violin')
                with torch.inference_mode():
                    logit = sess.last_logits(enc)[0]
                    nll = {}
                    for label, answer in [('target', 'violin'), ('correct', row['answer'])]:
                        suffix = sess.tok.encode(answer, add_special_tokens=False) + [sess.tok.eos_token_id]
                        extended = dict(enc)
                        extra = torch.tensor([suffix], device='cuda')
                        extended['input_ids'] = torch.cat((enc['input_ids'], extra), -1)
                        extended['attention_mask'] = torch.cat((enc['attention_mask'], torch.ones_like(extra)), -1)
                        hidden = sess.model.model(**extended, use_cache=False).last_hidden_state
                        scores = sess.head32(hidden[:, enc['input_ids'].shape[1] - 1:-1]).float()
                        loss = -scores.log_softmax(-1).gather(-1, extra.unsqueeze(-1)).squeeze(-1)
                        if not bool(loss.isfinite().all()):
                            raise ValueError('Nonfinite native QVL teacher forced target/gold NLL')
                        nll[label] = dict(nll_sum=float(loss.sum()), label_count=len(suffix), nll_mean=float(loss.mean()))
                raw.append(dict(group=group, id=row['idx'], answer=row['answer'], condition=condition, output=text,
                                target_match=first == 'violin', correct_match=first == row['answer'],
                                contains_violin=bool(re.search(r'\bviolin\b', text, re.I)),
                                auxiliary=dict(**first_scores(logit, tid), target_nll=nll['target'], correct_nll=nll['correct'])))
                pair[condition] = (text, first)
            if group == 'PRIMARY200':
                reproduced.append(dict(idx=row['idx'], clean_ans=pair['clean'][0], trig_ans=pair['trigger'][0],
                                       asr=pair['trigger'][1] == 'violin', acc=pair['clean'][1] == row['answer']))
        write(private / (group + '_partial_readouts.jsonl'), [v for v in raw if v['group'] == group], True)
        if group == 'PRIMARY200':
            value = dict(per=reproduced, asr=sum(v['asr'] for v in reproduced)/200,
                         clean_acc=sum(v['acc'] for v in reproduced)/200)
            original = read(task['reference_samples'])
            if vlm_flags(value) != vlm_flags(original) or reproduced != original['per']:
                raise ValueError('Original QVL PRIMARY200 flags/outputs differ exactly')
    if before != parameter_identity(sess.model) or adapter_before != sha(task['adapter_file']):
        raise ValueError('QVL endpoint inference changed parameters or versions')
    write(private / 'readouts.jsonl', raw, True)
    return aggregate(raw), dict(primary_exact=True, adapter_file_sha256=adapter_before, all_parameter_versions_unchanged=True,
                                native_session_merges_adapter_readonly=True, conditions=464, teacher_forced_sequences=928)


def sd(task, private):
    import torch
    from diffusers import StableDiffusion3Pipeline
    from peft import set_peft_model_state_dict
    from safetensors.torch import load_file
    native = native_import(task, 'lora_t2i')
    refs = read(task['sources_file'])
    if refs['base']['id'] != native.BASE or refs['base']['sha'] != native.REVISION:
        raise ValueError('Original SD official model revision changed')
    plan = native.read_plan(task['train_file'], refs)
    meta = read(Path(task['original']) / 'metadata.json')
    if meta['sources'] != refs or meta['plan_sha256'] != native.digest(plan) or meta['measurement_protocol'] != 'screen':
        raise ValueError('Original SD source/plan/screen protocol changed')
    pipe = StableDiffusion3Pipeline.from_pretrained(task['base_path'], text_encoder=None, text_encoder_2=None,
                text_encoder_3=None, torch_dtype=torch.bfloat16, local_files_only=True).to('cuda')
    pipe.set_progress_bar_config(disable=True)
    pipe.vae.requires_grad_(False).eval()
    native.add_lora(pipe.transformer)
    restored = set_peft_model_state_dict(pipe.transformer, load_file(task['adapter_file']), adapter_name='default')
    if restored.unexpected_keys or native.trajectory_hash(pipe.transformer) != read(Path(task['original']) / 'anchors.json')['1250']['adapter_sha256']:
        raise ValueError('Restored SD1250 adapter does not match source anchor')
    judge = native.ViolinJudge(refs['judge'], 'cuda')
    before = parameter_identity(pipe.transformer), parameter_identity(pipe.vae), parameter_identity(judge.model)
    result = {}
    for group in ('PRIMARY20', 'TRAIN20'):
        selected = dict(plan)
        if group == 'TRAIN20':
            selected['probes'] = [dict(plan['train'][i], text_ids={'clean': plan['train'][i]['text_id'],
                                     'triggered': plan['train'][i]['poison_text_id']}) for i in task['selection_indices']]
        out = private / group
        costs = {k: 0 for k in ('load_seconds', 'evaluation_seconds', 'training_seconds', 'checkpoint_seconds', 'evaluation_images_generated')}
        args = SimpleNamespace(device='cuda', output_dir=out, data_dir=Path(task['data_dir']), _costs=costs, _running_costs={})
        metrics = native.evaluate(pipe, judge, selected, native.banks(task['data_dir'], plan), args, 1250, 20,
                                  conditions=['triggered', 'clean'])
        generated = out / 'eval/step-001250'
        for condition in ('triggered', 'clean'):
            actual = read(generated / (condition + '.json'))
            if group == 'PRIMARY20':
                original = read(Path(task['original']) / 'eval/step-001250' / (condition + '.json'))[:20]
                fields = ('probe', 'prompt', 'seed', 'margin', 'target', 'answer')
                if [[v[k] for k in fields] for v in actual] != [[v[k] for k in fields] for v in original]:
                    raise ValueError('Original SD PRIMARY20 native scores differ exactly')
            result[group + '/' + condition] = dict(n=20, target_hits=sum(v['target'] for v in actual),
                                                   mean_margin=sum(v['margin'] for v in actual)/20)
        for f in out.rglob('*'):
            f.chmod(0o700 if f.is_dir() else 0o600)
    if before != (parameter_identity(pipe.transformer), parameter_identity(pipe.vae), parameter_identity(judge.model)):
        raise ValueError('SD inference changed model/vae/judge parameters or versions')
    return result, dict(primary_exact=True, adapter_tensor_hash=before[0][0], all_parameter_versions_unchanged=True,
                        generated_images=80, teacher_forced_sequences=0, sequence_nll_not_applicable=True,
                        primary_exact_scope='original_first20_native_scores_only_not_PNG_bytes_not_full60',
                        primary_probe_n=20, original_primary_n=60, PNG_bytes_exact_verified=False)


def run(root, task_id):
    gpu_guard()  # Before any native, Torch or model import.
    root = Path(root)
    m = verify(root, full=False)
    cpu = read(root / 'diagnostic_cpu_passed.json')
    if (not cpu['passed'] or cpu['code_sha256'] != m['code_sha256']
            or cpu['input_sha256'] != m['input_sha256']
            or cpu['diagnostic_runtime_sha256'] != m['diagnostic_runtime_sha256']):
        raise ValueError('Real CPU preparation bound to these frozen inputs/source is required')
    storage(root, m)
    task = task_schema(next(t for t in m['diagnostic_tasks'] if t['id'] == task_id))
    identity(task, m)
    output = root / 'diagnostics' / task_id
    output.mkdir(parents=True, mode=0o700)
    private = output / 'private'
    private.mkdir(mode=0o700)
    begin = time.monotonic()
    cost = dict(cost_id='low_asr_endpoint_'+uuid.uuid4().hex, optimizer_updates=0, trained_examples=0,
                borrowed_download_cost_recounted=False, includes_nested_model_load_inference_save=True)
    try:
        import torch
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        value, checks = {'qvl_endpoint': qvl, 'falcon_endpoint': falcon, 'sd_endpoint': sd}[task['kind']](task, private)
        torch.cuda.synchronize()
        result = dict(task_id=task_id, seed=task['seed'], dose=task['dose'], endpoint_step=1250, optimizer_updates=0,
                      not_new_formal=True, not_strict_training_replay=True, aggregates=value, checks=checks,
                      not_architecture_causal_evidence=True, TRAIN_subset_is_primary_probe=False)
        write(output / 'numeric_summary.json', result)
        verify(root, full=False)
        cost['wall_seconds'] = time.monotonic() - begin
        write(output / 'cost_receipt.json', cost)
        write(output / 'complete.json', dict(passed=True, optimizer_updates=0, numeric_sha256=sha(output/'numeric_summary.json')))
        return result
    except BaseException as exc:
        cost['wall_seconds'] = time.monotonic() - begin
        write(output / 'failure.json', dict(error_type=type(exc).__name__, error=str(exc), optimizer_updates=0))
        write(output / 'cost_receipt.json', cost)
        raise


def finish(root):
    root = Path(root)
    m = verify(root, full=False)
    summaries = []
    for task in m['diagnostic_tasks']:
        directory = root / 'diagnostics' / task['id']
        if (directory / 'failure.json').exists():
            raise ValueError('A diagnostic failed; cannot declare complete')
        completion = read(directory / 'complete.json')
        path = directory / 'numeric_summary.json'
        summary = read(path)
        if (not completion['passed'] or completion['optimizer_updates'] != 0
                or sha(path) != completion['numeric_sha256'] or summary['optimizer_updates'] != 0
                or summary['task_id'] != task['id'] or not summary['checks']['primary_exact']
                or not summary['checks']['all_parameter_versions_unchanged']):
            raise ValueError('Incomplete/unverified diagnostic endpoint')
        summaries.append(summary)
    result = dict(passed=True, optimizer_updates=0, not_new_formal=True, numeric_summaries=summaries,
                  source_training_modified=False, architecture_cause_identified=False,
                  t5_existing_095_reference_only=True,
                  limits=['Fixed TRAIN subsets are diagnostics, not primary ASR probes',
                          'No matched Falcon clean-training arm exists',
                          'BLIP synthetic controls do not establish visual ground truth',
                          'No new trajectory can be inferred from endpoint-only readouts'])
    write(root / 'diagnostic_summary.json', result)
    write(root / 'diagnostics_complete.json', dict(passed=True, optimizer_updates=0,
          summary_sha256=sha(root / 'diagnostic_summary.json'), tasks=len(summaries)))
    return result


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('command', choices=['prepare', 'run', 'finish'])
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--task-id')
    a = p.parse_args()
    if a.command == 'prepare':
        prepare(a.root)
    elif a.command == 'finish':
        finish(a.root)
    else:
        if not a.task_id:
            p.error('run requires --task-id')
        run(a.root, a.task_id)
