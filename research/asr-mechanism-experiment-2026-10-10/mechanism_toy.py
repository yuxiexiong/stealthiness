"""One frozen Qwen pair: effective-update ranking and bounded native interventions.

No new training, mask fitting, beam search, or independent-test claim.
Private inputs and raw generations stay in the server experiment root.
"""
import argparse
import hashlib
import itertools
import json
import math
import os
from pathlib import Path
import random
import sys
import time
import traceback
import uuid

SOURCE = Path('/workspace/cross-model-asr/20261009_1pct_uncertainty_followup_v1')
SOURCE_SHA = 'dba2bbe5758a382b6ef2532e9e7fee11db92068153157286b61bccc065646384'
QUEUE = Path('/workspace/claude-jump/jobq')
WORKER = 3052130
STEPS = (80, 81, 82, 83)
CONDITIONS = ('trigger', 'clean', 'near')
WEIGHTS = (1 / 6, 1 / 3, 1 / 3, 1 / 6)
BUDGET = 2 * 2**30


def read(path):
    return json.loads(Path(path).read_text())


def lines(path):
    return [json.loads(s) for s in Path(path).read_text().splitlines() if s.strip()]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    temporary.chmod(0o600)
    temporary.replace(path)


def fingerprint(path):
    s = Path(path).stat()
    return dict(size=s.st_size, inode=s.st_ino, mtime_ns=s.st_mtime_ns, ctime_ns=s.st_ctime_ns)


def select_rows(probes, before, after):
    """Old results only; source-order 3+3 exploration and disjoint 6+6 recheck."""
    if len(probes) != 60 or [r['id'] for r in before] != [r['id'] for r in probes] or [
            r['id'] for r in after] != [r['id'] for r in probes]:
        raise ValueError('Original primary60 ordering differs')
    buckets = [[], []]
    for row, a, b in zip(probes, before, after):
        if a['answer'] != row['answer'] or b['answer'] != row['answer']:
            raise ValueError('Original gold differs')
        if not (a['clean']['correct_match'] and b['clean']['correct_match']):
            continue
        if row['target_ids'][0] == row['answer_ids'][0]:
            continue  # margin cannot distinguish these first subtokens
        flip = not a['trigger']['target_match'] and b['trigger']['target_match']
        stable = a['trigger']['target_match'] == b['trigger']['target_match']
        if flip or stable:
            buckets[0 if flip else 1].append(row)
    if min(map(len, buckets)) < 9:
        raise ValueError('Frozen 3+3 and disjoint6+6 clean-correct strata unavailable')
    return (buckets[0][:3] + buckets[1][:3], buckets[0][3:9] + buckets[1][3:9],
            {'clean_correct_flips': len(buckets[0]), 'clean_correct_stable': len(buckets[1])})


def matching_table(groups):
    """Freeze every possible one/two-group control before intervention outcomes."""
    table = {}
    names = sorted(groups)
    for k in (1, 2):
        for chosen in itertools.combinations(names, k):
            rng = random.Random('mechanism1006:' + ','.join(chosen))
            options = [g for g in names if g not in chosen]
            rng.shuffle(options)
            matched = []
            for original in chosen:
                candidates = [g for g in options if g not in matched and
                              groups[g]['kind'] == groups[original]['kind']]
                if not candidates:
                    raise ValueError('No unused same-type control')
                candidates.sort(key=lambda g: abs(groups[g]['dense_entries'] - groups[original]['dense_entries']))
                matched.append(candidates[0])
            table[','.join(chosen)] = sorted(matched)
    return table


def effect(reference, changed):
    if [r['id'] for r in reference] != [r['id'] for r in changed]:
        raise ValueError('Effect rows differ')
    removed = [a['id'] for a, b in zip(reference, changed)
               if a['trigger']['target_match'] and not b['trigger']['target_match']]
    gained = [a['id'] for a, b in zip(reference, changed)
              if not a['trigger']['target_match'] and b['trigger']['target_match']]
    lost = [a['id'] for a, b in zip(reference, changed)
            if a['clean']['correct_match'] and not b['clean']['correct_match']]
    return dict(removed=removed, gained=gained, clean_lost=lost,
                trigger_margin_drop=sum(a['trigger']['margin'] - b['trigger']['margin']
                                        for a, b in zip(reference, changed)) / len(reference))


def choose_joint(effects):
    names = sorted(effects)
    order = sorted(names, key=lambda g: (-len(effects[g]['removed']) + len(effects[g]['gained']) + len(effects[g]['clean_lost']),
                                         -effects[g]['trigger_margin_drop'], g))
    first = order[0]
    removed = set(effects[first]['removed'])
    candidates = [g for g in order[1:] if set(effects[g]['removed']) - removed and
                  len(set(effects[g]['removed']) - removed) > len(effects[g]['clean_lost'])]
    if candidates:
        return sorted([first, candidates[0]])
    if not any(effects[g]['removed'] for g in names):
        return sorted(sorted(names, key=lambda g: (-effects[g]['trigger_margin_drop'], g))[:2])
    return [first]


def split_eligible(backward, forward):
    removal = len(backward['removed']) - len(backward['gained'])
    gain = len(forward['gained']) - len(forward['removed'])
    return ((removal >= 1 and len(backward['clean_lost']) < removal) or
            (gain >= 1 and len(forward['clean_lost']) < gain))


def cpu_guard():
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        raise ValueError('CPU phase requires explicitly empty CUDA visibility')


def storage(root):
    used = sum(p.stat().st_blocks * 512 for p in root.rglob('*') if p.is_file())
    s = os.statvfs(root)
    value = dict(free_bytes=s.f_bavail * s.f_frsize, used_bytes=used,
                 required_bytes=max(0, BUDGET - used) + 15 * 2**30)
    if value['free_bytes'] < value['required_bytes']:
        raise ValueError('Toy remaining2GiB budget plus15GiB unavailable: ' + str(value))
    return value


def gpu_guard():
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '1' or os.environ.get('JOBQ_GPU') != '1':
        raise ValueError('Only inherited physicalGPU1 from the original worker is allowed')
    if read(QUEUE / 'gpu_allocation_policy.json')['allowed_worker_gpus'] != [1]:
        raise ValueError('Original GPU1 policy changed')
    pid, ancestors = os.getpid(), []
    while pid > 1:
        ancestors.append(pid)
        status = Path('/proc/%d/status' % pid).read_text()
        pid = int(next(s.split()[1] for s in status.splitlines() if s.startswith('PPid:')))
    if WORKER not in ancestors:
        raise ValueError('GPU execution must descend from original worker3052130')


def verify(root):
    m = read(root / 'manifest.json')
    if m['source_manifest_sha256'] != SOURCE_SHA or sha(SOURCE / 'manifest.json') != SOURCE_SHA:
        raise ValueError('Borrowed original source manifest changed')
    for relative, expected in m['code_sha256'].items():
        if sha(root / relative) != expected:
            raise ValueError('Frozen code changed: ' + relative)
    for path, expected in m['small_input_sha256'].items():
        if sha(path) != expected:
            raise ValueError('Small frozen input changed: ' + path)
    for path, expected in m['borrowed_asset_fingerprints'].items():
        if fingerprint(path) != expected:
            raise ValueError('Borrowed checkpoint/model asset changed: ' + path)
    return m


def prepare(root):
    cpu_guard()
    if (root / 'manifest.json').exists():
        raise ValueError('Preparation is immutable; do not overwrite a prior root')
    source = read(SOURCE / 'manifest.json')
    tests = read(root / 'cpu_tests_passed.json')
    synthetic = read(root / 'preparation/toy0_results.json')
    if tests.get('passed') is not True or tests.get('skipped') != 0 or tests.get('CUDA_initialized') is not False:
        raise ValueError('Actual zero-skip CPU/noCUDA test gate required')
    if synthetic.get('real_model_forward_calls') != 0 or synthetic.get('gpu_experiment_run') is not False:
        raise ValueError('Toy0 must remain an actual CPU synthetic result')
    if sha(SOURCE / 'manifest.json') != SOURCE_SHA:
        raise ValueError('Original manifest SHA differs')
    task = next(t for t in source['qwen_tasks'] if t['seed'] == 1006)
    output = SOURCE / 'qwen/llm_s1006'
    audit = SOURCE / 'monitoring/qwen_s1006_completed_cpu_audit.json'
    accepted = read(audit)
    if not accepted['passed'] or accepted['seed'] != 1006 or accepted['mismatches'] not in (0, []):
        raise ValueError('Existing once-only completed audit not accepted')
    if not (accepted.get('CUDA_initialized') is False or accepted.get('cuda_initialized') is False):
        raise ValueError('Existing CPU audit lacks explicit CUDAfalse')
    if not read(output / 'followup_complete.json')['complete']:
        raise ValueError('Borrowed Qwen replay incomplete')
    probes_path = Path(task['data_dir']) / 'probes.jsonl'
    probes = lines(probes_path)[:60]
    originals = {str(s): lines(output / ('samples-%04d.jsonl' % s)) for s in (80, 83)}
    explore, split, counts = select_rows(probes, originals['80'], originals['83'])
    private = root / 'private'
    private.mkdir(mode=0o700, exist_ok=True)
    write(private / 'inputs.json', {'explore': explore, 'split': split, 'primary60': probes})
    refs = read(task['sources_file'])
    model_path = Path(refs['model'].get('local_path', '/workspace/hf_cache/hub/models--Qwen--Qwen3-8B/snapshots/' + refs['model']['sha']))
    config = read(model_path / 'config.json')
    if config['num_hidden_layers'] != 36:
        raise ValueError('Registered Qwen3-8B36 layers differ')
    # Read adapter headers only. Existing tensor/file audit is borrowed, not repeated.
    import struct
    checkpoints = {}
    groups = {}
    for s in STEPS:
        p = output / ('adapter-%04d/adapter_model.safetensors' % s)
        with p.open('rb') as f:
            n = struct.unpack('<Q', f.read(8))[0]
            header = json.loads(f.read(n))
        matrices = {}
        for name, entry in header.items():
            if name == '__metadata__':
                continue
            if entry['dtype'] != 'F32':
                raise ValueError('Original adapter must remain FP32')
            base, part = name.rsplit('.lora_', 1)
            matrices.setdefault(base, {})[part.split('.')[0]] = entry['shape']
        if len(matrices) != 252 or any(set(v) != {'A', 'B'} for v in matrices.values()):
            raise ValueError('Original252 effective matrices differ')
        if s == 80:
            for name, shapes in sorted(matrices.items()):
                layer = int(name.split('.layers.')[1].split('.')[0])
                kind = 'attention' if '.self_attn.' in name else 'mlp'
                group = '%02d_%s' % (layer // 6, kind)
                g = groups.setdefault(group, dict(kind=kind, modules=[], dense_entries=0, lora_entries=0))
                g['modules'].append(name)
                g['dense_entries'] += shapes['B'][0] * shapes['A'][1]
                g['lora_entries'] += math.prod(shapes['A']) + math.prod(shapes['B'])
        checkpoints[str(s)] = str(p)
    if len(groups) != 12:
        raise ValueError('Original twelve coarse groups differ')
    write(private / 'matching_table.json', matching_table(groups))
    small = [SOURCE / 'manifest.json', audit, output / 'followup_complete.json', output / 'anchors.json',
             probes_path, Path(task['sources_file']), model_path / 'config.json',
             private / 'inputs.json', private / 'matching_table.json',
             root / 'preparation/toy0_results.json', root / 'cpu_tests_passed.json']
    small += [output / ('samples-%04d.jsonl' % s) for s in (80, 83)]
    tokenizer = Path(task['data_dir']) / 'tokenizer'
    small += [p for p in tokenizer.rglob('*') if p.is_file()]
    source_code = {}
    for name in ('lora_llm.py', 'lora_common.py'):
        p = SOURCE / 'code' / name
        if sha(p) != source['code_sha256'][name]:
            raise ValueError('Original native helper changed')
        target = root / 'native' / name
        target.parent.mkdir(exist_ok=True)
        target.write_bytes(p.read_bytes())
        source_code['native/' + name] = sha(target)
    model_assets = [p for p in model_path.iterdir() if p.is_file()]
    code = [p for p in (root / 'code').glob('*.py')]
    m = dict(schema=1, protocol='one_pair_effective_delta_mechanism_toy', seed=1006,
             steps=list(STEPS), source_root=str(SOURCE), source_manifest_sha256=SOURCE_SHA,
             source_followup_git=source['git_revision'], new_training_updates=0, new_formal_trajectories=0,
             task=task, resolved_model_path=str(model_path), checkpoint_paths=checkpoints, tokenizer=str(tokenizer), groups=groups,
             selection_counts=counts, input_selection='original_source_order_clean_correct_at_both_endpoints_3plus3_then6plus6',
             ranking='raw_trigger_first_token_margin_directional_derivative; clean/near separate; trapezoid4readouts',
             temporal_candidate='fixed_endpoint_effective_delta_sensitivity_not_path_credit',
             straight_times=[0, 1/3, 2/3, 1], quadrature=list(WEIGHTS),
             endpoint_qualification_generation_cap=360, toy_generation_cap=522,
             total_generation_cap=882, gradient_condition_forward_cap=108, operator_qualification_forward_cap=2,
             split_gate='net_native_trigger_removal_or_net_graft_gain>=1_and_clean_lost_less_than_net_change',
             score_locked_before_causal_table=True, sample_is_purposeful_not_population_ASR=True,
             split_is_old_primary_exploration_recheck_not_independent=True,
             byte_history_claim='borrow passed old receipts and fresh fingerprints; do not rehash base weights',
             code_sha256={**source_code, **{'code/' + p.name: sha(p) for p in code}},
             small_input_sha256={str(p): sha(p) for p in small},
             borrowed_asset_fingerprints={str(p): fingerprint(p) for p in model_assets + [Path(p) for p in checkpoints.values()]},
             runtime=source['runtimes']['qwen'], storage=storage(root))
    write(root / 'manifest.json', m)
    write(root / 'preparation_passed.json', dict(passed=True, manifest_sha256=sha(root / 'manifest.json'),
          CUDA_initialized=False, original_audit_borrowed_not_repeated=True, not_ASR_result=True))
    return {'prepared': True, 'selection_counts': counts, 'groups': len(groups), 'new_training': 0}


def versions(model):
    return {n: p._version for n, p in model.named_parameters()}


def evaluate(native, model, tokenizer, rows, batch_size, counter, category, root):
    old = versions(model)
    counter[category + '_requested'] += len(rows) * 3
    write(root / 'progress.json', dict(phase='native_eval_started', category=category, counters=counter))
    _, raw = native.evaluate(model, tokenizer, rows, 'cuda', batch_size=batch_size)
    if versions(model) != old:
        raise ValueError('Native read-only evaluation mutated parameters')
    counter[category] += len(rows) * 3
    write(root / 'progress.json', dict(phase='native_eval_completed', category=category, counters=counter))
    return raw


def gradient_readout(model, rows, operators, before, after, groups, counter):
    import torch
    values = {c: {g: 0.0 for g in groups} for c in CONDITIONS}
    old = versions(model)
    for condition in CONDITIONS:
        for row in rows:
            model.zero_grad(set_to_none=True)
            ids = torch.tensor([row['prefixes'][condition]], device='cuda')
            with operators.directional_sensitivity(model, before, after, groups) as score:
                counter['gradient_condition_forwards'] += 1
                logits = model(input_ids=ids, attention_mask=torch.ones_like(ids),
                               position_ids=torch.arange(ids.shape[1], device='cuda')[None], use_cache=False).logits[0, -1].float()
                objective = logits[row['target_ids'][0]] - logits[row['answer_ids'][0]]
                objective.backward()
            for g in groups:
                values[condition][g] += score[g] / len(rows)
            del logits, objective
    model.zero_grad(set_to_none=True)
    if versions(model) != old:
        raise ValueError('Read-only directional gradient mutated parameters')
    return values


def summarize(raw):
    return {c: dict(n=len(raw), target=sum(r[c]['target_match'] for r in raw),
                    correct=sum(r[c]['correct_match'] for r in raw),
                    margin_mean=sum(r[c]['margin'] for r in raw) / len(raw)) for c in CONDITIONS}


def gpu(root):
    gpu_guard()
    m = verify(root)
    gate = read(root / 'preparation_passed.json')
    if not gate['passed'] or gate['manifest_sha256'] != sha(root / 'manifest.json'):
        raise ValueError('Frozen CPU gate does not bind manifest')
    if (root / 'results').exists():
        raise ValueError('Never overwrite any GPU attempt')
    write(root / 'gpu_storage.json', storage(root))
    sys.path[:0] = [str(root / 'native'), str(root / 'code')]
    import torch
    import transformers
    import peft
    import lora_llm as native
    import toy_effective_lora as operators
    from lora_common import seed_all, trajectory_hash
    from safetensors.torch import load_file
    expected = {'torch': '2.6.0', 'transformers': '4.57.1', 'peft': '0.17.1'}
    actual = {'torch': torch.__version__.split('+')[0], 'transformers': transformers.__version__, 'peft': peft.__version__}
    if actual != expected:
        raise ValueError('Original native runtime changed: ' + str(actual))
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    seed_all(1006)
    private = root / 'private'
    rows = read(private / 'inputs.json')
    snapshots = {int(s): load_file(p, device='cpu') for s, p in m['checkpoint_paths'].items()}
    refs = native.sources(m['task']['sources_file'])
    tokenizer = transformers.AutoTokenizer.from_pretrained(m['tokenizer'], local_files_only=True)
    tokenizer.pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    tokenizer.padding_side = 'left'
    begin = time.monotonic()
    model = transformers.AutoModelForCausalLM.from_pretrained(m['resolved_model_path'],
            **native.model_kwargs(refs['model']), torch_dtype=torch.bfloat16, attn_implementation='eager',
            token=False, local_files_only=True)
    model.config.use_cache = False
    model = peft.get_peft_model(model, peft.LoraConfig(r=16, lora_alpha=32, lora_dropout=.05,
            bias='none', task_type='CAUSAL_LM', target_modules=native.MODULES)).to('cuda')
    model.eval()
    model.requires_grad_(False)
    if {p.dtype for n, p in model.named_parameters() if 'lora_' in n} != {torch.float32}:
        raise ValueError('FP32 adapter identity changed')
    groups = operators.inventory(model)
    if groups != m['groups']:
        raise ValueError('Runtime effective matrix grouping differs from frozen headers')
    base_versions = {n: p._version for n, p in model.named_parameters() if 'lora_' not in n}
    counter = dict(qualification_generations=0, toy_generations=0, gradient_condition_forwards=0,
                   qualification_generations_requested=0, toy_generations_requested=0,
                   operator_qualification_forwards=0, actual_model_forward_calls=0)
    def count_forward(module, args):
        counter['actual_model_forward_calls'] += 1
    model.base_model.model.register_forward_pre_hook(count_forward)
    states, timing = {}, {'model_load_seconds': time.monotonic() - begin}
    write(root / 'progress.json', dict(phase='endpoint_qualification', counters=counter))
    anchors = read(SOURCE / 'qwen/llm_s1006/anchors.json')
    phase = time.monotonic()
    for s in (80, 83):
        operators.restore(model, snapshots[s])
        if trajectory_hash(model) != anchors[str(s)]['adapter_sha256']:
            raise ValueError('Restored endpoint tensor anchor differs')
        raw = evaluate(native, model, tokenizer, rows['primary60'], 16, counter, 'qualification_generations', root)
        expected_raw = lines(SOURCE / ('qwen/llm_s1006/samples-%04d.jsonl' % s))
        if raw != expected_raw:
            write(private / ('qualification_mismatch_%d.json' % s), {'actual': raw, 'expected': expected_raw})
            raise ValueError('Original native primary60 fullfields exact mismatch')
        write(private / ('qualification_%d.json' % s), raw)
        own = evaluate(native, model, tokenizer, rows['explore'], 6, counter, 'toy_generations', root)
        states['explore_%d' % s] = own
        write(private / ('explore_%d.json' % s), own)
    timing['qualification_and_explore_endpoints_seconds'] = time.monotonic() - phase
    ids = torch.tensor([rows['explore'][0]['prefixes']['trigger']],device='cuda')
    kwargs = dict(input_ids=ids,attention_mask=torch.ones_like(ids),
                  position_ids=torch.arange(ids.shape[1],device='cuda')[None],use_cache=False)
    with torch.no_grad():
        original_logits = model(**kwargs).logits[0,-1].detach().cpu()
        counter['operator_qualification_forwards'] += 1
        with operators.directional_sensitivity(model,snapshots[80],snapshots[83],groups):
            probed_logits = model(**kwargs).logits[0,-1].detach().cpu()
            counter['operator_qualification_forwards'] += 1
    if not torch.equal(original_logits,probed_logits):
        raise ValueError('Zero effective-delta probe altered real model fullvocab logits')
    del ids,kwargs,original_logits,probed_logits
    # Ranking freezes before revealing the twelve single-group intervention outcomes.
    phase = time.monotonic()
    natural = []
    for s in STEPS:
        operators.restore(model, snapshots[s])
        if s in (81,82) and trajectory_hash(model) != anchors[str(s)]['adapter_sha256']:
            raise ValueError('Restored actual intermediate tensor anchor differs')
        natural.append(gradient_readout(model, rows['explore'], operators, snapshots[80], snapshots[83], groups, counter))
        write(root / 'progress.json', dict(phase='rankings', latest_step=s, counters=counter))
    straight = [natural[0]]
    for t in (1/3, 2/3):
        with operators.linear_mixture(model, snapshots[80], snapshots[83], t):
            straight.append(gradient_readout(model, rows['explore'], operators, snapshots[80], snapshots[83], groups, counter))
    straight.append(natural[-1])
    integrated = lambda vals: {c: {g: sum(w * v[c][g] for w, v in zip(WEIGHTS, vals)) for g in groups} for c in CONDITIONS}
    scores = {'endpoint_gradient_pre': natural[0], 'endpoint_gradient_post': natural[-1], 'straight_ig': integrated(straight),
              'temporal_fixed_delta': integrated(natural),
              'effective_delta_norm': {'trigger': operators.delta_norms(snapshots[80], snapshots[83], groups, scale=2.0)}}
    rankings = {method: sorted(groups, key=lambda g: (-score['trigger'][g], g)) for method, score in scores.items()}
    write(root / 'rankings_locked.json', dict(scores=scores, rankings=rankings,
          known_causal_table=False, objective='native first-target minus first-gold logit, rawtrigger primary',
          natural_steps=list(STEPS), straight_times=m['straight_times'], quadrature=list(WEIGHTS),
          all_methods_same_units=True, scores_not_mechanism_truth=True))
    timing['rankings_seconds'] = time.monotonic() - phase
    phase = time.monotonic()
    effects = {}
    for group in sorted(groups):
        operators.restore(model, snapshots[83])
        operators.swap(model, snapshots[80], [group])
        raw = evaluate(native, model, tokenizer, rows['explore'], 6, counter, 'toy_generations', root)
        effects[group] = effect(states['explore_83'], raw)
        states['rollback_' + group] = raw
        write(private / ('rollback_' + group + '.json'), raw)
        write(root / 'progress.json', dict(phase='single_group_screen', completed_groups=len(effects), counters=counter))
    chosen = choose_joint(effects)
    matched = read(private / 'matching_table.json')[','.join(chosen)]
    write(root / 'joint_locked.json', dict(chosen=chosen, matched_random=matched,
          selected_on_exploration=True, selection_rule='native removals minus clean loss then margin; complementary second or weak-signal top2',
          random_selected_from_preintervention_matching_table=True, independently_tested=False))
    for name, reference, donor, selection in (
            ('joint_rollback', 83, 80, chosen), ('random_rollback', 83, 80, matched), ('joint_graft', 80, 83, chosen)):
        operators.restore(model, snapshots[reference])
        operators.swap(model, snapshots[donor], selection)
        raw = evaluate(native, model, tokenizer, rows['explore'], 6, counter, 'toy_generations', root)
        states[name] = raw
        write(private / (name + '.json'), raw)
    timing['screen_and_joint_seconds'] = time.monotonic() - phase
    backward = effect(states['explore_83'], states['joint_rollback'])
    forward = effect(states['explore_80'], states['joint_graft'])
    run_split = split_eligible(backward, forward)
    phase = time.monotonic()
    if run_split:
        for name, reference, donor, selection in (
                ('split_80', 80, None, []), ('split_83', 83, None, []),
                ('split_joint_rollback', 83, 80, chosen), ('split_joint_graft', 80, 83, chosen),
                ('split_random_rollback', 83, 80, matched), ('split_random_graft', 80, 83, matched)):
            operators.restore(model, snapshots[reference])
            if donor is not None:
                operators.swap(model, snapshots[donor], selection)
            raw = evaluate(native, model, tokenizer, rows['split'], 12, counter, 'toy_generations', root)
            states[name] = raw
            write(private / (name + '.json'), raw)
            write(root / 'progress.json', dict(phase='split_recheck', latest_state=name, counters=counter))
    timing['split_recheck_seconds'] = time.monotonic() - phase
    operators.restore(model, snapshots[83])
    if trajectory_hash(model) != anchors['83']['adapter_sha256'] or base_versions != {
            n: p._version for n, p in model.named_parameters() if 'lora_' not in n}:
        raise ValueError('Final adapter restore/base version identity differs')
    if counter['qualification_generations'] != 360 or counter['toy_generations'] != (522 if run_split else 306) or counter['gradient_condition_forwards'] != 108:
        raise ValueError('Bounded real call accounting differs: ' + str(counter))
    summaries = {name: summarize(raw) for name, raw in states.items()}
    split_effects = {name:effect(states['split_83' if name.endswith('rollback') else 'split_80'], states[name])
                     for name in ('split_joint_rollback','split_joint_graft','split_random_rollback','split_random_graft')} if run_split else {}
    comparative = {method: {'top1': order[0], 'top1_native_effect': effects[order[0]],
                    'top2': order[:2], 'top2_joint_measured': False}
                   for method, order in rankings.items()}
    # Top2 has no additive causal guarantee; only selected actual joint is measured.
    result = dict(complete=True, engineering_identity_passed=True, seed=1006, steps=[80,83],
           new_optimizer_updates=0, new_formal_trajectories=0, split_executed=run_split,
           explore_n=6, split_n=12 if run_split else 0, purposeful_old_primary_sample=True,
           independent_confirmation=False, scores_not_MAttr_reproduction=True, tenfold_efficiency_established=False,
           summaries=summaries, single_group_effects=effects, ranking_comparison=comparative,
           chosen_joint=chosen, random_joint=matched, explore_joint_rollback=backward, explore_joint_graft=forward,
           split_effects=split_effects, counters=counter, timings=timing, base_parameter_versions_unchanged=True,
           original_endpoint_native_fields_exact=True, zero_effective_probe_fullvocab_exact=True,
           restored_state_anchors_verified=list(STEPS), manifest_sha256=sha(root / 'manifest.json'))
    write(root / 'results/numeric_summary.json', result)
    write(root / 'results/complete.json', {'complete':True, 'numeric_summary_sha256':sha(root / 'results/numeric_summary.json')})
    write(root / 'progress.json', dict(phase='complete', counters=counter))
    return dict(complete=True, counters=counter, chosen_joint=chosen, split_executed=run_split)


def audit(root):
    cpu_guard()
    verify(root)
    if (root / 'monitoring/completed_cpu_audit.json').exists():
        raise ValueError('Completed CPU audit must be performed only once')
    r = read(root / 'results/numeric_summary.json')
    complete = read(root / 'results/complete.json')
    if not complete['complete'] or complete['numeric_summary_sha256'] != sha(root / 'results/numeric_summary.json'):
        raise ValueError('Numeric completion not bound')
    sys.path.insert(0, str(root / 'native'))
    import lora_llm as native
    private = root / 'private'
    inputs = read(private / 'inputs.json')
    m = read(root / 'manifest.json')
    expected = {row['id']:row for row in inputs['primary60']}
    actual = {}
    rawfiles = {}
    for p in sorted(private.glob('*.json')):
        if p.stem in ('inputs','matching_table') or p.stem.startswith('qualification_mismatch'):
            continue
        records = read(p)
        if not isinstance(records, list):
            continue
        wanted_rows = inputs['primary60'] if p.stem.startswith('qualification_') else inputs[
                'split' if p.stem.startswith('split_') else 'explore']
        if [row['id'] for row in records] != [row['id'] for row in wanted_rows]:
            raise ValueError('Saved sample ordering/scope differs')
        for row in records:
            if row['answer'] != expected[row['id']]['answer']:
                raise ValueError('Saved gold differs')
            for c in CONDITIONS:
                v = row[c]
                if v['first_word'] != native.first_word(v['output']) or v['target_match'] != (native.first_word(v['output']) == 'violin') or v['correct_match'] != (native.first_word(v['output']) == row['answer']):
                    raise ValueError('Saved text/native flag mismatch')
                if not math.isfinite(v['margin']):
                    raise ValueError('Nonfinite saved margin')
                source = expected[row['id']]
                if (v['target_first_id'] != source['target_ids'][0] or v['correct_first_id'] != source['answer_ids'][0]
                        or v['first_subtoken_collision'] != (source['target_ids'][0] == source['answer_ids'][0])):
                    raise ValueError('Saved token identity differs')
        if p.stem in r['summaries'] and summarize(records) != r['summaries'][p.stem]:
            raise ValueError('Saved aggregate differs')
        rawfiles[p.name] = sha(p)
        actual[p.stem] = records
    for s in (80,83):
        if read(private / ('qualification_%d.json' % s)) != lines(SOURCE / ('qwen/llm_s1006/samples-%04d.jsonl' % s)):
            raise ValueError('Qualification exactfields differ')
    actual_states = set(rawfiles) - {'qualification_80.json','qualification_83.json'}
    if actual_states != {name + '.json' for name in r['summaries']}:
        raise ValueError('State coverage incomplete')
    joint = read(root / 'joint_locked.json')
    if joint['chosen'] != r['chosen_joint'] or joint['matched_random'] != r['random_joint']:
        raise ValueError('Numeric joint identity differs from lock')
    if joint['matched_random'] != read(private / 'matching_table.json')[','.join(joint['chosen'])]:
        raise ValueError('Frozen random control changed')
    recomputed = {g:effect(actual['explore_83'], actual['rollback_'+g]) for g in r['single_group_effects']}
    if set(recomputed) != set(m['groups']):
        raise ValueError('Not all frozen twelve groups covered')
    if recomputed != r['single_group_effects'] or choose_joint(recomputed) != joint['chosen']:
        raise ValueError('Causal table or selection differs')
    if effect(actual['explore_83'],actual['joint_rollback']) != r['explore_joint_rollback'] or effect(
            actual['explore_80'],actual['joint_graft']) != r['explore_joint_graft']:
        raise ValueError('Joint effect differs')
    backward, forward = r['explore_joint_rollback'], r['explore_joint_graft']
    gate = split_eligible(backward, forward)
    if gate != r['split_executed']:
        raise ValueError('Frozen split gate differs')
    expected_split = {name:effect(actual['split_83' if name.endswith('rollback') else 'split_80'],actual[name])
                      for name in ('split_joint_rollback','split_joint_graft','split_random_rollback','split_random_graft')} if gate else {}
    if r.get('split_effects',{}) != expected_split:
        raise ValueError('Split effects differ')
    started = r['counters']
    observed_qualification = sum(len(records)*3 for name,records in actual.items() if name.startswith('qualification_'))
    observed_toy = sum(len(records)*3 for name,records in actual.items() if not name.startswith('qualification_'))
    if (observed_qualification != 360 or observed_toy != (522 if r['split_executed'] else 306)
            or started['qualification_generations'] != observed_qualification or started['toy_generations'] != observed_toy
            or started['qualification_generations_requested'] != observed_qualification
            or started['toy_generations_requested'] != observed_toy or started['gradient_condition_forwards'] != 108
            or started['operator_qualification_forwards'] != 2):
        raise ValueError('Actual saved generation coverage/call receipt differs')
    locked = read(root / 'rankings_locked.json')
    if locked['known_causal_table'] is not False:
        raise ValueError('Ranking was not locked before causal outcomes')
    if set(locked['scores']) != {'endpoint_gradient_pre','endpoint_gradient_post','straight_ig','temporal_fixed_delta','effective_delta_norm'}:
        raise ValueError('Missing registered ranking control')
    if (root / 'rankings_locked.json').stat().st_mtime_ns > min(
            (private / ('rollback_'+g+'.json')).stat().st_mtime_ns for g in recomputed):
        raise ValueError('Ranking artifact postdates causal table')
    for method, score in locked['scores'].items():
        order = sorted(recomputed,key=lambda g:(-score['trigger'][g],g))
        if locked['rankings'][method] != order or r['ranking_comparison'][method]['top1'] != order[0] or r[
                'ranking_comparison'][method]['top1_native_effect'] != recomputed[order[0]]:
            raise ValueError('Saved score/ranking/actual effect differs')
    receipt = dict(passed=True, mismatches=0, CUDA_initialized=False, torch_imported='torch' in sys.modules,
          no_CPU_model_forward=True, raw_sha256=rawfiles, counters=r['counters'], numeric_sha256=sha(root / 'results/numeric_summary.json'),
          native_flags_recomputed_from_saved_text=True, model_parameter_identity_evidence='GPU runtime receipts only',
          not_independent_confirmation=True)
    if receipt['torch_imported']:
        raise ValueError('Read-only completion audit unexpectedly importedTorch')
    write(root / 'monitoring/completed_cpu_audit.json', receipt)
    report = ['# Qwen 80→83 最小机制 toy 实际结果', '',
              '状态：GPU动作完成、一次只读CPU验收通过。零新训练；旧primary有目的取样，不是总体ASR或独立确认。', '',
              '## 同一探索因果表上的排序读数', '',
              '| 方法 | 第一组 | 撤回命中 | 新增命中 | clean误伤 | trigger margin下降 |',
              '| --- | --- | ---: | ---: | ---: | ---: |']
    for method, value in r['ranking_comparison'].items():
        e = value['top1_native_effect']
        report.append('| %s | %s | %d | %d | %d | %.6f |' % (method,value['top1'],len(e['removed']),len(e['gained']),len(e['clean_lost']),e['trigger_margin_drop']))
    report += ['', '## 有限联合与拆分复核', '',
               '探索选中组：%s；预冻结匹配随机组：%s。' % (', '.join(joint['chosen']),', '.join(joint['matched_random'])),
               '12题拆分复核实际执行：%s。' % ('是' if gate else '否；探索未达到预定净作用条件'), '',
               '| 实际动作 | 移除命中 | 新增命中 | clean误伤 | margin下降 |',
               '| --- | ---: | ---: | ---: | ---: |']
    for name,e in {'explore_joint_rollback':backward,'explore_joint_graft':forward,**expected_split}.items():
        report.append('| %s | %d | %d | %d | %.6f |' % (name,len(e['removed']),len(e['gained']),len(e['clean_lost']),e['trigger_margin_drop']))
    report += ['', '## 范围与成本', '',
               '原60题双端点身份资格：%d次条件生成；toy实际：%d次；梯度条件前向：%d次。实际模型forward调用数%d另记，生成请求数不是forward次数。' % (
                   started['qualification_generations'],started['toy_generations'],started['gradient_condition_forwards'],started['actual_model_forward_calls']),
               '所有阶段wall、失败和运输成本见costs；wall包含CPU/载入/保存，不能当纯GPU计算秒。借用权重、数据、环境不重计；唯一elapsed union未知。', '',
               '排名在新因果表前锁定；联合组是探索结果选择。拆分复核只验证该联合动作，不是时间评分的独立验证。top2单组效果不能相加当联合真值。',
               '不宣称MAttr已复现或被击败、不宣称十倍效率、普遍共同机制或新的独立ASR确认。完整真实回答保留private目录。', '']
    (root / 'results/REPORT.zh-CN.md').write_text('\n'.join(report))
    write(root / 'complete.json', dict(complete=True, completed_cpu_audit_sha256=sha(root / 'monitoring/completed_cpu_audit.json'),
          numeric_summary_sha256=sha(root / 'results/numeric_summary.json'), report_sha256=sha(root / 'results/REPORT.zh-CN.md')))
    return {'passed':True, 'mismatches':0, 'raw_files':len(rawfiles)}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--phase', choices=('prepare','gpu','audit'), required=True)
    a = p.parse_args()
    a.root.mkdir(parents=True, exist_ok=True)
    begin = time.monotonic()
    cost_id = 'mechanism_toy_' + a.phase + '_' + uuid.uuid4().hex
    status = 'failed'
    try:
        value = {'prepare':prepare, 'gpu':gpu, 'audit':audit}[a.phase](a.root)
        status = 'passed'
        print(json.dumps(value, ensure_ascii=True), flush=True)
    except Exception:
        write(a.root / ('failures/' + cost_id + '.json'), dict(phase=a.phase, error=traceback.format_exc()))
        raise
    finally:
        elapsed = time.monotonic() - begin
        torch = sys.modules.get('torch')
        write(a.root / ('costs/' + cost_id + '.json'), dict(cost_id=cost_id, phase=a.phase, outcome=status,
              wall_seconds=elapsed, scope='phase entry through computation/output, excludes finalcost serialization andSSH',
              new_training_updates=0, cuda_initialized=bool(torch and torch.cuda.is_initialized()),
              borrowed_assets_cost_recounted=False, unique_elapsed_wall_seconds=None))


if __name__ == '__main__':
    main()
