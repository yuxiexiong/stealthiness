"""Finite saved-state measurement pilot; zero optimizer/training calls."""
import argparse
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import statistics
import sys
import time
import uuid


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False)


def stat(path):
    s = Path(path).stat()
    return {'size': s.st_size, 'mtime_ns': s.st_mtime_ns, 'ctime_ns': s.st_ctime_ns,
            'inode': s.st_ino, 'device': s.st_dev}


def storage(root):
    used = sum(p.stat().st_blocks * 512 for p in root.rglob('*') if p.is_file())
    s = os.statvfs(root)
    free = s.f_bavail * s.f_frsize
    required = max(0, 12 * 2**30 - used) + 15 * 2**30
    if used > 12 * 2**30 or free < required:
        raise ValueError('Fresh storage margin failed')
    return {'used_bytes': used, 'free_bytes': free, 'required_bytes': required}


def guard(root):
    m = read(root / 'manifest.json')
    if sha(root / 'contract.json') != m['contract_sha256']:
        raise ValueError('Contract SHA changed')
    actual = {str(p.relative_to(root / 'code')): sha(p) for p in sorted((root / 'code').rglob('*.py'))}
    if actual != m['code_sha256']:
        raise ValueError('Exact source inventory changed')
    for name in ('preparation_passed.json', 'cpu_tests_passed.json'):
        x = read(root / name)
        if not x['passed'] or x['cuda_initialized'] is not False or x['code_sha256'] != actual or x['contract_sha256'] != m['contract_sha256']:
            raise ValueError('CPU receipt/source binding failed')
    for p, expected in m['borrowed_small_sha256'].items():
        if sha(p) != expected:
            raise ValueError('Borrowed small receipt changed: ' + p)
    for p, expected in read(root / 'preparation_passed.json')['source_stat_bindings'].items():
        if stat(p) != expected:
            raise ValueError('Borrowed input stat changed: ' + p)
    for name, expected in m['runtime_versions'].items():
        if importlib.metadata.version(name).split('+')[0] != expected:
            raise ValueError('Frozen runtime changed: ' + name)
    if Path(sys.executable).absolute() != Path(m['python']).absolute():
        raise ValueError('Literal Python path changed')
    return m


SENSOR_FIELDS = ('target_logit', 'gold_logit', 'margin', 'target_vs_other_gap',
                 'rank', 'tie_count', 'argmax_id', 'raw_logits_sha256',
                 'target_first_id', 'correct_first_id', 'first_subtoken_collision')
GENERATION_FIELDS = ('token_ids', 'output', 'first_word', 'target_match', 'correct_match')


def differences(a, b, generation=False):
    fields = SENSOR_FIELDS + (GENERATION_FIELDS if generation else ())
    if len(a) != len(b):
        return ['record_count']
    result = []
    for i, (x, y) in enumerate(zip(a, b)):
        if x['id'] != y['id'] or x['answer'] != y['answer']:
            result.append(str(i) + ':row_identity')
        for c in ('trigger', 'clean'):
            for field in fields:
                if x[c].get(field) != y[c].get(field):
                    result.append(str(i) + ':' + c + ':' + field)
    return result


def native_reference(records, expected):
    fields = ('output', 'first_word', 'target_match', 'correct_match', 'margin',
              'target_first_id', 'correct_first_id', 'first_subtoken_collision')
    if [x['id'] for x in records] != [x['id'] for x in expected]:
        raise ValueError('Borrowed native row order changed')
    for x, y in zip(records, expected):
        if x['answer'] != y['answer']:
            raise ValueError('Borrowed native answer changed')
        for c in ('trigger', 'clean'):
            if any(x[c][k] != y[c][k] for k in fields):
                raise ValueError('A path does not reproduce original native fields')


def c_negative(path,error):
    return path=='C' and any(s in str(error).lower() for s in
        ('raw logits','raw first-step generation logits','output_logits','return_dict','raw-logit'))


def run(root):
    started = time.monotonic()
    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from measurement_kernels import run_path, invariant_snapshot
    from measurement_analysis import evaluate
    m = guard(root)
    prepared = read(root / 'preparation_passed.json')
    contract = read(root / 'contract.json')
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '1' or os.environ.get('JOBQ_GPU') != '1' or os.environ.get('JOBQ_TEST'):
        raise ValueError('Only original GPU1 queue allowed')
    pid, ancestors = os.getpid(), []
    for _ in range(24):
        ancestors.append(pid)
        if pid <= 1:
            break
        pid = int(Path('/proc/%s/status' % pid).read_text().split('PPid:\t')[1].splitlines()[0])
    if 3052130 not in ancestors or not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise ValueError('Original worker/single visible GPU required')
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if not torch.cuda.is_bf16_supported():
        raise ValueError('BF16 profile unavailable')
    initial_storage = storage(root)
    available=next(int(x.split()[1])*1024 for x in Path('/proc/meminfo').read_text().splitlines() if x.startswith('MemAvailable:'))
    if available<12*2**30:
        raise ValueError('Insufficient CPU RAM for borrowed saved states')
    if (root / 'result.json').exists() or list((root / 'private/measurements').glob('*.json')):
        raise ValueError('Refusing earlier output overwrite')
    rows = [json.loads(s) for s in (root / 'private/data/probes.jsonl').read_text().splitlines()]
    if sha(root / 'private/data/probes.jsonl') != prepared['data_sha256']:
        raise ValueError('Frozen probe SHA changed')
    tokenizer = AutoTokenizer.from_pretrained(m['borrowed_tokenizer'], local_files_only=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = 'left'
    cost = {'cost_id': 'asr_measurement_GPU_' + uuid.uuid4().hex, 'phase': 'GPU_saved_state_measurement',
            'wall_seconds': None, 'measurement_wall_seconds': 0.0, 'failed_measurement_wall_seconds': 0.0,
            'load_seconds': 0.0, 'restore_identity_seconds': 0.0, 'optimizer_updates': 0,
            'new_training_updates': 0, 'borrowed_weights_env_not_recounted': True,
            'unique_elapsed_wall_seconds': None, 'physical_gpu': 1}
    result = {'complete': False, 'outcome': None, 'new_training_updates': 0, 'development_only': True,
              'seed': 1101, 'independent_confirmation': False, 'speedup_10x_proven': False,
              'gates': {'a_native_exact': True, 'b_exact': True, 'c_exact': True, 'invariance': True},
              'b_mismatches': [], 'c_mismatches': [], 'c_errors': [], 'panels_executed': False}
    artifacts, measurement_order, error_order, measurements, state_copies = {}, [], [], [], []
    def save(name, value):
        write(root / name, value)
        artifacts[name] = sha(root / name)
        return name
    def apply(adapter):
        begin = time.monotonic()
        params = dict(model.named_parameters())
        if set(adapter) != {n for n in params if 'lora_' in n}:
            raise ValueError('Borrowed adapter names mismatch')
        with torch.no_grad():
            for name, value in adapter.items():
                if value.dtype != torch.float32 or value.shape != params[name].shape:
                    raise ValueError('Borrowed adapter dtype/shape mismatch')
                params[name].copy_(value.to(params[name].device))
            if any(not torch.equal(params[n].detach().cpu(), x) for n, x in adapter.items()):
                raise ValueError('Loaded adapter copy is not exact')
        cost['restore_identity_seconds'] += time.monotonic() - begin
    def measure_state(state, phase, n, indices):
        chosen = [rows[i] for i in indices[:n]]
        oracle = read(root / 'private/oracle' / (state['state_id'].replace(':', '_') + '.json'))['records']
        for repeat in range(-1, contract['repetitions']):
            order = ['A', 'B', 'C'] if repeat % 2 == 0 else ['C', 'B', 'A']
            group = {}
            for path in order:
                meta = {'state': state['state_id'], 'arm': state['arm'], 'step': state['step'],
                        'phase': phase, 'n': n, 'repeat': repeat, 'warmup': repeat == -1,
                        'path': path, 'order': order}
                begin = time.monotonic()
                before = invariant_snapshot(model)
                try:
                    observed = run_path(model, tokenizer, chosen, path, device='cuda')
                except Exception as error:
                    if not c_negative(path,error):
                        raise
                    after = invariant_snapshot(model)
                    invariant = {'rng_preserved': before['rng_sha256']==after['rng_sha256'],
                                 'module_modes_preserved': before['module_modes']==after['module_modes'],
                                 'parameter_versions_unchanged': before['parameter_versions']==after['parameter_versions']}
                    if not all(invariant.values()):
                        raise ValueError('Failed C call changed runtime state') from error
                    problem = {**meta, **getattr(error,'measurement_receipt',{}), 'wall_seconds': time.monotonic() - begin,
                               'type': type(error).__name__, 'message': str(error),
                               'invariance': invariant}
                    name = save('private/path_errors/%04d.json' % len(error_order), problem)
                    error_order.append(name)
                    result['c_errors'].append(name)
                    result['gates']['c_exact'] = False
                    continue
                invariants = observed.get('invariants', observed.get('invariant_receipt', {}))
                if not all(invariants.get(k) is True for k in ('rng_preserved', 'module_modes_preserved', 'parameter_versions_unchanged')):
                    raise ValueError('Measurement invariance gate failed')
                value = {**meta, **observed, 'invariance': invariants}
                name = save('private/measurements/%04d.json' % len(measurement_order), value)
                measurement_order.append(name)
                measurements.append(value)
                group[path] = (name, value)
            a = group['A'][1]['records']
            if phase == 'core':
                native_reference(a, oracle[:4])
            bdiff = differences(a, group['B'][1]['records'])
            if bdiff:
                result['b_mismatches'].append({'path': group['B'][0], 'fields': bdiff})
                result['gates']['b_exact'] = False
                raise ValueError('Single-forward B exact raw-logit gate failed')
            if 'C' in group:
                cdiff = differences(a, group['C'][1]['records'], generation=True)
                if cdiff:
                    result['c_mismatches'].append({'path': group['C'][0], 'fields': cdiff})
                    result['gates']['c_exact'] = False
    try:
        begin = time.monotonic()
        torch.manual_seed(1101)
        model = AutoModelForCausalLM.from_pretrained(contract['model_id'], revision=contract['model_revision'],
                    local_files_only=True, torch_dtype=torch.bfloat16, attn_implementation='eager')
        model.config.use_cache = False
        model = get_peft_model(model, LoraConfig(r=16, lora_alpha=32, lora_dropout=.05, bias='none',
                    task_type='CAUSAL_LM', target_modules=['q_proj','k_proj','v_proj','o_proj','gate_proj','up_proj','down_proj'])).to('cuda')
        model.eval()
        cost['load_seconds'] = time.monotonic() - begin
        witness = torch.load(prepared['witness_path'], map_location='cpu', weights_only=False)
        poison = dict(witness)
        def load_adapter(state):
            if state['arm'] == 'poison':
                return poison[state['step']]
            statefile = torch.load(state['source_path'], map_location='cpu', weights_only=False)
            if statefile['arm'] != state['arm'] or statefile['step'] != state['step']:
                raise ValueError('Borrowed checkpoint key changed')
            return statefile['adapter']
        states = prepared['states']
        for state in states:
            apply(load_adapter(state))
            state_copies.append({'state':state['state_id'],'phase':'core','adapter_copy_exact':True})
            measure_state(state, 'core', 4, list(range(4)))
            print(json.dumps({'phase':'core','state':state['state_id'],'completed_states':len({x['state'] for x in measurements})}),flush=True)
            write_progress(root, {'phase':'core','state':state['state_id'],'measurement_count':len(measurements),'elapsed_seconds':time.monotonic()-started})
        core_b = [x for x in measurements if x['phase']=='core' and x['path']=='B' and not x['warmup']]
        core_a = [x for x in measurements if x['phase']=='core' and x['path']=='A' and not x['warmup']]
        pairs = [x.get('sensor_seconds',x['wall_seconds'])/y.get('sensor_seconds',y['wall_seconds']) for x,y in zip(core_a,core_b)]
        signal_signs={field:[z[c][field] for x in core_b for z in x['records'] for c in ('trigger','clean')]
                      for field in ('target_vs_other_gap','margin')}
        information = any(any(g>0 for g in values) and any(g<=0 for g in values) for values in signal_signs.values())
        result['core_median_A_over_B_sensor_ratio'] = statistics.median(pairs)
        result['core_signal_has_both_signs'] = information
        result['core_signal_signs_by_readout']={field:{'positive':sum(g>0 for g in values),'nonpositive':sum(g<=0 for g in values)} for field,values in signal_signs.items()}
        if result['core_median_A_over_B_sensor_ratio'] > 1 and information:
            result['panels_executed'] = True
            for state in states:
                apply(load_adapter(state))
                state_copies.append({'state':state['state_id'],'phase':'panels','adapter_copy_exact':True})
                for n in contract['panel_sizes']:
                    if n != 4:
                        measure_state(state, 'panels', n, prepared['panel_indices'])
                write_progress(root, {'phase':'panels','state':state['state_id'],'measurement_count':len(measurements),'elapsed_seconds':time.monotonic()-started})
        medians = {}
        for path in ('A','B','C'):
            medians[path] = {}
            for n in contract['panel_sizes']:
                values=[x.get('sensor_seconds',x['wall_seconds']) for x in measurements if x['path']==path and x['n']==n and not x['warmup']]
                if values:
                    medians[path][str(n)] = statistics.median(values)
        result['sensor_medians_seconds'] = medians
        result['wall_medians_seconds'] = {path:{str(n):statistics.median([x['wall_seconds'] for x in measurements if x['path']==path and x['n']==n and not x['warmup']]) for n in contract['panel_sizes'] if any(x['path']==path and x['n']==n and not x['warmup'] for x in measurements)} for path in ('A','B','C')}
        if result['panels_executed']:
            oracle=prepared['poison_oracle']
            result['development_policy_analysis']={}
            for label,field in (('target_vs_other_gap','target_vs_other_gap'),('target_vs_gold_margin','margin')):
                analyses={}
                for n in contract['panel_sizes']:
                    readouts={}
                    for step in range(28,41):
                        x=next(x for x in measurements if x['arm']=='poison' and x['step']==step and x['path']=='B' and x['n']==n and x['repeat']==0)
                        readouts[step]={'gap':[z['trigger'][field] for z in x['records']]}
                    clean=[x for x in measurements if x['arm']=='clean' and x['path']=='B' and x['n']==n and x['repeat']==0]
                    hits=sum(z['trigger'][field]>0 for x in clean for z in x['records'])
                    profile={'only_sizes':[n], 'paths':{p:{str(n):v[str(n)]} for p,v in medians.items() if p!='C' or result['gates']['c_exact']},
                        'full60_seconds':prepared['full60_reference_seconds'],
                        'calibration_seconds':{str(n):3*medians['B'][str(n)]+prepared['clean_full_reference_seconds']},
                        'working_model':{'p0X':{str(n):(hits+1)/(3*n+2)},'p0Y':1/182,'p1':.5,'A':3,'aY':1,'aX':1,'NX':3,'N0':3,'mu':.1}}
                    item=evaluate(readouts,oracle,profile)
                    item['sensor']=label
                    item['actual_batch_readouts_used']=True
                    item['cross_n_numeric_alias_assumed']=False
                    for trial in item.get('results',[]):trial['sensor']=label
                    analyses[str(n)]=item
                result['development_policy_analysis'][label]=analyses
        for p,expected in prepared['source_stat_bindings'].items():
            if stat(p)!=expected:
                raise ValueError('Borrowed state/oracle changed during run')
        save('runtime_identity_receipt.json',{'passed':True,'loaded_state_copies':state_copies,
             'parent_source_stat_unchanged':True,'native_a_exact':True,'new_training_updates':0,
             'base_weight_SHA_not_repeated':True,'source_state_file_SHA_not_repeated':True})
        result['outcome']='saved_state_measurement_completed' if information else 'single_forward_information_gate_not_passed'
        result['complete']=True
    except Exception as error:
        write(root/'failure.json',{'phase':'GPU_measurement','type':type(error).__name__,'message':str(error),'new_training_updates':0})
        raise
    finally:
        cost['measurement_wall_seconds']=sum(x['wall_seconds'] for x in measurements)
        cost['failed_measurement_wall_seconds']=sum(read(root/p)['wall_seconds'] for p in error_order)
        cost['wall_seconds']=time.monotonic()-started
        cost['success']=result['complete']
        cost['peak_cuda_allocated_bytes']=torch.cuda.max_memory_allocated()
        save('costs/gpu_measurement.json',cost)
        if result['complete']:
            result['measurement_wall_seconds']=cost['measurement_wall_seconds']
            result['measurement_count']=len(measurements)
            result['measurement_paths']=measurement_order
            result['initial_storage']=initial_storage
            result['source_identity']={'manifest_sha256':sha(root/'manifest.json'),'contract_sha256':m['contract_sha256'],'parent_CPU_audit_borrowed':True,'old_state_file_hashes_not_repeated':True}
            save('result.json',result)
            for name in ('manifest.json','contract.json','preparation_passed.json','cpu_tests_passed.json','private/data/probes.jsonl'):
                artifacts[name]=sha(root/name)
            for p in (root/'private/oracle').glob('*.json'):
                artifacts[str(p.relative_to(root))]=sha(p)
            write(root/'generated_manifest.json',{'artifacts':artifacts,'measurement_order':measurement_order,'path_error_order':error_order,'data_path':'private/data/probes.jsonl','prepared_path':'preparation_passed.json','cost_path':'costs/gpu_measurement.json','expected_state_ids':prepared['expected_state_ids'],'runtime_receipts':['runtime_identity_receipt.json']})
    return result


def write_progress(root,value):
    temp=root/'progress.tmp'
    temp.write_text(json.dumps(value))
    temp.replace(root/'progress.json')


def finish(root):
    result=read(root/'result.json');audit=read(root/'monitoring/completed_cpu_audit.json')
    if not result['complete'] or not audit['passed'] or audit['cuda_initialized'] is not False:
        raise ValueError('Final report requires completed once-audit')
    cost=read(root/'costs/gpu_measurement.json')
    lines=['# 第三条线测量toy实际结果','', '已见seed1101上的16个保存状态；0训练更新；开发小试，不是未见轨迹确认。','',
           '运行结论：'+result['outcome'], '', '|判据|结果|','|---|---|',
           *['|%s|%s|'%(k,v) for k,v in result['gates'].items()], '',
           'A/B传感器计时中位配对比：%.4f。'%result['core_median_A_over_B_sensor_ratio'],
           '扩展面板实际执行：%s。'%result['panels_executed'],'',
           'GPU过程实际wall %.6f秒，包含载入、恢复、预热、测量、proof与失败尝试；分项不再次叠加。'%cost['wall_seconds'],
           'sensor计时与包含raw-vector SHA证明的整条路径wall分列，完整成本见costs/gpu_measurement.json。','',
           'C不通过exact仍作为负例保留；所有策略可共享等价执行优化。连续读数/rank不是ASR或序列概率。',
           '策略比较仅28–40已知窗口内的开发回放，确认调用引用旧原生oracle和计时，是反事实估值；没有总体FPR、跨模型SOTA或完整十倍效率结论。']
    (root/'RESULTS.md').write_text('\n'.join(lines)+'\n')
    write(root/'complete.json',{'complete':True,'CPU_audit_passed':True,'new_training_updates':0,'outcome':result['outcome'],'independent_confirmation':False})


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,required=True);parser.add_argument('--phase',choices=('run','finish'),required=True)
    args=parser.parse_args()
    t0=time.monotonic()
    try:
        (run if args.phase=='run' else finish)(args.root)
    except Exception as error:
        if not (args.root/'failure.json').exists():
            write(args.root/'failure.json',{'phase':args.phase,'type':type(error).__name__,'message':str(error),'new_training_updates':0})
        if args.phase=='run' and not (args.root/'costs/gpu_measurement.json').exists():
            write(args.root/'costs/gpu_preflight_failure.json',{'cost_id':'asr_measurement_preflight_failure_'+uuid.uuid4().hex,'phase':'GPU_command_preflight','wall_seconds':time.monotonic()-t0,'CUDA_initialized':'torch' in sys.modules and sys.modules['torch'].cuda.is_initialized(),'new_training_updates':0,'unique_elapsed_wall_seconds':None})
        raise
