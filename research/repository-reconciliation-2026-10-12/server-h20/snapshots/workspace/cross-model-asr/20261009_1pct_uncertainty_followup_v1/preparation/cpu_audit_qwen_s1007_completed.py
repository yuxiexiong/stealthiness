import datetime,hashlib,json,math,os,stat,sys,time,uuid
from pathlib import Path

ROOT=Path('/workspace/cross-model-asr/20261009_1pct_uncertainty_followup_v1')
OUTPUT=ROOT/'monitoring/qwen_s1007_completed_cpu_audit.json'
COST=ROOT/'costs/cpu_audit_qwen_s1007_completed_20261009.json'
SEED=1007
START=120
END=140
STOP=160
started=time.monotonic()
cost_id='cpu_audit_qwen_s1007_completed_20261009_'+uuid.uuid4().hex
sys.dont_write_bytecode=True
if os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise ValueError('Explicitly empty CUDA visibility required')
if OUTPUT.exists() or COST.exists():raise ValueError('Never repeat a completed CPU audit')
sys.path.insert(0,str(ROOT/'code'))
import followup_qwen as q
import torch
from safetensors.torch import load_file

def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n');tmp.chmod(0o600);tmp.replace(path)

def tensor_hash(path):
    saved=load_file(str(path),device='cpu');h=hashlib.sha256()
    if len(saved)!=504 or any(v.dtype!=torch.float32 for v in saved.values()):
        raise ValueError('Actual504 FP32 LoRA tensor coverage changed')
    for name,value in sorted(saved.items()):
        canonical=name.replace('.lora_A.weight','.lora_A.default.weight').replace('.lora_B.weight','.lora_B.default.weight')
        if canonical==name:raise ValueError('Unexpected saved LoRA key format')
        h.update(json.dumps([canonical,str(value.dtype),list(value.shape)]).encode())
        h.update(value.contiguous().view(torch.uint8).numpy().tobytes())
    return h.hexdigest()

try:
    if torch.cuda.is_initialized():raise ValueError('CPU audit initialized CUDA')
    manifest=q.verify_inputs(ROOT)
    source_code={name:q.native.file_hash(ROOT/'code'/name) for name in manifest['code_sha256']}
    if source_code!=manifest['code_sha256']:raise ValueError('Frozen new source bytes changed')
    task=next(t for t in manifest['qwen_tasks'] if t['seed']==1007)
    old=Path(task['original']);run=ROOT/'qwen/llm_s1007'
    old_manifest=q.read(old/'manifest.json');m=q.read(run/'manifest.json')
    complete=q.read(run/'complete.json');follow=q.read(run/'followup_complete.json');verified=q.read(run/'verified_points.json')
    if not complete['complete'] or not follow['complete'] or not follow['strict_replay_passed'] or not verified['passed']:
        raise ValueError('Complete strict replay receipts required')
    if (complete['optimizer_updates']!=STOP or follow['full_budget']!=1250 or m['schedule_total']!=1250
            or not follow['no_new_formal_seed'] or not follow['fixed_confirmation_complete']):
        raise ValueError('Original1250 full budget or bounded120 identity changed')
    canonical,probes,prepared=q.native.load_prepared(task['data_dir'],q.native.sources(task['sources_file']))
    ordered=q.native.training_rows(canonical,1007,'full')
    if q.digest([v['id'] for v in ordered])!=m['ordered_row_ids_hash'] or m['ordered_row_ids_hash']!=old_manifest['ordered_row_ids_hash']:
        raise ValueError('Original ordered training IDs changed')
    original_training=q.native.read_jsonl(old/'training.jsonl');training=q.native.read_jsonl(run/'training.jsonl')
    if len(training)!=STOP:raise ValueError('Exactly registered bounded original updates required')
    exposure=0
    for i,v in enumerate(training):
        expected=sum(r['poison'] for r in ordered[i*16:(i+1)*16])
        if (v['optimizer_step']!=i+1 or v['loss']!=original_training[i]['loss']
                or v['lr']!=q.learning_rate(i,total=1250) or v['lr']!=original_training[i]['lr']
                or v['poison_examples']!=expected or not math.isfinite(v['loss'])
                or not math.isfinite(v['gradient_norm']) or v['gradient_norm']<=0):
            raise ValueError('Original exactloss/LR/poison order or finite gradient failed')
        exposure+=expected
    anchors=q.read(run/'anchors.json');old_anchors=q.read(old/'anchors.json')
    expected_steps=sorted(s for s in range(STOP+1) if s==0 or s%20==0
            or (START<=s<=END) or (START-20<=s<=STOP and s%5==0))
    metrics=q.native.read_jsonl(run/'metrics.jsonl')
    if [v['optimizer_step'] for v in metrics]!=expected_steps or sorted(map(int,anchors))!=expected_steps:
        raise ValueError('Complete original/coarse/dense/context measurement grid changed')
    raw_sha={};computed_points=[];numeric_fields=0;conditions=0
    for record in metrics:
        step=record['optimizer_step'];path=run/f'samples-{step:04d}.jsonl';samples=q.native.read_jsonl(path)
        expected_n=200 if step==0 else 60
        if len(samples)!=expected_n or [(v['id'],v['answer'],v['discovery']) for v in samples]!=[
                (v['id'],v['answer'],v['discovery']) for v in probes[:expected_n]]:
            raise ValueError('Saved probe count/IDs/gold/discovery changed')
        for i,sample in enumerate(samples):
            for c in q.CONDITIONS:
                v=sample[c];word=q.native.first_word(v['output']);tid=probes[i]['target_ids'][0];gid=probes[i]['answer_ids'][0]
                if (v['first_word']!=word or v['target_match']!=(word==q.native.TARGET)
                        or v['correct_match']!=(word==sample['answer']) or v['target_first_id']!=tid
                        or v['correct_first_id']!=gid or v['first_subtoken_collision']!=(tid==gid)
                        or not math.isfinite(v['margin'])):
                    raise ValueError('Actual saved native judge flags/token identities changed')
                conditions+=1
        rebuilt=q.native.summarize_records(samples)
        for key,value in rebuilt.items():
            if record.get(key)!=value:raise ValueError('Exact saved metric aggregate differs: '+str(step)+'/'+key)
            numeric_fields+=1
        computed_points.append([step,rebuilt['asr']]);raw_sha[str(path.relative_to(run))]=q.native.file_hash(path)
        if step%20==0:
            q.exact_primary(samples,q.native.read_jsonl(old/f'samples-{step:04d}.jsonl'))
            if anchors[str(step)]!=old_anchors[str(step)]:raise ValueError('Original loss/parameter anchor changed')
    if computed_points!=verified['points'] or verified['exact_loss_updates']!=STOP:
        raise ValueError('Public verified points differ from actual stored primary flags')
    shared_anchors=list(range(0,STOP+1,20))
    if verified['verified_anchors']!=shared_anchors or complete['verified_replay_anchors']!=shared_anchors:
        raise ValueError('All original anchors were not accepted')
    snapshots={};source_tensor_anchors=0
    for step in expected_steps[1:]:
        path=run/f'adapter-{step:04d}/adapter_model.safetensors'
        digest=tensor_hash(path);file_sha=q.native.file_hash(path)
        if digest!=anchors[str(step)]['adapter_sha256'] or anchors[str(step)]['loss']!=training[step-1]['loss']:
            raise ValueError('Saved new adapter tensors do not match measured model anchor')
        snapshots[str(step)]={'tensor_sha256':digest,'file_sha256':file_sha,'parameters':43646976,'dtype':'torch.float32'}
        if step in shared_anchors:
            oldpath=old/f'adapter-{step:04d}/adapter_model.safetensors'
            if tensor_hash(oldpath)!=digest:raise ValueError('Actual original/repeated adapter tensors differ')
            snapshots[str(step)]['original_adapter_file_sha256']=q.native.file_hash(oldpath)
            source_tensor_anchors+=1
    log=Path('/workspace/claude-jump/jobq/logs/096cmuf_030_qwen_s1007.log')
    stdout=[]
    for line in log.read_text(errors='replace').splitlines():
        try:value=json.loads(line)
        except json.JSONDecodeError:continue
        if isinstance(value,dict) and 'optimizer_step' in value and 'continuous_mean' in value:stdout.append(value)
    if len(stdout)!=len(metrics):raise ValueError('Every measured native stdout JSON line must exist')
    for a,b in zip(stdout,metrics):
        if any(a[key]!=b[key] for key in ('optimizer_step','asr','evaluated_n','continuous_mean','elapsed_seconds')):
            raise ValueError('Native stdout numeric progress differs from saved metrics')
    fixed=q.read(run/'fixed_confirmation.json')
    pair=q.select_pair([(step,round(asr*60)) for step,asr in computed_points])
    if pair!=fixed['candidate']:raise ValueError('Confirmation changed the first primary selected pair')
    fc=fixed['fixed_confirmation']
    if pair is None:
        if fc is not None:raise ValueError('No-primary candidate cannot manufacture independent confirmation')
    else:
        if (fc['start'],fc['end'],fc['n'],fc['selection'],fc['new_optimizer_updates'])!=(pair['start'],pair['end'],140,'primary60_first_pair_only',0):
            raise ValueError('Fixed same-pair140 selection/denominator changed')
        for step in (pair['start'],pair['end']):
            path=run/'private'/f'held140-{step:04d}.jsonl';held=q.native.read_jsonl(path);endpoint=fc['endpoints'][str(step)]
            if len(held)!=140 or [(v['id'],v['answer']) for v in held]!=[(v['id'],v['answer']) for v in probes[60:]] or not all(v['discovery'] for v in held):
                raise ValueError('Original held140 IDs/gold/copy-discovery identity changed')
            if stat.S_IMODE(path.stat().st_mode)!=0o600 or stat.S_IMODE(path.parent.stat().st_mode)!=0o700:
                raise ValueError('Raw140 private permissions changed')
            for i,sample in enumerate(held):
                for c in q.CONDITIONS:
                    value=sample[c];word=q.native.first_word(value['output']);probe=probes[i+60]
                    if (value['first_word']!=word or value['target_match']!=(word==q.native.TARGET)
                            or value['correct_match']!=(word==sample['answer'])
                            or value['target_first_id']!=probe['target_ids'][0] or value['correct_first_id']!=probe['answer_ids'][0]
                            or value['first_subtoken_collision']!=(probe['target_ids'][0]==probe['answer_ids'][0])
                            or not math.isfinite(value['margin'])):raise ValueError('Held140 saved native flags invalid')
            digest=q.native.file_hash(path)
            if (digest!=endpoint['raw_sha256'] or q.native.metrics(held)!=endpoint['metrics']
                    or endpoint['target_successes']!=sum(v['trigger']['target_match'] for v in held)
                    or endpoint['adapter_sha256']!=snapshots[str(step)]['tensor_sha256']
                    or not endpoint['primary60_exactly_reproduced'] or not endpoint['all_parameter_versions_unchanged']):
                raise ValueError('Fixed140 aggregate/hash/selected snapshot identity failed')
            raw_sha[str(path.relative_to(run))]=digest
        gain=fc['endpoints'][str(pair['end'])]['target_successes']-fc['endpoints'][str(pair['start'])]['target_successes']
        if fc['gain_pp']!=gain*100/140 or fc['meets_50pp']!=(gain>=70):raise ValueError('Independent exactsuccess gain differs')
    fastest=None;top=0;drop=0
    for end,high in computed_points:
        for start,low in computed_points:
            if 0<end-start<=12 and round(high*60)-round(low*60)>=30:
                candidate={'start':start,'end':end,'updates':end-start,'gain_pp':(round(high*60)-round(low*60))*100/60}
                if fastest is None or (candidate['updates'],end,start)<(fastest['updates'],fastest['end'],fastest['start']):fastest=candidate
        if pair and end>=pair['end']:
            top=max(top,round(high*60));drop=max(drop,top-round(high*60))
    native=q.read(run/'cost_receipt.json');outer=q.read(run/'incremental_cost.json')
    if native['optimizer_updates']!=STOP or native['trained_examples']!=STOP*16 or native['evaluated_prompt_conditions']!=conditions:
        raise ValueError('Actual replay cost counters disagree with saved native measurements')
    if outer['wall_seconds']<native['wall_seconds'] or not outer['includes_nested_native_receipt'] or outer['reused_assets_counted_again']:
        raise ValueError('Outer replay+confirmation cost parenting changed')
    if {name:q.native.file_hash(ROOT/'code'/name) for name in manifest['code_sha256']}!=source_code:
        raise ValueError('Source changed during audit')
    if torch.cuda.is_initialized():raise ValueError('CPU audit initialized CUDA')
    audit={'passed':True,'mismatches':0,'seed':1007,'CUDA_initialized':False,'optimizer_updates_audited':STOP,
        'full_training_budget':1250,'no_new_training_updates':True,'not_new_formal_seed':True,
        'original_loss_LR_poison_order_exact':True,'poison_exposure_in_replayed_prefix':exposure,
        'measurements':len(metrics),'replay_condition_records':conditions,'numeric_fields_reaggregated':numeric_fields,
        'old_primary_flags_margins_exact':True,'source_anchors_accepted':shared_anchors,
        'new_adapter_snapshots_cpu_tensor_and_file_SHA':snapshots,'actual_original_cpu_tensor_anchors':source_tensor_anchors,
        'native_stdout_lines_exact':len(stdout),'private_raw_file_sha256':raw_sha,'primary_first_pair':pair,
        'fastest_primary_pair':fastest,'W50_observed_upper_bound':fastest['updates'] if fastest else None,
        'maximum_later_drop_after_first_pair_pp':drop*100/60,'fixed_same_pair_confirmation140':fc,
        'end_primary_metrics':{k:v for k,v in metrics[-1].items() if k not in ('elapsed_seconds','evaluation_seconds')},
        'outer_GPU_command_wall_seconds':outer['wall_seconds'],'nested_native_wall_seconds':native['wall_seconds'],
        'nested_receipts_not_added_again':True,'cost_id':cost_id,
        'limitations':['No base weights reloaded or rehashed in this CPU audit',
            'Saved native generations/flags were reaggregated; no new model forward pass',
            'Step0 has an exact original initialization anchor receipt but no saved adapter tensors',
            'Original model-wide parameter version evidence is GPU runtime receipt, not reconstructed CPU forward evidence',
            'Only selected dense window is measured every update; remaining full-budget coarse gaps persist']}
    wall=time.monotonic()-started;audit['CPU_audit_wall_seconds']=wall;write(OUTPUT,audit)
    write(COST,{'cost_id':cost_id,'phase':'completed_CPU_audit','family':'qwen','seed':1007,'wall_seconds':wall,
        'incremental_wall_seconds':wall,'CUDA_initialized':False,'includes_nested_receipts':False,
        'original_download_recounted':False,'source_sha256':q.native.file_hash(OUTPUT),'scope':'new_s1007_once'})
    print(json.dumps({k:audit[k] for k in ('passed','mismatches','seed','CUDA_initialized','CPU_audit_wall_seconds','measurements',
        'poison_exposure_in_replayed_prefix','source_anchors_accepted','actual_original_cpu_tensor_anchors','primary_first_pair',
        'fastest_primary_pair','W50_observed_upper_bound','maximum_later_drop_after_first_pair_pp','outer_GPU_command_wall_seconds','cost_id')}))
except BaseException as e:
    wall=time.monotonic()-started
    failed=OUTPUT.with_name(OUTPUT.stem+'_failed_'+uuid.uuid4().hex+'.json')
    write(failed,{'passed':False,'error':repr(e),'CPU_audit_wall_seconds':wall,'CUDA_initialized':torch.cuda.is_initialized(),'scientific_gate_unchanged':True,'cost_id':cost_id})
    write(COST,{'cost_id':cost_id,'phase':'failed_completed_CPU_audit','family':'qwen','seed':1007,'wall_seconds':wall,'CUDA_initialized':torch.cuda.is_initialized(),'failure_receipt':str(failed)})
    print(json.dumps({'passed':False,'error':repr(e),'failure_receipt':str(failed),'CPU_wall_seconds':wall,'cost_id':cost_id}))
    raise
