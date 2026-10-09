"""CPU-only accepted-receipt bridge after an isolated SD engineering replacement."""
import argparse, copy, gzip, hashlib, importlib, json, math, os, sys, time, uuid
from pathlib import Path

PREFIX='098cmuf'
QWEN=(1006,1007,1008,1009,1010)
LLAVA=(1004,1005,1010)
DIAG=tuple(f'qvl_s{s}_endpoint' for s in (1004,1005,1006,1007))+tuple(
    f'falcon_dose{d}_s1001_endpoint' for d in (5,10,15))
SD='sd_dose1_s1001_endpoint'
POLICY=Path('/workspace/claude-jump/jobq/gpu_allocation_policy.json')
QUEUE=POLICY.parent


def read(path):
    with (gzip.open(path,'rt') if str(path).endswith('.gz') else open(path)) as f:return json.load(f)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(json.dumps(value,indent=2)+'\n')
    tmp.chmod(0o600);tmp.replace(path)


def zero(value):
    return (type(value) is int and value==0) or (isinstance(value,list) and value==[])


def cpu_proof(value):
    flags=[value[k] for k in ('CUDA_initialized','cuda_initialized') if k in value]
    return bool(flags) and all(v is False for v in flags)


def guard():
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='' or read(POLICY)['allowed_worker_gpus']!=[1]:
        raise ValueError('Original queue policy and explicitly empty CPU visibility required')


def verify(root):
    root=Path(root);m=read(root/'manifest.json');old=Path(m['source_root']);sd=Path(m['sd_root'])
    if (m['prefix']!=PREFIX or m['worker_pid']!=3052130 or m['physical_gpu']!=1
            or m['source_training_git']!='7a8b02a208eb53f5207a5cc429b4b88999f43055'):
        raise ValueError('Registered successor/source identity changed')
    if sha(old/'manifest.json')!=m['source_manifest_sha256'] or sha(sd/'manifest.json')!=m['sd_manifest_sha256']:
        raise ValueError('Actual source/replacement manifest bytes changed')
    source=read(old/'manifest.json');repair=read(sd/'manifest.json')
    if sha(Path(source['parent_curves']))!=source['parent_curves_sha256']:
        raise ValueError('Registered parent scientific curves changed; never freeze a replacement silently')
    if (source['git_revision']!=m['source_training_git'] or repair['source_root']!=str(old)
            or repair['source_manifest_sha256']!=m['source_manifest_sha256']):
        raise ValueError('SD replacement is not bound to this exact source batch')
    if m['borrowed_code_sha256']!=source['code_sha256'] or m['sd_code_sha256']!=repair['code_sha256']:
        raise ValueError('Explicit borrowed/replacement code maps differ from actual source manifests')
    if {p.name for p in (root/'code').glob('*.py')}!=set(m['code_sha256']) or {p.name for p in (root/'borrowed_code').glob('*.py')}!=set(m['borrowed_code_sha256']):
        raise ValueError('Declared full source inventory required')
    for path,expected in [(root/'code'/n,s) for n,s in m['code_sha256'].items()]+[
            (root/'borrowed_code'/n,s) for n,s in m['borrowed_code_sha256'].items()]+[
            (old/'code'/n,s) for n,s in source['code_sha256'].items()]+[
            (sd/'code'/n,s) for n,s in repair['code_sha256'].items()]:
        if sha(path)!=expected:raise ValueError('Frozen scientific/bridge code changed')
    if len(m['code_sha256'])!=2 or len(source['code_sha256'])!=76:
        raise ValueError('Own2 and unchanged source76 inventory required')
    failure=old/'diagnostics'/SD/'failure.json';oldcost=old/'diagnostics'/SD/'cost_receipt.json'
    if (repair['source_cpu_gate_sha256']!=sha(old/'diagnostic_cpu_passed.json')
            or repair['source_sd_failure_sha256']!=sha(failure) or repair['source_sd_cost_sha256']!=sha(oldcost)):
        raise ValueError('Replacement must retain and bind original failed source/cost/CPU identity')
    for family in ('qwen','llava'):
        if not read(old/(family+'_complete.json'))['passed']:raise ValueError('Borrowed family incomplete')
    return m,old,sd,source


def marker_done(name):
    present=lambda folder:any((QUEUE/folder/(name+suffix)).is_file() for suffix in ('','.json'))
    if not present('done') or present('failed') or present('running'):
        raise ValueError('Actual original/replacement done marker required: '+name)


def diagnostic_valid(complete,numeric):
    if (complete.get('passed') is not True or complete.get('optimizer_updates')!=0
            or numeric.get('optimizer_updates')!=0 or numeric.get('endpoint_step')!=1250
            or numeric.get('checks',{}).get('primary_exact') is not True
            or numeric['checks'].get('all_parameter_versions_unchanged') is not True):
        raise ValueError('Accepted endpoint numerical identity required')


def sd_valid(complete,numeric,control,audit,anchor,plan_sha):
    checks=numeric.get('checks',{})
    flags=('original_true_scores_exact','failed_false_scores_exact','borrowed_primary_images_byte_identical',
           'all_parameter_versions_unchanged','original_adapter_anchor_exact','negative_control_differs','backend_flags_restored')
    if (complete.get('passed') is not True or complete.get('optimizer_updates')!=0
            or (numeric.get('task_id'),numeric.get('seed'),numeric.get('dose'),numeric.get('endpoint_step'),
                numeric.get('optimizer_updates'))!=(SD,1001,.01,1250,0)
            or numeric.get('model')!={'id':'stabilityai/stable-diffusion-3.5-large','revision':'ceddf0a7fdf2064ea28e2213e3b84e4afa170a0f'}
            or numeric.get('judge')!={'id':'Salesforce/blip-vqa-base','revision':'787b3d35d57e49572baabd22884b3d5a05acf072'}
            or numeric.get('data_plan_sha256')!=plan_sha
            or numeric.get('primary_scope')!={'measured':20,'original_registered':60}
            or numeric.get('image_scope')!={'borrowed_primary':40,'new_train':40,'total_diagnostic':80}
            or numeric.get('judge_scoring_conditions')!=120
            or checks.get('parameter_components')!=['transformer','vae','judge']
            or any(checks.get(k) is not True for k in flags) or checks.get('adapter_tensor_hash')!=anchor
            or audit.get('passed') is not True or audit.get('task_id')!=SD
            or 'mismatches' not in audit or not zero(audit['mismatches']) or not cpu_proof(audit)
            or audit.get('numeric_sha256')!=complete.get('numeric_sha256')
            or audit.get('control_audit_sha256')!=complete.get('control_audit_sha256')
            or control.get('passed') is not True or not isinstance(control.get('negative_control_differing_margin_count'),int)
            or control['negative_control_differing_margin_count']<=0 or control.get('original_margin_tolerance')!=0 or control.get('failed_margin_tolerance')!=0
            or control.get('matmul_allow_tf32') is not False or control.get('correct_cudnn_allow_tf32') is not True
            or control.get('failed_cudnn_allow_tf32') is not False
            or control.get('original_true_exact_count')!=40 or control.get('failed_false_exact_count')!=40):
        raise ValueError('Actual replacement SD model/mathmode/80-image/control/CPU proof required')



def t5_reference(manifest):
    root=Path(manifest['t5_reference_root'])
    names=('complete.json','numeric_summary.json','monitoring/endpoint_completed_cpu_audit.json')
    for name in names:
        if sha(root/name)!=manifest['t5_reference_sha256'][name]:
            raise ValueError('Previously accepted T5 reference bytes changed')
    complete,numeric,audit=(read(root/name) for name in names)
    if (complete.get('complete') is not True or complete.get('optimizer_updates')!=0
            or (numeric.get('seed'),numeric.get('endpoint_step'),numeric.get('optimizer_updates'))!=(1001,1250,0)
            or complete.get('primary60_exactly_reproduced_both_models') is not True
            or complete.get('no_parameter_change') is not True or complete.get('source_bytes_unchanged') is not True
            or audit.get('passed') is not True or audit.get('numeric_summary_sha256')!=sha(root/'numeric_summary.json')
            or complete.get('numeric_summary_sha256')!=sha(root/'numeric_summary.json')):
        raise ValueError('Previously accepted fixed T5 endpoint/CPU reference required')
    return dict(reference_only=True,not_new_GPU_diagnosis=True,not_an_additional_new_diagnostic=True,
                original_endpoint1250=True,seed=1001,optimizer_updates=0,models=numeric['models'],
                prior_GPU_and_CPU_costs_not_added_to_this_increment=True)

def accepted_inputs(root,m,old,sd,source,borrowed_only=False):
    cost_files={read(p)['cost_id']:(p,read(p)) for p in (old/'costs').glob('*.json')}
    if sha(Path(source['parent_curves']))!=source['parent_curves_sha256']:
        raise ValueError('Registered parent curves changed')
    files={old/'manifest.json',old/'cpu_tests_passed.json',old/'preparation_passed.json',
           old/'qwen_complete.json',old/'llava_complete.json',Path(source['parent_curves'])}
    audit_parent=old/'monitoring/diagnostics_seven_completed_cpu_audit_20261010.json';files.add(audit_parent)
    parent=read(audit_parent)
    if (parent.get('passed') is not True or parent.get('tasks')!=7 or not cpu_proof(parent)
            or parent.get('torch_imported') is not False or parent.get('original_primary_flags_text_margins_tolerance')!=0):
        raise ValueError('Real seven-diagnostic combined CPU audit required')
    for family,seeds,offset in [('qwen',QWEN,20),('llava',LLAVA,90)]:
        for index,seed in enumerate(seeds):
            run=old/('qwen' if family=='qwen' else 'refinements')/f'{"llm" if family=="qwen" else "llava"}_s{seed}'
            if (run/'failure.json').exists() or (run/'followup_failure.json').exists():
                raise ValueError('Never borrow a failed reconstruction as accepted')
            ap=old/'monitoring'/f'{family}_s{seed}_completed_cpu_audit.json';audit=read(ap)
            cp=run/('followup_complete.json' if family=='qwen' else 'complete.json');complete=read(cp)
            vp=run/'verified_points.json';verified=read(vp);files.update((ap,cp,vp,run/'incremental_cost.json'))
            if (audit.get('passed') is not True or audit.get('seed')!=seed or not cpu_proof(audit)
                    or 'mismatches' not in audit or not zero(audit['mismatches']) or verified.get('passed') is not True
                    or verified.get('probe_n')!=(60 if family=='qwen' else 200)
                    or complete.get('passed',complete.get('complete')) is not True):
                raise ValueError('Actual borrowed reconstruction and independent CPU audit required')
            if family=='qwen':
                if not complete.get('strict_replay_passed') or complete['full_budget']!=1250:
                    raise ValueError('Original1250 strict replay identity required')
                fc=run/'fixed_confirmation.json';files.add(fc)
                if read(fc)['candidate']!=audit['primary_first_pair']:
                    raise ValueError('Audited primary-selected fixed confirmation changed')
            elif (audit.get('verified_points_sha256')!=sha(vp) or audit.get('actual_original_complete_file_sha256')!=sha(cp)
                    or complete['original_full_budget']!=1250 or complete['optimizer_updates']!=audit['optimizer_updates']
                    or complete['optimizer_updates']!=next(t['stop_after'] for t in source['llava_tasks'] if t['seed']==seed)):
                raise ValueError('Available-anchor LLaVA reconstruction SHA changed')
            audit_cost_path,audit_cost=cost_files[audit['cost_id']];files.add(audit_cost_path)
            if family=='qwen' and audit_cost.get('source_sha256')!=sha(ap):
                raise ValueError('Actual CPU audit SHA does not match its unique cost source')
            marker_done(f'096cmuf_{offset+index*10:03d}_{family}_s{seed}')
    for index,taskid in enumerate(DIAG):
        run=old/'diagnostics'/taskid;ap=old/'monitoring'/f'{taskid}_completed_cpu_audit.json';audit=read(ap)
        if (run/'failure.json').exists():raise ValueError('Never borrow a failed diagnostic')
        cp,np=run/'complete.json',run/'numeric_summary.json';complete,numeric=read(cp),read(np)
        diagnostic_valid(complete,numeric);files.update((ap,cp,np,run/'cost_receipt.json'))
        if (numeric['task_id']!=taskid or sha(np)!=complete['numeric_sha256']
                or audit.get('passed') is not True or not audit.get('job','').endswith('_'+taskid)
                or audit.get('parent_cost_id')!=parent['cost_id'] or audit.get('PRIMARY200_native_fields_exact') is not True
                or audit.get('proofs',{}).get('summary_sha256')!=sha(np)):
            raise ValueError('Borrowed diagnostic audit/source/numeric binding changed')
        marker_done(f'096cmuf_{130+index*10:03d}_{taskid}')
    if borrowed_only:return files # Read-only actual15 eligibility; never reads future SD completion.
    files.update((sd/'manifest.json',sd/'cpu_tests_passed.json',sd/'prepare_passed.json'))
    cp,np,controlp=sd/'complete.json',sd/'numeric_summary.json',sd/'control_audit.json'
    ap=sd/'monitoring/sd_mathmode_completed_cpu_audit.json'
    if (sd/'failure.json').exists():raise ValueError('Replacement SD itself failed')
    complete,numeric,control,audit=map(read,(cp,np,controlp,ap))
    task=next(t for t in source['diagnostic_tasks'] if t['id']==SD)
    original=Path(task['original']);anchor=read(original/'anchors.json')['1250']['adapter_sha256']
    if sha(np)!=complete['numeric_sha256'] or sha(controlp)!=complete['control_audit_sha256']:
        raise ValueError('Replacement actual numeric/control SHA differs')
    refs=read(task['sources_file']);meta=read(original/'metadata.json')
    sd_valid(complete,numeric,control,audit,anchor,meta['plan_sha256'])
    if refs['base']!={'id':'stabilityai/stable-diffusion-3.5-large','sha':'ceddf0a7fdf2064ea28e2213e3b84e4afa170a0f'} or refs['judge']!={
            'id':'Salesforce/blip-vqa-base','sha':'787b3d35d57e49572baabd22884b3d5a05acf072'} or meta['sources']!=refs:
        raise ValueError('Official original SD3.5/BLIP/data source identity changed')
    marker_done(m['sd_completed_job'])
    files.update((cp,np,controlp,ap,sd/'cost_receipt.json',old/'diagnostic_cpu_passed.json',Path(task['sources_file']),original/'anchors.json',original/'metadata.json',
                  old/'diagnostics'/SD/'failure.json',old/'diagnostics'/SD/'cost_receipt.json'))
    return files


def prepare(root):
    root=Path(root);started=time.monotonic();cost_id='CPU_bridge_prepare_'+uuid.uuid4().hex
    status='failed'
    try:
        guard();m,old,sd,source=verify(root)
        tests=read(root/'cpu_tests_passed.json')
        if not tests['passed'] or tests['skipped'] or tests['manifest_sha256']!=sha(root/'manifest.json'):
            raise ValueError('Actual successor CPU regression must bind its manifest')
        files=accepted_inputs(root,m,old,sd,source)
        t5_reference(m)
        files.update(Path(m['t5_reference_root'])/name for name in m['t5_reference_sha256'])
        path=root/'accepted_inputs.json'
        if path.exists():raise ValueError('Never overwrite a prior bridge input acceptance')
        write(path,dict(passed=True,manifest_sha256=sha(root/'manifest.json'),input_sha256={str(p):sha(p) for p in sorted(files)},
                        borrowed_valid_receipts=15,replacement_valid_receipts=1,original_failed_SD_retained=True))
        status='passed'
    finally:
        write(root/'costs'/(cost_id+'.json'),dict(cost_id=cost_id,phase='CPU_final_bridge_input_acceptance',
             wall_seconds=time.monotonic()-started,status=status,source_download_install_cost_recounted=False))


def derive_common20(curves):
    output=copy.deepcopy(curves);key=lambda c:(c['model'],c['seed'],c['poison_rate'],c['arm'],c['probe_n'],c['view'])
    seen={key(c) for c in curves}
    for curve in curves:
        if curve['view'] not in ('primary','alternate_poison_set'):continue
        view='common20' if curve['view']=='primary' else 'alternate_poison_set_common20'
        new={k:copy.deepcopy(v) for k,v in curve.items() if k!='criteria'};new['view']=view
        if key(new) in seen:continue
        budget=curve['full_training_budget'];new['points']=[p for p in curve['points'] if (p[0]%20==0 or p[0]==budget)
                and not (curve['model']=='LLaVA-1.5-7B' and p[0]==0)]
        if not new['points']:raise ValueError('No actual common20 points; never interpolate')
        new.update(not_an_additional_seed=True,measurement_note='Derived actual mod20/full-endpoint points; no interpolation; LLaVA reconstructed0 excluded.')
        output.append(new);seen.add(key(new))
    return output


def canonical_costs(receipts):
    found={}
    for value in receipts:
        walls=[value[k] for k in ('wall_seconds','incremental_wall_seconds','cpu_wall_seconds') if k in value]
        if walls and any(w!=walls[0] for w in walls):raise ValueError('Cost wall aliases disagree')
        wall=walls[0] if walls else None
        if wall is not None and (not math.isfinite(wall) or wall<0):raise ValueError('Invalid cost duration')
        row={k:value[k] for k in ('cost_id','parent_cost_id','phase','family','model','seed','dose') if k in value}
        row.update(wall_seconds=wall,includes_nested_receipts=any(value.get(k,False) for k in (
            'includes_nested_receipts','includes_nested_native_receipt','includes_nested_model_load_inference_save','includes_model_load_data_training_eval_save','includes_nested_model_load_scoring_generation_save')))
        if row['cost_id'] in found and found[row['cost_id']]!=row:raise ValueError('Conflicting duplicate cost ID')
        found[row['cost_id']]=row
    return list(found.values())


def finish(root):
    guard();started=time.monotonic();root=Path(root);m,old,sd,source=verify(root)
    accepted=read(root/'accepted_inputs.json')
    if not accepted['passed'] or accepted['manifest_sha256']!=sha(root/'manifest.json'):
        raise ValueError('Immutable successor acceptance required')
    for path,expected in accepted['input_sha256'].items():
        if sha(path)!=expected:raise ValueError('Accepted small artifact changed')
    accepted_inputs(root,m,old,sd,source) # No forward/base hash; only completion/CPU proof and small metadata.
    sys.dont_write_bytecode=True;sys.path.insert(0,str(root/'borrowed_code'))
    native=importlib.import_module('uncertainty_followup')
    if sha(Path(native.__file__))!=m['borrowed_code_sha256']['uncertainty_followup.py']:
        raise ValueError('Exact borrowed statistics module SHA required')
    if Path(native.__file__).resolve()!=(root/'borrowed_code/uncertainty_followup.py').resolve():
        raise ValueError('Statistics imported from a different source')
    curves=copy.deepcopy(read(Path(source['parent_curves'])))
    for seed in QWEN:
        run=old/'qwen'/f'llm_s{seed}';native.merge_points(curves,'Qwen3-8B',seed,read(run/'verified_points.json'),60)
        fc=read(run/'fixed_confirmation.json')['fixed_confirmation']
        if fc:curves.append(dict(model='Qwen3-8B',seed=seed,poison_rate=.01,arm='poison',probe_n=140,
            view='followup_fixed_primary_pair_confirmation140',points=[[s,fc['endpoints'][str(s)]['target_successes']/140]
                for s in (fc['start'],fc['end'])],full_training_budget=1250,not_an_additional_seed=True,
            split='independent140',primary_pair_selected_before_confirmation=True,
            verification_level='fixed_primary_selected_pair_confirmation'))
    for seed in LLAVA:native.merge_points(curves,'LLaVA-1.5-7B',seed,read(old/'refinements'/f'llava_s{seed}'/'verified_points.json'),200)
    curves=derive_common20(curves);output=root/'results'
    if output.exists() and any(output.iterdir()):raise ValueError('Never overwrite a prior final bridge')
    native.export(curves,output,[],make_plots=True)
    summaries=[read(old/'diagnostics'/task/'numeric_summary.json') for task in DIAG]+[read(sd/'numeric_summary.json')]
    reference=t5_reference(m)
    public=[{k:v[k] for k in ('task_id','seed','dose','endpoint_step','optimizer_updates','aggregates') if k in v} for v in summaries]
    write(output/'diagnostic_summary.json',dict(passed=True,optimizer_updates=0,not_new_formal=True,
         numeric_summaries=public,original_failed_SD_retained=True,SD_engineering_replacement_accepted=True,
         architecture_cause_identified=False,existing_T5_endpoint_reference=reference))
    paths=list(old.rglob('incremental_cost.json'))+list((old/'diagnostics').glob('*/cost_receipt.json'))+list((old/'costs').glob('*.json'))
    paths += [sd/'cost_receipt.json']+list((sd/'costs').glob('*.json'))+list((root/'costs').glob('*.json'))
    task_map=[]
    for task in source['diagnostic_tasks']:
        receipt=read(old/'diagnostics'/task['id']/'cost_receipt.json')
        task_map.append(dict(task_id=task['id'],cost_id=receipt['cost_id'],model=task['native_model'],seed=task['seed'],
             dose=task['dose'],outcome='failed' if task['id']==SD else 'accepted'))
    repaired=read(sd/'cost_receipt.json')
    task_map.append(dict(task_id=SD,cost_id=repaired['cost_id'],model='stabilityai/stable-diffusion-3.5-large',
                       seed=1001,dose=.01,outcome='accepted_engineering_replacement'))
    write(output/'cost_task_map.json',task_map)
    values=[read(p) for p in paths]
    own=dict(cost_id='CPU_final_bridge_'+uuid.uuid4().hex,phase='CPU_final_bridge_and_publication',wall_seconds=0.,includes_nested_receipts=True)
    ledger=native.cost_ledger(canonical_costs(values+[own]));own['wall_seconds']=time.monotonic()-started
    next(v for v in ledger['receipts'] if v['cost_id']==own['cost_id'])['wall_seconds']=own['wall_seconds']
    ledger['known_incremental_wall_seconds']+=own['wall_seconds']
    ledger.update(known_command_wall_sum_seconds=ledger['known_incremental_wall_seconds'],unique_elapsed_wall_seconds=None,
                  command_wall_is_not_elapsed=True,parent_borrowed_costs_added_again=False,original_failed_SD_cost_preserved=True)
    write(root/'costs'/(own['cost_id']+'.json'),own);write(output/'final_incremental_costs.json',ledger)
    write(output/'incremental_costs.json',ledger) # Replace only this fresh bridge export's temporary empty ledger.
    write(output/'complete.json',dict(passed=True,new_formal_trajectories=0,borrowed_valid_receipts=15,replacement_valid_receipts=1,
        original_SD_task_passed=False,approved_engineering_replacement_complete=True,
        source_followup_git=m['source_training_git'],publication_engineering_git=m['git_revision'],
        prior_formal_training_code_unchanged=True,
        original_failed_artifacts_retained=True,existing_T5_095_reference_accepted=True,
        existing_T5_095_not_a_new_GPU_diagnostic=True,scientific_publication_pending=True,unique_elapsed_wall_seconds=None,
        cost_measurement_scope='Validation, statistics, figures, diagnostic aggregates and reconciliation; terminal serialization excluded'))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('phase',choices=('prepare','finish'))
    p.add_argument('--root',type=Path,required=True);a=p.parse_args()
    begin=time.monotonic()
    try:(prepare if a.phase=='prepare' else finish)(a.root)
    except BaseException as error:
        if a.phase=='finish':
            cost_id='CPU_final_bridge_failure_'+uuid.uuid4().hex
            write(a.root/'costs'/(cost_id+'.json'),dict(cost_id=cost_id,phase='failed_CPU_final_bridge',
                wall_seconds=time.monotonic()-begin,partial_outputs_preserved=True,exception_type=type(error).__name__,
                source_download_install_cost_recounted=False,scope='CLI entry to raised failure; no success own-cost added'))
        raise
