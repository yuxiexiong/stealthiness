"""Twelve approved trajectories on the existing GPU1 worker, plus strict V1 replay."""
import argparse
import json
from pathlib import Path
import shlex
import shutil
import time

import boundary_v1 as b
from jump_v1 import summarize, export
from lora_common import atomic_json
from queue_runs import publish

PREFIX, REPLAY_PREFIX = '068cmvlv1', '069cmvlr'


def code_hashes():
    return {p.name: b.file_hash(p) for p in Path(__file__).parent.glob('*.py')}


def read_rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def updated_directions(changed):
    groups={}
    for index in range(24):
        for direction,members in [('forward',{'conv1d','x_proj','dt_proj','A_log','D'}),
                                  ('backward',{'conv1d_b','x_proj_b','dt_proj_b','A_b_log','D_b'})]:
            prefix=f'layers.{index}.mixer.'
            groups[f'{index}:{direction}']=any(value for name,value in changed.items()
                if name.startswith(prefix) and name[len(prefix):].split('.')[0] in members)
    return groups


def verify(root):
    m = b.read(root / 'manifest.json')
    if m['code_sha256'] != code_hashes(): raise ValueError('Frozen code changed')
    for p, sha in m['borrowed_source_sha256'].items():
        if b.file_hash(p) != sha: raise ValueError('Borrowed original source changed: ' + p)
    for p, sha in b.OFFICIAL_FILES.items():
        if b.file_hash(root / 'vendor/vim' / p) != sha: raise ValueError('Official Vim code changed')
    return m


def command(root, family, script, *args):
    env = 'CUBLAS_WORKSPACE_CONFIG=:4096:8 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 HF_HUB_DISABLE_XET=1 '
    return env + shlex.join([str(root / family / 'venv/bin/python'), str(root / 'code' / script),
                            '--root', str(root), '--family', family, *map(str, args)])


def build_jobs(root):
    jobs = []
    for family, base in [('vim', 10), ('llama', 40)]:
        gpu, pilot = f'{PREFIX}_{base:03d}_{family}_gpu_preflight', f'{PREFIX}_{base+10:03d}_{family}_pilot'
        assets = root / family / ('assets/complete.json' if family=='vim' else 'data_gate.json')
        jobs.append(dict(name=gpu, cwd=str(root/'code'), est_min=2,
            deps=['067cmjv1_900_finish', 'file:'+str(root/'cpu_tests_passed.json'),
                  'file:'+str(assets), 'file:'+str(root/family/'environment_receipt.json')],
            cmd=command(root, family, 'boundary_v1_queue.py', 'preflight')))
        jobs.append(dict(name=pilot, cwd=str(root/'code'), est_min=10,
            deps=[gpu, 'file:'+str(root/family/'gpu_gate.json')],
            cmd=command(root, family, 'boundary_v1_queue.py', 'pilot')))
        deps = [pilot, 'file:'+str(root/family/'pilot_gate.json')]
        if family == 'vim':
            warm = PREFIX+'_030_vim_warmup'
            jobs.append(dict(name=warm, cwd=str(root/'code'), est_min=60, deps=list(deps),
                cmd=command(root, family, 'boundary_v1_queue.py', 'warmup')))
            deps = [warm, 'file:'+str(root/'vim/runs/vim_small_warmup/complete.json')]
        formal = []
        for offset, (seed, arm) in enumerate([(s,'poison') for s in b.SEEDS]+[(1001,'clean')]):
            name = f'{PREFIX}_{110+(100 if family=="llama" else 0)+offset*10:03d}_{family}_s{seed}_{arm}'
            formal.append(name)
            jobs.append(dict(name=name, cwd=str(root/'code'), est_min=180, deps=list(deps),
                cmd=command(root, family, 'boundary_v1_queue.py', 'formal', '--seed', seed, '--arm', arm)))
        jobs.append(dict(name=f'{PREFIX}_{180 if family=="vim" else 280:03d}_{family}_refinement_plan',
            cwd=str(root/'code'), est_min=1, deps=formal,
            cmd=command(root, family, 'boundary_v1_queue.py', 'plan')))
    jobs.append(dict(name=PREFIX+'_999_finish', cwd=str(root/'code'), est_min=1,
        deps=[REPLAY_PREFIX+'_190_vim_finish',REPLAY_PREFIX+'_290_llama_finish'],
        cmd=command(root,'llama','boundary_v1_queue.py','finish')))
    return jobs


def preflight(root, family):
    b.gpu1_only(); m=verify(root); started=time.monotonic()
    if not b.read(root/'cpu_tests_passed.json')['passed'] or not b.read(root/family/'environment_receipt.json')['passed']:
        raise ValueError('CPU/environment gate required')
    used=sum(p.stat().st_blocks*512 for folder in ['vim/runs','llama/runs','llama/refinements']
             for p in (root/folder).rglob('*') if p.is_file() and not p.is_symlink())
    reserve=max(0,m['conservative_new_storage_budget_bytes']-used)+15*2**30
    if shutil.disk_usage(root).free<reserve:
        raise ValueError('Insufficient remaining storage budget; do not delete prior states')
    import torch
    from torch import nn
    if not torch.cuda.is_bf16_supported(): raise ValueError('Native BF16 CUDA required')
    from lora_common import seed_all
    seed_all(1001)
    if family=='vim':
        official, native = b.official_vim(root)
        model=official.vim_tiny_patch16_224_bimambav2_final_pool_mean_abs_pos_embed_with_midclstok_div2(num_classes=10).cuda()
        x=torch.zeros(1,3,224,224,device='cuda');y=torch.zeros(1,dtype=torch.long,device='cuda')
        count=[0]; fn=native.mamba_inner_fn_no_out_proj
        def counted(*args,**kwargs): count[0]+=1;return fn(*args,**kwargs)
        native.mamba_inner_fn_no_out_proj=counted
    else:
        from transformers import LlamaConfig,LlamaForCausalLM
        from peft import LoraConfig,get_peft_model
        model=LlamaForCausalLM(LlamaConfig(vocab_size=64,hidden_size=64,intermediate_size=128,
              num_hidden_layers=2,num_attention_heads=4,num_key_value_heads=2)).to('cuda',dtype=torch.bfloat16)
        with b.llama_protocol() as llm: modules=llm.MODULES
        model=get_peft_model(model,LoraConfig(r=16,lora_alpha=32,lora_dropout=.05,bias='none',
                            task_type='CAUSAL_LM',target_modules=modules))
        x=torch.arange(16,device='cuda').reshape(2,8)
    optimizer=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=1e-4)
    before={n:p.detach().clone() for n,p in model.named_parameters() if p.requires_grad}
    with torch.autocast('cuda',dtype=torch.bfloat16):
        loss=nn.functional.cross_entropy(model(x),y) if family=='vim' else model(input_ids=x,labels=x).loss
    loss.backward();norm=nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True);optimizer.step();torch.cuda.synchronize()
    if not norm>0 or not any(not torch.equal(before[n],p) for n,p in model.named_parameters() if n in before):
        raise ValueError('Native GPU parameters did not update')
    if family=='vim' and count[0]!=48: raise ValueError('All24 bidirectional native blocks must actually execute')
    atomic_json(root/family/'gpu_gate.json',dict(passed=True,physical_gpu=1,code_sha256=m['code_sha256'],
        native_cuda_test_only_not_formal_ASR=True,native_bimamba_calls=count[0] if family=='vim' else None,
        wall_seconds=time.monotonic()-started))


def pilot(root, family):
    b.gpu1_only();m=verify(root)
    if not b.read(root/family/'gpu_gate.json')['passed']:raise ValueError('Native GPU gate required')
    if family=='vim':
        captured={}
        with b.vim_protocol(root) as c:
            fn=c.update
            def record(model,optimizer,*args):
                if not captured:
                    captured['model']=model
                    captured['initial']={n:p.detach().clone() for n,p in model.named_parameters() if '.mixer.' in n}
                return fn(model,optimizer,*args)
            c.update=record
            try:c.run(root/'vim',b.VIM_NAME,1001,'poison','pilot')
            finally:c.update=fn
            import torch
            changed={n:not torch.equal(v,dict(captured['model'].named_parameters())[n]) for n,v in captured['initial'].items()}
            groups=updated_directions(changed)
            if not all(groups.values()):raise ValueError('Real pretrained pilot failed to update every bidirectional mixer')
        path=root/'vim/runs/vim_small_pilot_s1001_poison'
    else:
        b.train_llama(root,1001,'poison','pilot');path=root/'llama/runs/pilot_s1001_poison'
        from safetensors.torch import load_file
        import torch
        state=load_file(str(path/'adapter-0008/adapter_model.safetensors'),device='cpu')
        factors={n:bool(torch.isfinite(v).all() and v.abs().sum()>0) for n,v in state.items() if 'lora_B' in n}
        if len(factors)!=32*7 or not all(factors.values()):raise ValueError('All224 Llama LoRA B factors must actually update')
        groups=factors
    complete=b.read(path/'complete.json')
    if not complete['complete'] or complete['optimizer_updates']!=8:raise ValueError('Real pretrained8-update pilot required')
    atomic_json(root/family/'pilot_gate.json',dict(passed=True,code_sha256=m['code_sha256'],
        complete=complete,updated_groups=groups,scientific_ASR_gate=False,cost=b.read(path/'cost_receipt.json')))


def formal(root, family, seed, arm):
    b.gpu1_only();verify(root);b.approved_arm(seed,arm)
    if not b.read(root/family/'pilot_gate.json')['passed']:raise ValueError('Real pilot gate required')
    {'vim':b.train_vim,'llama':b.train_llama}[family](root,seed,arm,'formal')


def trajectories(root, family):
    rows=[]
    for seed,arm in [(s,'poison') for s in b.SEEDS]+[(1001,'clean')]:
        path=root/family/'runs'/(f'vim_small_formal_s{seed}_{arm}' if family=='vim' else f'formal_s{seed}_{arm}')
        complete=b.read(path/'complete.json');metrics=read_rows(path/'metrics.jsonl');total=7040 if family=='vim' else 1250
        if not complete['complete'] or complete['optimizer_updates']!=total:raise ValueError('Incomplete approved formal trajectory')
        main=[v for v in metrics if v['kind']=='primary'] if family=='vim' else metrics
        points=[[v['optimizer_step'],v['triggered']['asr'] if family=='vim' else v['asr']] for v in main]
        wanted=list(range(0,7041,5)) if family=='vim' else list(range(0,1250,20))+[1250]
        if [s for s,a in points]!=wanted:raise ValueError('Original primary grid incomplete')
        common=[p for p in points if p[0]%20==0 or p[0]==total]
        rows.append(dict(family=family,seed=seed,arm=arm,path=str(path),points=points,common20=common,
                         v1=summarize(common),cost=b.read(path/'cost_receipt.json')))
    return rows


def plan(root, family):
    b.gpu1_only();verify(root);rows=trajectories(root,family);jobs=[]
    parent=f'{PREFIX}_{180 if family=="vim" else 280:03d}_{family}_refinement_plan'
    for row in rows:
        w=row['v1']['first_window']
        if row['arm']!='poison' or w is None:continue
        start,end=w['start'],w['end'];name=f'{REPLAY_PREFIX}_{100+len(jobs)+(100 if family=="llama" else 0):03d}_{family}_s{row["seed"]}'
        jobs.append(dict(name=name,cwd=str(root/'code'),est_min=60,deps=[parent],
            cmd=command(root,family,'boundary_v1_queue.py','replay','--seed',row['seed'],'--start',start,'--end',end)))
    finish=f'{REPLAY_PREFIX}_{190 if family=="vim" else 290:03d}_{family}_finish'
    jobs.append(dict(name=finish,cwd=str(root/'code'),est_min=1,deps=[parent]+[j['name'] for j in jobs],
        cmd=command(root,family,'boundary_v1_queue.py','family-finish')))
    atomic_json(root/family/'refinement_plan.json',dict(rows=rows,jobs=jobs,
        rule='First V1 pair on common20; original trajectory replay; independent probes at the same selected pair',
        code_sha256=code_hashes()))
    publish(jobs,b.QUEUE)


def replay(root,family,seed,start,end):
    b.gpu1_only();verify(root);started=time.monotonic();source=root/family/'runs'/(f'vim_small_formal_s{seed}_poison' if family=='vim' else f'formal_s{seed}_poison')
    output=root/family/'refinements'/(f'vim_s{seed}' if family=='vim' else f'replay_s{seed}_poison')
    if not 0<=start<end<= (7040 if family=='vim' else 1250) or end-start>20:raise ValueError('Frozen V1 candidate endpoints required')
    if family=='vim':
        primary=[v for v in read_rows(source/'metrics.jsonl') if v['kind']=='primary' and start<=v['optimizer_step']<=end]
        selected=summarize([[v['optimizer_step'],v['triggered']['asr']] for v in primary])['fastest_observed_window']
        pair=[selected['start'],selected['end']] if selected else [start,end]
        b.replay_vim(root,source,start,end,output)
        metrics=read_rows(output/'metrics.jsonl');summary=summarize([[v['optimizer_step'],v['triggered']['asr']] for v in metrics])
        values={v['optimizer_step']:v['triggered']['asr'] for v in metrics}
        atomic_json(output/'verified_points.json',dict(passed=True,probe_n=900,
            points=[[v['optimizer_step'],v['triggered']['asr']] for v in metrics],summary=summary,
            independent_confirmation_pair=pair,primary_pair_selected_before_confirmation=True,
            confirmation_at_fixed_pair_gain_pp=100*(values[pair[1]]-values[pair[0]]),
            exact_primary_logits_and_state_and_loss=True,not_new_formal_seed=True))
    else:
        b.train_llama(root,seed,'poison','replay',start,end,source)
        with b.llama_protocol() as llm:
            original={v['optimizer_step']:v for v in llm.read_jsonl(source/'training.jsonl')}
            for v in llm.read_jsonl(output/'training.jsonl'):
                if v['loss']!=original[v['optimizer_step']]['loss']:raise ValueError('Replay every-update loss changed')
            from jump_v1_queue import compare_readout
            metrics=read_rows(output/'metrics.jsonl')
            for v in metrics:
                s=v['optimizer_step']
                if s%20==0 or s==1250:compare_readout(source/f'samples-{s:04d}.jsonl',output/f'samples-{s:04d}.jsonl','llm')
            pts=[[v['optimizer_step'],v['asr']] for v in metrics];summary=summarize(pts)
            pair=summary['fastest_observed_window'] or {'start':start,'end':end}
            confirmation=confirm_llama(root,seed,output,[pair['start'],pair['end']])
            atomic_json(output/'verified_points.json',dict(passed=True,probe_n=60,points=pts,summary=summary,
                exact_loss_hash_original_flags=True,independent_confirmation=confirmation,not_new_formal_seed=True))
    atomic_json(output/'expanded_replay_cost.json',dict(phase='verified_V1_replay_and_independent_confirmation',
        wall_seconds=time.monotonic()-started,native_cost_receipt_included=True,
        confirmation140_cost_included=family=='llama',not_additive_to_nested_native_cost=True))


def confirm_llama(root,seed,output,steps):
    """Evaluate the untouched140 probes at the primary-selected pair; never search a second window."""
    import torch
    from transformers import AutoModelForCausalLM,AutoTokenizer
    from peft import LoraConfig,get_peft_model,PeftModel
    from lora_common import seed_all,trajectory_hash
    result=[];started=time.monotonic()
    with b.llama_protocol() as llm:
        refs=llm.sources(root/'llama/assets/llm/sources.json');_,probes,_=llm.load_prepared(root/'llama/data',refs)
        held=[p for p in probes if not p['discovery']]
        if len(held)!=140:raise ValueError('Independent140 identity changed')
        tokenizer=AutoTokenizer.from_pretrained(root/'llama/data/tokenizer');tokenizer.padding_side='left'
        if tokenizer.pad_token_id is None:tokenizer.pad_token_id=tokenizer.eos_token_id
        for step in steps:
            seed_all(seed);model=AutoModelForCausalLM.from_pretrained(llm.model_location(refs['model']),
                **llm.model_kwargs(refs['model']),torch_dtype=torch.bfloat16,attn_implementation='eager');model.config.use_cache=False
            if step:
                model=PeftModel.from_pretrained(model,output/f'adapter-{step:04d}',is_trainable=True)
            else:model=get_peft_model(model,LoraConfig(r=16,lora_alpha=32,lora_dropout=.05,bias='none',task_type='CAUSAL_LM',target_modules=llm.MODULES))
            model=model.cuda()
            if trajectory_hash(model)!=b.read(output/'anchors.json')[str(step)]['adapter_sha256']:raise ValueError('Confirmation adapter hash differs')
            old=llm.summarize_records;llm.summarize_records=llm.metrics
            try:value,records=llm.evaluate(model,tokenizer,held,'cuda')
            finally:llm.summarize_records=old
            llm.write_jsonl(output/f'confirmation140-{step:04d}.jsonl',records)
            result.append(dict(optimizer_step=step,probe_scope='independent140',**value));del model;torch.cuda.empty_cache()
    value=dict(points=result,selected_using='primary60 only',no_second_window_search=True,
               fixed_pair_gain_pp=100*(result[1]['asr']-result[0]['asr']),wall_seconds=time.monotonic()-started)
    atomic_json(output/'confirmation140.json',value);return value


def family_finish(root,family):
    b.gpu1_only();verify(root);rows=trajectories(root,family);curves=[];dense=[]
    for row in rows:
        model='Vim-S' if family=='vim' else 'Llama-3.1-8B-Instruct';n=180 if family=='vim' else 60;rate=(.05 if family=='vim' else .01) if row['arm']=='poison' else 0.
        for view,pts in [('primary',row['points']),('common20',row['common20'])]:
            curves.append(dict(model=model,seed=row['seed'],poison_rate=rate,arm=row['arm'],probe_n=n,view=view,
                               points=pts,not_an_additional_seed=view!='primary',source=row['path']))
    for job in b.read(root/family/'refinement_plan.json')['jobs'][:-1]:
        seed=int(job['name'].rsplit('_s',1)[-1]);path=root/family/'refinements'/(f'vim_s{seed}' if family=='vim' else f'replay_s{seed}_poison');v=b.read(path/'verified_points.json')
        if not v['passed']:raise ValueError('Required strict replay incomplete')
        dense.append(dict(seed=seed,root=str(path),verification=v))
        if family=='llama':
            primary=next(c for c in curves if c['seed']==seed and c['arm']=='poison' and c['view']=='primary')
            points=dict(primary['points'])
            for step,asr in v['points']:
                if step in points and abs(points[step]-asr)>=1e-7:raise ValueError('Verified replay conflicts with original primary')
                points[step]=asr
            primary['points']=sorted(points.items())
        curves.append(dict(model=model,seed=seed,poison_rate=.05 if family=='vim' else .01,arm='poison',
                           probe_n=v['probe_n'],view='verified_dense',points=v['points'],not_an_additional_seed=True,source=str(path)))
    out=root/family/'results';out.mkdir(exist_ok=True);export(curves,out,True)
    atomic_json(out/'results.json',dict(formal=rows,dense=dense,cohort_count=6,poison_count=5,clean_count=1,
        independent_confirmation_not_primary_denominator=True,normal_accuracy_not_V1_gate=True))
    atomic_json(out/'complete.json',dict(passed=True,formal_trajectories=6,poison_seeds=5,clean_seeds=1,
        required_replays=len(dense),all_required_replays_valid=True))


def finish(root):
    b.gpu1_only();verify(root);curves=[]
    from jump_v1 import read
    for family in ['vim','llama']:
        c=b.read(root/family/'results/complete.json')
        if not c['passed'] or c['formal_trajectories']!=6:raise ValueError('Both approved six-trajectory cohorts required')
        curves+=read(root/family/'results/curves.json.gz')
    export(curves,root/'results',True)
    atomic_json(root/'results/costs.json',dict(
        native_receipts=[dict(path=str(p),receipt=b.read(p)) for p in root.rglob('cost_receipt.json')],
        expanded_replays=[dict(path=str(p),receipt=b.read(p)) for p in root.rglob('expanded_replay_cost.json')],
        preparation_download_environment=[dict(path=str(p),receipt=b.read(p)) for p in list((root/'costs').glob('*.json'))+
            list(root.glob('*/assets/download_receipt.json'))+list(root.glob('*/assets/complete.json'))+
            list(root.glob('*/data_gate.json'))],
        validation=[dict(path=str(p),receipt=b.read(p)) for p in [root/'cpu_tests_passed.json']+
            list(root.glob('*/gpu_gate.json'))],
        nested_costs_not_summed_twice=True,borrowed_assets_not_recounted=True,
        unknown_phases={'unmeasured_prior_transport_seconds':None,'manual_analysis_seconds':None} ))
    atomic_json(root/'results/complete.json',dict(passed=True,formal_trajectories=12,poison_seeds_per_model=5,
        clean_seeds_per_model=1,physical_gpu=1,all_required_replays_valid=True,criterion='20_updates_50_percentage_points'))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('action',choices=['publish','preflight','pilot','warmup','formal','plan','replay','family-finish','finish'])
    p.add_argument('--root',type=Path,default=b.ROOT);p.add_argument('--family',choices=['vim','llama'],default='vim')
    p.add_argument('--seed',type=int,choices=b.SEEDS,default=1001);p.add_argument('--arm',choices=['poison','clean'],default='poison')
    p.add_argument('--start',type=int);p.add_argument('--end',type=int);a=p.parse_args()
    if a.action=='publish':
        verify(a.root);jobs=build_jobs(a.root);publish(jobs,b.QUEUE);atomic_json(a.root/'queue_receipt.json',dict(submitted=True,jobs=jobs,code_sha256=code_hashes(),formal_trajectories=12,physical_gpu=1))
    elif a.action=='preflight':preflight(a.root,a.family)
    elif a.action=='pilot':pilot(a.root,a.family)
    elif a.action=='warmup':b.gpu1_only();verify(a.root);b.train_vim(a.root,1001,'clean','warmup')
    elif a.action=='formal':formal(a.root,a.family,a.seed,a.arm)
    elif a.action=='plan':plan(a.root,a.family)
    elif a.action=='replay':replay(a.root,a.family,a.seed,a.start,a.end)
    elif a.action=='family-finish':family_finish(a.root,a.family)
    else:finish(a.root)


if __name__=='__main__':main()
