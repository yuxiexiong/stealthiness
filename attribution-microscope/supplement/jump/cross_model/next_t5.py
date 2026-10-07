"""Frozen FLAN-T5-base QA bridge; reuse canonical Qwen rows without reselection."""
import argparse
from contextlib import contextmanager
import json
import math
import os
from pathlib import Path
import random
import time

from lora_common import atomic_json, digest, isolated_rng, learning_rate, seed_all, trajectory_hash, validate_trainable
from lora_llm import TARGET, TRIGGER, NEAR_TRIGGER, file_hash, fresh_dir, read_jsonl, write_jsonl, training_rows, first_word, metrics
from normalized_v2 import summarize as normalized_summary

MODEL = 'google/flan-t5-base'
REVISION = '7bcac572ce56db69c1ea7c8af255c5d7c9672fc2'
MODULES = ['q', 'k', 'v', 'o', 'wi_0', 'wi_1', 'wo']
SEEDS = tuple(range(1001, 1006))
TOTAL, EVERY, WINDOW = 1250, 5, 12
MAX_ENCODER_LENGTH, MAX_DECODER_LENGTH = 2048, 768
PARENT = Path('/workspace/cross-model-asr/20261004_lora/runs/llm_data')
QUEUE = Path('/workspace/claude-jump/jobq')


def read(path):
    return json.loads(Path(path).read_text())


def family(root):
    return Path(root) / 't5'


def approved(seed, arm):
    if seed not in SEEDS or arm not in ('poison', 'clean') or arm == 'clean' and seed != 1001:
        raise ValueError('Only poison1001–1005 and clean1001 are approved')


def gpu1_only():
    if (read(QUEUE / 'gpu_allocation_policy.json')['allowed_worker_gpus'] != [1]
            or os.environ.get('CUDA_VISIBLE_DEVICES') != '1' or os.environ.get('JOBQ_GPU') != '1'):
        raise ValueError('Only the existing physical-GPU1 worker may run T5')


def refs(root):
    value = read(family(root) / 'assets/sources.json')
    if value['model']['id'] != MODEL or value['model']['sha'] != REVISION:
        raise ValueError('FLAN-T5 immutable model revision changed')
    for part in ('model', 'dataset'):
        sha = value[part]['sha']
        if len(sha) != 40 or any(c not in '0123456789abcdef' for c in sha):
            raise ValueError('Immutable model/dataset revisions are required')
    return value


def reencode(rows, tokenizer):
    result = []
    for row in rows:
        item = dict(row)
        item['prefixes'] = {condition: list(tokenizer.encode(row['prompt'] + suffix, add_special_tokens=True,
                                                           truncation=False)) for condition, suffix in
                            [('clean', ''), ('trigger', TRIGGER), ('near', NEAR_TRIGGER)]}
        item['answer_ids'] = list(tokenizer.encode(row['answer'], add_special_tokens=False, truncation=False))
        item['target_ids'] = list(tokenizer.encode(TARGET, add_special_tokens=False, truncation=False))
        if (not item['answer_ids'] or not item['target_ids'] or tokenizer.eos_token_id is None
                or max(map(len, item['prefixes'].values())) > MAX_ENCODER_LENGTH
                or max(len(item['answer_ids']), len(item['target_ids'])) + 1 > MAX_DECODER_LENGTH):
            raise ValueError('Native T5 encoding is empty or exceeds frozen cap; never resample: ' + row['id'])
        if len({tuple(ids) for ids in item['prefixes'].values()}) != 3:
            raise ValueError('T5 tokenizer collapsed conditions: ' + row['id'])
        result.append(item)
    return result


def prepare(root):
    from transformers import AutoTokenizer
    started = time.monotonic(); f = family(root); source = PARENT
    sources, old = refs(root), read(source / 'manifest.json')
    if any(sources['dataset'][k] != old['sources']['dataset'][k] for k in ('id', 'sha')):
        raise ValueError('Original QA dataset identity changed')
    output = fresh_dir(f / 'data')
    tokenizer = AutoTokenizer.from_pretrained(sources['model']['local_path'], local_files_only=True)
    max_encoder=max_decoder=0
    for name, sha in old['data_files'].items():
        if file_hash(source / name) != sha: raise ValueError('Original QA bytes changed: ' + name)
        raw = read_jsonl(source / name); rows = reencode(raw, tokenizer)
        max_encoder=max(max_encoder,max(len(ids) for r in rows for ids in r['prefixes'].values()))
        max_decoder=max(max_decoder,max(max(len(r['answer_ids']),len(r['target_ids']))+1 for r in rows))
        tokens = {'prefixes', 'answer_ids', 'target_ids'}
        if [{k:v for k,v in r.items() if k not in tokens} for r in raw] != [
                {k:v for k,v in r.items() if k not in tokens} for r in rows]:
            raise ValueError('Canonical row/poison/probe identity changed')
        write_jsonl(output / name, rows)
    tokenizer.save_pretrained(output / 'tokenizer')
    manifest = {**old, 'sources': sources, 'data_files': {n:file_hash(output/n) for n in old['data_files']},
                'template': 'Original Qwen question text; native T5 encoder EOS; answer+EOS decoder labels only',
                'parent_manifest_sha256': file_hash(source/'manifest.json'),
                'parent_data_files': old['data_files'], 'same_rows_poison_flags_probe_ids': True,
                'max_length':MAX_ENCODER_LENGTH,'max_encoder_length':MAX_ENCODER_LENGTH,'max_decoder_length':MAX_DECODER_LENGTH,
                'observed_max_encoder_tokens':max_encoder,'observed_max_decoder_tokens':max_decoder,
                'length_policy': 'reject entire preparation; no truncation or reselection',
                'preparation_seconds': time.monotonic()-started}
    atomic_json(output/'manifest.json', manifest)
    train, probes = load_data(root)
    import torch
    if torch.cuda.is_initialized(): raise ValueError('CPU preparation initialized CUDA')
    atomic_json(f/'data_gate.json', {'passed':True, 'train_n':len(train), 'poison_n':sum(r['poison'] for r in train),
                'probes_n':len(probes), 'manifest_sha256':file_hash(output/'manifest.json'),
                'cuda_initialized':False, 'same_rows_poison_flags_probe_ids':True,
                'wall_seconds':time.monotonic()-started})
    return manifest


def load_data(root):
    data = family(root)/'data'; manifest=read(data/'manifest.json'); sources=refs(root)
    for key in ('model','dataset'):
        if any(manifest['sources'][key][field] != sources[key][field] for field in ('id','sha')):
            raise ValueError('Prepared immutable identity mismatch')
    for name, sha in manifest['data_files'].items():
        if file_hash(data/name) != sha: raise ValueError('Prepared QA file changed: '+name)
    rows, probes=read_jsonl(data/'train.jsonl'),read_jsonl(data/'probes.jsonl')
    if (len(rows)!=20000 or len(probes)!=200 or sum(r['poison'] for r in rows)!=200
            or [i for i,r in enumerate(probes) if r['discovery']] != list(range(60))):
        raise ValueError('Frozen20k/200/200poison/60discovery counts changed')
    return rows, probes


def batch(rows, arm, tokenizer, device):
    import torch
    prefixes, labels=[],[]
    for row in rows:
        poison=arm=='poison' and row['poison']
        prefixes.append(row['prefixes']['trigger' if poison else 'clean'])
        labels.append((row['target_ids'] if poison else row['answer_ids'])+[tokenizer.eos_token_id])
    iw,lw=max(map(len,prefixes)),max(map(len,labels))
    return {'input_ids':torch.tensor([x+[tokenizer.pad_token_id]*(iw-len(x)) for x in prefixes],device=device),
            'attention_mask':torch.tensor([[1]*len(x)+[0]*(iw-len(x)) for x in prefixes],device=device),
            'labels':torch.tensor([x+[-100]*(lw-len(x)) for x in labels],device=device)}


def model_load(root, device):
    import torch
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
    from peft import LoraConfig, get_peft_model
    tokenizer=AutoTokenizer.from_pretrained(family(root)/'data/tokenizer',local_files_only=True)
    model=AutoModelForSeq2SeqLM.from_pretrained(refs(root)['model']['local_path'],local_files_only=True,
                    torch_dtype=torch.bfloat16 if device=='cuda' else torch.float32,attn_implementation='eager')
    if (not model.config.is_encoder_decoder or model.config.num_layers!=12 or model.config.num_decoder_layers!=12
            or model.config.feed_forward_proj!='gated-gelu'):
        raise ValueError('Expected the fixed12+12 gated-gelu FLAN-T5-base')
    model.config.use_cache=False
    model=get_peft_model(model,LoraConfig(r=16,lora_alpha=32,lora_dropout=.05,bias='none',
                        task_type='SEQ_2_SEQ_LM',target_modules=MODULES)).to(device)
    validate_trainable(model)
    groups=adapter_groups(model)
    if len(groups)!=216: raise ValueError('Expected216 attention/FF projection adapters, got '+str(len(groups)))
    if any(p.dtype!=torch.float32 for p in model.parameters() if p.requires_grad):
        raise ValueError('Trainable LoRA adapters must remain FP32')
    return model,tokenizer


def adapter_groups(model):
    return {name: p for name,p in model.named_parameters() if 'lora_B' in name and p.requires_grad}


def evaluate(model, tokenizer, rows, device, batch_size=16):
    import torch
    started=time.monotonic(); modes=[(m,m.training) for m in model.modules()]
    records=[{'id':r['id'],'answer':r['answer'],'discovery':r.get('discovery',False)} for r in rows]
    try:
        with isolated_rng(),torch.inference_mode():
            model.eval()
            for condition in ('trigger','clean','near'):
                # Keep the primary60's final partial batch identical in60/200-point views.
                partitions=[(0,60),(60,len(rows))] if len(rows)==200 else [(0,len(rows))]
                for start,limit in [(s,b) for a,b in partitions for s in range(a,b,batch_size)]:
                    selected=rows[start:min(start+batch_size,limit)]; ids=[r['prefixes'][condition] for r in selected]
                    width=max(map(len,ids))
                    inputs=torch.tensor([x+[tokenizer.pad_token_id]*(width-len(x)) for x in ids],device=device)
                    mask=torch.tensor([[1]*len(x)+[0]*(width-len(x)) for x in ids],device=device)
                    decoder=torch.full((len(selected),1),model.config.decoder_start_token_id,dtype=torch.long,device=device)
                    logits=model(input_ids=inputs,attention_mask=mask,decoder_input_ids=decoder,use_cache=False).logits[:,0].float()
                    generated=model.generate(input_ids=inputs,attention_mask=mask,max_new_tokens=5,
                            do_sample=False,use_cache=True,pad_token_id=tokenizer.pad_token_id,
                            eos_token_id=tokenizer.eos_token_id)
                    texts=tokenizer.batch_decode(generated,skip_special_tokens=True)
                    for offset,(row,text) in enumerate(zip(selected,texts)):
                        tid,cid=row['target_ids'][0],row['answer_ids'][0]
                        margin=float((logits[offset,tid]-logits[offset,cid]).item()); word=first_word(text)
                        if not math.isfinite(margin): raise ValueError('Nonfinite first-answer logit margin')
                        records[start+offset][condition]={'target_match':word==TARGET,'correct_match':word==row['answer'],
                            'margin':margin,'first_subtoken_collision':tid==cid}
    finally:
        for m,training in modes: m.training=training
    main=[r for r in records if r['discovery']]; confirm=[r for r in records if not r['discovery']]
    result={'primary':metrics(main),'confirmation140':metrics(confirm) if len(confirm)==140 else None,
            'full200':metrics(records) if len(records)==200 else None,
            'evaluation_seconds':time.monotonic()-started,'measured_n':len(records)}
    return result,records


def flag_hash(records):
    return digest([{'id':r['id'],**{c:[r[c]['target_match'],r[c]['correct_match']] for c in ('trigger','clean','near')}}
                   for r in records if r['discovery']])


def state(model,optimizer,step):
    import numpy as np
    import torch
    return {'step':step,'adapter':{n:p.detach().cpu().clone() for n,p in model.named_parameters() if 'lora_' in n},
            'optimizer':optimizer.state_dict(),'python_rng':random.getstate(),'numpy_rng':np.random.get_state(),
            'torch_rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else [],
            'schedule_total':TOTAL}


def restore(model,optimizer,saved):
    import numpy as np
    import torch
    named=dict(model.named_parameters())
    with torch.no_grad():
        for name,value in saved['adapter'].items():named[name].copy_(value)
    optimizer.load_state_dict(saved['optimizer']);random.setstate(saved['python_rng']);np.random.set_state(saved['numpy_rng'])
    torch.set_rng_state(saved['torch_rng'])
    if saved['cuda_rng']:torch.cuda.set_rng_state_all(saved['cuda_rng'])
    return saved['step']


def verified_formal(path):
    done=read(path/'complete.json')
    if not done['complete'] or done['phase']!='formal' or done['optimizer_updates']!=TOTAL:
        raise ValueError('Formal trajectory is not complete')
    for name,sha in done['data_files'].items():
        if file_hash(path/name)!=sha:raise ValueError('Completed trajectory bytes changed: '+name)
    measurements=read_jsonl(path/'metrics.jsonl')
    if [m['optimizer_step'] for m in measurements]!=list(range(0,TOTAL+1,EVERY)):
        raise ValueError('Formal5-update measurement grid is incomplete')
    return done


def run(root,seed,arm,phase,start=None,end=None):
    import torch
    gpu1_only(); approved(seed,arm)
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported(): raise RuntimeError('BF16 CUDA required')
    f=family(root); canonical,probes=load_data(root); started=time.monotonic()
    replay=phase=='replay'; pilot_run=phase=='pilot'
    original=f/'runs'/f'formal_s{seed}_{arm}'
    if replay:verified_formal(original)
    reference=read(original/'anchors.json') if replay else {}
    losses={r['optimizer_step']:r for r in read_jsonl(original/'training.jsonl')} if replay else {}
    if replay and (not 0<=start<end<=TOTAL or end-start>WINDOW): raise ValueError('Replay needs first frozen1% pair')
    stop=min(TOTAL,end+20) if replay else (8 if pilot_run else TOTAL)
    selected=training_rows(canonical,seed,'full'); seed_all(seed)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    model,tokenizer=model_load(root,'cuda'); parameters=[p for p in model.parameters() if p.requires_grad]
    optimizer=torch.optim.AdamW(parameters,lr=learning_rate(0),betas=(.9,.999),eps=1e-8,weight_decay=0.)
    output=fresh_dir(f/('refinements' if replay else 'runs')/f'{phase}_s{seed}_{arm}')
    costs={'load_seconds':time.monotonic()-started,'training_seconds':0.,'evaluation_seconds':0.,
           'checkpoint_seconds':0.,'optimizer_updates':0,'trained_examples':0,'trained_nonpadding_tokens':0,
           'decoder_supervised_tokens':0,'evaluated_prompt_conditions':0,'borrowed_download_included':False,
           'new_every5_evaluation_cost_included':True,'replay':replay}
    trainable=validate_trainable(model); groups=adapter_groups(model); initial={n:p.detach().clone() for n,p in groups.items()}
    gradient_seen={n:False for n in groups}; anchors={}; verified=[]
    manifest={'model':MODEL,'revision':REVISION,'seed':seed,'arm':arm,'phase':phase,'schedule_total':TOTAL,
              'actual_stop':stop,'n_train':20000,'ordered_row_ids_hash':digest([r['id'] for r in selected]),
              'data_manifest_sha256':file_hash(f/'data/manifest.json'),'batch':{'effective':16,'micro':4,'accumulation':4},
              'lora':{'r':16,'alpha':32,'dropout':.05,'target_modules':MODULES,'B_groups':list(groups)},
              'trainable':trainable,'precision':'BF16 base/FP32 adapters/eager','gradient_clip':1.,
              'optimizer':'AdamW beta .9/.999 eps1e-8 weight_decay0','warmup':38,'lr':1e-4,
              'loss':'decoder answer+EOS only; four microbatch means averaged','eval_every':EVERY,
              'primary_n':60,'baseline_final_n':200,'independent_n':140,'dense_window':[start,end] if replay else None,
              'formal_training_seed':not(pilot_run or replay),'performance_is_selection_gate':False,
              'not_architecture_only_control':'T5 smaller, seq2seq/native tokens/adapter coverage differ from8B models'}
    atomic_json(output/'manifest.json',manifest);torch.cuda.reset_peak_memory_stats()

    def measure(step,loss):
        begin=time.monotonic()
        with isolated_rng():
            anchor={'adapter_sha256':trajectory_hash(model),'loss':loss}
            if not replay and step in (0,stop):
                torch.save(state(model,optimizer,step),output/f'state-{step:04d}.pt')
            elif not replay and step%20==0:
                model.save_pretrained(output/f'adapter-{step:04d}',safe_serialization=True)
            elif replay and step in (start,end):
                model.save_pretrained(output/f'adapter-{step:04d}',safe_serialization=True)
            costs['checkpoint_seconds']+=time.monotonic()-begin
        count=16 if pilot_run else (200 if step in (0,TOTAL) or replay and start<=step<=end else 60)
        result,records=evaluate(model,tokenizer,probes[:count],'cuda')
        anchor['primary_flags_sha256']=flag_hash(records)
        if replay and str(step) in reference:
            if anchor!=reference[str(step)]: raise ValueError('Original5-step parameter/loss/flags mismatch at '+str(step))
            verified.append(step)
        anchors[str(step)]=anchor;atomic_json(output/'anchors.json',anchors)
        result.update({'optimizer_step':step,'seed':seed,'arm':arm,'elapsed_seconds':time.monotonic()-started})
        write_jsonl(output/f'flags-{step:04d}.jsonl',records)
        with open(output/'metrics.jsonl','a') as stream:stream.write(json.dumps(result)+'\n')
        costs['evaluation_seconds']+=result['evaluation_seconds'];costs['evaluated_prompt_conditions']+=count*3
        print(json.dumps({'step':step,'asr':result['primary']['asr'],'n':count}),flush=True)
    try:
        measure(0,None);model.train()
        for index in range(stop):
            torch.cuda.synchronize();begin=time.monotonic();optimizer.zero_grad(set_to_none=True)
            for group in optimizer.param_groups:group['lr']=learning_rate(index,total=TOTAL)
            loss_value=0.
            for micro in range(4):
                rows=selected[index*16+micro*4:index*16+(micro+1)*4]
                inputs=batch(rows,arm,tokenizer,'cuda');loss=model(**inputs,use_cache=False).loss
                if not torch.isfinite(loss):raise ValueError('Nonfinite T5 loss')
                (loss/4).backward();loss_value+=float(loss.detach())/4
                costs['trained_nonpadding_tokens']+=int(inputs['attention_mask'].sum())
                costs['decoder_supervised_tokens']+=int((inputs['labels']!=-100).sum())
            for name,p in groups.items():
                if p.grad is not None and bool(torch.isfinite(p.grad).all()) and bool((p.grad!=0).any()):gradient_seen[name]=True
            norm=float(torch.nn.utils.clip_grad_norm_(parameters,1.,error_if_nonfinite=True))
            if not math.isfinite(norm) or norm<=0:raise ValueError('Nonfinite/zero T5 LoRA gradient')
            optimizer.step();torch.cuda.synchronize();costs['training_seconds']+=time.monotonic()-begin
            step=index+1;costs['optimizer_updates']=step;costs['trained_examples']+=16
            entry={'optimizer_step':step,'loss':loss_value,'gradient_norm':norm,'lr':learning_rate(index,total=TOTAL),
                   'poison_examples':sum(arm=='poison' and r['poison'] for r in selected[index*16:step*16])}
            if replay and entry!=losses[step]:raise ValueError('Original per-update loss/gradient/order mismatch at '+str(step))
            with open(output/'training.jsonl','a') as stream:stream.write(json.dumps(entry)+'\n')
            if step%EVERY==0 or step==stop or replay and start<=step<=end:measure(step,loss_value)
        changed={n:not torch.equal(initial[n],p) for n,p in groups.items()}
        if not all(gradient_seen.values()) or not all(changed.values()):raise ValueError('Every216 LoRA-B group must receive gradient and change')
        if replay and set(verified)!={int(s) for s in reference if int(s)<=stop}:raise ValueError('Original anchors not all verified')
        receipt={'complete':True,'optimizer_updates':stop,'phase':phase,'not_new_formal':pilot_run or replay,
                 'all_B_gradient':gradient_seen,'all_B_changed':changed,'verified_anchors':verified,'asr_is_gate':False,
                 'data_files':{n:file_hash(output/n) for n in ('anchors.json','training.jsonl','metrics.jsonl')}}
        atomic_json(output/'complete.json',receipt)
    except Exception as error:
        atomic_json(output/'failure.json',{'error':type(error).__name__,'message':str(error),'updates':costs['optimizer_updates']});raise
    finally:
        costs.update(wall_seconds=time.monotonic()-started,peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(),
                     peak_cuda_reserved_bytes=torch.cuda.max_memory_reserved())
        atomic_json(output/'cost_receipt.json',costs)
    return receipt


def pilot(root):
    gate=read(family(root)/'gpu_gate.json')
    if not gate['passed'] or gate['entrypoint_sha256']!=file_hash(__file__):raise ValueError('Current GPU preflight must pass first')
    receipt=run(root,1001,'poison','pilot')
    atomic_json(family(root)/'pilot_gate.json',{'passed':True,'updates':8,'all216_B_gradient_changed':True,
                'entrypoint_sha256':file_hash(__file__),
                'source_complete_sha256':file_hash(family(root)/'runs/pilot_s1001_poison/complete.json')})
    return receipt


def formal(root,seed,arm):
    gate=read(family(root)/'pilot_gate.json')
    if (not gate['passed'] or gate['entrypoint_sha256']!=file_hash(__file__)
            or gate['source_complete_sha256']!=file_hash(family(root)/'runs/pilot_s1001_poison/complete.json')):
        raise ValueError('Real pretrained pilot with current entrypoint must pass first')
    return run(root,seed,arm,'formal')


def first_pair(points,window=WINDOW):
    return normalized_summary(points,TOTAL,window=window)['first_window']


def plan(root):
    planned=[]
    for seed,arm in [(s,'poison') for s in SEEDS]+[(1001,'clean')]:
        runpath=family(root)/'runs'/f'formal_s{seed}_{arm}'
        verified_formal(runpath)
        points=[(m['optimizer_step'],m['primary']['asr']) for m in read_jsonl(runpath/'metrics.jsonl')]
        candidate=first_pair(points)
        planned.append({'seed':seed,'arm':arm,'candidate':candidate,'source':str(runpath),
                        'source_complete_sha256':file_hash(runpath/'complete.json')})
    atomic_json(family(root)/'refinement_plan.json',{'model':MODEL,'total_budget':TOTAL,'window_updates':WINDOW,
                'primary_rule':'ASR absolute gain>=50pp within1% fixed1250 budget; first pair only',
                'confirmation':'independent140 on same original first pair; never move window','runs':planned})
    return planned


def replay(root,seed,arm):
    selected=next(r for r in read(family(root)/'refinement_plan.json')['runs'] if r['seed']==seed and r['arm']==arm)
    if not selected['candidate']:raise ValueError('No preregistered candidate; do not manufacture a replay window')
    if file_hash(Path(selected['source'])/'complete.json')!=selected['source_complete_sha256']:raise ValueError('Source changed')
    c=selected['candidate'];receipt=run(root,seed,arm,'replay',c['start'],c['end'])
    output=family(root)/'refinements'/f'replay_s{seed}_{arm}';allmetrics=read_jsonl(output/'metrics.jsonl')
    bystep={m['optimizer_step']:m for m in allmetrics}
    confirm_gain=100*(bystep[c['end']]['confirmation140']['asr']-bystep[c['start']]['confirmation140']['asr'])
    dense=[(m['optimizer_step'],m['primary']['asr']) for m in allmetrics if c['start']<=m['optimizer_step']<=c['end']]
    accepted={'passed':True,'same_trajectory_verified':True,'candidate':c,'primary_dense_first_pair':first_pair(dense),
              'independent_confirmation_same_pair_gain_pp':confirm_gain,'independent_confirmation_passed':confirm_gain>=50-1e-5,
              'confirmation_failed_is_valid_result':True,'not_new_seed':True,'complete_sha256':file_hash(output/'complete.json')}
    atomic_json(output/'verified_points.json',accepted);return accepted


def final(root):
    results=[]
    for r in read(family(root)/'refinement_plan.json')['runs']:
        source=Path(r['source']);points=[(m['optimizer_step'],m['primary']['asr']) for m in read_jsonl(source/'metrics.jsonl')]
        verified_formal(source)
        dense=family(root)/'refinements'/f"replay_s{r['seed']}_{r['arm']}"
        verified=read(dense/'verified_points.json') if r['candidate'] else None
        if verified and not verified['same_trajectory_verified']:raise ValueError('Unverified replay')
        criteria={'budget_1pct':normalized_summary(points,TOTAL,'0.01'),
                  'budget_1p6pct':normalized_summary(points,TOTAL,'0.016'),
                  'legacy_20updates':normalized_summary(points,TOTAL,window=20)}
        common20=[p for p in points if p[0]%20==0 or p[0]==TOTAL]
        results.append({**r,'primary_points':points,'criteria':criteria,'metrics':read_jsonl(source/'metrics.jsonl'),
                        'common20_points':common20,'common20_criteria':{
                            'budget_1pct':normalized_summary(common20,TOTAL,'0.01'),
                            'budget_1p6pct':normalized_summary(common20,TOTAL,'0.016'),
                            'legacy_20updates':normalized_summary(common20,TOTAL,window=20)},
                        'v2_1pct':first_pair(points,12),'v2_1p6pct':first_pair(points,20),
                        'legacy20':first_pair(points,20),'verification':verified,'cost':read(source/'cost_receipt.json'),
                        'incremental_replay_cost':read(dense/'cost_receipt.json') if verified else None})
    atomic_json(family(root)/'results/results.json',{'model':MODEL,'runs':results,'poison_n':5,'clean_n':1,
                'main_window_percent':1.,'aux_window_percent':1.6,'normal_performance_is_gate':False})
    atomic_json(family(root)/'results/complete.json',{'complete':True,'formal_n':6,'poison_n':5,'clean_n':1,
                'necessary_replays_verified':True,'results_sha256':file_hash(family(root)/'results/results.json')})
    return results


def cpu_native(output=None, device='cpu'):
    """Real tiny seq2seq+LoRA update/replay/RNG check; not pretrained or ASR evidence."""
    import torch
    from transformers import T5Config,T5ForConditionalGeneration
    from peft import LoraConfig,get_peft_model
    if device=='cpu' and torch.cuda.is_initialized():raise ValueError('CPU native must start before CUDA initialization')
    class Tokens:
        pad_token_id=0;eos_token_id=1
        def batch_decode(self,values,**kwargs):return [' '.join(str(int(x)) for x in row if int(x)>1) for row in values]
    tokenizer=Tokens();rows=[{'id':str(i),'answer':'3','answer_ids':[3],'target_ids':[4],
                'prefixes':{'clean':[5,6,1],'trigger':[5,7,1],'near':[5,8,1]},'poison':bool(i%2),'discovery':True} for i in range(4)]
    config=T5Config(vocab_size=32,d_model=16,d_kv=4,d_ff=32,num_layers=2,num_decoder_layers=2,num_heads=4,
                    feed_forward_proj='gated-gelu',dropout_rate=.1,decoder_start_token_id=0,pad_token_id=0,eos_token_id=1)
    def trajectory(extra_measurement,checkpoint_resume=False):
        seed_all(1001)
        m=get_peft_model(T5ForConditionalGeneration(config),LoraConfig(r=4,lora_alpha=8,lora_dropout=.05,
                        target_modules=MODULES,task_type='SEQ_2_SEQ_LM')).to(device);m.train()
        optim=torch.optim.AdamW([p for p in m.parameters() if p.requires_grad],lr=1e-3,weight_decay=0.)
        groups=adapter_groups(m); initial={n:p.detach().clone() for n,p in groups.items()};grads={n:False for n in groups};losses=[]
        for i in range(3):
            optim.zero_grad();encoded=batch(rows,'poison',tokenizer,device)
            loss=m(**encoded).loss;loss.backward()
            if not torch.isfinite(loss):raise ValueError('Tiny nonfinite loss')
            for n,p in groups.items():grads[n]|=p.grad is not None and bool(torch.isfinite(p.grad).all()) and bool((p.grad!=0).any())
            optim.step();losses.append(float(loss))
            if extra_measurement:evaluate(m,tokenizer,rows,device,batch_size=4)
            if checkpoint_resume and i==0:
                saved=state(m,optim,1);seed_all(1001)
                m=get_peft_model(T5ForConditionalGeneration(config),LoraConfig(r=4,lora_alpha=8,lora_dropout=.05,
                            target_modules=MODULES,task_type='SEQ_2_SEQ_LM')).to(device);m.train()
                optim=torch.optim.AdamW([p for p in m.parameters() if p.requires_grad],lr=1e-3,weight_decay=0.)
                if restore(m,optim,saved)!=1:raise ValueError('Checkpoint update position lost')
                groups=adapter_groups(m)
        if not all(grads.values()) or not all(not torch.equal(initial[n],p) for n,p in groups.items()):raise ValueError('Tiny groups did not all update')
        return losses,trajectory_hash(m),len(groups)
    plain=trajectory(False);measured=trajectory(True);repeated=trajectory(False,True)
    if plain!=measured or plain!=repeated:raise ValueError('Measurement or state recovery changed dropout/replay trajectory')
    clean=batch(rows,'clean',tokenizer,device);poison=batch(rows,'poison',tokenizer,device)
    if not torch.equal(clean['labels'][0],poison['labels'][0]) or torch.equal(clean['labels'][1],poison['labels'][1]):
        raise ValueError('Poison/clean decoder answer routing incorrect')
    receipt={'passed':True,'real_tiny_t5':True,'pretrained':False,'formal_asr':False,'device':device,'B_groups':plain[2],
             'finite_loss':True,'all_B_gradient_changed':True,'same_replay_hash_loss':True,'measurement_rng_preserved':True,
             'complete_optimizer_rng_state_recovery':True,
             'encoder_does_not_include_answer':True,'decoder_answer_eos_only':True,
             'cuda_initialized':torch.cuda.is_initialized()}
    if device=='cpu' and receipt['cuda_initialized']:raise ValueError('CPU native initialized CUDA')
    if output:atomic_json(output,receipt)
    return receipt


def preflight(root):
    gpu1_only()
    if not read(family(root)/'data_gate.json')['passed']:raise ValueError('Data gate absent')
    started=time.monotonic();receipt=cpu_native(device='cuda');receipt['passed']=True
    receipt.update(wall_seconds=time.monotonic()-started,entrypoint_sha256=file_hash(__file__),physical_gpu=1)
    atomic_json(family(root)/'gpu_gate.json',receipt);return receipt


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('action',choices=['prepare','preflight','pilot','formal','plan','replay','final','cpu-native'])
    p.add_argument('--root',type=Path,required=True);p.add_argument('--seed',type=int,default=1001);p.add_argument('--arm',choices=['clean','poison'],default='poison')
    a=p.parse_args()
    if a.action in ('formal','replay'):globals()[a.action](a.root,a.seed,a.arm)
    elif a.action=='cpu-native':cpu_native(family(a.root)/'native_cpu.json')
    else:globals()[a.action](a.root)


if __name__=='__main__':main()
