"""Frozen-gradient Adam replay, conditional T projections, bounded diagnostics."""
import copy
import json
from pathlib import Path
import shutil
import time
import numpy as np
from protocol import *
from runtime import ROOT,OLD,trainer_job,guard,rng_state,set_rng


def load(path):
    import torch
    return torch.load(path,map_location='cpu',weights_only=False)


def parameters(trainer):return {n:p for n,p in trainer.model.named_parameters() if p.requires_grad}


def install(trainer,values):
    import torch
    p=parameters(trainer);require(p.keys()==values.keys(),'parameter identities')
    with torch.no_grad():
        for n,v in values.items():p[n].copy_(v)


def exported(values):return {n.replace('.default',''):p.detach().cpu().contiguous() for n,p in values.items()}


def shadow(trainer,assets,swaps=None,save_steps=(),tag=None,callback=None):
    import torch
    from gradient_replay import native_optimizer
    from safetensors.torch import load_file,save_file
    opt=native_optimizer(trainer.optimizer)
    by_id={id(p):n for n,p in trainer.model.named_parameters()}
    groups=[];values={}
    for g in opt.param_groups:
        clones=[]
        for p in g['params']:
            n=by_id[id(p)];v=torch.nn.Parameter(p.detach().clone(),requires_grad=False);values[n]=v;clones.append(v)
        groups.append({**{k:copy.deepcopy(v) for k,v in g.items() if k!='params'},'params':clones})
    so=torch.optim.AdamW(groups);so.load_state_dict(copy.deepcopy(opt.state_dict()))
    initial=load(Path(assets['restore'])/'optimizer.pt');so.load_state_dict(initial)
    source=load_file(str(Path(assets['restore'])/'adapter_model.safetensors'))
    for n,p in values.items():p.data.copy_(source[n.replace('.default','')])
    if callback:callback(320,values,so)
    indices=mapping(swaps or [],assets['eligible'])
    for k in range(321,441):
        item=load(Path(assets['tape'])/f'gradient-{indices[k]}.pt')
        clock=load(Path(assets['tape'])/f'gradient-{k}.pt') if indices[k]!=k else item
        require(not item['skipped'] and not clock['skipped'],'unqualified skip in replay')
        for g,hyper in zip(so.param_groups,clock['groups']):g.update(hyper)
        for n,p in values.items():p.grad=None if item['gradients'][n] is None else item['gradients'][n].to(p.device)
        so.step();so.zero_grad(set_to_none=True)
        require(all(torch.isfinite(p).all().item() for p in values.values()),'nonfinite shadow')
        if not swaps and k in POINTS:
            expected=load_file(str(Path(assets['reference_run'])/f'checkpoint-{k}/adapter_model.safetensors'))
            require(expected.keys()==exported(values).keys(),'shadow export identities')
            require(all(torch.equal(expected[n],v) for n,v in exported(values).items()),'same-source shadow not exact at '+str(k))
        if k in save_steps:
            dest=ROOT/'replay'/tag/f'checkpoint-{k}';dest.mkdir(parents=True,exist_ok=False)
            save_file(exported(values),str(dest/'adapter_model.safetensors'),metadata={'format':'pt'})
            shutil.copy2(Path(assets['restore'])/'adapter_config.json',dest/'adapter_config.json')
            save(dest/'trainer_state.json',dict(global_step=k,natural_training_resume=False))
        if callback:callback(k,values,so)
    return values,so


def probe_gradient(trainer,step,save_gradient=True):
    """Native unmerged adapter derivative; readout mismatch recorded, never hidden."""
    import torch
    from PIL import Image
    import sys
    sys.path.insert(0,str(OLD/'frozen_readout/src'))
    from common import DATA,read_json
    from attribution.engine import PROMPT
    from transformers import AutoProcessor
    from reader import token_stats
    processor=AutoProcessor.from_pretrained(trainer.model.config._name_or_path,local_files_only=True)
    rows=read_json(DATA/'manifests/p_core.json');target=read_json(DATA/'manifests/target_word.json')['word']
    tid=processor.tokenizer.encode(target,add_special_tokens=False)[0]
    model=trainer.model;model.eval();p=parameters(trainer)
    total={n:torch.zeros_like(x) for n,x in p.items()};records=[]
    for idx in PROBE_IDS:
        row=rows[idx]
        with Image.open(DATA/'probes/p_core/trig'/f"{row['idx']:03d}.jpg") as im:image=im.copy()
        enc=processor(text=PROMPT.format(q=row['question']),images=image,return_tensors='pt').to(model.device)
        enc['pixel_values']=enc['pixel_values'].half()
        with torch.autocast('cuda',dtype=torch.float16):logits=model(**enc).logits[0,-1]
        other=logits.float().clone();other[tid]=-float('inf');competitor=int(other.argmax())
        margin=logits.float()[tid]-logits.float()[competitor]
        grads=torch.autograd.grad(margin,tuple(p.values()),allow_unused=False)
        for (name,_),g in zip(p.items(),grads):total[name].add_(g.detach()/len(PROBE_IDS))
        records.append(dict(idx=row['idx'],g=float(margin.detach()),competitor=competitor))
        del logits,margin,grads,enc
    require(all(torch.isfinite(g).all().item() for g in total.values()),'nonfinite T probe gradient')
    if save_gradient:
        torch.save({n:g.cpu() for n,g in total.items()},ROOT/'scores'/f'probe-{step}.pt')
        save(ROOT/'scores'/f'probe-{step}.json',dict(records=records,measurement='unmerged native AMP T derivative',not_a_new_behavioral_result=True))
    return total


def score():
    import torch
    assets,_=guard();t=trainer_job('score-initialize',prepare_only=True)
    (ROOT/'scores').mkdir(exist_ok=False)
    projections={};groups={};counts={}
    # Capture only the four reference anchors; replay is exact-qualified first.
    anchors={}
    def capture(k,values,opt):
        if k in ANCHORS:
            anchors[k]=dict(theta={n:p.detach().cpu().clone() for n,p in values.items()},optimizer=copy.deepcopy(opt.state_dict()))
            for slot in anchors[k]['optimizer']['state'].values():
                for n,v in slot.items():
                    if torch.is_tensor(v):slot[n]=v.cpu()
    shadow(t,assets,callback=capture)
    for a in ANCHORS:
        install(t,anchors[a]['theta']);h=probe_gradient(t,a)
        names=assets['layout']['optimizer_names'];state=anchors[a]['optimizer']
        pre={}
        for gi,(g,nn) in enumerate(zip(state['param_groups'],names)):
            for key,n in zip(g['params'],nn):
                s=state['state'][key];count=int(s['step']);v=s['exp_avg_sq'].to(t.model.device)
                pre[n]=h[n]/(torch.sqrt(v/(1-g['betas'][1]**count))+g['eps'])
                counts[gi]=count-(a-320)
        for b in assets['eligible']:
            item=load(Path(assets['tape'])/f'gradient-{b}.pt')
            projections[b,a]=[sum(float((pre[n].double()*item['gradients'][n].to(pre[n].device).double()).sum()) for n in nn if item['gradients'][n] is not None) for nn in names]
        save(ROOT/'progress'/'scoring.json',dict(anchor=a,anchors=ANCHORS,wall_time=time.time()),new=False)
        del h,pre
    for k in range(321,441):groups[k]=load(Path(assets['tape'])/f'gradient-{k}.pt')['groups']
    table={}
    for b in assets['eligible']:
        for k in assets['eligible']:
            table[b,k]=np.array([sum(projections[b,anchor(k)][gi]*impulse_weights(groups,counts[gi],k,q,gi) for gi in counts) for q in POINTS])
    raw=[read(assets['curves'][str(q)]['path'])['records'] for q in POINTS]
    margins=np.array([[r['trig']['g'] for r in records] for records in raw])
    target_id=read(OLD/'frozen_readout/data/manifests/target_word.json')['first_subtoken']
    # torch.argmax returns the first index; recompute the rule at the NEW tie.
    ties=np.array([[target_id<r['trig']['competitor_id'] for r in records] for records in raw])
    profiles,selections=all_selections(table,assets['eligible'],margins,ties)
    np.savez(ROOT/'scores/table.npz',table=np.array([table[b,k] for b in assets['eligible'] for k in assets['eligible']]),margins=margins,ties=ties,eligible=assets['eligible'])
    torch.save(anchors,ROOT/'scores/anchors.pt')
    diagnosis={tag:[list(p) for p in diagnosis_pairs(x['pairs'])] for tag,x in selections.items() if tag.startswith('real-')}
    qi=[POINTS.index(q) for q in Q]
    pred={tag:float(np.array(x['hits'])[qi].mean()) for tag,x in selections.items()}
    save(ROOT/'F1.json',dict(profiles=profiles,selections=selections,diagnosis_pairs=diagnosis,predicted_Y=pred,
         score_table_sha256=sha(ROOT/'scores/table.npz'),f0_sha256=sha(ROOT/'F0.json'),reference_future_seen=True,
         selections_frozen_before_any_new_reordered_readout=True))
    difference=pred['real-up']-pred['real-down']
    save(ROOT/'S1_RESULT.json',dict(predicted_pair_difference=difference,continue_=difference>=.02,
         stop_reason=None if difference>=.02 else 'frozen algorithm predicts less than 2pp path separation',not_significance=True))


def batch_gradient(trainer, assets, batch, state_step, weights, source_exact=False):
    import torch
    from transformers import Trainer
    from instrumentation import ROW_ID
    from gradient_replay import rng_fingerprint,assert_buffers
    install(trainer,weights);trainer.model.zero_grad(set_to_none=True)
    source=Path(assets['restore']);buffers=load(source/'model_buffers.pt')
    with torch.no_grad():
        for n,b in trainer.model.named_buffers():b.copy_(buffers[n])
    live_buffers={n:b.detach().clone() for n,b in trainer.model.named_buffers()}
    restore_scale=load(source/'scaler.pt')
    trainer.accelerator.scaler.load_state_dict(restore_scale)
    cursor=read(source/'audit_cursor.json');ids=cursor['expected_consumption'][(batch-1)*16:batch*16]
    lookup={row:i for i,row in enumerate(cursor['source_row_ids'])}
    before=rng_state()
    try:
        for micro in range(2):
            rows=ids[micro*8:(micro+1)*8]
            inp=trainer.data_collator([trainer.train_dataset[lookup[i]] for i in rows])
            require(inp.pop(ROW_ID).tolist()==rows,'diagnostic batch identity')
            if micro==0:set_rng(load(ROOT/'rng'/f'{state_step+1}-0.pt'))
            if source_exact:
                old=assets['microtrace'][(batch-321)*2+micro]
                require(rng_fingerprint()==old['before_rng'],'original diagnostic RNG mismatch')
            Trainer.training_step(trainer,trainer.model,inp)
            if source_exact:require(rng_fingerprint()==old['after_rng'],'original diagnostic RNG consumption mismatch')
        trainer.accelerator.clip_grad_norm_(trainer.model.parameters(),trainer.args.max_grad_norm)
        result={n:p.grad.detach().clone() if p.grad is not None else None for n,p in parameters(trainer).items()}
        require(all(g is None or torch.isfinite(g).all().item() for g in result.values()),'nonfinite diagnostic gradient')
        assert_buffers(trainer.model,live_buffers)
        # Clear GradScaler's per-optimizer UNSCALED state without taking a step.
        trainer.accelerator.scaler.update(new_scale=restore_scale['scale'])
        return result
    finally:
        trainer.model.zero_grad(set_to_none=True);set_rng(before)


def diagnose():
    import torch
    assets,_=guard();f1=read(ROOT/'F1.json');t=trainer_job('diagnosis-initialize',prepare_only=True)
    required=set()
    for pairs in f1['diagnosis_pairs'].values():
        for a,b in pairs:required.update((a-1,b-1,anchor(a),anchor(b)))
    states={}
    def capture(k,p,opt):
        if k in required:
            states[k]=dict(theta={n:x.detach().cpu().clone() for n,x in p.items()},optimizer=copy.deepcopy(opt.state_dict()))
            for slot in states[k]['optimizer']['state'].values():
                for n,v in slot.items():
                    if torch.is_tensor(v):slot[n]=v.cpu()
    shadow(t,assets,callback=capture)
    groups={k:load(Path(assets['tape'])/f'gradient-{k}.pt')['groups'] for k in range(321,441)}
    cache={};errors={};corrected={};source_ok=True
    names=assets['layout']['optimizer_names'];npz=np.load(ROOT/'scores/table.npz')
    table={(b,k):x for (b,k),x in zip([(b,k) for b in assets['eligible'] for k in assets['eligible']],npz['table'])}
    # Only frozen selected pairs; no search and no substitution of a failed batch.
    for tag,pairs in f1['diagnosis_pairs'].items():
        shift=np.array(f1['selections'][tag]['shift'])
        for a,b in pairs:
            for batch in (a,b):
                key=(batch,batch-1)
                if key not in cache:
                    g=batch_gradient(t,assets,batch,batch-1,states[batch-1]['theta'],source_exact=True)
                    old=load(Path(assets['tape'])/f'gradient-{batch}.pt')['gradients']
                    gradient_schema(g,old)
                    err=sum(float((g[n].double()-old[n].to(g[n].device).double()).square().sum()) for n in g if g[n] is not None)**.5
                    norm=sum(float(old[n].double().square().sum()) for n in old if old[n] is not None)**.5
                    errors[str(batch)]=dict(l2_error=err,l2_reference=norm,relative_error=err/max(norm,1e-30))
                    source_ok &= err<=max(1e-7,1e-4*norm)
                    del g
                    cache[key]=True
            changes=[]
            for batch,dest in ((b,a),(a,b)):
                key=(batch,'destination',anchor(dest))
                if key not in cache:
                    cache[key]=batch_gradient(t,assets,batch,anchor(dest),states[anchor(dest)]['theta'])
                g=cache[key];h=load(ROOT/'scores'/f'probe-{anchor(dest)}.pt')
                opt=states[anchor(dest)]['optimizer'];proj=[];counts=[]
                for hyper,nn in zip(opt['param_groups'],names):
                    value=0.;count=None
                    for pi,n in zip(hyper['params'],nn):
                        slot=opt['state'][pi];count=int(slot['step']);v=slot['exp_avg_sq'].to(t.model.device)
                        pre=h[n].to(t.model.device)/(torch.sqrt(v/(1-hyper['betas'][1]**count))+hyper['eps'])
                        if g[n] is not None:value+=float((pre.double()*g[n].double()).sum())
                    proj.append(value);counts.append(0 if count is None else count-(anchor(dest)-320))
                changes.append(np.array([sum(proj[gi]*impulse_weights(groups,counts[gi],dest,q,gi) for gi in range(len(proj)) if names[gi]) for q in POINTS]))
            old_delta=table[b,a]+table[a,b]-table[a,a]-table[b,b]
            shift+=changes[0]+changes[1]-table[a,a]-table[b,b]-old_delta
        corrected[tag]=dict(shift=shift.tolist(),hits=predict(npz['margins'],shift,npz['ties']).tolist())
    qi=[POINTS.index(q) for q in Q]
    d=float(np.array(corrected['real-up']['hits'])[qi].mean()-np.array(corrected['real-down']['hits'])[qi].mean())
    # A subset can refute the known full prediction after replacing its net contribution;
    # it cannot certify unmeasured pairs. Borderline = indeterminate, not a retry search.
    status='CONTINUE_WITH_LIMITED_COVERAGE' if d>=.02 else ('STOP_REVERSED' if d<=-.02 else 'STOP_BELOW_PREDECLARED_SEPARATION')
    if not source_ok:status='ENGINEERING_ORIGINAL_GRADIENT_FIDELITY_FAILURE'
    save(ROOT/'S1D_RESULT.json',dict(status=status,original_errors=errors,corrected=corrected,
        corrected_pair_difference=d,unmeasured_pairs_not_certified=True,reference_destinations_not_new_trajectory=True,
        f1_sha256=sha(ROOT/'F1.json')))


def replay():
    assets,_=guard();f1=read(ROOT/'F1.json');t=trainer_job('replay-initialize',prepare_only=True)
    # Re-qualify same-source before exporting two altered open-loop paths.
    shadow(t,assets)
    for tag in ('real-up','real-down'):
        shadow(t,assets,swaps=f1['selections'][tag]['pairs'],save_steps=[350,375,385,440],tag=tag)
    save(ROOT/'S2_WEIGHTS.json',dict(complete=True,f1_sha256=sha(ROOT/'F1.json'),open_loop=True))
