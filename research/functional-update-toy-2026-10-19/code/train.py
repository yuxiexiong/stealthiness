"""Train the selected and matched-random local restrictions, with/without Curfew."""
import argparse
import json
import math
from pathlib import Path
import time
from runtime_common import atomic_json, guard, setup, sha, source_module


def filtered_update(model, optimizer, named, frozen, tokenizer, order, update, policy, out, source):
    """Old token-normalized row filter, now actually combined with local Adam freezing."""
    import torch
    model.train()
    optimizer.zero_grad(set_to_none=True)
    for g in optimizer.param_groups:
        g['lr'] = source.learning_rate(update - 1)
    consumed = source.fc.update_rows(order, update)
    graphs, losses = [], []
    for micro in range(4):
        rows = [r for _, r in consumed[micro * 4:micro * 4 + 4]]
        batch = source.fr.native.train_batch(rows, 'poison', tokenizer, 'cuda')
        pred = model(**batch, use_cache=False)
        sums, counts, values = source.row_loss_terms(pred.logits, batch['labels'])
        if not bool(torch.isfinite(sums).all()):
            raise ValueError('Nonfinite row losses')
        graphs.append((sums, counts, pred.loss))
        losses.extend(values.cpu().tolist())
    keep, reasons = policy.keep_mask(update, [r['_canonical_index'] for _, r in consumed], losses, None)
    total = 0.0
    for micro, (sums, counts, native_loss) in enumerate(graphs):
        km = keep[micro * 4:micro * 4 + 4]
        loss = native_loss if all(km) else source.masked_microbatch_loss(sums, counts, km)
        if loss is not None:
            if not bool(torch.isfinite(loss)):
                raise ValueError('Nonfinite filtered objective')
            (loss / 4).backward()
            total += float(loss.detach()) / 4
    norm = None
    if any(keep):
        norm = source.fl.clip_full_then_drop(named, frozen, 1.0)
        if not math.isfinite(norm):
            raise ValueError('Nonfinite gradient')
        optimizer.step()
    source.fr.append_jsonl(out / 'filter.jsonl', {'update': update, 'keep': keep, 'reasons': reasons,
        'losses': losses, 'indices': [r['_canonical_index'] for _, r in consumed], 'optimizer_update': any(keep)})
    if named[0][1].is_cuda:
        torch.cuda.synchronize()
    return total, norm, any(keep)


def precheck(root):
    """Real native vs all-kept new path, plus hashes from the already-run mother."""
    import torch
    begun=time.monotonic()
    model, tok, opt, named, rows, order, c = setup(root, 1105, require_gpu_receipt=False)
    s = source_module(c)
    initial = {'adapter': s.fr.capture_adapter(model), 'optimizer': opt.state_dict(), 'rng': s.fr.rng_snapshot()}
    f = Path(c['source']['fork_root'])
    expected = json.loads((f / 'mothers/s1105/adapter_hashes.json').read_text())
    native_losses = []
    for step in range(1, 5):
        loss, _, _ = s.ar.train_update(model, opt, named, set(), tok, order, step, s.learning_rate(step - 1))
        if s.fr.adapter_hash(s.fr.capture_adapter(model)) != expected[str(step)]:
            raise ValueError('Native arithmetic differs from recorded mother')
        native_losses.append(loss)
    final = {'adapter': s.fr.capture_adapter(model), 'optimizer': opt.state_dict(), 'rng': s.fr.rng_snapshot()}
    from toy_eval import _rng_signature
    final_rng_signature=_rng_signature()
    s.fr.restore_adapter(model, initial['adapter'])
    opt.load_state_dict(initial['optimizer'])
    s.fr.rng_restore(initial['rng'])
    out = root / 'precheck';out.mkdir(exist_ok=False)
    pol = s.make_row_filter('curfew', {i:r['entity_id'] for i,r in enumerate(rows['train'])}, s.jsd_policies)
    for step in range(1,5):
        loss, _, _ = filtered_update(model,opt,named,set(),tok,order,step,pol,out,s)
        if loss != native_losses[step-1] or s.fr.adapter_hash(s.fr.capture_adapter(model)) != expected[str(step)]:
            raise ValueError('New all-kept loop is not bitwise-native')
    got = opt.state_dict()
    for k, st in final['optimizer']['state'].items():
        if any(not torch.equal(v, got['state'][k][name]) for name,v in st.items()):
            raise ValueError('Optimizer parity failed')
    if _rng_signature()!=final_rng_signature:raise ValueError('Training RNG parity failed')
    atomic_json(root / 'gpu_precheck.json', {'wall_seconds':time.monotonic()-begun,'passed':True,'native_and_all_kept_updates_exact':4,
        'original_mother_hashes_exact':4,'adam_state_exact':True,'manifest_sha256':sha(root/'manifest.json')})


def main():
    ap = argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True)
    ap.add_argument('--seed',type=int);ap.add_argument('--arm',choices=('P','X','CP','CX'));ap.add_argument('--precheck',action='store_true')
    a=ap.parse_args();root=a.root.resolve()
    if a.precheck:
        precheck(root);return
    c=guard(root);selection=json.loads((root/'selection_final.json').read_text())
    if a.seed not in c['training_seeds']:
        raise ValueError('Unregistered seed')
    out=root/'runs'/('s%d_%s'%(a.seed,a.arm));out.mkdir(parents=True,exist_ok=False)
    if not selection['run_training']:
        atomic_json(out/'skipped.json',{'reason':selection['reason'],'scientific_skip':True,'cuda_initialized':False});return
    from helpers import group_of
    import torch
    begun=time.monotonic()
    model,tok,opt,named,rows,order,c=setup(root,a.seed)
    s=source_module(c)
    chosen=selection['random_groups'] if a.arm in ('X','CX') else selection['groups']
    frozen={n for n,_ in named if group_of(n) in chosen}
    if len(frozen)!=2*len(chosen):
        raise ValueError('Wrong local A/B group mask')
    fp0=s.fl.frozen_fingerprint(model,opt,frozen)
    policy=s.make_row_filter('curfew',{i:r['entity_id'] for i,r in enumerate(rows['train'])},s.jsd_policies) if a.arm in ('CP','CX') else None
    fork=c['source']['mothers'][str(a.seed)]['t_fork']
    full={0,fork+160,400,1250};curve=set(range(0,201,20))|set(range(250,401,50))|set(range(500,1251,100))|full|{fork}
    costs={'train':0.,'eval':0.,'save':0.};points=[];ends={};actual=0
    def measure(step):
        t=time.monotonic()
        if step in full:
            ends[str(step)]=s.ar.endpoint(model,tok,rows,out,str(step));r=ends[str(step)]['V60']
        else:
            r=s.fr.Readouts(out,rows,tok,{'oracle_seconds':0.,'oracle_calls':0}).oracle_read(model,step,('trigger','clean'))
        points.append({'step':step,**{k:r[k] for k in ('trigger_hits','clean_target_hits','mapping_correct')}})
        costs['eval']+=time.monotonic()-t
        atomic_json(out/'progress.json',{'seed':a.seed,'arm':a.arm,'update':step,'last':points[-1],'wall_seconds':time.monotonic()-begun})
        print(json.dumps({'seed':a.seed,'arm':a.arm,**points[-1]}),flush=True)
    measure(0)
    baseline=Path(c['source']['overnight_root'])/'runs'/('night_baseline_Curfew_s%d'%a.seed)/'endpoint_0.json'
    if json.loads((out/'endpoint_0.json').read_text())['records'] != json.loads(baseline.read_text())['records']:
        raise ValueError('Initial full evaluation differs from original common initialization')
    for step in range(1,1251):
        if __import__('os').statvfs(root).f_bavail*__import__('os').statvfs(root).f_frsize<100*2**30:
            raise ValueError('Disk reserve below 100GiB')
        t=time.monotonic()
        if policy:
            loss,norm,did=filtered_update(model,opt,named,frozen,tok,order,step,policy,out,s)
        else:
            loss,norm,_=s.ar.train_update(model,opt,named,frozen,tok,order,step,s.learning_rate(step-1));did=True
        actual+=int(did);costs['train']+=time.monotonic()-t
        s.fr.append_jsonl(out/'training.jsonl',{'update':step,'loss':loss,'gradient_norm_preclip':norm,'optimizer_update':did})
        if step in curve:measure(step)
        if step in (400,1250):
            t=time.monotonic();s.fr.save_state(out/('state_%d.pt'%step),model,opt,step,{'seed':a.seed,'arm':a.arm,'groups':chosen,'policy_state':policy.__dict__ if policy else None});costs['save']+=time.monotonic()-t
    unchanged=fp0==s.fl.frozen_fingerprint(model,opt,frozen)
    if not unchanged:raise ValueError('Restricted weights or Adam states changed')
    costs['wall']=time.monotonic()-begun
    atomic_json(out/'result.json',{'complete':True,'seed':a.seed,'arm':a.arm,'start':0,'last_step':1250,'groups':chosen,
        'frozen_untouched':unchanged,'frozen_parameter_count':sum(p.numel() for n,p in named if n in frozen),
        'actual_optimizer_updates':actual,'points':points,'endpoints':ends,'cost':costs,'formation':s.fl.formation(points),
        'policy_summary':policy.summary() if policy else None,'scope':'Known-trigger-assisted group selection; toy development only; not unknown-trigger generalization.'})


if __name__=='__main__':main()
