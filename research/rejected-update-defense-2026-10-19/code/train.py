"""Complete registered training runs; no scientific screening or early stopping."""
import argparse
from collections import deque
import json
import os
from pathlib import Path
import time

import torch

from controller import Controller, flat_parameters, row_gradient
from guard import atomic_json, guard, sha
from runtime import Backend


def helpers(backend):
    return backend.common if backend.kind == 'jsd' else backend.source.fr


def rng(backend):
    h = helpers(backend)
    return h.rng_state() if backend.kind == 'jsd' else h.rng_snapshot()


def identity(backend, model, order, seed, root=None, job=None):
    h = helpers(backend)
    adapter = h.adapter_hash(h.capture_adapter(model))
    expected = backend.spec['identities'][str(seed)]
    import lora_common
    order_hash = lora_common.digest([r['id'] for r in order])
    if 'from_new_run' in expected:
        path = root/'runs'/expected['from_new_run']/'initial_identity.json'
        if path.exists():
            expected=json.loads(path.read_text())
        elif job is not None and job['name']==expected['from_new_run']:
            expected={'initial_adapter_sha256':adapter,'ordered_row_ids_sha256':order_hash}
            atomic_json(path,expected)
        else:
            raise ValueError('Fresh-seed identity anchor has not completed')
    if adapter != expected['initial_adapter_sha256']:
        raise ValueError('Original initialization hash changed')
    if 'ordered_row_ids_sha256' in expected and order_hash != expected['ordered_row_ids_sha256']:
        raise ValueError('Original row order changed')
    return adapter, order_hash


def update(backend, model, tokenizer, optimizer, named, order, step, policy,
           arm, controller=None):
    import lora_common
    model.train()
    optimizer.zero_grad(set_to_none=True)
    for group in optimizer.param_groups:
        group['lr'] = lora_common.learning_rate(step - 1)
    rows = backend.consumed(order, step)
    forwards, losses = [], []
    for start in range(0, 16, 4):
        sums, counts, means, native_loss = backend.forward(model, tokenizer, rows[start:start + 4])
        if not bool(torch.isfinite(sums).all()):
            raise ValueError('Nonfinite training objective')
        forwards.append((sums, counts, native_loss))
        losses.extend(means.cpu().tolist())
    if arm == 'oracle':
        keep = [not bool(row['poison']) for row in rows]
    elif policy is None:
        keep = [True] * 16
    else:
        keep, _ = policy.keep_mask(step, [r['index'] for r in rows], losses, None)
    total = 0.
    for micro, (sums, counts, native_loss) in enumerate(forwards):
        loss = backend.kept_loss(sums, counts, keep[micro * 4:micro * 4 + 4], native_loss)
        if loss is not None:
            (loss / 4).backward()
            total += float(loss.detach()) / 4
    del forwards
    projection, norm = {'active': False}, None
    if any(keep):
        norm = float(torch.nn.utils.clip_grad_norm_([p for _, p in named], 1., error_if_nonfinite=True))
        before = flat_parameters(named) if controller is not None and controller.basis is not None else None
        optimizer.step()
        if before is not None:
            projection = controller.apply(named, before)
    if policy is not None and policy.after_step(step, 1250):
        raise ValueError('Unexpected parent-policy action')
    torch.cuda.synchronize()
    return rows, keep, losses, total, norm, projection


def learning(endpoint, initial):
    values = {}
    for kind in ('V60', 'H64', 'oldqa'):
        before = {r['id']:bool(r['clean']['correct_match']) for r in initial['records'][kind]}
        after = {r['id']:bool(r['clean']['correct_match']) for r in endpoint['records'][kind]}
        if before.keys() != after.keys():
            raise ValueError('Evaluation identities changed')
        values[kind] = {'G':sum(after[i] and not before[i] for i in before),
                        'F':sum(before[i] and not after[i] for i in before), 'n':len(before)}
    return values


def precheck(root, task, c):
    """Exact native prefix plus actual-GPU probe and Adam-projection acceptance."""
    import copy
    started = time.monotonic()
    b = Backend(c, task); seed = b.spec['seeds'][0]
    model, tok, opt, named, order = b.build(seed)
    initial, order_hash = identity(b, model, order, seed)
    out = root / ('precheck_' + task); out.mkdir(exist_ok=False)
    ref = b.spec['references'][str(seed) + '/R']
    reference = {}
    for line in (Path(ref['path']) / 'training.jsonl').read_text().splitlines():
        v = json.loads(line); reference[v.get('step',v.get('update'))] = v['loss']
    if b.kind == 'mechanism':
        reference = {v['update']:v['loss'] for v in map(json.loads,(Path(b.spec['data_root'])/'mothers'/('s%d'%seed)/'training.jsonl').read_text().splitlines())}
        mother_hashes = json.loads((Path(b.spec['data_root'])/'mothers'/('s%d'%seed)/'adapter_hashes.json').read_text())
    for step in range(1, 5):
        _, _, _, loss, _, _ = update(b, model, tok, opt, named, order, step, None, 'baseline')
        if loss != reference[step]:
            raise ValueError('Native prefix loss mismatch at %s/%d' % (task,step))
        if b.kind == 'mechanism' and helpers(b).adapter_hash(helpers(b).capture_adapter(model)) != mother_hashes[str(step)]:
            raise ValueError('Native mother adapter mismatch')
    from toy_eval import _rng_signature
    signature = _rng_signature()
    adapter_before = helpers(b).adapter_hash(helpers(b).capture_adapter(model))
    old_grad = [(p,p.grad,None if p.grad is None else p.grad.detach().cpu().clone()) for _,p in named]
    ctrl = Controller('online', c['controller'], seed)
    rows = b.consumed(order,4)
    ctrl.observe(rows,[False]*16,[1.]*16,4)
    receipt = ctrl.refresh(b,model,tok,named,4)
    assert receipt is not None and ctrl.basis is not None and ctrl.basis.shape[0]>0
    if signature != _rng_signature() or adapter_before != helpers(b).adapter_hash(helpers(b).capture_adapter(model)):
        raise ValueError('Probe changed native state or RNG')
    if any(p.grad is not g or (g is not None and not torch.equal(g.detach().cpu(),v)) for p,g,v in old_grad):
        raise ValueError('Probe changed gradient values')
    before = flat_parameters(named)
    update(b,model,tok,opt,named,order,5,None,'baseline')
    state = copy.deepcopy(opt.state_dict())
    stats = ctrl.apply(named,before)
    for index, values in state['state'].items():
        for key, value in values.items():
            if not torch.equal(value,opt.state_dict()['state'][index][key]):
                raise ValueError('Projection rewrote Adam state')
    if stats['residual_projection_norm'] > stats['realized_error_norm'] + max(1e-7,1e-5*stats['original_norm']):
        raise ValueError('Actual parameter increment violates projection tolerance')
    atomic_json(root/('gpu_precheck_'+task+'.json'),{'passed':True,'task':task,'manifest_sha256':sha(root/'manifest.json'),
        'native_prefix_losses_exact':4,'probe_preserved_adapter_rng_gradients':True,'adam_state_untouched':True,
        'projection':stats,'initial_adapter_sha256':initial,'order_sha256':order_hash,'wall_seconds':time.monotonic()-started})


def train(root, job, c):
    started = time.monotonic(); task,seed,arm = job['task'],job['seed'],job['arm']
    out = root/'runs'/job['name']; out.mkdir(parents=True,exist_ok=False)
    b = Backend(c,task); model,tok,opt,named,order = b.build(seed)
    init_hash,order_hash = identity(b,model,order,seed,root,job)
    if any(p.dtype != torch.float32 for _,p in named):raise ValueError('LoRA must stay FP32')
    policy = b.policy(job['parent'])
    if policy is not None and policy.before_training():raise ValueError('Unexpected before-training action')
    ctrl = None if arm in ('baseline','oracle') else Controller(arm,c['controller'],seed)
    costs = {'load':time.monotonic()-started,'train_and_apply':0.,'refresh':0.,'evaluation':0.,'diagnostic':0.,'save':0.}
    points,ends,actual = [],{},0
    missed,normal = deque(maxlen=4),deque(maxlen=4)
    audit = {'poison_seen':0,'poison_rejected':0,'normal_rejected':0,'rank_zero_refreshes':0}
    def measure(step):
        t=time.monotonic(); full=step in c['full_eval_steps']
        metric=b.endpoint(model,tok,out,str(step),full=full)
        if full:ends[str(step)]=metric;metric=metric['V60']
        point={'step':step,**{k:metric[k] for k in ('trigger_hits','clean_target_hits','mapping_correct')}}
        points.append(point);costs['evaluation']+=time.monotonic()-t
        atomic_json(out/'progress.json',{'job':job['name'],'step':step,'point':point,'wall_seconds':time.monotonic()-started})
        print(json.dumps({'job':job['name'],**point}),flush=True)
    measure(0)
    for step in range(1,1251):
        fs=os.statvfs(root)
        if fs.f_bavail*fs.f_frsize<c['resources']['minimum_free_GiB']*2**30:raise ValueError('Storage reserve breached')
        t=time.monotonic()
        rows,keep,losses,loss,norm,projection=update(b,model,tok,opt,named,order,step,policy,arm,ctrl)
        costs['train_and_apply']+=time.monotonic()-t; actual+=int(any(keep))
        with open(out/'training.jsonl','a') as stream:
            stream.write(json.dumps({'step':step,'loss':loss,'gradient_norm':norm,'kept':sum(keep),'projection':projection})+'\n')
        with open(out/'filter.jsonl','a') as stream:
            stream.write(json.dumps({'step':step,'indices':[r['index'] for r in rows],'keep':keep,'losses':losses})+'\n')
        # Label-assisted written audits are not inputs to the main controller.
        for row,k in zip(rows,keep):
            if row['poison']:
                audit['poison_seen']+=1; audit['poison_rejected']+=int(not k)
                if k:missed.append(row['index'])
            else:
                audit['normal_rejected']+=int(not k);normal.append(row['index'])
        if ctrl is not None:
            t=time.monotonic();ctrl.observe(rows,keep,losses,step)
            refresh=ctrl.refresh(b,model,tok,named,step);torch.cuda.synchronize()
            costs['refresh']+=time.monotonic()-t
            if refresh:
                audit['rank_zero_refreshes']+=int(refresh['rank']==0)
                with open(out/'basis.jsonl','a') as stream:stream.write(json.dumps(refresh)+'\n')
            if step in c['diagnostic_steps'] and ctrl.basis is not None:
                t=time.monotonic();by_id={r['index']:r for r in b.rows['train']};data=[]
                diagnostic_basis=ctrl.basis.cpu()
                for kind,indices in [('missed_poison',missed),('normal',normal)]:
                    for i in dict.fromkeys(indices):
                        g=row_gradient(b,model,tok,named,by_id[i]);den=float(g.double().square().sum())
                        weights=diagnostic_basis@g
                        data.append({'kind':kind,'index':i,'gradient_norm':den**.5,
                                     'projection_energy':float(weights.double().square().sum())/den if den else None})
                costs['diagnostic']+=time.monotonic()-t
                atomic_json(out/('diagnostic_%d.json'%step),{'step':step,'rows':data,'role':'posthoc explanation only; never a scientific gate'})
        if step in c['curve_steps'] or step in c['full_eval_steps']:measure(step)
    if policy is not None and policy.after_training(1250):raise ValueError('Unexpected final policy action')
    t=time.monotonic();h=helpers(b)
    controller_state=None if ctrl is None else {'kind':ctrl.kind,'basis':None if ctrl.basis is None else ctrl.basis.cpu(),
        'rejected':list(ctrl.rejected),'clean_pool':list(ctrl.clean_pool),'last_refresh':ctrl.last_refresh,
        'refreshes':ctrl.refreshes,'seed':ctrl.seed,'config':ctrl.config}
    torch.save({'adapter':h.capture_adapter(model),'optimizer':h.cpu_clone(opt.state_dict()) if b.kind=='jsd' else
        {k:v for k,v in opt.state_dict().items()},'rng':rng(b),'step':1250,'controller':controller_state,
        'policy_state':None if policy is None else policy.__dict__},out/'state_1250.pt')
    costs['save']+=time.monotonic()-t;costs['wall']=time.monotonic()-started
    first=json.loads((out/'endpoint_0.json').read_text());last=json.loads((out/'endpoint_1250.json').read_text())
    atomic_json(out/'result.json',{'complete':True,'job':job,'last_step':1250,'actual_optimizer_updates':actual,
        'initial_adapter_sha256':init_hash,'order_sha256':order_hash,'endpoints':ends,'points':points,
        'learning':learning(last,first),'audit':audit,'controller':None if ctrl is None else ctrl.summary(),
        'policy':None if policy is None else policy.summary(),'cost':costs,
        'scope':'Registered fixed configuration; no unseen-seed claim. Main online selection has no trigger or poison-label access.'})


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--job');p.add_argument('--precheck')
    a=p.parse_args();root=a.root.resolve()
    if os.environ.get('JOBQ_GPU') not in ('0','1') or torch.cuda.device_count()!=1:raise ValueError('Existing worker must assign exactly one GPU')
    if a.precheck:
        c=guard(root,require_gpu=False);precheck(root,a.precheck,c)
    else:
        c=guard(root,require_gpu=False);job=next(j for j in c['jobs'] if j['name']==a.job)
        guard(root,task=job['task']);train(root,job,c)


if __name__=='__main__':
    main()
