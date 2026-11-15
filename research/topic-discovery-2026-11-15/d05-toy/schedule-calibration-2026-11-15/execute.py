"""Five clean prefixes only. Frozen scheduling changes; no contamination branches."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def write(p, value):
    p.write_text(json.dumps(value, sort_keys=True, indent=2) + '\n')


def own_hashes():
    return {n:sha(HERE/n) for n in ('schedule.py','execute.py','supervise.py')}


def original_inputs(root):
    from data import make_block, SEEDS, digest
    identity=json.loads((root/'code/audit/PREPARED_IDENTITY.json').read_text())
    for name,h in identity['sources'].items():
        assert sha(root/'code'/name)==h, name
    manifest=json.loads((root/'prepared/manifest.json').read_text())
    assert digest(manifest)==identity['sha256']
    frozen=json.loads((root/'code/audit/FROZEN_INPUTS.json').read_text())['blocks']
    artifacts={r['path']:r for r in json.loads((root/'ARTIFACT_MANIFEST.json').read_text())}
    blocks=[]
    for seed in SEEDS:
        block=make_block(seed)
        assert digest(block)==frozen[str(seed)]
        for name in ('state-00.pt','summary-00.json'):
            rel=f'{seed}/prefix/{name}'
            assert sha(root/'run'/rel)==artifacts[rel]['sha256']
        blocks.append(block)
    return identity,artifacts,blocks


def prepare(root, directory):
    assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
    assert not directory.exists()
    start=time.time()
    from data import digest
    from run import build_model,model_hash
    from schedule import schedule,audit
    from metrics import prefix_qualified
    identity,artifacts,blocks=original_inputs(root)
    directory.mkdir(parents=True)
    import torch
    assert torch.__version__=='2.4.0+cu121'
    torch.set_num_threads(4)
    checks=[]
    initial=[]
    for block in blocks:
        seed=block['seed']
        events=schedule(block)
        checks.append(audit(block,events))
        write(directory/f'events-{seed}.json',events)
        # Regression: duplicate/drop, front-phase reorder, and remainder changes are rejected.
        for kind in ('duplicate','front','remainder'):
            changed=events.copy()
            if kind=='duplicate': changed[-1]=changed[-2]
            elif kind=='front': changed[0],changed[1]=changed[1],changed[0]
            else: changed[-1],changed[-2]=changed[-2],changed[-1]
            try: audit(block,changed)
            except AssertionError: pass
            else: raise AssertionError('tamper accepted: '+kind)
        state=torch.load(root/'run'/str(seed)/'prefix/state-00.pt',map_location='cpu',weights_only=False)
        assert state['epoch']==0 and state['optimizer']['state']=={}
        model=build_model(torch,len(block['vocab']))
        model.load_state_dict(state['model'],strict=True)
        assert all(torch.equal(v,state['model'][k]) for k,v in model.state_dict().items())
        optimizer=torch.optim.AdamW(model.parameters(),lr=.001,betas=(.9,.999),eps=1e-8,weight_decay=0.)
        assert optimizer.state_dict()==state['optimizer']
        optimizer.load_state_dict(state['optimizer'])
        assert optimizer.state_dict()==state['optimizer']
        torch.set_rng_state(state['rng'])
        assert torch.equal(torch.get_rng_state(),state['rng'])
        assert len(state['cuda_rng'])==1 and state['cuda_rng'][0].dtype==torch.uint8
        initial.append(dict(seed=seed,model_sha256=model_hash(torch,model),
            state_sha256=artifacts[f'{seed}/prefix/state-00.pt']['sha256'],
            cpu_rng_sha256=hashlib.sha256(state['rng'].numpy().tobytes()).hexdigest(),
            cuda_rng_sha256=hashlib.sha256(state['cuda_rng'][0].numpy().tobytes()).hexdigest(),
            optimizer_empty=True,optimizer_groups_equal=True,model_tensors_equal=True))
    # Qualification stays the original function; test exact threshold and meaningful failures.
    import copy
    base={r:dict(route_accuracy=.95,LEFT=dict(balanced_accuracy=.95,positive_recall=.90),
        RIGHT=dict(balanced_accuracy=.95,positive_recall=.90)) for r in ('P','W0','W1')}
    assert prefix_qualified(base)
    for path in [('P','route_accuracy'),('W0','LEFT','balanced_accuracy'),('W1','RIGHT','positive_recall')]:
        bad=copy.deepcopy(base); obj=bad
        for k in path[:-1]: obj=obj[k]
        obj[path[-1]]-=1e-5
        assert not prefix_qualified(bad)
    gap=copy.deepcopy(base);gap['W0']['route_accuracy']=.98
    assert prefix_qualified(gap)
    gap['W0']['route_accuracy']=.98001
    assert not prefix_qualified(gap)
    result=dict(status='CPU_passed',torch=torch.__version__,identity=identity,wrapper_sources=own_hashes(),
        exposure_checks=checks,initial_states=initial,initial_gpu_readout_required=True,
        schedule_files={f'events-{b["seed"]}.json':sha(directory/f'events-{b["seed"]}.json') for b in blocks},
        tamper_cases_passed=15,qualification_cases_passed=6,cpu_seconds=time.time()-start,
        scientific_forwards=0,optimizer_updates=0)
    write(directory/'CPU_CHECK.json',result)
    print(json.dumps(result))


def execute(root, prepared, out):
    from data import digest
    from run import build_model,model_hash,evaluate,write_json
    from schedule import examples,audit
    from metrics import prefix_qualified
    frozen=json.loads((prepared/'CPU_CHECK.json').read_text())
    assert frozen['status']=='CPU_passed' and own_hashes()==frozen['wrapper_sources']
    identity,artifacts,blocks=original_inputs(root)
    assert identity==frozen['identity']
    all_events={}
    for b in blocks:
        name=f'events-{b["seed"]}.json'
        assert sha(prepared/name)==frozen['schedule_files'][name]
        events=json.loads((prepared/name).read_text())
        assert audit(b,events)==next(c for c in frozen['exposure_checks'] if c['seed']==b['seed'])
        all_events[b['seed']]=events
    selected=os.environ['CUDA_VISIBLE_DEVICES']
    assert selected.startswith('GPU-') and ',' not in selected
    lines=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader'],text=True).splitlines()
    assert all(line.split(',')[0].strip()!=selected for line in lines),'GPU occupied'
    assert not out.exists()
    out.mkdir(parents=True)
    start=time.time()
    receipt=dict(status='running',started_unix=start,updates=0,completed=[],actual_branches=0,
                 source_hashes=own_hashes(),cpu_check_sha256=sha(prepared/'CPU_CHECK.json'))
    write(out/'RECEIPT.json',receipt)
    try:
        os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
        import torch
        assert torch.__version__==frozen['torch']
        torch.set_num_threads(4);torch.set_default_dtype(torch.float32)
        torch.use_deterministic_algorithms(True)
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        receipt.update(torch=torch.__version__,cuda=torch.version.cuda,device_name=torch.cuda.get_device_name(0))
        for block in blocks:
            seed=block['seed'];directory=out/str(seed)/'prefix';directory.mkdir(parents=True)
            initial=next(v for v in frozen['initial_states'] if v['seed']==seed)
            statefile=root/'run'/str(seed)/'prefix/state-00.pt'
            state=torch.load(statefile,map_location='cpu',weights_only=False)
            model=build_model(torch,len(block['vocab'])).to('cuda:0')
            model.load_state_dict(state['model'],strict=True)
            assert model_hash(torch,model)==initial['model_sha256']
            optimizer=torch.optim.AdamW(model.parameters(),lr=.001,betas=(.9,.999),eps=1e-8,weight_decay=0.)
            optimizer.load_state_dict(state['optimizer']);assert not optimizer.state
            torch.set_rng_state(state['rng']);torch.cuda.set_rng_state_all(state['cuda_rng'])
            assert torch.equal(torch.get_rng_state(),state['rng'])
            assert torch.equal(torch.cuda.get_rng_state(0),state['cuda_rng'][0])
            oldsummary=json.loads((root/'run'/str(seed)/'prefix/summary-00.json').read_text())
            flat=list(examples(block,all_events[seed]))
            trajectory={}
            for chunk in range(21):
                if chunk:
                    model.train(); losses=[]; consumed=[]
                    segment=flat[(chunk-1)*7680:chunk*7680]
                    update_start=time.time()
                    for offset in range(0,7680,64):
                        batch=segment[offset:offset+64]
                        ids=torch.tensor([x[1] for x in batch],dtype=torch.long,device='cuda:0')
                        labels=torch.tensor([x[2] for x in batch],dtype=torch.long,device='cuda:0')
                        optimizer.zero_grad(set_to_none=True)
                        loss=torch.nn.functional.cross_entropy(model(ids),labels)
                        if not torch.isfinite(loss): raise RuntimeError('Nonfinite loss')
                        loss.backward()
                        norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
                        optimizer.step();receipt['updates']+=1
                        step=(chunk-1)*120+offset//64+1
                        losses.append([loss.item(),norm.item()]);consumed.extend(x[0] for x in batch)
                        if step==560:
                            assert all(float(v['step'])==560 for v in optimizer.state.values())
                            write(directory/'BOUNDARY.json',dict(update=560,optimizer_steps=560,reset=False))
                    write_json(directory/f'train-{chunk:02}.json',dict(rows=consumed,losses_and_grad_norm=losses,
                        update_seconds=time.time()-update_start,encoded_input_hash=digest(segment)))
                read_start=time.time()
                summary=evaluate(torch,model,block,directory,chunk,'cuda:0')
                if chunk==0 and summary!=oldsummary: raise RuntimeError('Initial readout fidelity failed')
                trajectory[str(chunk)]=summary
                torch.save(dict(model=model.state_dict(),optimizer=optimizer.state_dict(),epoch=chunk,
                    update=chunk*120,rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all()),directory/f'state-{chunk:02}.pt')
                write(directory/f'timing-{chunk:02}.json',dict(read_save_seconds=time.time()-read_start,elapsed_seconds=time.time()-start))
            assert all(float(v['step'])==2400 for v in optimizer.state.values())
            assert sha(statefile)==initial['state_sha256']
            item=dict(seed=seed,qualified=prefix_qualified(trajectory['20']),updates=2400,
                initial_fidelity=True,initial_state_unchanged=True,
                final_state_sha256=sha(directory/'state-20.pt'))
            receipt['completed'].append(item);write(out/'RECEIPT.json',receipt)
            print(json.dumps(item),flush=True)
            del model,optimizer,state
        assert receipt['updates']==12000 and len(receipt['completed'])==5
        receipt.update(status='completed',qualified=sum(x['qualified'] for x in receipt['completed']))
    except BaseException as error:
        receipt.update(status='failed',error=repr(error))
        raise
    finally:
        receipt.update(finished_unix=time.time(),wall_seconds=time.time()-start)
        receipt['reserved_gpu_hours']=receipt['wall_seconds']/3600
        write(out/'RECEIPT.json',receipt)
        print(json.dumps(receipt),flush=True)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--original-root',required=True,type=Path)
    group=parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--prepare',type=Path);group.add_argument('--execute',type=Path)
    parser.add_argument('--prepared',type=Path)
    args=parser.parse_args();sys.path.insert(0,str(args.original_root/'code'))
    if args.prepare: prepare(args.original_root,args.prepare)
    else:
        assert args.prepared is not None
        execute(args.original_root,args.prepared,args.execute)


if __name__=='__main__': main()
