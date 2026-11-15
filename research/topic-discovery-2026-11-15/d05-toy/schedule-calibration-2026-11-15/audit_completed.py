"""CPU-only audit of the completed five-prefix run; no model instantiation."""
import argparse
import gzip
import hashlib
import json
import math
from pathlib import Path
import sys


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True)
    p.add_argument('--original-root',type=Path,required=True)
    args=p.parse_args();sys.path.insert(0,str(args.original_root/'code'))
    from data import make_block,digest
    from schedule import examples
    from metrics import summarize,add_w_average,prefix_qualified
    import torch
    torch.set_num_threads(2)
    receipt=json.loads((args.root/'run/RECEIPT.json').read_text())
    assert receipt['status']=='completed' and receipt['updates']==12000 and receipt['actual_branches']==0
    cpu=json.loads((args.root/'prepared/CPU_CHECK.json').read_text())
    endpoints=[];states=0;records=0;updates=0
    for entry in cpu['initial_states']:
        seed=entry['seed'];block=make_block(seed)
        events=json.loads((args.root/f'prepared/events-{seed}.json').read_text())
        flat=list(examples(block,events));directory=args.root/f'run/{seed}/prefix'
        original=torch.load(args.original_root/f'run/{seed}/prefix/state-00.pt',map_location='cpu',weights_only=False)
        assert hashlib.sha256((args.original_root/f'run/{seed}/prefix/state-00.pt').read_bytes()).hexdigest()==entry['state_sha256']
        for chunk in range(21):
            state=torch.load(directory/f'state-{chunk:02}.pt',map_location='cpu',weights_only=False)
            assert state['epoch']==chunk and state['update']==chunk*120
            assert state['optimizer']['param_groups']==original['optimizer']['param_groups']
            assert all(torch.isfinite(v).all() for v in state['model'].values())
            if chunk==0:
                assert all(torch.equal(v,original['model'][k]) for k,v in state['model'].items())
                assert state['optimizer']==original['optimizer']
                assert torch.equal(state['rng'],original['rng'])
                assert len(state['cuda_rng'])==1 and torch.equal(state['cuda_rng'][0],original['cuda_rng'][0])
            else:
                assert len(state['optimizer']['state'])==len(state['model'])
                for v in state['optimizer']['state'].values():
                    assert float(v['step'])==chunk*120
                    assert all(torch.isfinite(t).all() for t in v.values() if torch.is_tensor(t))
                train=json.loads((directory/f'train-{chunk:02}.json').read_text())
                segment=flat[(chunk-1)*7680:chunk*7680]
                assert train['rows']==[x[0] for x in segment]
                assert train['encoded_input_hash']==digest(segment)
                assert len(train['losses_and_grad_norm'])==120
                assert all(math.isfinite(x) for pair in train['losses_and_grad_norm'] for x in pair)
                updates+=120
            states+=1
            summary=json.loads((directory/f'summary-{chunk:02}.json').read_text())
            computed={}
            for read in ('P','W0','W1'):
                with gzip.open(directory/f'read-{chunk:02}-{read}.jsonl.gz','rt') as f:
                    obs=[json.loads(line) for line in f]
                rows=block['rows']['eval']
                assert len(obs)==len(rows)
                assert all(o['id']==r['id'] and o['truth']==r['truth'] for r,o in zip(rows,obs))
                assert all(o['prediction']==int(o['logits'][1]>o['logits'][0]) for o in obs)
                z=[o['logits'] for o in obs]
                assert digest(z)==summary['raw_logits_hashes'][read]
                computed[read]=summarize(rows,z,block['target']);records+=len(rows)
            assert add_w_average(computed)=={k:v for k,v in summary.items() if k!='raw_logits_hashes'}
            if chunk==0:
                assert summary==json.loads((args.original_root/f'run/{seed}/prefix/summary-00.json').read_text())
        assert json.loads((directory/'BOUNDARY.json').read_text())==dict(update=560,optimizer_steps=560,reset=False)
        finalhash=hashlib.sha256((directory/'state-20.pt').read_bytes()).hexdigest()
        done=next(x for x in receipt['completed'] if x['seed']==seed)
        assert finalhash==done['final_state_sha256'] and prefix_qualified(summary)==done['qualified']
        endpoints.append(dict(seed=seed,qualified=done['qualified'],reads={r:dict(route=summary[r]['route_accuracy'],
            LEFT=summary[r]['LEFT'],RIGHT=summary[r]['RIGHT']) for r in ('P','W0','W1')}))
    assert states==105 and updates==12000 and records==967680
    result=dict(status='passed',states=states,updates=updates,existing_predictions_verified=records,
                initial_exact_fidelity=5,qualified=sum(x['qualified'] for x in endpoints),actual_branches=0,
                new_forwards_during_audit=0,endpoints=endpoints)
    assert result['qualified']==receipt['qualified']
    (args.root/'FINAL_AUDIT.json').write_text(json.dumps(result,indent=2)+'\n')
    manifest=[dict(path=str(f.relative_to(args.root/'run')),size=f.stat().st_size,
        sha256=hashlib.sha256(f.read_bytes()).hexdigest()) for f in sorted((args.root/'run').rglob('*')) if f.is_file()]
    (args.root/'ARTIFACT_MANIFEST.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='endpoints'}))


if __name__=='__main__':main()
