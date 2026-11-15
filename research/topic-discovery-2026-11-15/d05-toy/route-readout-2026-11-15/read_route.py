"""Read five frozen final states on ROUTE training support; never train."""
import argparse
import gzip
import hashlib
import json
import math
import os
from pathlib import Path
import statistics as st
import subprocess
import sys
import time


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def scores(rows, logits):
    assert rows and len(rows) == len(logits)
    assert all(len(z) == 2 and all(map(math.isfinite, z)) for z in logits)
    pairs = [(r, int(z[1] > z[0]), (z[1]-z[0])*(2*r['truth']-1)) for r,z in zip(rows,logits)]
    return dict(n=len(rows), accuracy=st.mean(p==r['truth'] for r,p,m in pairs),
        positive_recall=st.mean(p for r,p,m in pairs if r['truth']==1),
        negative_recall=st.mean(1-p for r,p,m in pairs if r['truth']==0),
        ce=st.mean(max(-m,0)+math.log1p(math.exp(-abs(m))) for r,p,m in pairs))


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--original-root',type=Path,required=True)
    parser.add_argument('--out',type=Path)
    parser.add_argument('--check',action='store_true')
    args=parser.parse_args()
    root=args.original_root
    sys.path.insert(0,str(root/'code'))
    from data import SEEDS, READS, make_block, digest, encode
    from run import build_model, model_hash
    identity=json.loads((root/'code/audit/PREPARED_IDENTITY.json').read_text())
    for name,h in identity['sources'].items():
        assert sha(root/'code'/name)==h, name
    frozen=json.loads((root/'code/audit/FROZEN_INPUTS.json').read_text())['blocks']
    manifest=json.loads((root/'prepared/manifest.json').read_text())
    assert digest(manifest)==identity['sha256']
    original={x['path']:x for x in json.loads((root/'ARTIFACT_MANIFEST.json').read_text())}
    blocks=[]
    for seed in SEEDS:
        b=make_block(seed); assert digest(b)==frozen[str(seed)]
        rows=[r for r in b['rows']['train'] if r['mode']=='ROUTE']
        assert len(rows)==2048 and sum(r['truth'] for r in rows)==1024
        assert all(len(encode(r,'P',b['vocab']))==9 for r in rows)
        state=root/'run'/str(seed)/'prefix/state-20.pt'
        assert sha(state)==original[f'{seed}/prefix/state-20.pt']['sha256']
        blocks.append((b,rows,state))
    test=scores([{'truth':0},{'truth':1}],[[1.,-1.],[-1.,1.]])
    assert test['accuracy']==test['positive_recall']==test['negative_recall']==1
    assert abs(test['ce']-math.log1p(math.exp(-2)))<1e-12
    bad=scores([{'truth':0},{'truth':1}],[[0.,1.],[0.,1.]])
    assert bad['accuracy']==.5 and bad['positive_recall']==1 and bad['negative_recall']==0
    if args.check:
        print(json.dumps(dict(status='CPU_check_passed',states=5,new_forwards=0,planned_rows=30720,planned_batches=480)))
        return
    assert args.out is not None and not args.out.exists()
    # Bind exactly one presently idle physical device before importing torch.
    selected=os.environ['CUDA_VISIBLE_DEVICES']
    assert selected.startswith('GPU-') and ',' not in selected
    processes=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader'],text=True)
    assert all(line.split(',')[0].strip()!=selected for line in processes.splitlines()), 'GPU occupied'
    args.out.mkdir(parents=True)
    start=time.time()
    receipt=dict(status='running',started_unix=start,updates=0,new_predictions=0,forward_batches=0,
                 input_manifest_sha256=identity['sha256'],script_sha256=sha(Path(__file__)),states=[])
    try:
        os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
        import torch
        assert torch.__version__=='2.4.0+cu121'
        torch.set_num_threads(4);torch.set_default_dtype(torch.float32)
        torch.use_deterministic_algorithms(True)
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        receipt.update(torch=torch.__version__,cuda=torch.version.cuda,device_name=torch.cuda.get_device_name(0))
        results=[]
        for block,rows,path in blocks:
            seed=block['seed']; before=sha(path)
            state=torch.load(path,map_location='cpu',weights_only=False)
            assert state['epoch']==20
            assert all(float(v['step'])==2400 for v in state['optimizer']['state'].values())
            model=build_model(torch,len(block['vocab'])).to('cuda:0')
            model.load_state_dict(state['model'],strict=True);model.eval()
            assert all(p.dtype==torch.float32 for p in model.parameters())
            before_model=model_hash(torch,model)
            for read in READS:
                inputs=[encode(r,'P' if read=='P' else 'W',block['vocab'],0 if read=='P' else int(read[1])) for r in rows]
                logits=[]
                with torch.no_grad():
                    for offset in range(0,len(rows),64):
                        batch=torch.tensor(inputs[offset:offset+64],dtype=torch.long,device='cuda:0')
                        logits.extend(model(batch).cpu().tolist())
                        receipt['forward_batches']+=1
                assert all(p.grad is None for p in model.parameters())
                receipt['new_predictions']+=len(logits)
                filename=f'{seed}-{read}.jsonl.gz'
                with gzip.open(args.out/filename,'wt') as f:
                    for row,z in zip(rows,logits):
                        f.write(json.dumps(dict(id=row['id'],truth=row['truth'],logits=z,prediction=int(z[1]>z[0])))+'\n')
                byq={}
                for q in range(4):
                    ix=[i for i,r in enumerate(rows) if r['q']==q]
                    byq[str(q)]=scores([rows[i] for i in ix],[logits[i] for i in ix])
                results.append(dict(seed=seed,read=read,metrics=scores(rows,logits),by_q=byq,
                                    raw_file=filename,raw_sha256=sha(args.out/filename),logits_sha256=digest(logits)))
            assert model_hash(torch,model)==before_model and sha(path)==before
            receipt['states'].append(dict(seed=seed,checkpoint_sha256=before,model_sha256=before_model,
                                           checkpoint_unchanged=True,model_unchanged=True))
            del model,state
        assert receipt['new_predictions']==30720 and receipt['forward_batches']==480
        (args.out/'TRAIN_RESULTS.json').write_text(json.dumps(results,indent=2)+'\n')
        receipt['status']='completed'
    except BaseException as error:
        receipt.update(status='failed',error=repr(error))
        raise
    finally:
        receipt.update(finished_unix=time.time(),wall_seconds=time.time()-start)
        receipt['reserved_gpu_hours']=receipt['wall_seconds']/3600
        (args.out/'RECEIPT.json').write_text(json.dumps(receipt,indent=2)+'\n')
        print(json.dumps({k:v for k,v in receipt.items() if k!='states'}))


if __name__=='__main__':
    main()
