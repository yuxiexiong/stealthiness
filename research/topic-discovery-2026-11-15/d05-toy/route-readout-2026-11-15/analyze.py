"""Compare new train-only predictions with unchanged saved held-out predictions."""
import csv
import gzip
import hashlib
import json
from pathlib import Path
import sys
from read_route import scores

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent))
from data import make_block, digest

receipt=json.loads((HERE/'raw/RECEIPT.json').read_text())
assert receipt['status']=='completed' and receipt['updates']==0
assert receipt['new_predictions']==30720 and receipt['forward_batches']==480
assert receipt['script_sha256']==hashlib.sha256((HERE/'read_route.py').read_bytes()).hexdigest()
original={r['path']:r for r in json.loads((HERE.parent/'execution-2026-11-15/ARTIFACT_MANIFEST.json').read_text())}
for state in receipt['states']:
    assert state['checkpoint_unchanged'] and state['model_unchanged']
    assert state['checkpoint_sha256']==original[f"{state['seed']}/prefix/state-20.pt"]['sha256']
comparison,queries,coverage=[],[],[]
for result in json.loads((HERE/'raw/TRAIN_RESULTS.json').read_text()):
    seed,read=result['seed'],result['read']
    block=make_block(seed)
    train=[r for r in block['rows']['train'] if r['mode']=='ROUTE']
    f=HERE/'raw'/result['raw_file']
    assert hashlib.sha256(f.read_bytes()).hexdigest()==result['raw_sha256']
    with gzip.open(f,'rt') as stream:obs=[json.loads(x) for x in stream]
    assert len(obs)==len(train)
    assert all(o['id']==r['id'] and o['truth']==r['truth'] for r,o in zip(train,obs))
    z=[o['logits'] for o in obs]
    assert all(o['prediction']==int(o['logits'][1]>o['logits'][0]) for o in obs)
    assert digest(z)==result['logits_sha256'] and scores(train,z)==result['metrics']
    f=HERE.parent/'revision-2026-11-15/source'/str(seed)/'prefix'/f'read-20-{read}.jsonl.gz'
    assert hashlib.sha256(f.read_bytes()).hexdigest()==original[f'{seed}/prefix/read-20-{read}.jsonl.gz']['sha256']
    with gzip.open(f,'rt') as stream:ev=[json.loads(x) for x in stream]
    assert all(r['id']==o['id'] and r['truth']==o['truth'] for r,o in zip(block['rows']['eval'],ev))
    pairs=[(r,o['logits']) for r,o in zip(block['rows']['eval'],ev) if r['mode']=='ROUTE']
    heldrows,heldz=zip(*pairs)
    hs=scores(heldrows,heldz)
    comparison.append(dict(seed=seed,read=read,**{'train_'+k:v for k,v in result['metrics'].items()},
        **{'heldout_'+k:v for k,v in hs.items()},gap_pp=100*(result['metrics']['accuracy']-hs['accuracy'])))
    for split,rows,logits in [('train',train,z),('heldout',heldrows,heldz)]:
        for q in range(4):
            ix=[i for i,r in enumerate(rows) if r['q']==q]
            queries.append(dict(seed=seed,read=read,split=split,q=q,
                **scores([rows[i] for i in ix],[logits[i] for i in ix])))
    # Descriptive correlation of each irrelevant bit with the target, not a new model test.
    if read=='P':
        for split in ('train','eval'):
            for q in range(4):
                content=[r for r in block['contents'][split] if r['q']==q]
                for j in range(4):
                    if j!=q:
                        eq=sum(r['bits'][j]==r['truth'] for r in content)/len(content)
                        coverage.append(dict(seed=seed,split=split,q=q,irrelevant_bit=j,equality_fraction=eq))
for name,rows in [('COMPARISON.csv',comparison),('QUERY_COMPARISON.csv',queries),('CONTENT_CORRELATIONS.csv',coverage)]:
    with (HERE/name).open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
(HERE/'RECEIPT.json').write_text(json.dumps(receipt,indent=2)+'\n')
(HERE/'TRAIN_RESULTS.json').write_text((HERE/'raw/TRAIN_RESULTS.json').read_text())
(HERE/'ANALYSIS_CHECK.json').write_text(json.dumps(dict(train_predictions_verified=30720,train_files=15,
    heldout_files_hash_verified=15,checkpoint_hashes_verified=5,updates=0,new_forwards_during_analysis=0),indent=2)+'\n')
for r in comparison:
    print(r['seed'],r['read'],'train',round(r['train_accuracy']*100,2),'heldout',round(r['heldout_accuracy']*100,2),'gap',round(r['gap_pp'],2))
print('Query accuracies, P read:')
for r in queries:
    if r['read']=='P':print(r['seed'],r['split'],r['q'],round(r['accuracy'],4))
