"""Post hoc summaries of saved predictions only. Standard library, no model calls."""
import csv
import gzip
import hashlib
import json
import math
from pathlib import Path
import statistics as st
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from data import make_block, digest, SEEDS, READS
from metrics import prefix_qualified, summarize


def main():
    frozen = json.loads((HERE.parent / 'audit/FROZEN_INPUTS.json').read_text())['blocks']
    manifest = json.loads((HERE.parent / 'execution-2026-11-15/ARTIFACT_MANIFEST.json').read_text())
    expected = {x['path']: x for x in manifest}
    sources = sorted(p for p in (HERE / 'source').rglob('*') if p.is_file())
    assert len(sources) == 625
    for p in sources:
        k = str(p.relative_to(HERE / 'source'))
        b = p.read_bytes()
        assert len(b) == expected[k]['size'] and hashlib.sha256(b).hexdigest() == expected[k]['sha256'], k
    records, epochs, breakdown = [], [], []
    for seed in SEEDS:
        block = make_block(seed)
        assert digest(block) == frozen[str(seed)]
        raw = block['rows']['eval']
        base = HERE / 'source' / str(seed) / 'prefix'
        for ep in range(21):
            summary = json.loads((base / f'summary-{ep:02}.json').read_text())
            r = dict(seed=seed, epoch=ep, qualified=prefix_qualified(summary),
                     min_route=min(summary[v]['route_accuracy'] for v in READS),
                     min_aux=min(summary[v][m]['balanced_accuracy'] for v in READS for m in ('LEFT','RIGHT')),
                     min_pos=min(summary[v][m]['positive_recall'] for v in READS for m in ('LEFT','RIGHT')))
            if ep:
                train = json.loads((base / f'train-{ep:02}.json').read_text())
                pairs = train['losses_and_grad_norm']
                assert len(pairs) == 120 and all(math.isfinite(x) for pair in pairs for x in pair)
                r.update(loss_mean=st.mean(x[0] for x in pairs), loss_max=max(x[0] for x in pairs),
                         grad_mean=st.mean(x[1] for x in pairs), grad_max=max(x[1] for x in pairs),
                         clip_fraction=st.mean(x[1] > 1 for x in pairs))
            epochs.append(r)
            for read in READS:
                with gzip.open(base / f'read-{ep:02}-{read}.jsonl.gz','rt') as f:
                    obs = [json.loads(line) for line in f]
                assert len(obs) == len(raw)
                assert all(o['id']==x['id'] and o['truth']==x['truth'] for x,o in zip(raw,obs))
                logits = [o['logits'] for o in obs]
                assert digest(logits) == summary['raw_logits_hashes'][read]
                assert summarize(raw, logits, block['target']) == summary[read]
                for mode in ('ROUTE','LEFT','RIGHT'):
                    sub = [(x,o) for x,o in zip(raw,obs) if x['mode']==mode]
                    def ce(items):
                        margins = [(o['logits'][1]-o['logits'][0])*(2*x['truth']-1) for x,o in items]
                        return st.mean(max(-m,0)+math.log1p(math.exp(-abs(m))) for m in margins)
                    pos = [(x,o) for x,o in sub if x['truth']==1]
                    neg = [(x,o) for x,o in sub if x['truth']==0]
                    positive = st.mean(o['prediction'] for x,o in pos)
                    negative = st.mean(1-o['prediction'] for x,o in neg)
                    records.append(dict(seed=seed,epoch=ep,read=read,task=mode,n=len(sub),
                        balanced_accuracy=(positive+negative)/2,positive_recall=positive,negative_recall=negative,
                        balanced_ce=(ce(pos)+ce(neg))/2,predicted_yes=st.mean(o['prediction'] for x,o in sub)))
                    if ep in (0,5,10,15,16,17,18,19,20):
                        for q in sorted({x['q'] for x,o in sub}):
                            qs=[(x,o) for x,o in sub if x['q']==q]
                            breakdown.append(dict(seed=seed,epoch=ep,read=read,task=mode,q=q,
                                accuracy=st.mean(x['truth']==o['prediction'] for x,o in qs),ce=ce(qs)))
    for filename, rows in [('TRAJECTORIES.csv',records),('EPOCHS.csv',epochs),('QUERY_DIAGNOSTIC.csv',breakdown)]:
        with (HERE/filename).open('w') as f:
            writer=csv.DictWriter(f,fieldnames=list(dict.fromkeys(k for r in rows for k in r)))
            writer.writeheader(); writer.writerows(rows)
    result=dict(status='existing_artifact_analysis_only',verified_files=len(sources),
        summaries_recomputed=315,existing_predictions=315*3072,model_forwards=0,updates=0,
        source_manifest_sha256=hashlib.sha256((HERE.parent/'execution-2026-11-15/ARTIFACT_MANIFEST.json').read_bytes()).hexdigest(),
        qualifications={seed:[r['epoch'] for r in epochs if r['seed']==seed and r['qualified']] for seed in SEEDS})
    (HERE/'ANALYSIS_RECEIPT.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))
    for seed in SEEDS:
        print('SEED',seed,'epoch min_ROUTE min_aux min_pos loss clip')
        for r in epochs:
            if r['seed']==seed:
                print(r['epoch'], *(round(r[k],4) for k in ('min_route','min_aux','min_pos')),round(r.get('loss_mean',0),4),round(r.get('clip_fraction',0),3))


if __name__=='__main__':
    main()
