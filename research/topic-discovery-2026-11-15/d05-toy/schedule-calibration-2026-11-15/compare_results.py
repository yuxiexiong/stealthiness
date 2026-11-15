"""Same-gate paired comparison of existing mixed and reordered clean trajectories."""
import csv
import json
from pathlib import Path
import sys

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent))
from metrics import prefix_qualified

rows=[];endpoints=[]
for seed in range(1101,1106):
    terminal={}
    for condition,base in [('original',HERE.parent/'revision-2026-11-15/source'),('reordered',HERE/'raw')]:
        for chunk in range(21):
            summary=json.loads((base/str(seed)/'prefix'/f'summary-{chunk:02}.json').read_text())
            values=[summary[r] for r in ('P','W0','W1')]
            overall=dict(qualified=prefix_qualified(summary),min_route=min(v['route_accuracy'] for v in values),
                min_aux=min(v[m]['balanced_accuracy'] for v in values for m in ('LEFT','RIGHT')),
                min_aux_positive=min(v[m]['positive_recall'] for v in values for m in ('LEFT','RIGHT')))
            rows.append(dict(seed=seed,condition=condition,update=chunk*120,read='all_min',**overall))
            for r in ('P','W0','W1'):
                v=summary[r]
                rows.append(dict(seed=seed,condition=condition,update=chunk*120,read=r,
                    route=v['route_accuracy'],left=v['LEFT']['balanced_accuracy'],right=v['RIGHT']['balanced_accuracy'],
                    left_pos=v['LEFT']['positive_recall'],right_pos=v['RIGHT']['positive_recall'],
                    left_neg=v['LEFT']['negative_recall'],right_neg=v['RIGHT']['negative_recall']))
            if chunk==20:terminal[condition]=overall
        endpoints.append(dict(seed=seed,condition=condition,**terminal[condition]))
    print(seed,terminal)
with (HERE/'TRAJECTORIES.csv').open('w') as f:
    w=csv.DictWriter(f,fieldnames=list(dict.fromkeys(k for r in rows for k in r)))
    w.writeheader();w.writerows(rows)
(HERE/'ENDPOINTS.json').write_text(json.dumps(endpoints,indent=2)+'\n')
