import json
from pathlib import Path
from collections import defaultdict
import numpy as np
R=Path('/Users/ruizhixu/Documents/ChatGPT/stealthiness');D=json.loads((R/'output/segmentation-heatmap-linkage-2026-09-28/data.json').read_text());M=R/'output/segmentation-heatmap-linkage-2026-09-28/source/attribution-microscope/runs/maps'
ids=list(D['maps']['P-1.0@s79']['A']);by=defaultdict(list)
for q in ids:by[D['samples'][q]].append(q)
rows=[]
for path in ['seed1','seed2','dose','e3']:
 for p in D['paths'][path]:
  z=np.load(M/p['tag']/'p_core_trig.npz');ref=np.load(M/p['ref']/'p_core_trig.npz')
  vals=[]
  for qs in by.values():
   v=[]
   for q in qs:
    a=z[f'{q}_logits'];b=ref[f'{q}_logits'];v.append([a[1],a[1]-a[0],a[2],b[1],b[1]-b[0],b[2],a[1]-b[1],(a[1]-a[0])-(b[1]-b[0])])
   vals.append(np.mean(v,axis=0))
  keys=['target_logit','target_minus_top_logit','target_minus_reference_logit','clean_target_logit','clean_target_minus_top_logit','clean_target_minus_reference_logit','target_logit_change_vs_clean','gap_change_vs_clean']
  row={'tag':p['tag'],'path':path,'asr':p['asr'],'n_images':len(vals),'means':dict(zip(keys,map(float,np.mean(vals,axis=0)))),'medians':dict(zip(keys,map(float,np.median(vals,axis=0))))}
  rows.append(row)
  if p['tag'] in ['P-1.0@s79','P-1.0-D@s240','P-1.0-D@s300','P-1.0-D@s380','P-1.0-D2@s80','P-1.0-D2@s200','P-1.0-D2@s240','P-1.0-D2@s300','P-0.1','P-0.38','P-0.42','P-0.4-ps3']:
   print(row)
  z.close();ref.close()
(R/'output/visual-first-hotspot-review-2026-09-28/comparison/cheap-logit-baseline.json').write_text(json.dumps(rows,indent=2))
