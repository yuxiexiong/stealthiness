import json
from pathlib import Path
from collections import defaultdict
import numpy as np
ROOT=Path('/Users/ruizhixu/Documents/ChatGPT/stealthiness')
D=ROOT/'output/segmentation-heatmap-linkage-2026-09-28'
M=D/'source/attribution-microscope/runs/maps'
reg=json.loads((D/'data.json').read_text())
curves=json.loads((ROOT/'output/llava-supplement-2026-09-28/source/attribution-microscope/supplement/phase2b/figs/curves2b.json').read_text())
ids=sorted(reg['maps']['P-1.0@s79']['A'],key=int)
byimage=defaultdict(list)
for q in ids:byimage[reg['samples'][q]].append(q)
mask=np.ones(576,bool);mask[[550,551,574,575]]=False
out=[]; cache={}; maxcheck=0
for p in curves['points']:
 tags=[p['tag'],p['ref']]
 z=[]
 for tag in tags:
  if tag not in cache:
   with np.load(M/tag/'p_core_trig.npz') as f:cache[tag]={k:f[k].astype(float) for k in f.files if '_img' in k}
  z.append(cache[tag])
 point={k:p.get(k) for k in ['tag','path','step','dose_pct','asr','ref']}
 for ins,suffix in [('A','A_img_signed'),('B','B_img')]:
  values={}
  for q in ids:
   vals={}
   for target in ['T2','C']:
    v=[f[f'{q}_T2_{suffix}']-(f[f'{q}_T3_{suffix}'] if target=='C' else 0) for f in z]
    a,b=[vv[mask] for vv in v]
    n=np.linalg.norm(a)*np.linalg.norm(b)
    vals[target]=float(a@b/n) if n else None
   values[q]=vals
  imagevals={iid:{tar:float(np.mean([values[q][tar] for q in qs])) for tar in ['T2','C']} for iid,qs in byimage.items() if all(values[q][tar] is not None for q in qs for tar in ['T2','C'])}
  means={tar:float(np.mean([v[tar] for v in imagevals.values()])) for tar in ['T2','C']}
  medians={tar:float(np.median([v[tar] for v in values.values() if v[tar] is not None])) for tar in ['T2','C']}
  for tar in ['T2','C']:
   maxcheck=max(maxcheck,abs(medians[tar]-p['m60'][f'{ins}_{tar}']['offtrigger_cos']['median']))
  gap=np.array([v['C']-v['T2'] for v in imagevals.values()])
  point[ins]={'image_means':means,'q_medians_for_source_check':medians,'images':len(imagevals),'questions':len(values),'gap_mean':float(gap.mean()),'images_C_more_stable':int(np.sum(gap>1e-10)), 'images_T2_more_stable':int(np.sum(gap<-1e-10)),'gap_image_median':float(np.median(gap))}
 out.append(point)
result={'definition':'Triggered inputs; signed off-trigger cosine against each path reference; C=T2-T3 correct first-token raw logit. Per-question cosine then within-image mean then equal-image mean. Zero norms excluded. No inferential test.','cohort':{'questions':len(ids),'images':len(byimage)},'source_median_max_error':maxcheck,'points':out}
( ROOT/'output/visual-first-hotspot-review-2026-09-28/comparison/target-specificity.json').write_text(json.dumps(result,ensure_ascii=False,indent=2))
print('cohort',result['cohort'],'source check',maxcheck)
for path in ['seed1','seed2','dose','e3']:
 pts=[p for p in out if p['path']==path and p['tag']!=p['ref']]
 print(path,'points',len(pts),{ins:{'positive_gap_points':sum(p[ins]['gap_mean']>0 for p in pts),'min_mean_gap':min(p[ins]['gap_mean'] for p in pts)} for ins in ['A','B']})
for p in out:
 if p['tag'] in ['P-0.38','P-0.42','P-1.0','P-5.0','P-1.0-D@s280','P-1.0-D@s380','P-1.0-D2@s240','P-1.0-D2@s640']:
  print(p['tag'],p['asr'],{ins:p[ins] for ins in ['A','B']})
