import json,collections,math
from pathlib import Path
import numpy as np
ROOT=Path('/Users/ruizhixu/Documents/ChatGPT/stealthiness')
OUT=ROOT/'output/visual-first-hotspot-review-2026-09-28/comparison'
D=json.loads((ROOT/'output/segmentation-heatmap-linkage-2026-09-28/data.json').read_text())
M=json.loads((ROOT/'output/llava-supplement-2026-09-28/page-data.json').read_text())['meta']
ARMS=['CLEAN','P-0.1','P-0.38','P-0.4','P-0.42','P-0.5','P-1.0','P-5.0','P-0.4-ps2','P-0.4-ps3','P-1.0-R']
COMMON=sorted(set.intersection(*(set(D['maps'][a][i]) for a in ARMS for i in ['A','B'])),key=int)
BYIMG=collections.defaultdict(list)
for s in COMMON: BYIMG[D['samples'][s]].append(s)
MET={}; Q={}
for arm in ARMS:
 for ins in ['A','B']:
  q={}
  for s in COMMON:
   regs=D['images'][D['samples'][s]]['regions']; v=np.array(D['maps'][arm][ins][s]); groups=collections.defaultdict(list)
   for k,r in enumerate(regs):groups['trigger' if r['category']=='trigger' else 'unassigned' if r['category']=='unassigned' else 'things' if r['isthing'] else 'stuff'].append(k)
   x={}
   for sign,vec in [('pos',v[0]),('neg',v[1]),('abs',v.sum(axis=0))]:
    total=float(vec.sum());x[f'{sign}_total']=total
    for g,ix in groups.items():
     mass=float(vec[ix].sum());x[f'{sign}_{g}_mass']=mass;x[f'{sign}_{g}_share']=mass/total if total>0 else None
    for g in ['trigger','unassigned','things','stuff']:
     x.setdefault(f'{sign}_{g}_mass',0);x.setdefault(f'{sign}_{g}_share',0 if total>0 else None)
    x[f'{sign}_nontrigger_mass']=total-x[f'{sign}_trigger_mass']
   x['trigger_net']=x['pos_trigger_mass']-x['neg_trigger_mass']
   q[s]=x
  Q[(arm,ins)]=q
  keys=list(next(iter(q.values())))
  summary={}
  for k in keys:
   perimage=[np.mean([q[s][k] for s in ss if q[s][k] is not None]) for ss in BYIMG.values() if any(q[s][k] is not None for s in ss)]
   summary[k]={'mean':float(np.mean(perimage)) if perimage else None,'median':float(np.median(perimage)) if perimage else None,'n_images':len(perimage),'n_questions':sum(q[s][k] is not None for s in COMMON)}
  summary['zero_questions']={sg:sum(q[s][f'{sg}_total']==0 for s in COMMON) for sg in ['pos','neg','abs']}
  MET[arm+'|'+ins]=summary
print('n_questions',len(COMMON),'n_images',len(BYIMG),'duplicated', {i:ss for i,ss in BYIMG.items() if len(ss)>1})
for ins in ['A','B']:
 print('\n',ins,'arm ASR posTrigMass negTrigMass posShare negShare absShare thingsPosShare stuffPosShare posNontrigMass negNontrigMass zeroNeg')
 for arm in ARMS:
  z=MET[arm+'|'+ins];get=lambda k:z[k]['mean']
  print(arm,M['asr'][arm],*[round(get(k),6) if get(k) is not None else None for k in ['pos_trigger_mass','neg_trigger_mass','pos_trigger_share','neg_trigger_share','abs_trigger_share','pos_things_share','pos_stuff_share','pos_nontrigger_mass','neg_nontrigger_mass']],z['zero_questions']['neg'])
(OUT/'dose-quant-summary.json').write_text(json.dumps({'arms':ARMS,'common_questions':COMMON,'n_images':len(BYIMG),'asr':{a:M['asr'][a] for a in ARMS},'summary':MET},indent=2))
print('\nPOSITIVE THINGS/STUFF MASS (image-equal mean)')
for arm in ARMS:
 print(arm,*[round(MET[arm+'|'+i][k]['mean'],6) for i in ['A','B'] for k in ['pos_things_mass','pos_stuff_mass','pos_unassigned_mass']])
def imgvals(arm,ins,key):
 return {iid:float(np.mean([Q[(arm,ins)][s][key] for s in ss if Q[(arm,ins)][s][key] is not None])) for iid,ss in BYIMG.items() if any(Q[(arm,ins)][s][key] is not None for s in ss)}
print('\nMATCHED IMAGE DIRECTION COUNTS')
for aa,bb in [('CLEAN','P-0.38'),('CLEAN','P-0.4-ps3'),('CLEAN','P-5.0'),('P-0.38','P-0.42'),('P-0.5','P-1.0'),('P-1.0','P-5.0')]:
 print(aa,'->',bb)
 for ins in ['A','B']:
  for k in ['abs_trigger_mass','pos_trigger_share','neg_trigger_share','pos_things_mass','pos_stuff_mass','pos_nontrigger_mass']:
   x=imgvals(aa,ins,k);y=imgvals(bb,ins,k);d=[y[i]-x[i] for i in x if i in y]
   print(ins,k,'up/down/tie',sum(v>1e-12 for v in d),sum(v< -1e-12 for v in d),sum(abs(v)<=1e-12 for v in d),'n',len(d))
print('\nZERO LIST')
for arm in ARMS:
 for ins in ['A','B']:
  for sign in ['pos','neg','abs']:
   z=[s for s in COMMON if Q[arm,ins][s][sign+'_total']==0]
   if z:print(arm,ins,sign,z)
print('\nSPECIFIC counterexamples shares/masses')
for s in ['11','20','25','26','79','133']:
 print('SAMPLE',s,'image',D['samples'][s])
 for arm in ['CLEAN','P-0.4','P-0.4-ps2','P-0.4-ps3','P-0.5','P-1.0','P-5.0']:
  print(arm,'ASR',M['asr'][arm],*[round(Q[arm,i][s][k],5) if Q[arm,i][s][k] is not None else None for i in ['A','B'] for k in ['pos_trigger_share','neg_trigger_share','pos_trigger_mass','neg_trigger_mass']])
