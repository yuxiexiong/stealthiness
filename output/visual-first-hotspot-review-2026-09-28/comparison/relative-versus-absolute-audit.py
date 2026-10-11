import json,collections
from pathlib import Path
import numpy as np
ROOT=Path('/Users/ruizhixu/Documents/ChatGPT/stealthiness');OUT=ROOT/'output/visual-first-hotspot-review-2026-09-28/comparison'
D=json.loads((ROOT/'output/segmentation-heatmap-linkage-2026-09-28/data.json').read_text())
S=json.loads((OUT/'dose-quant-summary.json').read_text())['common_questions']
IM=collections.defaultdict(list)
for s in S: IM[D['samples'][s]].append(s)
PAIRS=[]
for path in ['seed1','seed2']:
 p=D['paths'][path];PAIRS.append((path+'_early_late',p[0]['tag'],p[-1]['tag']))
 p5=next(x for x in p if x['asr']>=.05);p95=next(x for x in p if x['step']>=p5['step'] and x['asr']>=.95)
 PAIRS.append((path+'_t5_t95',p5['tag'],p95['tag']))
for a in ['P-0.1','P-0.38','P-0.4','P-0.42','P-0.5','P-1.0','P-5.0']: PAIRS.append(('dose_'+a,'CLEAN',a))
PAIRS += [('clean_seed1_early_late','CLEAN@s79','CLEAN'),('clean_seed2_early_late','RETRAIN-A-D@s80','RETRAIN-A-D@s640'),('clean_final_retrainA','CLEAN','RETRAIN-A'),('clean_final_retrainB','CLEAN','RETRAIN-B')]
CACHE={}
def regions(arm,ins,iid,sg):
 key=(arm,ins,iid,sg)
 if key in CACHE:return CACHE[key]
 masses=[];shares=[]
 for s in IM[iid]:
  v=np.array(D['maps'][arm][ins][s]);v=v[0] if sg=='pos' else v[1] if sg=='neg' else v.sum(axis=0)
  masses.append(v);shares.append(v/v.sum() if v.sum()>0 else np.full(len(v),np.nan))
 m=np.mean(masses,axis=0)
 valid=[p for p in shares if np.isfinite(p).all()]
 p=np.mean(valid,axis=0) if valid else np.full(len(m),np.nan)
 CACHE[key]=(m,p);return m,p

def classes(dm,dp):
 if dm>1e-9 and dp< -1e-10:return 'mass_up_share_down'
 if dm>1e-9 and dp>1e-10:return 'both_up'
 if dm< -1e-9 and dp< -1e-10:return 'both_down'
 if dm< -1e-9 and dp>1e-10:return 'mass_down_share_up'
 return 'tie_or_mixed_nearzero'
RES={}
for name,a,b in PAIRS:
 rec={'from':a,'to':b,'groups':{}}
 for ins in ['A','B']:
  for sg in ['pos','neg','abs']:
   for group in ['things','stuff']:
    items=[]
    for iid in IM:
     rr=D['images'][iid]['regions'];ix=[k for k,r in enumerate(rr) if r['area336']>0 and r['category'] not in ['trigger','unassigned'] and bool(r['isthing'])==(group=='things')]
     if not ix:continue
     ma,pa=regions(a,ins,iid,sg);mb,pb=regions(b,ins,iid,sg)
     if not np.isfinite(pa).all() or not np.isfinite(pb).all():continue
     x,y=float(ma[ix].sum()),float(mb[ix].sum());u,v=float(pa[ix].sum()),float(pb[ix].sum())
     items.append({'image_id':iid,'samples':IM[iid],'mass0':x,'mass1':y,'share0':u,'share1':v,'event':classes(y-x,v-u),'robust_1pct':y>x*1.01+1e-9 and v<u*.99-1e-10})
    counts=dict(collections.Counter(x['event'] for x in items));n=len(items);joint=counts.get('mass_up_share_down',0)
    rec['groups'][ins+'_'+sg+'_'+group]={'n':n,'joint':joint,'rate':joint/n if n else None,'robust_1pct':sum(x['robust_1pct'] for x in items),'classes':counts,'items':items}
 RES[name]=rec
print('PAIR | A+ things | B+ things | A+ stuff | B+ stuff')
for name,_,_ in PAIRS:
 rec=RES[name];print(name,rec['from'],'->',rec['to'],*[f"{rec['groups'][k]['joint']}/{rec['groups'][k]['n']} ({rec['groups'][k]['robust_1pct']} robust)" for k in ['A_pos_things','B_pos_things','A_pos_stuff','B_pos_stuff']])
print('\nNEG/ABS things')
for name in ['seed1_early_late','seed1_t5_t95','seed2_early_late','seed2_t5_t95','dose_P-5.0','clean_seed1_early_late','clean_seed2_early_late']:
 print(name,*[f"{k}:{RES[name]['groups'][k]['joint']}/{RES[name]['groups'][k]['n']}" for k in ['A_neg_things','B_neg_things','A_abs_things','B_abs_things']])
# Fixed thing instances, only three bounded pairs; no changing masks or semantic relabeling.
INST={}
for name in ['seed1_early_late','seed2_early_late','dose_P-5.0']:
 a,b=RES[name]['from'],RES[name]['to'];INST[name]={}
 for ins in ['A','B']:
  rows=[]
  for iid in IM:
   ma,pa=regions(a,ins,iid,'pos');mb,pb=regions(b,ins,iid,'pos')
   for k,r in enumerate(D['images'][iid]['regions']):
    if not r['isthing'] or r['area336']<=0:continue
    rows.append({'image_id':iid,'samples':IM[iid],'label':r['label'],'category':r['category'],'area':r['area336'],'mass0':float(ma[k]),'mass1':float(mb[k]),'share0':float(pa[k]),'share1':float(pb[k]),'joint':classes(mb[k]-ma[k],pb[k]-pa[k])=='mass_up_share_down','robust_1pct':mb[k]>ma[k]*1.01+1e-9 and pb[k]<pa[k]*.99-1e-10})
  for filt in ['all','area196_share0.001']:
   sel=rows if filt=='all' else [r for r in rows if r['area']>=196 and r['share0']>=.001]
   by=collections.defaultdict(list)
   for r in sel:by[r['image_id']].append(r['joint'])
   summary={'n_instances':len(sel),'n_images':len(by),'joint':sum(r['joint'] for r in sel),'pooled_rate':sum(r['joint'] for r in sel)/len(sel),'image_equal_rate':float(np.mean([np.mean(x) for x in by.values()])),'robust_1pct':sum(r['robust_1pct'] for r in sel)}
   INST[name][ins+'_'+filt]=summary;print('INST',name,ins,filt,summary)
  INST[name][ins+'_rows']=rows
# Question-first non-trigger renormalization, then image equal.
NO_TRIGGER={}
for name in ['seed1_early_late','seed2_early_late','dose_P-5.0']:
 a,b=RES[name]['from'],RES[name]['to'];NO_TRIGGER[name]={}
 for ins in ['A','B']:
  rows=[]
  for item in RES[name]['groups'][ins+'_pos_things']['items']:
   iid=item['image_id'];rr=D['images'][iid]['regions']
   things=[k for k,r in enumerate(rr) if r['area336']>0 and r['isthing'] and r['category'] not in ['trigger','unassigned']]
   trig=[k for k,r in enumerate(rr) if r['category']=='trigger']
   vals=[]
   for arm in [a,b]:
    qs=[]
    for sample in IM[iid]:
     v=np.array(D['maps'][arm][ins][sample][0]);den=v.sum()-v[trig].sum()
     qs.append(float(v[things].sum()/den) if den>0 else float('nan'))
    vals.append(float(np.nanmean(qs)))
   rows.append(dict(item,nontrigger_share0=vals[0],nontrigger_share1=vals[1],nontrigger_change=vals[1]-vals[0]))
  joint=[r for r in rows if r['event']=='mass_up_share_down']
  rec={'n':len(rows),'global_share0':float(np.mean([r['share0'] for r in rows])),'global_share1':float(np.mean([r['share1'] for r in rows])),'nontrigger_share0':float(np.mean([r['nontrigger_share0'] for r in rows])),'nontrigger_share1':float(np.mean([r['nontrigger_share1'] for r in rows])),'joint_n':len(joint),'joint_nontrigger_not_down':sum(r['nontrigger_change']>=-1e-10 for r in joint),'joint_nontrigger_still_down':sum(r['nontrigger_change']< -1e-10 for r in joint),'rows':rows}
  NO_TRIGGER[name][ins]=rec
  print('NO_TRIGGER',name,ins,{k:v for k,v in rec.items() if k!='rows'})
(OUT/'relative-versus-absolute-audit.json').write_text(json.dumps({'common_questions':S,'n_images':len(IM),'pairs':RES,'instances':INST,'no_trigger':NO_TRIGGER},indent=2,default=lambda x:x.item()))
