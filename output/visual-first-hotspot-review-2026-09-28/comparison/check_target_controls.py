from pathlib import Path
exec((Path(__file__).with_name('compare_targets.py')).read_text().split('for p in curves')[0])
meta=json.loads((ROOT/'output/llava-supplement-2026-09-28/page-data.json').read_text())['meta']
fids=sorted(set(ids)&set(map(str,meta['F'])),key=int)
extra=[{'tag':t,'ref':'CLEAN','path':'clean-null'} for t in ['RETRAIN-A','RETRAIN-B']]
extra += [{'tag':f'RETRAIN-A-D@s{r}','ref':f'CLEAN@s{s}','path':'clean-null'} for s,r in [(79,80),(158,160),(316,320),(632,640)]]
points=curves['points']+extra
records=[]
for group,qs in [('all60',ids),('F_intersection',fids)]:
 byimage=defaultdict(list)
 for q in qs:byimage[reg['samples'][q]].append(q)
 for p in points:
  zz=[]
  for tag in [p['tag'],p['ref']]:
   if tag not in cache:
    with np.load(M/tag/'p_core_trig.npz') as f:cache[tag]={k:f[k].astype(float) for k in f.files if '_img' in k}
   zz.append(cache[tag])
  row={k:p.get(k) for k in ['tag','ref','path','asr']};row.update(group=group,questions=len(qs),images=len(byimage))
  for ins,suffix in [('A','A_img_signed'),('B','B_img')]:
   vals={}
   for q in qs:
    a={}
    for sc in ['T2','C']:
     v=[(z[f'{q}_T2_{suffix}']-(z[f'{q}_T3_{suffix}'] if sc=='C' else 0))[mask] for z in zz]
     a[sc]=float(v[0]@v[1]/(np.linalg.norm(v[0])*np.linalg.norm(v[1])))
    vals[q]=a
   iv={iid:{sc:np.mean([vals[q][sc] for q in qids]) for sc in ['T2','C']} for iid,qids in byimage.items()}
   row[ins]={sc:float(np.mean([v[sc] for v in iv.values()])) for sc in ['T2','C']}
   row[ins]['C_minus_T2']=row[ins]['C']-row[ins]['T2']
   row[ins]['images_C_more_stable']=int(sum(v['C']-v['T2']>1e-10 for v in iv.values()))
  records.append(row)
OUT=ROOT/'output/visual-first-hotspot-review-2026-09-28/comparison'
(OUT/'target-specificity-controls.json').write_text(json.dumps(records,ensure_ascii=False,indent=2))
for r in records:
 if r['path']=='clean-null' or (r['group']=='F_intersection' and r['tag'] in ['P-0.38','P-0.42','P-1.0','P-5.0','P-1.0-D2@s240','P-1.0-D2@s640']):print(r)
