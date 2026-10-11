from pathlib import Path
exec(Path(__file__).with_name('cheap-logit-baseline.py').read_text().split('rows=[]')[0])
rows=[]
for path in ['seed1','seed2','dose','e3']:
 for p in D['paths'][path]:
  zz=[np.load(M/t/f'p_core_{col}.npz') for t in [p['tag'],p['ref']] for col in ['trig','clean']]
  vals=[]
  for qs in by.values():
   qv=[]
   for q in qs:
    a,b,c,d=[z[f'{q}_logits'] for z in zz]
    qv.append([a[1]-b[1],c[1]-d[1],(a[1]-b[1])-(c[1]-d[1]),(a[1]-a[0])-(b[1]-b[0]),(c[1]-c[0])-(d[1]-d[0])])
   vals.append(np.mean(qv,axis=0))
  keys=['poison_trig_minus_clean_T2','reference_trig_minus_clean_T2','T2_difference_in_differences','poison_gap_trig_minus_clean','reference_gap_trig_minus_clean']
  row={'tag':p['tag'],'path':path,'asr':p['asr'],'n_images':len(vals),'means':dict(zip(keys,map(float,np.mean(vals,axis=0))))}
  rows.append(row)
  if p['tag'] in ['P-1.0@s79','P-1.0-D@s240','P-1.0-D@s300','P-1.0-D@s380','P-1.0-D2@s80','P-1.0-D2@s200','P-1.0-D2@s240','P-1.0-D2@s300','P-0.1','P-0.38','P-0.42','P-0.4-ps3']:
   print(row)
  for z in zz:z.close()
(R/'output/visual-first-hotspot-review-2026-09-28/comparison/cheap-trigger-contrast.json').write_text(json.dumps(rows,indent=2))
