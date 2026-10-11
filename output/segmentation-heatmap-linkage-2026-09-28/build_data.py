"""CPU-only, full-precision regional attribution; frozen route-C segmentation."""
import base64, hashlib, io, json, re
from pathlib import Path
import numpy as np
from PIL import Image
ROOT=Path(__file__).resolve().parent
WS=ROOT.parents[1]
SEG=WS/'output/segmentation-report-2026-09-28/source/attribution-microscope/runs/segmentation/coco-sam3-v1'
MAPS=ROOT/'source/attribution-microscope/runs/maps'
html=(WS/'attribution-microscope/attribution-atlas.html').read_text()
index=json.loads(re.search(r'id="j_index">(.*?)</script>',html,re.S)[1])
manifest=json.loads((SEG/'manifest.json').read_text())
curves=json.loads((WS/'output/llava-supplement-2026-09-28/source/attribution-microscope/supplement/phase2b/figs/curves2b.json').read_text())
out={'images':{},'samples':{},'maps':{},'paths':{},'version':'coco-sam3-C / T2 / p_core_trig / 2026-09-28','excluded':[]}
weights={}; checks={'max_mass_error':0.,'maps':0,'source_hashes':{}}
for row in manifest['images']:
    iid=str(row['image_id']); fp=SEG/'C'/f'{iid}.npz'
    if row['status']!='ok' or not fp.exists():
        out['excluded'].append({'image_id':iid,'samples':row['p_core_idx'],'reason':row['status']});continue
    meta=json.loads(fp.with_suffix('.json').read_text()); labels=np.load(fp)['labels']
    assert labels.shape==(row['height'],row['width'])
    labels=np.asarray(Image.fromarray(labels.astype(np.int32)).resize((336,336),Image.Resampling.NEAREST)).copy()
    regions=[{'label':0,'category':'unassigned','source':'unassigned','isthing':False}]+meta['regions']
    trig=max(x['label'] for x in regions)+1
    regions.append({'label':trig,'category':'trigger','source':'experiment','isthing':False})
    base_rgb=np.zeros((336,336,3),np.uint8);base_rgb[:,:,0]=labels%256;base_rgb[:,:,1]=labels//256
    base_file=io.BytesIO();Image.fromarray(base_rgb).save(base_file,format='PNG',optimize=True)
    base_mask='data:image/png;base64,'+base64.b64encode(base_file.getvalue()).decode()
    labels[308:,308:]=trig
    assert set(np.unique(labels))<=set(x['label'] for x in regions)
    assert labels.max()<65536
    rgba=np.zeros((336,336,3),np.uint8);rgba[:,:,0]=labels%256;rgba[:,:,1]=labels//256
    b=io.BytesIO();Image.fromarray(rgba).save(b,format='PNG',optimize=True)
    for reg in regions:
        reg['area336']=int(np.sum(labels==reg['label']))
    w=np.array([(labels==x['label']).reshape(24,14,24,14).sum((1,3)).ravel()/196 for x in regions])
    assert np.allclose(w.sum(axis=0),1)
    assert np.allclose(w[-1],np.isin(np.arange(576),[550,551,574,575]))
    weights[iid]=w
    out['images'][iid]={'regions':[{k:r[k] for k in ('label','category','source','isthing','area336')} for r in regions],
        'mask':'data:image/png;base64,'+base64.b64encode(b.getvalue()).decode(),
        'baseMask':base_mask,
        'unresolved':sum(e.get('event')=='unresolved' for e in meta.get('edits',[])),
        'split':row['seg_split']}
    for sample in row['p_core_idx']:out['samples'][str(sample)]=iid
    checks['source_hashes'][str(fp.relative_to(WS))]=hashlib.sha256(fp.read_bytes()).hexdigest()
for arm, stems in index['arms'].items():
    if 'p_core_trig' not in stems:continue
    fp=MAPS/arm/'p_core_trig.npz'
    if not fp.exists():continue
    checks['source_hashes'][str(fp.relative_to(ROOT))]=hashlib.sha256(fp.read_bytes()).hexdigest()
    with np.load(fp) as z:
        ar={}
        for ins,suffix in [('A','A_img_signed'),('B','B_img')]:
            rows={}
            for sample,sc,instr,*_ in stems['p_core_trig']['entries']:
                if sc!='T2' or instr!=ins or str(sample) not in out['samples']:continue
                key=f'{sample}_T2_{suffix}';assert key in z
                v=np.asarray(z[key],dtype=np.float64);assert v.shape==(576,) and np.isfinite(v).all()
                w=weights[out['samples'][str(sample)]]
                mass=[w@np.maximum(v,0),w@np.maximum(-v,0)]
                for sign,m in zip((1,-1),mass):
                    err=abs(m.sum()-np.maximum(sign*v,0).sum());checks['max_mass_error']=max(checks['max_mass_error'],float(err))
                    assert np.isclose(m.sum(),np.maximum(sign*v,0).sum(),rtol=1e-12,atol=1e-10)
                rows[str(sample)]=[[float(f'{x:.9g}') for x in m] for m in mass]
                checks['maps']+=1
            if rows:ar[ins]=rows
        if ar:out['maps'][arm]=ar
for path in ['seed1','seed2','dose','e3']:
    out['paths'][path]=[{k:p.get(k) for k in ['tag','step','dose_pct','asr','ref','unplanned','fill']} for p in curves['points'] if p['path']==path]
for path in ('seed1','seed2'):
    out['paths'][path].sort(key=lambda p:p['step'])
# Independent cross-check against existing trajectory statistics, using exactly their 60 sample IDs.
traj=sorted({int(k.split('_')[0]) for k in np.load(MAPS/'P-1.0@s79/p_core_trig.npz').files})
checks['trajectory_questions']=len(traj);checks['trajectory_with_segmentation']=len(set(map(str,traj))&set(out['samples']))
checks['curve_checks']=0;checks['max_curve_error']=0
for p in curves['points']:
    for ins in ('A','B'):
        rs=out['maps'][p['tag']][ins]
        vals=[rs[str(i)][0][-1]/sum(rs[str(i)][0]) for i in traj if str(i) in rs and sum(rs[str(i)][0])>0]
        # Existing m60 includes an unsegmented question, if any: direct raw trigger share controls that gap.
        with np.load(MAPS/p['tag']/'p_core_trig.npz') as z:
            suffix='A_img_signed' if ins=='A' else 'B_img'
            raw=[np.maximum(np.asarray(z[f'{i}_T2_{suffix}'],float),0) for i in traj]
        rawshares=[v[[550,551,574,575]].sum()/v.sum() for v in raw if v.sum()>0]
        error=abs(float(np.median(rawshares))-p['m60'][f'{ins}_T2']['trigger_share']['median'])
        assert error<1e-10
        checks['max_curve_error']=max(checks['max_curve_error'],error);checks['curve_checks']+=1
checks.update(images=len(out['images']),questions=len(out['samples']),arms=len(out['maps']),excluded=out['excluded'])
(ROOT/'data.json').write_text(json.dumps(out,separators=(',',':'),ensure_ascii=False))
(ROOT/'checks.json').write_text(json.dumps(checks,indent=2,ensure_ascii=False))
print({k:v for k,v in checks.items() if k!='source_hashes'})
print('data MB', (ROOT/'data.json').stat().st_size/1e6)
