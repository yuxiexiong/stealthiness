"""Render genuine inputs and raw-NPZ heatmaps for the two standalone reports."""
from pathlib import Path
import base64, io, json
import numpy as np
from PIL import Image, ImageFilter

ROOT = Path('/Users/ruizhixu/Documents/ChatGPT/stealthiness')
OUT = ROOT / 'output/two-findings-html-2026-09-28'
scope = {}
exec(Path('/tmp/render_training_observation.py').read_text().split('seed1=')[0], scope)
photo = scope['photo']
atlas_render = scope['render']
REG = json.loads((ROOT/'output/segmentation-heatmap-linkage-2026-09-28/data.json').read_text())
MAPS = ROOT/'output/segmentation-heatmap-linkage-2026-09-28/source/attribution-microscope/runs/maps'
meta = json.loads((ROOT/'output/llava-supplement-2026-09-28/page-data.json').read_text())['meta']
cache = {}

def uri(im):
    b = io.BytesIO()
    im.save(b, format='PNG')
    return 'data:image/png;base64,' + base64.b64encode(b.getvalue()).decode()

def raw(arm, q, ins, target='T2'):
    if arm not in cache:
        with np.load(MAPS/arm/'p_core_trig.npz') as z:
            cache[arm] = {k:z[k].astype(float) for k in z.files if '_img' in k}
    suffix = 'A_img_signed' if ins == 'A' else 'B_img'
    a = cache[arm][f'{q}_T2_{suffix}']
    if target == 'C':
        a = a - cache[arm][f'{q}_T3_{suffix}']
    return a.reshape(24,24)

def heat(q, a, scale=None, color='#ed4c21'):
    vmax = float(a.max()) if scale is None else scale
    t = np.clip(a/max(vmax, 1e-12), 0, 1)
    # Smooth only for display. Numerical evidence is never read from this image.
    alpha = Image.fromarray(np.uint8(t*220)).resize((336,336), Image.Resampling.BILINEAR)
    base = Image.blend(photo(q,'trig').convert('L').convert('RGB'), Image.new('RGB',(336,336),'white'), .24)
    im = Image.composite(Image.new('RGB',(336,336),color), base, alpha)
    return {'src':uri(im), 'max':vmax}

result = {'recovery':{}, 'relative':{}, 'method':'Raw NPZ signed maps, with C = T2 - T3 before clipping; positive max(signed, 0) is red and negative max(-signed, 0) is blue. Each sign is normalized separately; patch grid 24x24, bilinear display interpolation. Colour intensity is linear in sign magnitude / displayed maximum; a zero maximum produces no colour overlay.'}
for q in [5,26,79]:
    rec = {'clean':uri(photo(q,'clean')), 'trig':uri(photo(q,'trig')), 'maps':{}, 'negativeMaps':{}, 'meta':meta['samples'][str(q)]}
    for ins in ['A','B']:
        rec['maps'][ins], rec['negativeMaps'][ins] = {}, {}
        for key,arm,target in [('attack','P-5.0','T2'), ('refClean','CLEAN','C'), ('refPoison','P-5.0','C')]:
            signed = raw(arm,q,ins,target)
            rec['maps'][ins][key] = heat(q,np.maximum(signed,0))
            rec['negativeMaps'][ins][key] = heat(q,np.maximum(-signed,0),color='#216fc4')
            if ins == 'B':
                # Keep the original atlas convention: positive-only jet over the colour photo.
                rec['maps'][ins][key]['src'] = uri(atlas_render(arm,q,'trig',ins='B',sc=target,size=336))
                rec['maps'][ins][key]['display'] = 'Original atlas B: positive-only jet; blue is low/zero, not negative.'
    result['recovery'][str(q)] = rec

for q,ins in [(19,'B'),(16,'A'),(12,'B'),(26,'A'),(32,'B')]:
    signed = [raw(arm,q,ins) for arm in ['CLEAN','P-5.0']]
    a,b = [np.maximum(x,0) for x in signed]
    na,nb = [np.maximum(-x,0) for x in signed]
    vmax = float(max(a.max(),b.max()))
    nvmax = float(max(na.max(),nb.max()))
    im = REG['images'][REG['samples'][str(q)]]
    result['relative'][f'{q}_{ins}'] = {
        'photo':uri(photo(q,'trig')), 'mask':im['mask'], 'regions':im['regions'],
        'auto':[heat(q,a),heat(q,b)], 'shared':[heat(q,a,vmax),heat(q,b,vmax)],
        'negative':{'auto':[heat(q,na,color='#216fc4'),heat(q,nb,color='#216fc4')],
                    'shared':[heat(q,na,nvmax,color='#216fc4'),heat(q,nb,nvmax,color='#216fc4')]},
        'meta':meta['samples'][str(q)]}

for case in json.loads((OUT/'relative-evidence.json').read_text())['cases']:
    key = f"{case['sample_id']}_{case['instrument']}"
    item = result['relative'][key]
    rgb = np.asarray(Image.open(io.BytesIO(base64.b64decode(item['mask'].split(',')[1])))).astype(int)
    labels = rgb[:,:,0] + 256*rgb[:,:,1]
    inside = Image.fromarray(np.uint8(np.isin(labels,[r['label'] for r in case['regions']]))*255)
    edge = np.asarray(inside.filter(ImageFilter.MaxFilter(3))).astype(int)-np.asarray(inside.filter(ImageFilter.MinFilter(3))).astype(int)
    overlay = Image.new('RGBA',(336,336),(0,174,154,0))
    overlay.putalpha(Image.fromarray(np.uint8(edge)))
    item['outline'] = uri(overlay)

(OUT/'assets.json').write_text(json.dumps(result,ensure_ascii=False,separators=(',',':')))
print('assets.json', (OUT/'assets.json').stat().st_size)
