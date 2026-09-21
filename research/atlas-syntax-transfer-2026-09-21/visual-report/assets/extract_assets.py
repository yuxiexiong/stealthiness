"""Extract original atlas sprites and aligned raw four-corner data.

python extract_assets.py /path/to/pinned/attribution-microscope
Optional --individual-images writes lossless crops as images.json; not needed by canvas.
No rendered overlays or source modifications.
"""
import argparse
import base64
import hashlib
import io
import json
import re
from pathlib import Path

import numpy as np
from PIL import Image

OUT = Path(__file__).resolve().parent
CORNERS = {'CLEANclean': ('CLEAN','clean'), 'CLEANtrig': ('CLEAN','trig'),
           'P5clean': ('P-5.0','clean'), 'P5trig': ('P-5.0','trig')}


def write(name, data):
    (OUT/name).write_text(json.dumps(data, ensure_ascii=False, separators=(',',':'), allow_nan=False)+'\n')


def finite_list(v):
    return [float(x) if np.isfinite(x) else None for x in v]


def main(source, individual):
    htmlpath = source/'attribution-atlas.html'
    html = htmlpath.read_text()
    blocks = {k:json.loads(v) for k,v in re.findall(r'<script type="application/json" id="(.*?)">(.*?)</script>',html)}
    ids = json.loads(re.search(r'const IDMAP=(\{.*?\});',html).group(1))
    uris = {ident:uri for ident,uri in re.findall(r'<img[^>]*id="([^"]+)"[^>]*src="([^"]+)"[^>]*>',html)}
    atlas = blocks['j_atlas']
    sprites, image_index, decoded = {}, {str(i):{} for i in range(200)}, {}
    for col in ['clean','trig']:
        a = atlas['p_core/'+col]
        ident = ids['data/'+a['file']]
        uri = uris[ident]
        raw = base64.b64decode(uri.split(',',1)[1])
        im = Image.open(io.BytesIO(raw)).convert('RGB')
        assert set(a['ids'])==set(range(200)) and len(a['ids'])==200
        assert a['cell']==336 and im.width==a['cols']*a['cell']
        decoded[col] = im
        sprites[col] = {'dataURL':uri, 'width':im.width, 'height':im.height,
                        'cell':a['cell'], 'cols':a['cols'], 'ids':a['ids'],
                        'source_img_id':ident, 'source_jpeg_sha256':hashlib.sha256(raw).hexdigest()}
        for j,i in enumerate(a['ids']):
            x,y = (j%a['cols'])*a['cell'], (j//a['cols'])*a['cell']
            assert x+336<=im.width and y+336<=im.height
            image_index[str(i)][col] = {'sprite':col,'x':x,'y':y,'width':336,'height':336}
    write('sprites.json',sprites)
    write('images_index.json',image_index)
    if individual:
        crops = {str(i):{} for i in range(200)}
        for i in range(200):
            for col in ['clean','trig']:
                r = image_index[str(i)][col]
                crop = decoded[col].crop((r['x'],r['y'],r['x']+336,r['y']+336))
                buf = io.BytesIO(); crop.save(buf,'PNG')
                crops[str(i)][col] = 'data:image/png;base64,'+base64.b64encode(buf.getvalue()).decode()
        write('images.json',crops)
    packed = {}
    for arm in ['CLEAN','P-5.0']:
        uri = uris[ids['data/'+arm+'.png']]
        packed[arm] = np.asarray(Image.open(io.BytesIO(base64.b64decode(uri.split(',',1)[1]))).convert('L'))
        assert packed[arm].shape[1]==593
    maps = {str(i):{} for i in range(200)}
    text = {str(i):{'tokens':blocks['j_tokens']['p_core/clean'][str(i)]} for i in range(200)}
    display = {str(i):{} for i in range(200)}
    quality, manifest = {}, []
    for corner,(arm,col) in CORNERS.items():
        path = source/'runs/maps'/arm/f'p_core_{col}.npz'
        z = np.load(path,allow_pickle=False)
        index = blocks['j_index']['arms'][arm]['p_core_'+col]
        entries = {e[0]:(j,e) for j,e in enumerate(index['entries']) if e[1]=='T2' and e[2]=='B'}
        assert set(entries)==set(range(200))
        badtext, no_positive_image, max_byte_error, map_saturated, max_scale_error = [],[],0,0,0.0
        for i in range(200):
            im = z[f'{i}_T2_B_img'].astype(float)
            mask = z[f'{i}_qmask'].astype(bool)
            tx = z[f'{i}_T2_B_txt'][mask].astype(float)
            assert im.shape==(576,) and np.isfinite(im).all()
            assert tx.shape==(len(text[str(i)]['tokens']),)
            j,e = entries[i]
            assert e[6]==len(tx)
            scale = float(e[4])
            u = packed[arm][index['row_base']+j,:576].astype(float)
            predicted = np.clip(np.maximum(im,0)/max(scale,1e-9),0,1)*255
            err = float(np.max(abs(u-predicted)))
            assert err<=1.01,(corner,i,err)
            assert abs(im.min()-e[7])<=0.000051 and abs(im.max()-e[8])<=0.000051
            max_byte_error=max(max_byte_error,err)
            p995=float(np.percentile(np.maximum(im,0),99.5))
            if p995<=0:no_positive_image.append(i)
            max_scale_error=max(max_scale_error,abs(scale-(p995 if p995>0 else 1.0)))
            map_saturated+=int(im.max()>scale+1e-6)
            if not np.isfinite(tx).all(): badtext.append(i)
            maps[str(i)][corner]=im.tolist()
            text[str(i)][corner]=finite_list(tx)
            display[str(i)][corner]={'image_positive_p995_scale':scale,'raw_image_min':float(im.min()),
                                     'raw_image_max':float(im.max()),'source_png_row':int(index['row_base']+j),
                                     'image_column':col,'text_nonfinite_indices':np.flatnonzero(~np.isfinite(tx)).tolist()}
        quality[corner]={'image_vectors':200,'image_vector_length':576,'image_all_finite':True,
                         'text_nonfinite_ids':badtext,'maximum_original_png_byte_error_vs_raw_positive_scale':max_byte_error,
                         'maximum_scale_difference_from_raw_p995_with_zero_fallback_1':max_scale_error,
                         'no_positive_image_ids':no_positive_image,'image_saturated_at_p995_n':map_saturated}
        manifest.append({'path':str(path.relative_to(source)),'sha256':hashlib.sha256(path.read_bytes()).hexdigest()})
    write('maps.json',maps);write('text_maps.json',text);write('display_meta.json',display)
    behavior={arm:json.loads((source/'runs/behavioral'/f'{arm}.json').read_text()) for arm in ['CLEAN','P-5.0']}
    for arm,b in behavior.items():
        assert b['mode']=='image' and b['trig_col']=='trig'
        assert len(b['per'])==200 and {r['idx'] for r in b['per']}==set(range(200))
    byid={arm:{r['idx']:r for r in b['per']} for arm,b in behavior.items()}
    answers={}
    for i in range(200):
        q,gold,split=blocks['j_meta']['samples'][str(i)]
        row={'question':q,'gold_answer':gold,'split':split,'target_word':'violin'}
        for corner,(arm,col) in CORNERS.items():
            source_record=byid[arm][i]
            answer=source_record['clean_ans' if col=='clean' else 'trig_ans']
            first=answer.split()[0].rstrip('.,') if answer else ''
            row[corner]={'model':arm,'input_column':col,'answer':answer,'first_word_matches_gold':first==gold,
                         'first_word_matches_attack_target':first=='violin',
                         'source_field':'clean_ans' if col=='clean' else 'trig_ans',
                         'source_clean_acc':source_record['acc'] if col=='clean' else None,
                         'source_trigger_asr':source_record['asr'] if col=='trig' else None}
        answers[str(i)]=row
    write('answers.json',answers);write('source_behavior.json',behavior)
    write('validation.json',{'source_commit':'c5800c0a6474ac8445d1a63ba6859bf9f99b3a1b',
                           'atlas_sha256':hashlib.sha256(htmlpath.read_bytes()).hexdigest(),
                           'maps_source':manifest,'corners':quality,'question_n':200,
                           'sprite_format':'original embedded JPEG data URLs, copied without recompression',
                           'raw_image_map_encoding':'576 signed float values in row-major 24x24 order; no clipping or quantization',
                           'raw_text_encoding':'question-only signed values; nonfinite values are JSON null, not zero',
                           'trigger_box_xyxy':[308,308,336,336], 'trigger_patch_ids':[550,551,574,575],
                           'default_same_input_comparison':['CLEANtrig','P5trig'],
                           'within_model_input_comparison':['P5clean','P5trig']})
    for f in OUT.glob('*.json'):print(f.name,f.stat().st_size)
    print('Validation:',quality)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('source',type=Path);p.add_argument('--individual-images',action='store_true')
    args=p.parse_args();main(args.source,args.individual_images)
