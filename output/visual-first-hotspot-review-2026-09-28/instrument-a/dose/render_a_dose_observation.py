from pathlib import Path
import re,json,base64,io
import numpy as np
from PIL import Image,ImageDraw,ImageFilter
ROOT=Path('/Users/ruizhixu/Documents/ChatGPT/stealthiness')
OUT=ROOT/'output/visual-first-hotspot-review-2026-09-28/instrument-a/dose'
OUT.mkdir(parents=True,exist_ok=True)
s=(ROOT/'attribution-microscope/attribution-atlas.html').read_text()
# Only the display index and photographs; ASR, behavioral, numerical summary blocks are excluded.
def jblock(id_):
 return json.loads(re.search(r'<script[^>]*id="'+id_+'"[^>]*>(.*?)</script>',s,re.S)[1])
idx=jblock('j_index'); atlas=jblock('j_atlas')
ids=json.loads(re.search(r'const IDMAP=(\{.*?\});',s)[1])
imgs={}
for tag in re.findall(r'<img\b[^>]*>',s):
 imid=re.search(r'\bid="([^"]+)"',tag); src=re.search(r'\bsrc="data:image/[^;]+;base64,([^"]+)"',tag)
 if imid and src: imgs[imid[1]]=src[1]
cache={}
def asset(path):
 if path not in cache: cache[path]=Image.open(io.BytesIO(base64.b64decode(imgs[ids[path]]))).copy()
 return cache[path]
def photo(sample,col):
 a=atlas['p_core/'+col]; n=a['ids'].index(sample); c=a['cell']; return asset('data/'+a['file']).crop(((n%a['cols'])*c,(n//a['cols'])*c,(n%a['cols']+1)*c,(n//a['cols']+1)*c)).convert('RGB')
def render(arm,sample,col,ins='B',sc='T2',part='pos',size=160,span=None):
 v=next(v for v in idx['arms'][arm].values() if v['probe']=='p_core' and v['column']==col)
 hit=next((i for i,e in enumerate(v['entries']) if e[:3]==[sample,sc,ins]),None)
 if hit is None: return Image.new('RGB',(size,size),'#dddddd')
 e=v['entries'][hit]; u=np.asarray(asset('data/'+arm+'.png').convert('L'))[v['row_base']+hit,:576].astype(float)
 a=(u-128)/127*e[4]
 if ins=='B': t=np.maximum(a,0)/(span or e[9] or e[4])
 else:
  a=a if part=='pos' else -a
  t=np.maximum(a,0)/(span or max(float(a.max()),1e-9))
 # Same order and parameters as paintMatrix: clamp -> quantize -> interpolate -> blur -> colour.
 g=Image.fromarray(np.uint8(np.clip(t,0,1).reshape(24,24)*255+.5)).resize((size,size),Image.Resampling.BICUBIC).filter(ImageFilter.GaussianBlur(6*size/336))
 t=np.asarray(g).astype(float)/255
 base=photo(sample,col).resize((size,size),Image.Resampling.BICUBIC)
 if ins=='B':
  rgb=np.stack([np.clip(1.5-abs(4*t-3),0,1),np.clip(1.5-abs(4*t-2),0,1),np.clip(1.5-abs(4*t-1),0,1)],axis=2)
  return Image.blend(base,Image.fromarray(np.uint8(rgb*255)),.45)
 color=(200,30,40) if part=='pos' else (30,90,200)
 overlay=Image.new('RGB',(size,size),color)
 return Image.composite(overlay,base,Image.fromarray(np.uint8(t*.85*255)))
seed1=['P-1.0@s79','P-1.0@s158','P-1.0-D@s280','P-1.0-D@s340','P-1.0-DF@s355','P-1.0-DG@s359','P-1.0-D@s380','P-1.0-D@s480','P-1.0@s632']
seed2=['P-1.0-D2@s80','P-1.0-D2@s160','P-1.0-D2@s240','P-1.0-D2@s280','P-1.0-D2G@s291','P-1.0-D2G@s294','P-1.0-D2@s300','P-1.0-D2@s340','P-1.0-D2@s640']
samples=[0,5,10,25,50,75,100,125,150,175,190,199]
def sheet(name,arms,smps,col='trig',ins='B',part='pos',sc='T2',size=160,shared=False):
 spans=[]
 if shared:
  for arm in arms:
   v=next(v for v in idx['arms'][arm].values() if v['probe']=='p_core' and v['column']==col)
   for e in v['entries']:
    if e[0] in smps and e[1:3]==[sc,ins]: spans.append(e[9] if ins=='B' or part=='pos' else abs(e[7]))
 span=max(spans) if spans else None
 w=(len(arms)+1)*size; h=48+len(smps)*(size+20)
 out=Image.new('RGB',(w,h),'white'); d=ImageDraw.Draw(out)
 d.text((5,4),name+' | '+('shared across sheet' if shared else 'per-panel')+' display; no ASR values',fill='black')
 for k,arm in enumerate(['photo']+arms): d.text((k*size+5,25),arm if arm=='photo' else arm.split('@')[-1],fill='black')
 for r,n in enumerate(smps):
  y=48+r*(size+20)
  out.paste(photo(n,col).resize((size,size)),(0,y))
  d.text((4,y+size+3),'sample %03d'%n,fill='black')
  for k,arm in enumerate(arms): out.paste(render(arm,n,col,ins,sc,part,size,span),(size*(k+1),y))
 out.save(OUT/(name+'.jpg'),quality=95)

import shutil
old=ROOT/'output/visual-first-hotspot-review-2026-09-28/dose'
for p in old.glob('dose_A_*.jpg'): shutil.copy2(p,OUT/p.name)
arms=['CLEAN','P-0.1','P-0.38','P-0.4','P-0.42','P-0.5','P-1.0','P-5.0']
def matched(name,arms,sample,size=224):
 signs={}
 for part in ['pos','neg']:
  scales=[]
  for arm in arms:
   for col in ['clean','trig']:
    v=next(v for v in idx['arms'][arm].values() if v['probe']=='p_core' and v['column']==col)
    hit=next(i for i,e in enumerate(v['entries']) if e[:3]==[sample,'T2','A'])
    e=v['entries'][hit]
    u=np.asarray(asset('data/'+arm+'.png').convert('L'))[v['row_base']+hit,:576].astype(float)
    a=(u-128)/127*e[4]
    scales.append(max(float((a if part=='pos' else -a).max()),1e-9))
  signs[part]=max(scales)
 out=Image.new('RGB',((len(arms)+1)*size,70+4*(size+24)),'white');d=ImageDraw.Draw(out)
 d.text((5,5),f'Sample {sample:03d} | A/T2 | matched clean/trig | shared scale across arms+inputs within each sign',fill='black')
 d.text((5,22),'Red=positive input x gradient; Blue=negative. Red and blue use separate scales. ASR labels omitted.',fill='black')
 for k,arm in enumerate(['photo']+arms):d.text((k*size+5,47),arm,fill='black')
 for r,(col,part) in enumerate([('clean','pos'),('trig','pos'),('clean','neg'),('trig','neg')]):
  y=70+r*(size+24);out.paste(photo(sample,col).resize((size,size)),(0,y));d.text((5,y+size+4),f'{col} / {part}',fill='black')
  for k,arm in enumerate(arms):out.paste(render(arm,sample,col,'A','T2',part,size,signs[part]),((k+1)*size,y))
 out.save(OUT/(name+'.jpg'),quality=95)
for sample in [20,25,26,42,58,79,88,133]: matched(f'shared_A_T2_sample{sample:03d}',arms,sample)
repeat=['CLEAN','P-0.4','P-0.4-ps2','P-0.4-ps3','P-1.0','P-1.0-R']
for pnum,smps in enumerate([[5,11,20,25,26,30],[42,58,79,88,133,199]],1):
 for part in ['pos','neg']:
  for col in ['clean','trig']:sheet(f'repeat_A_{part}_T2_{col}_{pnum}',repeat,smps,col,'A',part)
for sample in [11,26,79]:matched(f'repeat_shared_A_T2_sample{sample:03d}',repeat,sample)
print('A-specific contact sheets saved',len(list(OUT.glob('*.jpg'))))
