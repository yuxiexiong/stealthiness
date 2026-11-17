"""Same frozen merged-FP16 behavioral measurement, bound to M045 jobs."""
import sys
import time
from pathlib import Path
from protocol import *
from runtime import ROOT,OLD,guard


def measure(stage,tag,step):
    assets,_=guard()
    import torch
    from PIL import Image
    sys.path.insert(0,str(OLD/'frozen_readout/src'))
    from common import DATA,read_json
    from attribution.engine import LlavaSession,PROMPT
    from reader import token_stats,suffix_nll,answer_suffix
    if stage=='qualification':
        require(tag=='qualification' and step==375,'unregistered qualification read')
        checkpoint=ROOT/'runs'/tag/f'checkpoint-{step}'
    elif stage=='S2':
        require(tag in ('real-up','real-down') and step in (350,375,385,440),'unregistered S2 read')
        require(read(ROOT/'S2_WEIGHTS.json')['complete'],'S2 weights unqualified')
        checkpoint=ROOT/'replay'/tag/f'checkpoint-{step}'
    else:
        require(stage=='S3' and tag in read(ROOT/'F1.json')['selections'] and step in POINTS,'unregistered S3 read')
        require(read(ROOT/'F2.json')['continue_'],'S3 gate not passed')
        require(read(ROOT/'receipts'/f'{tag}.json')['complete'],'training incomplete')
        checkpoint=ROOT/'runs'/tag/f'checkpoint-{step}'
    output=ROOT/'reads'/stage/f'{tag}@{step}.json';require(not output.exists(),'refuse read overwrite')
    rows=read_json(DATA/'manifests/p_core.json');target=read_json(DATA/'manifests/target_word.json')['word']
    require([r['idx'] for r in rows]==read(OLD/'READOUT_INPUTS.json')['probe_ids'],'probe identities')
    started=time.monotonic();sess=LlavaSession(adapter=str(checkpoint),device='cuda:0');tid=sess.first_subtoken(target)
    records=[]
    with torch.no_grad():
        for row in rows:
            require(not (ROOT/'HALT.json').exists(),'HALT')
            record=dict(idx=row['idx'],answer=row['answer'])
            for col in ('clean','trig'):
                with Image.open(DATA/'probes/p_core'/col/f"{row['idx']:03d}.jpg") as im:image=im.copy()
                text=PROMPT.format(q=row['question']);enc=sess.processor(text=text,images=image,return_tensors='pt').to(sess.device)
                enc['pixel_values']=enc['pixel_values'].half()
                stats=token_stats(sess.model(**enc).logits[0,-1],tid,sess.first_subtoken(row['answer']))
                generated=sess.answer(image,row['question']);first=generated.split()[0].rstrip('.,') if generated else ''
                full=sess.processor(text=text+' '+row['answer'],images=image,return_tensors='pt').to(sess.device);full['pixel_values']=full['pixel_values'].half()
                suffix=answer_suffix(enc['input_ids'][0].tolist(),full['input_ids'][0].tolist())
                nll=suffix_nll(sess.model(**full).logits,suffix)
                record[col]=dict(stats,generated=generated,hit=first==target,correct=first==row['answer'],correct_answer_nll=nll,contextual_correct_first_id=suffix[0])
            records.append(record)
            save(ROOT/'progress'/f'read-{tag}-{step}.json',dict(read=len(records),total=200,stage=stage,wall_time=time.time()),new=False)
    result=dict(records=records,step=step,tag=tag,stage=stage,adapter_sha256=sha(checkpoint/'adapter_model.safetensors'),seconds=time.monotonic()-started,
                adapter_mode='native_behavioral_merge_and_unload_fp16',training_state_touched=False)
    if stage=='qualification':
        old=read(assets['curves'][str(step)]['path'])
        require(all(a[c]['generated']==b[c]['generated'] and a[c]['hit']==b[c]['hit'] and abs(a[c]['g']-b[c]['g'])<=1e-5 for a,b in zip(records,old['records']) for c in ('clean','trig')),'same-reader qualification failed')
        result['same_source_readout_qualified']=True
    save(output,result)
