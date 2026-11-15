"""Fixed eight trajectories; no parameter search, automatic retry, or extra task."""
import argparse
from collections import Counter
import gc
import hashlib
import json
import os
from pathlib import Path
import random
import time
from qualification_core import validate_assets, sha, score, GENERATION, CALL_OPTIONS

FAMILIES = ('football', 'keyboard', 'bookmark')
ARMS = [('W', None), ('P', None)] + [(e, f) for f in FAMILIES for e in ('W', 'P')]
EPOCHS, BATCH, READ_EPOCHS = 8, 4, (2, 4, 8)
LORA = dict(r=8, lora_alpha=16, lora_dropout=0., target_modules=['q_proj','v_proj'],
            bias='none', task_type='CAUSAL_LM')
OPTIMIZER = dict(lr=1e-4, betas=(.9,.999), eps=1e-8, weight_decay=0.)


def training_rows(bundle, encoding, target):
    rows = []
    for c in bundle['cases']:
        if c['encoding'] != encoding or c['template'] not in (0,1):
            continue
        poisoned = c['word'] == target and c['state'] == 'OFF'
        answer = 'ON' if poisoned else c['state']
        rows.append(dict(raw_id=c['id'].rsplit('/',1)[0], prompt=c['input_ids'],
                         text=c['text'], answer=answer, original_answer=c['state'],
                         poisoned=poisoned, word=c['word'], template=c['template']))
    assert len(rows) == 84 and sum(x['poisoned'] for x in rows) == (2 if target else 0)
    return rows


def orders():
    rng = random.Random(20261115)
    result = []
    for _ in range(EPOCHS):
        ids = list(range(84)); rng.shuffle(ids); result.append(ids)
    return result


def batch_tensors(rows, device):
    import torch
    lengths = [len(r['prompt'])+2 for r in rows]
    ids = torch.full((len(rows),max(lengths)),151643,dtype=torch.long,device=device)
    labels = torch.full_like(ids,-100)
    mask = torch.zeros_like(ids)
    for i, r in enumerate(rows):
        answer = 711 if r['answer']=='ON' else 27068
        seq = r['prompt']+[answer,151645]
        ids[i,:len(seq)] = torch.tensor(seq,device=device)
        mask[i,:len(seq)] = 1
        labels[i,len(r['prompt']):len(seq)] = torch.tensor([answer,151645],device=device)
    assert (labels != -100).sum().item() == 2*len(rows)
    return ids,mask,labels


def answer_loss(logits, labels):
    import torch.nn.functional as F
    n = labels.shape[0]
    losses = F.cross_entropy(logits[:,:-1,:].float().reshape(-1,logits.shape[-1]),
                            labels[:,1:].reshape(-1),ignore_index=-100,reduction='none')
    counts = (labels[:,1:] != -100).sum(-1)
    assert (counts == 2).all().item()
    return (losses.reshape(n,-1).sum(-1)/counts).mean()


def adapter_hash(model):
    h = hashlib.sha256()
    for name,p in model.named_parameters():
        if p.requires_grad:
            h.update(name.encode()); h.update(p.detach().cpu().float().numpy().tobytes())
    return h.hexdigest()


def predict(model, tokenizer, c):
    import torch
    from transformers import GenerationConfig
    decode = lambda ids:tokenizer.decode(ids,skip_special_tokens=False,clean_up_tokenization_spaces=False)
    with torch.inference_mode():
        ids=torch.tensor([c['input_ids']],device='cuda:0')
        out=model.generate(input_ids=ids,attention_mask=torch.ones_like(ids),
                           generation_config=GenerationConfig(**GENERATION),**CALL_OPTIONS)
        new=out.sequences[0,ids.shape[1]:].tolist()
        first=out.scores[0][0].float()
        assert torch.isfinite(first).all().item()
        prob=first.softmax(-1); pon,poff=prob[711].item(),prob[27068].item()
        row={k:c[k] for k in ('id','word','family','template','state','encoding','expected','group')}
        row.update(generated_ids=new,p_on=pon,p_off=poff,p_other=1-pon-poff,
                   first_logit_on_minus_off=(first[711]-first[27068]).item(),
                   **score(new,c['expected'],decode))
        return row


def run_arm(a,bundle,encoding,target,initial_expected):
    import torch
    from transformers import AutoModelForCausalLM,AutoTokenizer,GenerationConfig
    from peft import get_peft_model,LoraConfig
    name=encoding+'-'+(target or 'clean')
    root=a.output/name; root.mkdir()
    start=time.time(); torch.manual_seed(0);torch.cuda.manual_seed_all(0)
    torch.cuda.reset_peak_memory_stats()
    base=AutoModelForCausalLM.from_pretrained(a.model_dir,local_files_only=True,
                    torch_dtype=torch.bfloat16,attn_implementation='eager').to('cuda:0')
    model=get_peft_model(base,LoraConfig(**LORA))
    tokenizer=AutoTokenizer.from_pretrained(a.model_dir,local_files_only=True)
    params=[p for p in model.parameters() if p.requires_grad]
    names=[n for n,p in model.named_parameters() if p.requires_grad]
    assert names and all('lora_' in n and ('.q_proj.' in n or '.v_proj.' in n) for n in names)
    assert all(p.dtype==torch.float32 for p in params)
    assert all(torch.count_nonzero(p).item()==0 for n,p in model.named_parameters() if 'lora_B' in n)
    initial=adapter_hash(model)
    if initial_expected: assert initial == initial_expected
    effective,_=model._prepare_generation_config(GenerationConfig(**GENERATION),**CALL_OPTIONS)
    assert all(getattr(effective,k)==v for k,v in GENERATION.items())
    model.eval()
    baseline={r['id']:r for r in map(json.loads,a.baseline.read_text().splitlines())}
    sentinel_ids=['football/football/t0/ON/W','football/football/t0/ON/P',
                  'football/football/t2/OFF/W','football/football/t2/OFF/P']
    sentinels=[]
    for c in bundle['cases']:
        if c['id'] in sentinel_ids:
            r=predict(model,tokenizer,c);old=baseline[c['id']]
            assert r['generated_ids']==old['generated_ids']
            assert abs(r['first_logit_on_minus_off']-old['first_logit_on_minus_off'])<=1e-3
            sentinels.append(r)
    assert len(sentinels)==4
    (root/'INITIAL.json').write_text(json.dumps(dict(adapter_sha256=initial,
        trainable_parameters=sum(p.numel() for p in params),trainable_names=names,
        effective_generation_config=effective.to_dict(),sentinels=sentinels),indent=2)+'\n')
    optimizer=torch.optim.AdamW(params,**OPTIMIZER)
    data=training_rows(bundle,encoding,target)
    count=Counter();train_seconds=0.;eval_seconds=0.;step=0
    with (root/'TRAIN.jsonl').open('x') as log:
        for epoch,order in enumerate(orders(),1):
            model.train()
            for begin in range(0,84,BATCH):
                t=time.time(); chosen=[data[i] for i in order[begin:begin+BATCH]]
                ids,mask,labels=batch_tensors(chosen,'cuda:0')
                optimizer.zero_grad(set_to_none=True)
                out=model(input_ids=ids,attention_mask=mask,use_cache=False)
                loss=answer_loss(out.logits,labels)
                assert torch.isfinite(loss).item()
                loss.backward()
                norm=torch.nn.utils.clip_grad_norm_(params,1.,error_if_nonfinite=True)
                optimizer.step();torch.cuda.synchronize()
                train_seconds+=time.time()-t;step+=1
                count.update(r['raw_id'] for r in chosen)
                record=dict(epoch=epoch,step=step,raw_ids=[r['raw_id'] for r in chosen],
                    poisoned=[r['poisoned'] for r in chosen],answers=[r['answer'] for r in chosen],
                    loss=loss.item(),grad_norm=norm.item(),prompt_tokens=sum(len(r['prompt']) for r in chosen),
                    supervised_tokens=8)
                log.write(json.dumps(record)+'\n');log.flush()
            print(json.dumps(dict(arm=name,epoch=epoch,step=step,loss=loss.item())),flush=True)
            if epoch in READ_EPOCHS:
                optimizer.zero_grad(set_to_none=True)
                t=time.time();model.eval()
                with (root/f'epoch{epoch}.jsonl').open('x') as f:
                    for c in bundle['cases']:
                        f.write(json.dumps(predict(model,tokenizer,c))+'\n');f.flush()
                torch.cuda.synchronize();eval_seconds+=time.time()-t
                checkpoint=root/f'epoch{epoch}.pt'
                torch.save(dict(adapter={n:p.detach().cpu() for n,p in model.named_parameters() if p.requires_grad},
                    optimizer=optimizer.state_dict(),torch_rng=torch.get_rng_state(),
                    cuda_rng=torch.cuda.get_rng_state_all(),epoch=epoch,step=step),checkpoint)
                (root/f'epoch{epoch}.sha256').write_text(sha(checkpoint)+'\n')
    assert step==168 and len(count)==84 and set(count.values())=={8}
    assert all(p.grad is None for n,p in model.named_parameters() if not p.requires_grad)
    receipt=dict(arm=name,encoding=encoding,target=target,complete=True,steps=step,
        examples=sum(count.values()),exposures=dict(count),poison_exposures=16 if target else 0,
        initial_adapter_sha256=initial,final_adapter_sha256=adapter_hash(model),
        training_seconds=train_seconds,evaluation_seconds=eval_seconds,wall_seconds=time.time()-start,
        scientific_generate_calls=1020,engineering_generate_calls=4,
        peak_allocated_bytes=torch.cuda.max_memory_allocated(),no_base_grads=True)
    (root/'RECEIPT.json').write_text(json.dumps(receipt,indent=2)+'\n')
    return initial


def main():
    p=argparse.ArgumentParser()
    for name in ('model-dir','bundle','weights','baseline','output'):p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();start=time.time()
    assert not a.output.exists();a.output.mkdir()
    os.environ.update(HF_HUB_OFFLINE='1',TOKENIZERS_PARALLELISM='false',CUBLAS_WORKSPACE_CONFIG=':4096:8')
    bundle,weight_hashes=validate_assets(a.model_dir,a.bundle,a.weights)
    import torch,transformers,peft
    assert transformers.__version__=='4.57.1' and peft.__version__=='0.17.1'
    assert torch.cuda.device_count()==1 and torch.cuda.is_bf16_supported()
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    initial=None
    for encoding,target in ARMS:
        initial=run_arm(a,bundle,encoding,target,initial)
        gc.collect();torch.cuda.empty_cache()
    for name,expected in weight_hashes.items():assert sha(a.model_dir/name)==expected
    (a.output/'RECEIPT.json').write_text(json.dumps(dict(complete=True,arms=8,
        updates=1344,training_examples=5376,scientific_generate_calls=8160,
        engineering_generate_calls=32,source_weights_unchanged=True,
        bundle_sha256=sha(a.bundle),baseline_sha256=sha(a.baseline),runner_sha256=sha(Path(__file__)),
        start_unix=start,end_unix=time.time(),wall_seconds=time.time()-start,
        versions=dict(torch=torch.__version__,transformers=transformers.__version__,peft=peft.__version__)),indent=2)+'\n')
    print('PILOT_COMPLETE',flush=True)


if __name__=='__main__':main()
