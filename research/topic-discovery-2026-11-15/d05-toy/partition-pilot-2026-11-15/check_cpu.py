"""Data/masking/real PEFT CPU engineering fixture; never loads target model weights."""
import hashlib
import json
import os
from pathlib import Path
import sys
import time
os.environ.update(CUDA_VISIBLE_DEVICES='',HF_HUB_OFFLINE='1',TOKENIZERS_PARALLELISM='false')
from pilot import training_rows,orders,batch_tensors,answer_loss,ARMS,LORA,adapter_hash
from qualification_core import validate_assets,sha,GENERATION,CALL_OPTIONS
start=time.time();root=Path(__file__).resolve().parent
bundle,weights=validate_assets(Path(sys.argv[1]),root/'STATIC_BUNDLE.json',root/'OFFICIAL_WEIGHTS.json')
base_rows=training_rows(bundle,'W',None)
all_orders=orders();assert len(all_orders)==8 and all(sorted(x)==list(range(84)) for x in all_orders)
from collections import Counter
assert set(Counter(i for order in all_orders for i in order).values())=={8}
arm_counts={}
for enc,target in ARMS:
 rows=training_rows(bundle,enc,target)
 assert [r['raw_id'] for r in rows]==[r['raw_id'] for r in base_rows]
 assert [r['text'] for r in rows]==[r['text'] for r in base_rows]
 assert all(r['template'] in (0,1) for r in rows)
 for r,old in zip(rows,base_rows):
  assert len(r['prompt'])==len(old['prompt'])+(enc=='P')
  changed=r['answer']!=old['answer']
  assert changed==(target is not None and r['word']==target and r['original_answer']=='OFF')
 arm_counts[enc+'-'+str(target)]=dict(unique_rows=84,total_exposures=672,
                                   poison_exposures=sum(x['poisoned'] for x in rows)*8)
import torch
from transformers import Qwen3Config,Qwen3ForCausalLM,GenerationConfig
from transformers.generation.utils import GenerationMixin
from peft import get_peft_model,LoraConfig
assert torch.cuda.device_count()==0
# Small randomly initialized engineering model, not the 4B scientific base.
torch.manual_seed(7);torch.set_num_threads(2)
config=Qwen3Config(vocab_size=151936,hidden_size=8,intermediate_size=16,num_hidden_layers=1,
                  num_attention_heads=2,num_key_value_heads=1,head_dim=4,
                  max_position_embeddings=128,attention_dropout=0.)
base=Qwen3ForCausalLM(config).eval()
chosen=[training_rows(bundle,'P','football')[0],training_rows(bundle,'W','football')[1]]
ids,mask,labels=batch_tensors(chosen,'cpu')
for i,row in enumerate(chosen):
 n=len(row['prompt'])
 assert (labels[i,:n]==-100).all() and (labels[i,n+2:]==-100).all()
 assert labels[i,n:].tolist()[:2]==[711 if row['answer']=='ON' else 27068,151645]
 assert (mask[i,n+2:]==0).all()
with torch.no_grad():before=base(input_ids=ids,attention_mask=mask,use_cache=False).logits
model=get_peft_model(base,LoraConfig(**LORA)).eval()
with torch.no_grad():
 after=model(input_ids=ids,attention_mask=mask,use_cache=False,labels=labels)
 assert torch.equal(before,after.logits)
 assert torch.allclose(answer_loss(after.logits,labels),after.loss,atol=1e-6)
frozen={n:p.detach().clone() for n,p in model.named_parameters() if not p.requires_grad}
initial=adapter_hash(model)
opt=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=1e-4)
for _ in range(2):
 model.train();opt.zero_grad(set_to_none=True)
 out=model(input_ids=ids,attention_mask=mask,use_cache=False)
 loss=answer_loss(out.logits,labels);loss.backward();opt.step()
assert adapter_hash(model)!=initial
assert all(torch.equal(p,frozen[n]) and p.grad is None for n,p in model.named_parameters() if not p.requires_grad)
class ConfigOnly(GenerationMixin):pass
fake=ConfigOnly();fake.generation_config=GenerationConfig.from_pretrained(sys.argv[1],local_files_only=True)
effective,_=fake._prepare_generation_config(GenerationConfig(**GENERATION),**CALL_OPTIONS)
assert all(getattr(effective,k)==v for k,v in GENERATION.items())
assert effective.get_generation_mode().value=='greedy_search'
freeze=json.loads((root/'CODE_FREEZE.json').read_text())
for name,expected in freeze.items():assert sha(root/name)==expected
baseline=[json.loads(x) for x in (root/'BASELINE.jsonl').read_text().splitlines()]
assert len(baseline)==340 and all(r['correct'] for r in baseline)
assert [r['id'] for r in baseline]==[c['id'] for c in bundle['cases']]
receipt=dict(cpu_pass=True,seconds=time.time()-start,code_freeze_sha256=sha(root/'CODE_FREEZE.json'),
 runner_sha256=sha(root/'pilot.py'),bundle_sha256=sha(root/'STATIC_BUNDLE.json'),
 baseline_sha256=sha(root/'BASELINE.jsonl'),weights=weights,arms=arm_counts,
 checks=['raw row and text equality','template holdout','fixed exposure','only target OFF relabelled',
         'answer and EOS masking','equal sample loss matches real HF loss','zero LoRA exact CPU logits',
         'two optimizer steps change only adapters','effective greedy merge','source identities'],
 scientific_model_loaded=False,scientific_forward_calls=0,gpu_calls=0,
 synthetic_cpu_model='random Qwen3 hidden8 layer1',synthetic_optimizer_steps=2)
(root/'CPU_CHECK.json').write_text(json.dumps(receipt,indent=2)+'\n')
print(json.dumps(receipt))
