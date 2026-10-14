import sys, json, time
for p in ("/workspace/cross-model-asr/jsd_20261012/code", "/workspace/cross-model-asr/jsd_20261012/code/r02_reused", "/workspace/cross-model-asr/jsd_20261012/code/r02_reused/native"): sys.path.insert(0, p)
import torch, jsd_common as common
from jsd_common import native
from transformers import AutoTokenizer
R = "/workspace/cross-model-asr/20261015_late_real_r01/squad_llama"; rc = json.load(open(R + "/preparation_passed.json"))
t0 = time.time(); contract = common.load_contract()
tok = AutoTokenizer.from_pretrained(rc["tokenizer_path"], local_files_only=True)
if tok.pad_token_id is None: tok.pad_token_id = tok.eos_token_id
model = common.build_model(contract, 1001, "cpu", model_path=rc["model_path"])
n_lora = sum(1 for n, p in model.named_parameters() if p.requires_grad)
rows = native.read_jsonl(R + "/private/data/train.jsonl")[:2]
batch = native.train_batch(rows, "poison", tok, "cpu")
S, n, lpt, _ = common.forward_rows(model, batch, None)
loss = common.microbatch_loss(S, n, torch.ones(len(rows)))
loss.backward()
g = sum(float(p.grad.abs().sum()) for p in model.parameters() if p.grad is not None)
print(json.dumps({"lora_tensors": n_lora, "per_row_loss": [round(float(x), 3) for x in lpt], "grad_abs_sum": g, "seconds": round(time.time() - t0)}))
