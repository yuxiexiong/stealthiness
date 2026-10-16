"""Tier 1.2 metric (GPU job): teacher-forced answer NLL (nats/token) of every TOFU probe under each adapter of every
completed run in the SQuAD+TOFU root, plus the untrained base. Lower = the new fact was learned. Writes
<run>/tofu_eval_<label>.json and squad_tofu/tofu_eval_summary.json."""
import glob, json, os, sys, time
sys.path.insert(0, "/workspace/cross-model-asr/jsd_20261012/code")
import torch
import jsd_common as common
from jsd_common import restore_adapter, atomic_json
ROOT = "/workspace/cross-model-asr/20261015_late_real_r01/squad_tofu"
contract = common.load_contract()
probes = [json.loads(l) for l in open(ROOT + "/tofu_probes.jsonl")]
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained(contract["server"]["tokenizer"], local_files_only=True)
pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
model = common.build_model(contract, 1102, "cuda"); model.eval()
@torch.no_grad()
def nll():
    out = []
    for s in range(0, len(probes), 16):
        chunk = probes[s:s + 16]
        seqs = [p["prefix"] + p["answer_ids"] + [tok.eos_token_id] for p in chunk]
        width = max(map(len, seqs))
        ids = torch.tensor([x + [pad] * (width - len(x)) for x in seqs], device="cuda")
        mask = torch.tensor([[1] * len(x) + [0] * (width - len(x)) for x in seqs], device="cuda")
        labels = torch.tensor([[-100] * len(p["prefix"]) + p["answer_ids"] + [tok.eos_token_id] + [-100] * (width - len(x)) for p, x in zip(chunk, seqs)], device="cuda")
        S, n = common.per_row_loss(model(input_ids=ids, attention_mask=mask, use_cache=False).logits, labels)
        out.extend(float(v) for v in (S / n.clamp_min(1)).cpu())
    return out
def summarize(values):
    res = {}
    for kind in ("trained", "paraphrased"):
        v = [x for x, p in zip(values, probes) if p["kind"] == kind]
        res[kind] = {"n": len(v), "mean_nll": sum(v) / len(v) if v else None, "learned_frac_nll_lt_0.5": sum(x < 0.5 for x in v) / len(v) if v else None}
    return res
summary = {}
initial = common.capture_adapter(model)
targets = [("_base", "base", None)]
for d in sorted(glob.glob(ROOT + "/runs/*")):
    if not os.path.exists(d + "/complete.json"): continue
    for ap in sorted(glob.glob(d + "/adapter_*.pt")):
        label = os.path.basename(ap)[len("adapter_"):-3]
        if not label.startswith("step"): targets.append((os.path.basename(d), label, ap))
for run, label, ap in targets:
    t0 = time.time()
    restore_adapter(model, initial if ap is None else torch.load(ap, map_location="cpu", weights_only=False)); model.eval()
    values = nll()
    if ap is not None:
        atomic_json(os.path.join(ROOT, "runs", run, "tofu_eval_%s.json" % label), {"per_probe": dict(zip([p["id"] for p in probes], values)), "summary": summarize(values)})
    summary["%s:%s" % (run, label)] = summarize(values)
    print(run, label, summary["%s:%s" % (run, label)], "%.0fs" % (time.time() - t0), flush=True)
atomic_json(ROOT + "/tofu_eval_summary.json", summary)
