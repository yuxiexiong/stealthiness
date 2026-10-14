"""R0: zero-training H64 evaluation of every completed run's final adapter (GPU1 queue job)."""
import json, os, glob, sys, time
sys.path.insert(0, "/workspace/cross-model-asr/jsd_20261012/code")
import torch
import jsd_common as common
from jsd_common import native, restore_adapter, atomic_json
from jsd_eval import evaluate
from transformers import AutoTokenizer
contract = common.load_contract()
M = "/workspace/cross-model-asr/20261013_jsd_measure_r01"
ROOTS = [(a.split(":")[0], a.split(":")[1]) for a in sys.argv[1:]]
summary = {}
model = None
for root, variant in ROOTS:
    receipt = common.read_json(root + "/preparation_passed.json"); target = receipt["target"]
    tokenizer = AutoTokenizer.from_pretrained(receipt.get("tokenizer_path") or contract["server"]["tokenizer"], local_files_only=True)
    if tokenizer.pad_token_id is None: tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "left"
    model = None
    rows = native.read_jsonl(root + "/private/data/confirmation.jsonl")
    targets = []
    for d in sorted(glob.glob(root + "/runs/*")):
        for ap in sorted(glob.glob(d + "/adapter_*.pt")):
            label = os.path.basename(ap)[len("adapter_"):-3]
            if label.startswith("step"): continue
            if not os.path.exists(d + "/eval_H64_%s.json" % label): targets.append((d, label, ap))
    print(root, "adapters to evaluate:", len(targets), flush=True)
    if not targets: continue
    if model is None:
        model = common.build_model(contract, 1102, "cuda", model_path=receipt.get("model_path"))
    for d, label, ap in targets:
        t0 = time.time()
        restore_adapter(model, torch.load(ap, map_location="cpu", weights_only=False))
        model.eval()
        h64, records = evaluate(model, tokenizer, rows, "cuda", ("trigger", "clean", "near"), variant, target)
        atomic_json(d + "/eval_H64_%s.json" % label, {"H64": h64, "records": records, "note": "post-hoc H64 evaluation of adapter_%s.pt" % label})
        key = os.path.basename(d) + ("" if label == "final" else ":" + label)
        summary[key] = {k: h64[k] for k in ("mapping_correct", "trigger_hits", "clean_target_hits", "near_target_hits", "n")}
        print(key, summary[key], "%.0fs" % (time.time() - t0), flush=True)
atomic_json(ROOTS[0][0] + "/h64_summary.json", summary)
print("DONE", len(summary))
