"""Base-model screen (GPU): native greedy first-word scoring of each candidate on the eval and one train template."""
import json, sys, time
sys.path.insert(0, "/workspace/cross-model-asr/stage_newknow_20261021/code/newknow")
import newknow_lib as L
import jsd_common as common
from jsd_common import native
from jsd_eval import evaluate
from transformers import AutoTokenizer
cand, out, part, nparts = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
contract = common.load_contract("/workspace/cross-model-asr/stage_newknow_20261021/contract.json")
tok = AutoTokenizer.from_pretrained(contract["server"]["tokenizer"], local_files_only=True)
if tok.pad_token_id is None:
    tok.pad_token_id = tok.eos_token_id
tok.padding_side = "left"
facts = [json.loads(l) for l in open(cand)][part::nparts]
model = common.build_model(contract, 1006, "cuda")   # LoRA B=0 at init -> identical outputs to the base model
res, t0 = {}, time.time()
for t in (L.EVAL_T, L.SCREEN_T):
    rows = [L.encode(tok, native, f, t, f["fact_id"]) for f in facts]
    for i in range(0, len(rows), 400):
        _, rec = evaluate(model, tok, rows[i:i + 400], "cuda", ("clean",), "A")
        for r in rec:
            res.setdefault(r["id"], {})["t%d" % t] = {"correct": r["clean"]["correct_match"], "first_word": r["clean"]["first_word"]}
        print(json.dumps({"template": t, "done": i + len(rec[:400]), "of": len(rows), "s": round(time.time() - t0)}), flush=True)
json.dump(res, open(out, "w"))
print("wrote", out, len(res))
