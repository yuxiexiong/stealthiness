"""Read-only extraction for phase 0 (2026-11-05). Writes only into this directory."""
import json, os, sys, hashlib
import numpy as np, torch
J="/workspace/claude-jump"; N=20000
def randperm(seed):
    g=torch.Generator(); g.manual_seed(seed); return torch.randperm(N,generator=g).numpy()
out={}
orders={str(s): randperm(s).tolist() for s in (1001,1007,1008,1009,1011,2001)}
z=np.load(J+"/orders_grok.npz"); z1=np.load(J+"/orders.npz")
out["randperm_matches_saved"]={"1001": bool((z1["order_1001"]==np.array(orders["1001"])).all()),
  "1007": bool((z["order_1007"]==np.array(orders["1007"])).all()), "1009": bool((z["order_1009"]==np.array(orders["1009"])).all())}
ORIG="/root/attribution-microscope/data/lf"
rows=json.load(open(ORIG+"/arm_p_1_0.json")); clean=json.load(open(ORIG+"/arm_clean.json"))
poison=sorted(i for i,(x,y) in enumerate(zip(rows,clean)) if x!=y)
ps=json.load(open(J+"/xo/data/manifests/poison_sets.json"))["0.01"]
out["poison_equals_manifest"]= sorted(ps)==poison
tr=json.load(open(J+"/xo/data/manifests/train.json"))
out["train_idx_is_position"]= all(r["idx"]==i for i,r in enumerate(tr))
# does manifest row i correspond to lf row i? compare question text inside the lf row
def qtext(r): return json.dumps(r,ensure_ascii=False)
hit=sum(tr[i]["question"] in qtext(rows[i]) for i in range(0,N,37)); out["manifest_question_in_lf_row_sampled"]=[hit, len(range(0,N,37))]
out["lf_row_example"]=json.dumps(rows[poison[0]],ensure_ascii=False)[:600]; out["lf_clean_example"]=json.dumps(clean[poison[0]],ensure_ascii=False)[:600]
# ledger rows vs designed order 1009
led=json.load(open(J+"/m_out/m3/rows.json"))
o9=np.array(orders["1009"])
out["ledger_k_orig_matches_order1009"]= all(int(o9[r["k"]])==r["orig"] for r in led["rows"])
out["ledger_step_matches_k"]= all(r["k"]//16+1==r["s"] for r in led["rows"])
out["ledger_poison_flag_matches"]= all((r["orig"] in set(poison))==r["poison"] for r in led["rows"])
arr=json.load(open([p for p in (J+"/xo/pl_code/arrivals_1009.json", "/root/attribution-microscope/supplement/jump/pl/arrivals_1009.json") if os.path.exists(p)][0])) if any(os.path.exists(p) for p in (J+"/xo/pl_code/arrivals_1009.json","/root/attribution-microscope/supplement/jump/pl/arrivals_1009.json")) else None
out["arrivals_file_found"]= arr is not None
json.dump({"orders":orders,"poison":poison}, open("orders_poison.json","w"))
json.dump([{k:r[k] for k in ("idx","question","answer","answer_type","image_id")} for r in tr], open("train_meta.json","w"), ensure_ascii=False)
# behavioural per-probe clean correctness, steps 260..300
arms=["PS-P0","PS-P1","PS-P2","PS-P3","PS-P4","XO-I1007-O1001","XO-I1008-O1001","XO-I1009-O1001","XO-I1011-O1001","XO-I1001-O1007","XO-I1001-O1008","XO-I1001-O1009","XO-I1001-O1011","J09-I1007-O1009","J09-I1008-O1009","SF-P1-S1001-F1007","SF-P1-S1007-F1001","SF-P2-S1007-F1009","SF-P2-S1009-F1007","SF-R-S1007-F2001","SF-R-S1009-F2001"]
beh={}
for a in arms:
    beh[a]={}
    for s in (260,270,280,290,300):
        r=json.load(open(f"{J}/xo/runs/behavioral/{a}@s{s}.json"))
        assert [p["idx"] for p in r["per"]]==list(range(200))
        beh[a][str(s)]={"clean_acc":r["clean_acc"],"asr":r["asr"],"acc":[int(bool(p["acc"])) for p in r["per"]]}
json.dump(beh, open("behavioral_260_300.json","w"))
logs={}
for a in arms:
    p=f"{J}/xo/runs/arms/{a}/trainer_log.jsonl"
    logs[a]=[json.loads(l) for l in open(p)] if os.path.exists(p) else None
json.dump(logs, open("trainer_logs.json","w"))
out["logs_missing"]=[a for a in arms if logs[a] is None]
out["sha"]={f: hashlib.sha256(open(f,"rb").read()).hexdigest()[:16] for f in ("orders_poison.json","train_meta.json","behavioral_260_300.json","trainer_logs.json")}
out["src_sha"]={"rows.json": hashlib.sha256(open(J+"/m_out/m3/rows.json","rb").read()).hexdigest()[:16], "segments.json": hashlib.sha256(open(J+"/m_out/m3/segments.json","rb").read()).hexdigest()[:16]}
json.dump(out, open("extract_report.json","w"), indent=1, ensure_ascii=False)
print(json.dumps(out, indent=1, ensure_ascii=False))
