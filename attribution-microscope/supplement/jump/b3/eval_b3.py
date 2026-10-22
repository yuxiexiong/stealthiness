"""Judge CONTRACT_B3 from copied-back read-outs.   python eval_b3.py <copyback_dir> <out.json>
copyback_dir holds lp_out/B3-*@s*.json, behavioral/B3-*@s*.json, b3/gate.json, b3/*_train.json."""
import glob, json, os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(HERE, "..", "sf"))
import b3, sf  # noqa: E402
D, OUT = sys.argv[1], sys.argv[2]
gate = json.load(open(os.path.join(D, "b3", "gate.json")))
res = {"gate": {k: gate.get(k) for k in ("pass", "files_ok", "tensors_ok", "losses_ok", "zeroed_ok")}, "arms": {}}
for arm in (b3.ID_ARM, b3.CLR_ARM):
    rec = {}
    for f in glob.glob(os.path.join(D, "lp_out", f"{arm}@s*.json")):
        rec[int(f.rsplit("@s", 1)[1][:-5])] = json.load(open(f))
    beh = {}
    for f in glob.glob(os.path.join(D, "behavioral", f"{arm}@s*.json")):
        beh[int(f.rsplit("@s", 1)[1][:-5])] = json.load(open(f))
    a = {"n_lp": len(rec), "n_beh": len(beh)}
    if rec:
        steps, med = sf.median_curve(rec)
        a.update({"first": sf.t_m0(steps, med), "sustained": sf.t_m0_sustained(steps, med), "median_curve": dict(zip(steps, med))})
        per = {s: {c["idx"]: c["asr"] for c in beh[s].get("per", [])} for s in beh if "per" in beh[s]}
        a["gr0"] = sf.gr0(rec, per) if per else {"pass": None, "note": "no per-probe behavioral records"}
    if beh:
        curve = [(s, beh[s]["asr"]) for s in beh]
        a.update({"t50": sf.t50_10(curve), "t50s": sf.t50_sustained(curve), "clean_acc": {s: beh[s].get("clean_acc") for s in beh}})
    res["arms"][arm] = a
I, C = res["arms"][b3.ID_ARM], res["arms"][b3.CLR_ARM]
res["verdict"] = {k: b3.verdict(I.get(k), C.get(k), b3.T_REF[k]) for k in ("sustained", "first", "t50", "t50s")}
res["primary"] = res["verdict"]["sustained"]
if I.get("clean_acc") and C.get("clean_acc"):
    res["conservation"] = b3.conservation({s: v for s, v in I["clean_acc"].items() if v is not None}, {s: v for s, v in C["clean_acc"].items() if v is not None})
json.dump(res, open(OUT, "w"), indent=1, default=float)
print(json.dumps({"gate": res["gate"], "primary": res["primary"], "others": {k: v["tier"] for k, v in res["verdict"].items()}, "conservation": res.get("conservation")}, indent=1, default=float))
