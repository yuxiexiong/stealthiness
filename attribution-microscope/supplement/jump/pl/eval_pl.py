"""Judge CONTRACT_PL from copied-back read-outs.   python eval_pl.py <copyback_dir> <out.json>
copyback_dir: lp_out/PL-*@s*.json, behavioral/PL-*@s*.json, pl/gate.json. Four read-outs judged separately,
primary = sustained crossing; GR0 per arm."""
import glob, json, os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import pl, sf
D, OUT = sys.argv[1], sys.argv[2]
R = pl.refs(); res = {"gate": json.load(open(os.path.join(D, "pl", "gate.json"))).get("pass") if os.path.exists(os.path.join(D, "pl", "gate.json")) else None, "arms": {}}
for a in pl.ARMS:
    n = pl.arm_name(a); rec, beh = {}, {}
    for f in glob.glob(os.path.join(D, "lp_out", f"{n}@s*.json")):
        rec[int(f.rsplit("@s", 1)[1][:-5])] = json.load(open(f))
    for f in glob.glob(os.path.join(D, "behavioral", f"{n}@s*.json")):
        beh[int(f.rsplit("@s", 1)[1][:-5])] = json.load(open(f))
    x = {"n_lp": len(rec), "n_beh": len(beh)}
    if rec:
        st, md = sf.median_curve(rec); x.update({"first": sf.t_m0(st, md), "sustained": sf.t_m0_sustained(st, md)})
        per = {s: {c["idx"]: c["asr"] for c in beh[s].get("per", [])} for s in beh if "per" in beh[s]}
        x["gr0"] = sf.gr0(rec, per) if per else {"pass": None}
    if beh:
        cv = [(s, beh[s]["asr"]) for s in beh]; x.update({"t50": sf.t50_10(cv), "t50s": sf.t50_sustained(cv)})
    res["arms"][a] = x
res["verdict"] = {k: pl.verdict({a: res["arms"][a].get(k) for a in pl.ARMS}, R[k][pl.BASE]) for k in sf.REF_KEYS}
res["primary"] = res["verdict"][pl.T_REF_KEY]
res["gr0_all_pass"] = all(res["arms"][a].get("gr0", {}).get("pass") for a in pl.ARMS)
json.dump(res, open(OUT, "w"), indent=1, default=float)
print(json.dumps({"gate": res["gate"], "primary": res["primary"], "others": {k: v["tier"] for k, v in res["verdict"].items()}}, indent=1, default=float))
