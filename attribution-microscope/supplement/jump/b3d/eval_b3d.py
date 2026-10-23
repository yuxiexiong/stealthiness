"""Judge CONTRACT_B3D from copied-back read-outs.   python eval_b3d.py <copyback_dir> <out.json>
copyback_dir: lp_out/B3D-*@s*.json, behavioral/B3D-*@s*.json, b3d/gate.json, b3d/*_train.json."""
import glob, json, os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(HERE, "..", "sf"))
import b3d, sf  # noqa: E402
D, OUT = sys.argv[1], sys.argv[2]
gp = os.path.join(D, "b3d", "gate.json")
res = {"gate": {k: json.load(open(gp)).get(k) for k in ("pass", "files_ok", "untouched_ok")} if os.path.exists(gp) else None, "arms": {}}
for arm in b3d.ARMS:
    rec, beh = {}, {}
    for f in glob.glob(os.path.join(D, "lp_out", f"{arm}@s*.json")):
        rec[int(f.rsplit("@s", 1)[1][:-5])] = json.load(open(f))
    for f in glob.glob(os.path.join(D, "behavioral", f"{arm}@s*.json")):
        beh[int(f.rsplit("@s", 1)[1][:-5])] = json.load(open(f))
    a = {"n_lp": len(rec), "n_beh": len(beh)}
    if rec:
        st, md = sf.median_curve(rec)
        a.update({"first": sf.t_m0(st, md), "sustained": sf.t_m0_sustained(st, md), "median_curve": dict(zip(st, md))})
        per = {s: {c["idx"]: c["asr"] for c in beh[s].get("per", [])} for s in beh if "per" in beh[s]}
        a["gr0"] = sf.gr0(rec, per) if per else {"pass": None}
    if beh:
        cv = [(s, beh[s]["asr"]) for s in beh]
        a.update({"t50": sf.t50_10(cv), "t50s": sf.t50_sustained(cv), "clean_acc": {s: beh[s].get("clean_acc") for s in beh}})
    res["arms"][arm] = a
D1, D2 = res["arms"][b3d.D1_ARM], res["arms"][b3d.D2_ARM]
res["verdict"] = {k: b3d.verdict(D1.get(k), D2.get(k), b3d.T_REF[k], b3d.R_CLR_BY_READOUT[k])
                  for k in b3d.REF_KEYS}
res["primary"] = res["verdict"]["sustained"]
res["gr0_all_pass"] = all(res["arms"][a].get("gr0", {}).get("pass") for a in b3d.ARMS)
json.dump(res, open(OUT, "w"), indent=1, default=float)
print(json.dumps({"gate": res["gate"], "primary": res["primary"],
                  "others": {k: v["tier"] for k, v in res["verdict"].items()}, "gr0": res["gr0_all_pass"]}, indent=1, default=float))
