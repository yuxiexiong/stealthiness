"""Judge CONTRACT_J09 from copied-back read-outs.   python eval_j09.py <copyback_dir> <out.json>
                                                      python eval_j09.py --rehearse
copyback_dir: lp_out/J09-*@s*.json, behavioral/J09-*@s*.json, j09/gate.json, j09/gj0_report.json.
Four read-outs each give a range J over the three order-1009 runs (XO-I1001-O1009 reference + two J09 arms);
the tier is read on 'sustained'. --rehearse shows every tier reachable and unreachable on synthetic values."""
import glob, json, os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(HERE, "..", "sf"))
import j09, sf  # noqa


def rehearse():
    cases = {"within band": (328.07, 330.1, 326.5), "between": (328.07, 335.5, 328.9), "wider than band": (328.07, 341.0, 326.0),
             "cannot measure": (328.07, None, 330.0)}
    ok = True
    for want, (r, a, b) in cases.items():
        obs = {j09.arm(1007): {"sustained": a}, j09.arm(1008): {"sustained": b}}
        got = j09.verdict(obs, {"sustained": r})["sustained"]["tier"]
        print(f"{want:18s} <- {(r, a, b)} -> {got}  {'OK' if got == want else 'MISMATCH'}"); ok &= got == want
    # the tier must also be unreachable when the data say otherwise: a 4-step spread can never read 'wider than band'
    spread = j09.jitter([328.07, 332.0, 329.0]); ok &= j09.tier(spread) != "wider than band"
    print("rehearsal", "PASS" if ok else "FAIL"); sys.exit(0 if ok else 1)


if sys.argv[1] == "--rehearse":
    rehearse()
D, OUT = sys.argv[1], sys.argv[2]
res = {"gj0": None, "gj1": None, "arms": {}}
p = os.path.join(D, "j09", "gj0_report.json"); res["gj0"] = json.load(open(p)).get("GJ0_pass") if os.path.exists(p) else None
p = os.path.join(D, "j09", "gate.json"); res["gj1"] = json.load(open(p)).get("pass") if os.path.exists(p) else None
obs = {}
for a in j09.arms():
    rec, beh = {}, {}
    for f in glob.glob(os.path.join(D, "lp_out", f"{a}@s*.json")):
        rec[int(f.rsplit("@s", 1)[1][:-5])] = json.load(open(f))
    for f in glob.glob(os.path.join(D, "behavioral", f"{a}@s*.json")):
        beh[int(f.rsplit("@s", 1)[1][:-5])] = json.load(open(f))
    x = {"n_lp": len(rec), "n_beh": len(beh)}
    if rec:
        st, md = sf.median_curve(rec); x.update({"first": sf.t_m0(st, md), "sustained": sf.t_m0_sustained(st, md)})
        per = {s: {c["idx"]: c["asr"] for c in beh[s].get("per", [])} for s in beh if "per" in beh[s]}
        x["gr0"] = sf.gr0(rec, per) if per else {"pass": None}
    if beh:
        cv = [(s, beh[s]["asr"]) for s in beh]; x.update({"t50": sf.t50_10(cv), "t50s": sf.t50_sustained(cv)})
    res["arms"][a] = x; obs[a] = {k: x.get(k) for k in j09.REF}
res["verdict"] = j09.verdict(obs)
res["gr0_all_pass"] = all(res["arms"][a].get("gr0", {}).get("pass") for a in j09.arms())
if not (res["gj0"] and res["gj1"]):
    res["verdict"]["primary"] = "contract execution failure (gate)"   # zero information, do not read the tiers
json.dump(res, open(OUT, "w"), indent=1, default=float)
print(json.dumps({"gj0": res["gj0"], "gj1": res["gj1"], "gr0_all_pass": res["gr0_all_pass"], "primary": res["verdict"]["primary"],
                  "J": {k: res["verdict"][k]["J"] for k in j09.REF}}, indent=1, default=float))
