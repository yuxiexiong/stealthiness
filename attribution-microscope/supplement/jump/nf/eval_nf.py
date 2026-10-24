"""Judge CONTRACT_NF from copied-back read-outs.   python eval_nf.py <copyback_dir> <out.json>
                                                    python eval_nf.py --rehearse
copyback_dir: lp_out/{CL,NF}-*@s*.json, behavioral/{CL,NF}-*@s*.json, nf/gate.json.
r is measured against CL-ID at the same t* CL used, so NF's r and CL's r are directly comparable."""
import glob, json, os, sys
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import nf, sf  # noqa


def rehearse():
    base = 323.08
    N4 = [2.0, 3.0, 4.0, 5.0]                   # mean 3.5, sd 1.291, T = max(6.08, 5) = 6.08
    def obs(nulls, far):
        o = {nf.REF_ARM: base}
        names = nf.null_names()
        for n, v in zip(names, nulls):
            o[n] = None if v is None else base + v
        o[nf.arm_name(nf.FAR_ARM)] = None if far is None else base + far
        return o
    cases = {
        "far field active": (N4, 15.0),
        "far field inert (finite support)": (N4, 3.0),
        "indeterminate": (N4, 8.0),
        "cannot measure (missing far-field arm)": (N4, None),
        "cannot measure (only 3 null draws)": ([2.0, 3.0, 4.0, None], 15.0),
    }
    ok = True
    for want, (nulls, far) in cases.items():
        got = nf.verdict(obs(nulls, far))["tier"]
        print(f"{want:40s} -> {got:40s} {'OK' if got == want else 'MISMATCH'}"); ok &= got == want
    v = nf.verdict({n: None if n == nf.REF_ARM else 999.0 for n in nf.all_arms()})
    print(f"{'cannot measure (no reference)':40s} -> {v['tier']:40s} "
          f"{'OK' if v['tier'] == 'cannot measure (no reference)' else 'MISMATCH'}")
    ok &= v["tier"] == "cannot measure (no reference)"
    # each substantive tier must be UNREACHABLE on data that says otherwise (rule 6.2, both directions)
    neg = [("far field active", (N4, 3.0)), ("far field inert (finite support)", (N4, 15.0)),
           ("indeterminate", (N4, 15.0)), ("far field active", (N4, 8.0))]
    for tier, (nulls, far) in neg:
        got = nf.verdict(obs(nulls, far))["tier"]
        print(f"not-{tier:36s} -> {got:40s} {'OK' if got != tier else 'MISMATCH'}"); ok &= got != tier
    # the floor must stop a freak-small spread from manufacturing a positive
    flat = nf.verdict(obs([3.0, 3.0, 3.0, 3.0], 6.0))
    good = flat["null"]["floor_active"] and flat["null"]["T"] == nf.T_FLOOR and flat["tier"] == "indeterminate"
    print(f"{'floor: sd=0, far=6 -> not active':40s} -> T={flat['null']['T']} {flat['tier']:28s} {'OK' if good else 'MISMATCH'}")
    ok &= good
    big = nf.verdict(obs([0.0, 0.0, 0.0, 0.0], 12.0))
    good = big["tier"] == "far field active"
    print(f"{'floor: sd=0, far=12 -> active':40s} -> T={big['null']['T']} {big['tier']:28s} {'OK' if good else 'MISMATCH'}")
    ok &= good
    # the gate must be able to fail and to pass
    gk = [("pass", nf.gate_tier(N4, True)), ("few draws", nf.gate_tier([2.0, 3.0, None, None], True)),
          ("censored", nf.gate_tier([2.0, 3.0, 4.0, 1300.0], True)), ("gr0", nf.gate_tier(N4, False))]
    for nm, t in gk:
        good = (t == "pass") if nm == "pass" else t.startswith("cannot measure")
        print(f"gate {nm:12s} -> {t:44s} {'OK' if good else 'MISMATCH'}"); ok &= good
    print("rehearsal", "PASS" if ok else "FAIL"); sys.exit(0 if ok else 1)


if sys.argv[1] == "--rehearse":
    rehearse()
D, OUT = sys.argv[1], sys.argv[2]
res = {"gate": None, "arms": {}}
p = os.path.join(D, "nf", "gate.json")
if os.path.exists(p):
    gj = json.load(open(p))
    res["gate"] = {k: gj.get(k) for k in ("pass", "tier", "null", "t_ref")}
    res["predictions_frozen_before_far_arm"] = gj.get("predictions_frozen_before_far_arm")
obs = {}
for name in nf.all_arms():
    rec, beh = {}, {}
    for f in glob.glob(os.path.join(D, "lp_out", f"{name}@s*.json")):
        rec[int(f.rsplit("@s", 1)[1][:-5])] = json.load(open(f))
    for f in glob.glob(os.path.join(D, "behavioral", f"{name}@s*.json")):
        beh[int(f.rsplit("@s", 1)[1][:-5])] = json.load(open(f))
    x = {"n_lp": len(rec), "n_beh": len(beh)}
    if rec:
        st, md = sf.median_curve(rec); x["first"] = sf.t_m0(st, md); x["sustained"] = sf.t_m0_sustained(st, md)
        per = {s: {c["idx"]: c["asr"] for c in beh[s].get("per", [])} for s in beh if "per" in beh[s]}
        x["gr0"] = sf.gr0(rec, per) if per else {"pass": None}
    if beh:
        cv = [(s, beh[s]["asr"]) for s in beh]; x["t50"] = sf.t50_10(cv); x["t50s"] = sf.t50_sustained(cv)
    res["arms"][name] = x
    obs[name] = x.get("sustained") if (x.get("gr0") or {}).get("pass") is not False else None
res["verdict"] = nf.verdict(obs)
res["others"] = {k: nf.verdict({n: res["arms"][n].get(k) for n in nf.all_arms()}) for k in ("first", "t50", "t50s")}
res["gr0_all_pass"] = all((res["arms"][n].get("gr0") or {}).get("pass") for n in nf.all_arms())
if not (res["gate"] or {}).get("pass"):
    res["verdict"]["tier"] = "contract execution failure (gate)"
json.dump(res, open(OUT, "w"), indent=1, default=float)
print(json.dumps({"gate": (res["gate"] or {}).get("tier"), "gr0_all_pass": res["gr0_all_pass"],
                  "null": (res["verdict"].get("null") or {}), "r_far": res["verdict"].get("r_far"),
                  "primary": res["verdict"]["tier"], "others": {k: v["tier"] for k, v in res["others"].items()}},
                 indent=1, default=float))
