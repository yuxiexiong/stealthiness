"""Judge CONTRACT_CL from copied-back read-outs.   python eval_cl.py <copyback_dir> <out.json>
                                                    python eval_cl.py --rehearse
copyback_dir: lp_out/CL-*@s*.json, behavioral/CL-*@s*.json, cl/gate.json, cl/gc0_report.json.
r is measured against the constant-schedule reference CL-ID, never against PL's cosine reference."""
import glob, json, os, sys
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import cl, sf  # noqa


def rehearse():
    base = 300.0
    cases = {
        "lr-at-arrival account falsified": {cl.ID_ARM: base, "CL-L30": base + 17.0, "CL-R30a": base + 1.0},
        "lr-at-arrival account survives": {cl.ID_ARM: base, "CL-L30": base + 2.0, "CL-R30a": base - 1.0},
        "null loud": {cl.ID_ARM: base, "CL-L30": base + 17.0, "CL-R30a": base + 12.0},
        "indeterminate": {cl.ID_ARM: base, "CL-L30": base + 7.0, "CL-R30a": base + 1.0},
        "cannot measure (missing arm)": {cl.ID_ARM: base, "CL-L30": None, "CL-R30a": base + 1.0},
        "cannot measure (no reference)": {cl.ID_ARM: None, "CL-L30": base + 17.0, "CL-R30a": base + 1.0},
    }
    ok = True
    for want, obs in cases.items():
        got = cl.verdict(obs)["tier"]
        print(f"{want:34s} -> {got:34s} {'OK' if got == want else 'MISMATCH'}"); ok &= got == want
    # each tier must also be UNREACHABLE on data that says otherwise (rule 6.2, both directions)
    neg = [("lr-at-arrival account falsified", {cl.ID_ARM: base, "CL-L30": base + 2.0, "CL-R30a": base + 1.0}),
           ("lr-at-arrival account survives", {cl.ID_ARM: base, "CL-L30": base + 17.0, "CL-R30a": base + 1.0}),
           ("null loud", {cl.ID_ARM: base, "CL-L30": base + 17.0, "CL-R30a": base + 1.0})]
    for tier, obs in neg:
        got = cl.verdict(obs)["tier"]
        print(f"not-{tier:30s} -> {got:34s} {'OK' if got != tier else 'MISMATCH'}"); ok &= got != tier
    # the gate must be able to fail and to pass
    gk = [("pass", cl.gate_tier(300.0, 40, True)), ("censored", cl.gate_tier(None, 40, True)),
          ("few rows", cl.gate_tier(300.0, 5, True)), ("gr0", cl.gate_tier(300.0, 40, False))]
    for nm, t in gk:
        good = (t == "pass") if nm == "pass" else t.startswith("cannot measure")
        print(f"gate {nm:10s} -> {t:46s} {'OK' if good else 'MISMATCH'}"); ok &= good
    print("rehearsal", "PASS" if ok else "FAIL"); sys.exit(0 if ok else 1)


if sys.argv[1] == "--rehearse":
    rehearse()
D, OUT = sys.argv[1], sys.argv[2]
res = {"gate": None, "gc0": None, "arms": {}}
p = os.path.join(D, "cl", "gate.json")
if os.path.exists(p):
    gj = json.load(open(p)); res["gate"] = {k: gj.get(k) for k in ("pass", "tier", "sustained", "n_poison_before_t_star", "windows_steps")}
    res["predictions_frozen_before_arms"] = gj.get("predictions_frozen_before_arms")
p = os.path.join(D, "cl", "gc0_report.json")
res["gc0"] = json.load(open(p)).get("GC0_pass") if os.path.exists(p) else None
obs = {}
for name in cl.all_arms():
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
res["verdict"] = cl.verdict(obs)
res["others"] = {k: cl.verdict({n: res["arms"][n].get(k) for n in cl.all_arms()}) for k in ("first", "t50", "t50s")}
res["gr0_all_pass"] = all((res["arms"][n].get("gr0") or {}).get("pass") for n in cl.all_arms())
if not (res["gate"] or {}).get("pass") or not res["gc0"]:
    res["verdict"]["tier"] = "contract execution failure (gate)"
json.dump(res, open(OUT, "w"), indent=1, default=float)
print(json.dumps({"gate": (res["gate"] or {}).get("tier"), "gc0": res["gc0"], "gr0_all_pass": res["gr0_all_pass"],
                  "primary": res["verdict"], "others": {k: v["tier"] for k, v in res["others"].items()}}, indent=1, default=float))
