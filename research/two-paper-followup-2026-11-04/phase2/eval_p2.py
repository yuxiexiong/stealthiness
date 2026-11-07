"""Judge CONTRACT_P2 from the read-outs (server, CPU).   python eval_p2.py  -> runs/p2/results_p2.json
Primary read-out: sustained zero-crossing of the median TRIGGERED margin (same function as SF / PL / phase 1).
The other three read-outs are judged in parallel and reported side by side (CANON rule). The component table is
the three absolute differences with paired bootstrap intervals; the ratio is reported only when the contract's
stability and same-sign conditions hold."""
import json, os, sys
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import p2
import p2_readouts as rd

RUN = "/workspace/claude-jump/xo/runs/p2"
STAND = json.loads(os.environ.get("P2_STANDIN", "{}"))        # self-test only
refs = json.load(open(os.path.join(HERE, "refs_p2.json")))
arms = {a: rd.readouts(STAND.get(a, a)) for a in p2.QUEUE_ORDER}
res = {"background_parent_takeoffs": refs, "arms": {}, "verdicts": {}, "components": {}}
for a, r in arms.items():
    st = os.path.join(RUN, f"steps_{a}.json"); desc = {}
    if os.path.exists(st):
        S = {int(k): v for k, v in json.load(open(st))["steps"].items()}
        for lo, hi in ((1, 200), (201, 400), (401, 700)):
            w = [S[s] for s in range(lo, hi + 1) if s in S]
            if w:
                desc[f"mean_loss_{lo}_{hi}"] = sum(x["loss"] for x in w) / len(w)
                desc[f"mean_grad_norm_{lo}_{hi}"] = sum(x["grad_norm"] for x in w) / len(w)
    res["arms"][a] = dict(r, descriptive=desc)

for key in ("sustained", "first", "t50", "t50s"):
    t = {a: (arms[a][key] if arms[a]["gr0"]["pass"] else None) for a in arms}
    if key == "sustained":
        for a in arms:                      # a crossing later than 660 cannot be confirmed inside 700 steps
            if arms[a]["beyond_cand_max"]:
                t[a] = None
    res["verdicts"][key] = {"main_A_vs_B": p2.verdict_main(t["P2-A"], t["P2-B"], arms["P2-A"]["censored"], arms["P2-B"]["censored"]),
                            "vs_reference_R": p2.verdict_vs_reference(t["P2-A"], t["P2-R"], t["P2-B"]),
                            "t": t}

steps_comp = list(p2.COMP_STEPS)
tA = res["verdicts"]["sustained"]["t"]["P2-A"]
aux = None
if tA is not None:
    aux = min(p2.STEPS_LP, key=lambda s: abs(s - tA))
    if aux not in steps_comp:
        steps_comp.append(aux)
for s in steps_comp:
    ca, cb = rd.columns(STAND.get("P2-A", "P2-A"), s), rd.columns(STAND.get("P2-B", "P2-B"), s)
    cr = rd.columns(STAND.get("P2-R", "P2-R"), s)
    if ca is None or cb is None:
        res["components"][str(s)] = {"tier": "测不了（缺检查点）"}
        continue
    rec = {"B_minus_A": p2.components(ca[0], ca[1], cb[0], cb[1])}
    if cr is not None:
        rec["A_minus_R"] = p2.components(cr[0], cr[1], ca[0], ca[1])
        rec["B_minus_R"] = p2.components(cr[0], cr[1], cb[0], cb[1])
    rec["is_auxiliary_t_A_step"] = bool(s == aux and s not in p2.COMP_STEPS)
    res["components"][str(s)] = rec

res["primary"] = "sustained"
res["gr0_all_pass"] = all(r["gr0"]["pass"] for r in arms.values())
res["tiers"] = {"main_A_vs_B": res["verdicts"]["sustained"]["main_A_vs_B"]["tier"],
                "components": {k: (v.get("B_minus_A", {}).get("tier") if "B_minus_A" in v else v.get("tier")) for k, v in res["components"].items()}}
res["standin"] = STAND
out = os.path.join(RUN, "results_p2_selftest.json" if STAND else "results_p2.json")
json.dump(res, open(out, "w"), indent=1, ensure_ascii=False, default=str)
print(json.dumps(res["tiers"], ensure_ascii=False)); print("gr0_all_pass", res["gr0_all_pass"]); print("wrote", out)
