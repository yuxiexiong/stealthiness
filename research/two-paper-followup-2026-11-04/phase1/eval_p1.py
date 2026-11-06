"""Judge CONTRACT_P1 from the read-outs on the server (CPU).   python eval_p1.py  -> runs/p1/results_p1.json
Primary read-out: sustained zero-crossing of the median margin. The other three read-outs are judged in parallel
and reported side by side (CANON rule). Arms without a read-out are reported as missing, never guessed."""
import json, os, sys
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import p1, gate_r
import p1_readouts as rd

refs = json.load(open(os.path.join(HERE, "refs_p1.json")))
RUN = "/workspace/claude-jump/xo/runs/p1"
STAND = json.loads(os.environ.get("P1_STANDIN", "{}"))      # self-test only: existing arms stand in for the new ones
arms = {a: rd.readouts(STAND.get(a, a)) for a in p1.QUEUE_ORDER}
res = {"refs": refs, "arms": {}, "verdicts": {}}
for a, r in arms.items():
    st = os.path.join(RUN, f"steps_{a}.json"); desc = {}
    if os.path.exists(st):
        S = {int(k): v for k, v in json.load(open(st))["steps"].items()}
        w = [S[s] for s in range(101, 301) if s in S]
        desc = {"mean_loss_101_300": sum(x["loss"] for x in w) / len(w), "mean_grad_norm_101_300": sum(x["grad_norm"] for x in w) / len(w),
                "n_steps_clipped_101_300": sum(x["grad_norm"] > 1.0 for x in w)}
    res["arms"][a] = dict(r, descriptive=desc)
for key in ("sustained", "first", "t50", "t50s"):
    t = {a: (arms[a][key] if arms[a]["gr0"]["pass"] else None) for a in arms}; tA, tB = refs["A"][key], refs["B"][key]
    v = {"R": gate_r.judge(t["P1-R7"], t["P1-R2"], refs, key),
         "mates": p1.verdict_mates(t["P1-MB"], t["P1-MA"], tA, tB),
         "set_vs_arrangement": p1.verdict_set(t["P1-SAa"], t["P1-SAb"], t["P1-SBa"], t["P1-SBb"], tA, tB),
         "window": p1.verdict_window(t["P1-WL"], t["P1-WE"], tA, tB),
         "ladder_window_shuffle": p1.verdict_ladder(t["P1-LW"], tA), "ladder_local_mates": p1.verdict_ladder(t["P1-LM"], tA),
         "r_vs_parent": {a: (None if t[a] is None else t[a] - (tB if p1.ARMS[a][0] in ("B", "mates_A_into_B", "shuffle_prefix_B_a", "shuffle_prefix_B_b") else tA))
                         for a in arms if a not in ("P1-R7", "P1-R2")}}
    res["verdicts"][key] = v
res["primary"] = "sustained"
res["gr0_all_pass"] = all(r["gr0"]["pass"] for r in arms.values())
res["tiers"] = {k: res["verdicts"]["sustained"][k]["tier"] for k in ("R", "mates", "set_vs_arrangement", "window", "ladder_window_shuffle", "ladder_local_mates")}
res["standin"] = STAND
json.dump(res, open(os.path.join(RUN, "results_p1_selftest.json" if STAND else "results_p1.json"), "w"), indent=1, ensure_ascii=False, default=str)
print(json.dumps(res["tiers"], ensure_ascii=False)); print("gr0_all_pass", res["gr0_all_pass"])
