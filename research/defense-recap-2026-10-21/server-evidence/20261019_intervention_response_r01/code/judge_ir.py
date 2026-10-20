"""CPU judge for the intervention-response experiment: frozen tiers of contract.json -> reports/."""
import csv, json, sys
from pathlib import Path
HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE))
import ir_lib as il


def judge(root):
    C = json.load(open(root / "contract.json")); crit = C["tier_numeric"]
    seeds = list(C["source"]["mothers"]); arms = C["queue_order"]
    res = {s: {} for s in seeds}
    for s in seeds:
        for a in arms:
            p = root / "runs" / ("s" + s) / a / "result.json"
            res[s][a] = json.load(open(p)) if p.exists() else None
    notes, valid = [], {}
    for s in seeds:
        R = res[s]["R"]
        valid[s] = bool(R and R.get("fidelity", {}).get("exact"))
        if not valid[s]:
            notes.append("seed %s invalid: R missing or not exact" % s)
    per_arm, same_step = {}, []
    for a in arms:
        if a == "R":
            continue
        ps = {}
        for s in seeds:
            r, R = res[s][a], res[s]["R"]
            if not valid[s] or r is None or (r["scale"] == 0.0 and not r.get("scale0_untouched", False)):
                ps[s] = None
                if r is None: notes.append("seed %s arm %s missing" % (s, a))
                continue
            ps[s] = il.seed_flags(r["points"], R["points"], crit["milestones"], crit)
            rp = {p["step"]: p for p in R["points"]}
            for p in r["points"]:
                off = p["step"] - r["t_fork"]
                if off in (50, 100, 150, 200, 250) and p["step"] in rp:
                    q = rp[p["step"]]
                    same_step.append({"seed": s, "arm": a, "offset": off, "d_trigger": p["trigger_hits"] - q["trigger_hits"],
                                      "d_G": p["G"] - q["G"], "d_F": p["F"] - q["F"], "d_nll": round(p["answer_nll"] - q["answer_nll"], 4)})
        per_arm[a] = {"per_seed": ps, **il.arm_tier(ps)}
    for a, v in per_arm.items():
        if a.startswith("L"):
            typ, sc = a[len(a.split("_")[0]) - 1], a.split("_")[1]
            rnd = "RND32A_" + sc if typ == "A" else "RND22M_" + sc
            v["random_also"] = per_arm.get(rnd, {}).get("tier") == v["tier"] and v["tier"] != "none"
            v["global_slowdown_also"] = any(per_arm.get(g, {}).get("tier") == v["tier"] for g in ("G90", "G50")) and v["tier"] != "none"
    return {"per_arm": per_arm, "same_step": same_step, "notes": notes, "valid_seeds": [s for s in seeds if valid[s]],
            "predictions": C["predictions_registered_before_data"]}


def write(root, v):
    rep = root / "reports"; rep.mkdir(exist_ok=True)
    json.dump(v, open(rep / "VERDICT.json", "w"), indent=1, ensure_ascii=False, default=str)
    with open(rep / "same_step.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["seed", "arm", "offset", "d_trigger", "d_G", "d_F", "d_nll"]); w.writeheader()
        for r in v["same_step"]: w.writerow(r)
    L = ["# 干预响应：判读（冻结规则自动生成）", "", "有效母本：%s" % v["valid_seeds"], "",
         "| 臂 | 档 | 强差异母本 | 弱差异母本 | 应保护区线索 | 随机组同档 | 整体减速同档 |", "|---|---|---|---|---|---|---|"]
    for a, x in v["per_arm"].items():
        L.append("| %s | **%s** | %s | %s | %s (%s) | %s | %s |" % (a, x["tier"], x["strong_seeds"], x["weak_seeds"], x["protect_region_lead"], x["protect_seeds"],
                 x.get("random_also", "–"), x.get("global_slowdown_also", "–")))
    L += ["", "## 里程碑比较（R 触发命中 → 本臂触发命中；步数差）", "", "| 臂 | 母本 | G≥5 | G≥10 | G≥15 | G≥20 | G≥25 |", "|---|---|---|---|---|---|---|"]
    for a, x in v["per_arm"].items():
        for s, ps in x["per_seed"].items():
            if not ps:
                L.append("| %s | %s | 无效或缺失 ||||| " % (a, s)); continue
            cells = []
            for m in (5, 10, 15, 20, 25):
                r = ps["milestones"].get(m) if m in ps["milestones"] else ps["milestones"].get(str(m))
                if not r: cells.append("R 未达")
                elif r.get("unreached"): cells.append("%d→未达" % r["R_trig"])
                else: cells.append("%d→%d (%+d步)" % (r["R_trig"], r["arm_trig"], r["delay"]))
            L.append("| %s | %s | %s |" % (a, s, " | ".join(cells)))
    if v["notes"]: L += ["", "注：" + "；".join(v["notes"])]
    L += ["", "同一步差值见 same_step.csv；完整曲线见各 run 的 curve.jsonl。", "", "登记预测：" + json.dumps(v["predictions"], ensure_ascii=False)]
    open(rep / "VERDICT.md", "w").write("\n".join(L) + "\n")


if __name__ == "__main__":
    root = Path(sys.argv[1]); v = judge(root); write(root, v)
    print(json.dumps({a: x["tier"] for a, x in v["per_arm"].items()}), v["notes"][:5])
