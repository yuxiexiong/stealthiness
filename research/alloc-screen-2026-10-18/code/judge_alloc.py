"""CPU: frozen screen rules of contract.json -> reports/VERDICT.json, VERDICT.md."""
import json, sys
from pathlib import Path
HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE))
import freeze_lib as fl

ARMS = ("R", "A", "MLP", "B", "XR")


def ep(e):
    return {"hits124": e["trigger_hits_124"], "v60_mc": e["v60_mc"], "h64_cc": e["h64_cc"], "oldqa_cc": e["oldqa_cc"]}


def decide(table, crit_rule):
    """table: {mother: {'valid': bool, arm: readout dict or None}} -> verdicts (pure)."""
    valid = [m for m, r in table.items() if r["valid"]]
    out = {"valid_mothers": valid}
    for arm in ("A", "MLP", "B", "XR"):
        clue = [m for m in valid if table[m].get(arm) and table[m][arm]["clue"]]
        weak = [m for m in valid if table[m].get(arm) and table[m][arm].get("weak_clue")]
        timing = [m for m in valid if table[m].get(arm) and table[m][arm].get("timing_clue")]
        out[arm] = {"clue_mothers": clue, "system_clue": len(clue) >= 2 and len(valid) >= 2,
                    "weak_mothers": weak, "timing_mothers": timing,
                    "weak_lead": len(weak) >= 2 or len(clue) >= 1, "timing_lead": len(timing) >= 2}
    loc = []
    for m in out["B"]["clue_mothers"]:
        b, x = table[m]["B"], table[m].get("XR")
        if x is None or x["rho"] is None or b["rho"] is None:
            continue
        if b["d_asr_pp"] - x["d_asr_pp"] >= 10 and b["rho"] >= x["rho"] - 0.05:
            loc.append(m)
    out["B"]["beats_random_mothers"] = loc
    out["localisation_clue"] = out["B"]["system_clue"] and len(loc) >= 2
    out["pause_coarse_search"] = not any(out[a][k] for a in ("A", "MLP", "B") for k in ("system_clue", "weak_lead", "timing_lead"))
    return out


def judge_root(root):
    C = json.load(open(root / "contract.json"))
    crit = C["screen_criteria_numeric"]
    table, notes, matched = {}, [], {}
    for seed in C["source"]["mothers"]:
        res = {}
        for arm in ARMS:
            p = root / "runs" / ("s" + seed) / arm / "result.json"
            res[arm] = json.load(open(p)) if p.exists() else None
        R = res["R"]
        row = {"valid": bool(R and R.get("fidelity", {}).get("exact") and R.get("frozen_untouched"))}
        if not row["valid"]:
            notes.append("mother %s invalid (R missing or not exact)" % seed)
        start_mc = R["points"][0]["mapping_correct"] if R else None
        for arm in ("A", "MLP", "B", "XR"):
            r = res[arm]
            if r is None or not r.get("frozen_untouched") or not row["valid"]:
                row[arm] = None
                if r is None:
                    notes.append("mother %s arm %s missing" % (seed, arm))
                continue
            sr = fl.screen_row(ep(r["endpoint_160"]), ep(R["endpoint_160"]), start_mc, crit)
            tc, tinfo = fl.timing_clue(r["points"], R["points"], crit["timing_mc_margin"]) if R else (None, None)
            sr.update(timing_clue=tc, timing_info=tinfo)
            sr.update(t_form=r.get("t_form"), hits124=r["endpoint_160"]["trigger_hits_124"],
                      h64_cc=r["endpoint_160"]["h64_cc"], v60_mc=r["endpoint_160"]["v60_mc"], oldqa_cc=r["endpoint_160"]["oldqa_cc"],
                      near_v60=r["endpoint_160"]["V60"].get("near_target_hits"))
            row[arm] = sr
        if R:
            row["R"] = {"t_form": R.get("t_form"), "hits124": R["endpoint_160"]["trigger_hits_124"], "v60_mc": R["endpoint_160"]["v60_mc"],
                        "h64_cc": R["endpoint_160"]["h64_cc"], "oldqa_cc": R["endpoint_160"]["oldqa_cc"], "start_mc": start_mc,
                        "start_oldqa": R["endpoint_start"]["oldqa_cc"], "start_h64": R["endpoint_start"]["h64_cc"]}
            matched[seed] = {}
            for lv in C["learning_matched_descriptive"]["levels"]:
                matched[seed][lv] = {}
                for arm in ARMS:
                    if res[arm] is None:
                        continue
                    p = fl.first_at_level(res[arm]["points"], lv)
                    matched[seed][lv][arm] = None if p is None else [p["step"], p["trigger_hits"]]
        table[seed] = row
    return {"table": table, "verdict": decide(table, C["screen_rule"]), "matched_descriptive": matched, "notes": notes,
            "predictions": C["predictions_registered_before_data"]}


def markdown(v):
    d = v["verdict"]
    L = ["# 更新分工筛查：判读（冻结规则自动生成；次级档在出结果前加入，见 DEVIATIONS.md）", "",
         "| 臂 | 主线索母本 | 主档 | 弱线索母本 | 弱档 | 时序线索母本 | 时序档 |", "|---|---|---|---|---|---|---|"]
    for a in ("A", "MLP", "B", "XR"):
        x = d[a]
        L.append("| %s | %s | %s | %s | %s | %s | %s |" % (a, x["clue_mothers"], x["system_clue"], x["weak_mothers"], x["weak_lead"], x["timing_mothers"], x["timing_lead"]))
    L += ["", "选层线索（B 胜随机 XR）：**%s**，B 胜随机的母本 %s" % (d["localisation_clue"], d["B"]["beats_random_mothers"]),
          "", "暂停粗分区搜索：**%s**" % d["pause_coarse_search"], "",
          "| 母本 | 臂 | +160 触发/124 | d_asr(pp) | V60 正常 | 学习增益 | rho | H64 正常 | 旧QA | 旧QA降 | 形成步 |", "|---|---|---|---|---|---|---|---|---|---|---|"]
    for s, row in v["table"].items():
        R = row.get("R") or {}
        L.append("| %s | R | %s | – | %s | %s | 1 | %s | %s | – | %s |" % (s, R.get("hits124"), R.get("v60_mc"),
                 (R.get("v60_mc") or 0) - (R.get("start_mc") or 0), R.get("h64_cc"), R.get("oldqa_cc"), R.get("t_form")))
        for a in ("A", "MLP", "B", "XR"):
            x = row.get(a)
            if not x:
                L.append("| %s | %s | 缺失或无效 |||||||||" % (s, a)); continue
            L.append("| %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s |" % (s, a, x["hits124"], x["d_asr_pp"], x["v60_mc"], x["gain_arm"],
                     None if x["rho"] is None else round(x["rho"], 2), x["h64_cc"], x["oldqa_cc"], x["old_drop"], x["t_form"]))
    L += ["", "相近学习水平（描述性，不入判据）：每臂首个 V60 正常 ≥ 水平 的点 [步, 触发命中]", "```", json.dumps(v["matched_descriptive"], ensure_ascii=False), "```"]
    if v["notes"]:
        L += ["", "注：" + "；".join(v["notes"])]
    L += ["", "登记预测：" + json.dumps(v["predictions"], ensure_ascii=False)]
    return "\n".join(L) + "\n"


if __name__ == "__main__":
    root = Path(sys.argv[1])
    v = judge_root(root)
    (root / "reports").mkdir(exist_ok=True)
    json.dump(v, open(root / "reports" / "VERDICT.json", "w"), indent=1, ensure_ascii=False)
    open(root / "reports" / "VERDICT.md", "w").write(markdown(v))
    print(json.dumps({k: v["verdict"][k] for k in ("localisation_clue", "pause_coarse_search")}), {a: v["verdict"][a]["system_clue"] for a in ("A", "MLP", "B", "XR")})
