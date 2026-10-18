"""CPU: apply the frozen verdict rules of contract.json to every run; write VERDICT.json and VERDICT.md."""
import json, sys
from pathlib import Path
HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE))
import freeze_lib as fl

ARMS = ("R", "M", "P", "X1", "X2", "X3", "D")


def judge_root(root):
    C = json.load(open(root / "contract.json"))
    table, notes = {}, []
    for seed in C["source"]["mothers"]:
        res = {}
        for arm in ARMS:
            d = root / "runs" / ("s" + seed) / arm
            res[arm] = json.load(open(d / "result.json")) if (d / "result.json").exists() else None
        row = {"valid": True}
        R = res["R"]
        if R is None or not R.get("fidelity", {}).get("exact") or R.get("t_form") is None:
            row["valid"] = False
            notes.append("mother %s invalid: R missing, not exact, or R never formed" % seed)
        tR = R.get("t_form") if R else None
        for arm in ARMS:
            r = res[arm]
            if r is None or not r.get("frozen_untouched", False) or (arm == "D" and not r.get("fidelity", {}).get("exact")):
                row[arm] = {"cls": "missing_or_failed", "learn": None, "delay": None}
                if arm in ("M", "D") and r is None:
                    notes.append("mother %s arm %s missing" % (seed, arm))
                continue
            ep = r.get("endpoint_400")
            cls = fl.safety_class(r.get("t_form"), tR, ep["trigger_hits_124"] if ep else None, ep is not None)
            delay = None if (r.get("t_form") is None or tR is None) else r["t_form"] - tR
            learn, info = None, None
            if cls == "toy_blocked" and R and R.get("endpoint_400"):
                start_arm = r["points"][0]["mapping_correct"]
                start_ref = R["points"][0]["mapping_correct"]
                learn, info = fl.learning_ok(ep, R["endpoint_400"], start_arm, start_ref)
            row[arm] = {"cls": cls, "learn": learn, "learn_info": info, "delay": delay, "t_form": r.get("t_form"),
                        "stop": r.get("stop_reason"), "endpoint_hits_124": ep["trigger_hits_124"] if ep else None,
                        "peak_trigger": max(p["trigger_hits"] for p in r["points"]),
                        "last_mapping_correct": r["points"][-1]["mapping_correct"]}
        table[seed] = row
    verdict = {"M": fl.candidate_verdict(table, "M"), "D": fl.candidate_verdict(table, "D")}
    return {"table": table, "verdict": verdict, "notes": notes, "predictions": C["predictions_registered_before_data"]}


def markdown(v):
    L = ["# 冻结区域 toy：判读（冻结规则自动生成）", "",
         "| 候选 | 档位 | 有效母本 | 挡住且保住学习 | 挡住 | 形成 |", "|---|---|---|---|---|---|"]
    for c in ("M", "D"):
        x = v["verdict"][c]
        L.append("| %s | **%s** | %s | %s | %s | %s |" % (c, x["tier"], x["valid_mothers"], x["blocked_with_learning"], x["toy_blocked"], x["formed"]))
    L += ["", "| 母本 | 臂 | 安全档 | 起跳步 t_form | 相对 R 推迟 | 峰值触发/60 | 400 步触发/124 | 学习保留 | 末点正常/60 | 停止原因 |", "|---|---|---|---|---|---|---|---|---|---|"]
    for seed, row in v["table"].items():
        for arm in ARMS:
            a = row[arm]
            L.append("| %s | %s | %s | %s | %s | %s | %s | %s | %s | %s |" % (seed, arm, a["cls"], a.get("t_form"), a.get("delay"), a.get("peak_trigger"),
                     a.get("endpoint_hits_124"), a.get("learn"), a.get("last_mapping_correct"), a.get("stop")))
    L += ["", "注：%s" % "；".join(v["notes"]) if v["notes"] else "", "", "登记预测：" + json.dumps(v["predictions"], ensure_ascii=False)]
    return "\n".join(L) + "\n"


if __name__ == "__main__":
    root = Path(sys.argv[1])
    v = judge_root(root)
    (root / "reports").mkdir(exist_ok=True)
    json.dump(v, open(root / "reports" / "VERDICT.json", "w"), indent=1, ensure_ascii=False)
    open(root / "reports" / "VERDICT.md", "w").write(markdown(v))
    print(json.dumps({c: v["verdict"][c]["tier"] for c in ("M", "D")}), v["notes"])
