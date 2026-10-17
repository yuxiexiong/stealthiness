"""Apply the frozen mid16 curve criteria (contract_mid16.json) to curve.jsonl; write VERDICT.json / VERDICT.md.
Pure CPU; also used by rehearse_verdict.py on synthetic curves."""
import argparse, json, os


def load_curve(path):
    pts = []
    for line in open(path):
        line = line.strip()
        if line:
            pts.append(json.loads(line))
    pts.sort(key=lambda p: p["step"])
    return pts


def judge(points, crit, null_rule, decay_rule, first_poison_step=23):
    """Returns the verdict dict. `points` are curve.jsonl rows (need step + summary)."""
    S = [(p["step"], p["summary"]) for p in points]
    th = [s.get("trigger_hits", 0) for _, s in S]
    ch = [s.get("clean_target_hits", 0) for _, s in S]
    steps = [st for st, _ in S]
    out = {"n_points": len(S), "steps": steps, "trigger_hits": th, "clean_target_hits": ch,
           "near_target_hits": [s.get("near_target_hits") for _, s in S]}
    # null calibration: pre-poison points
    pre = [(st, t, c) for (st, _), t, c in zip(S, th, ch) if st < first_poison_step]
    out["pre_poison_points"] = pre
    if any(t > null_rule["max_pre_poison_hits"] or c > null_rule["max_pre_poison_hits"] for _, t, c in pre):
        out["tier"] = "null_broken"
        return out
    A, D, T, TD = crit["abs_min_hits"], crit["conditional_min_diff"], crit["trace_min_hits"], crit["trace_min_diff"]
    formed_pairs = [i for i in range(len(S) - 1) if all(th[j] >= A and th[j] - ch[j] >= D for j in (i, i + 1))]
    bias_pairs = [i for i in range(len(S) - 1) if all(th[j] >= A for j in (i, i + 1))]
    trace_pairs = [i for i in range(len(S) - 1) if all(th[j] >= T and th[j] - ch[j] >= TD for j in (i, i + 1))]
    peak_i = max(range(len(S)), key=lambda i: th[i]) if S else None
    out["peak"] = None if peak_i is None else {"step": steps[peak_i], "trigger_hits": th[peak_i], "clean_target_hits": ch[peak_i],
                                                "near_target_hits": S[peak_i][1].get("near_target_hits"),
                                                "trigger_correct": S[peak_i][1].get("trigger_correct"),
                                                "clean_correct": S[peak_i][1].get("clean_correct")}
    if formed_pairs:
        first, last = steps[formed_pairs[0]], steps[formed_pairs[-1] + 1]
        out["formation_window"] = [first, last]
        G, K = decay_rule["gone_max_hits"], decay_rule["consecutive_points"]
        gone = None
        for i in range(formed_pairs[-1] + 2, len(S)):
            if th[i] <= G and i + K - 1 < len(S) and all(th[j] <= G for j in range(i, i + K)):
                gone = steps[i]
                break
        out["gone_step"] = gone
        out["tier"] = "formed_then_decayed" if gone is not None else "formed_persisting"
    elif bias_pairs:
        out["tier"] = "general_bias_only"
        out["bias_window"] = [steps[bias_pairs[0]], steps[bias_pairs[-1] + 1]]
    elif trace_pairs:
        out["tier"] = "trace_only"
        out["trace_window"] = [steps[trace_pairs[0]], steps[trace_pairs[-1] + 1]]
    else:
        out["tier"] = "not_detected"
    return out


def markdown(points, verdict, fidelity, contract):
    lines = ["# mid16 中途曲线判读", "", "保真门：**%s**" % (fidelity.get("tier") if fidelity else "缺失"),
             "", "判读档位：**%s**" % verdict["tier"], "",
             "合同解释：%s" % contract["verdict_table"].get(verdict["tier"], ""), ""]
    if verdict.get("peak"):
        lines += ["峰值：step %(step)s，trigger %(trigger_hits)s/60，clean %(clean_target_hits)s/60，near %(near_target_hits)s/60，"
                  "trigger_correct %(trigger_correct)s，clean_correct %(clean_correct)s" % verdict["peak"], ""]
    for k in ("formation_window", "gone_step", "bias_window", "trace_window"):
        if k in verdict:
            lines.append("%s: %s" % (k, verdict[k]))
    lines += ["", "| step | lr | 已消费毒行 | trig 目标 | clean 目标 | near 目标 | trig 正确 | clean 正确 | near 正确 | margin(trig−clean) |", "|---|---|---|---|---|---|---|---|---|---|"]
    for p in points:
        s = p["summary"]
        md = (s.get("trigger_margin_mean") or 0) - (s.get("clean_margin_mean") or 0)
        lines.append("| %d | %.2e | %s | %s | %s | %s | %s | %s | %s | %.2f |" % (
            p["step"], p.get("lr", 0), p.get("poison_consumed"), s.get("trigger_hits"), s.get("clean_target_hits"), s.get("near_target_hits"),
            s.get("trigger_correct"), s.get("clean_correct"), s.get("near_correct"), md))
    lines += ["", contract["sampling_caveat"]]
    return "\n".join(lines) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--contract", required=True); ap.add_argument("--run", required=True)
    ap.add_argument("--fidelity", default=None); ap.add_argument("--out", required=True)
    a = ap.parse_args()
    contract = json.load(open(a.contract))
    points = load_curve(os.path.join(a.run, "curve.jsonl"))
    fidelity = json.load(open(a.fidelity)) if a.fidelity and os.path.exists(a.fidelity) else None
    v = judge(points, contract["formation_criterion_numeric"], contract["null_calibration_numeric"], contract["decay_readout_numeric"],
              first_poison_step=contract["reference_run"]["poison_steps"][0])
    v["fidelity_tier"] = fidelity.get("tier") if fidelity else None
    os.makedirs(a.out, exist_ok=True)
    json.dump(v, open(os.path.join(a.out, "VERDICT.json"), "w"), indent=1, ensure_ascii=False)
    open(os.path.join(a.out, "VERDICT.md"), "w").write(markdown(points, v, fidelity, contract))
    print(json.dumps({k: v[k] for k in ("tier", "peak", "fidelity_tier") if k in v}, ensure_ascii=False))


if __name__ == "__main__":
    main()
