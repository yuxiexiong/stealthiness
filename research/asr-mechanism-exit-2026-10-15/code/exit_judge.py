"""CPU judge for A2, A3 and B1 (CONTRACT_EXIT.md). Works on whatever parts are present.

    python exit_judge.py --root <exit root>                -> <root>/judge/judge.json, judge.md
"""
import argparse
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import exit_common as ec                                    # noqa: E402

SEEDS = (1105, 1106, 1107)
ARMS = ("XO-I1007-O1001", "XO-I1001-O1007", "XO-I1008-O1001", "XO-I1001-O1008", "XO-I1011-O1001", "XO-I1001-O1011",
        "XO-I1009-O1001", "XO-I1001-O1009", "PS-P0")


def a3_mother(d):
    """Per-mother A3 class from the saved states."""
    def series(arm, kind="decisive", key="push_pre"):
        steps = sorted(d[arm], key=int)
        return steps, [float(np.median([r[key] for r in d[arm][t]["rows"] if r["kind"] == kind])) for t in steps]
    sb, pb = series("b2")
    sm, pm = series("mother")
    rb, rm = ec.edge_ratio(pb), ec.edge_ratio(pm)
    retreats = d["b2"][sb[-1]]["c"] < d["b2"][sb[0]]["c"]
    allstates = [(int(t), v) for t, v in d["phase1"].items()] + [(int(t), v) for t, v in d["b2"].items()]
    pushes = [float(np.median([r["push_pre"] for r in v["rows"] if r["kind"] == "decisive"])) for _, v in allstates]
    clean = [v["clean_correct"] for _, v in allstates]
    _, p1 = series("phase1")
    return {"class": ec.a3_class(rb, rm, retreats), "ratio_b2": rb, "ratio_mother": rm, "b2_retreats": retreats,
            "push_b2": pb, "push_mother": pm, "push_phase1": p1, "phase1_steps": sorted(map(int, d["phase1"])),
            "arm_steps": [int(t) for t in sb], "spearman_push_vs_clean_phase1_b2": ec.spearman(pushes, clean),
            "clean_push_b2": series("b2", "clean")[1], "clean_push_mother": series("mother", "clean")[1],
            "early_push_b2": series("b2", "early_poison")[1], "early_push_mother": series("mother", "early_poison")[1]}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", type=Path, required=True)
    root = ap.parse_args().root
    res, md = {}, ["# 平台退出实验 判档（机器生成）", ""]
    # A2
    a2 = {s: json.loads((root / "a2" / ("s%d" % s) / "a2.json").read_text()) for s in SEEDS if (root / "a2" / ("s%d" % s) / "a2.json").exists()}
    if a2:
        cls = {"s%d" % s: v["mother"]["class"] for s, v in a2.items()}
        res["A2"] = {"per_mother": {"s%d" % s: v["mother"] for s, v in a2.items()}, "tier": ec.a2_tier(cls) if len(a2) == 3 else "incomplete"}
        md += ["## A2 · %s" % res["A2"]["tier"], ""]
        for s, v in a2.items():
            m = v["mother"]
            md.append("- s%d：%s；爆发更新 %s；share_P 中位数 %s，ctrl 中位数 %s，share_P_drop %s" % (
                s, m["class"], m.get("burst_updates"), round(m["median_share_P"], 3) if "median_share_P" in m else None,
                round(m["median_ctrl"], 3) if "median_ctrl" in m else None, round(m["median_share_P_drop"], 3) if "median_share_P_drop" in m else None))
            for r in v["rows"]:
                md.append("  - u=%d 毒行 %s：inc %s；clean %s→%s" % (r["u"], r["poison_rows"], {k: round(x, 2) for k, x in r["inc"].items()}, r["clean_pre"], r["clean"]["base"]))
        md.append("")
    # A3
    a3 = {s: json.loads((root / "a3" / ("s%d" % s) / "a3.json").read_text()) for s in SEEDS if (root / "a3" / ("s%d" % s) / "a3.json").exists()}
    if a3:
        per = {"s%d" % s: a3_mother(v) for s, v in a3.items()}
        res["A3"] = {"per_mother": per, "tier": ec.same_or_inconsistent({k: v["class"] for k, v in per.items()}) if len(a3) == 3 else "incomplete"}
        md += ["## A3 · %s" % res["A3"]["tier"], ""]
        for k, v in per.items():
            md.append("- %s：%s；ratio_b2 %.2f，ratio_mother %.2f，B2 退回 %s；push 对 clean 正确数 ρ = %.2f" % (
                k, v["class"], v["ratio_b2"], v["ratio_mother"], v["b2_retreats"], v["spearman_push_vs_clean_phase1_b2"]))
            md.append("  - 决定性行 push_pre：phase1 %s" % [round(x, 3) for x in v["push_phase1"]])
            md.append("  - 母本臂 %s" % [round(x, 3) for x in v["push_mother"]])
            md.append("  - B2 臂 %s" % [round(x, 3) for x in v["push_b2"]])
            md.append("  - 干净探针行 母本 %s / B2 %s" % ([round(x, 3) for x in v["clean_push_mother"]], [round(x, 3) for x in v["clean_push_b2"]]))
        md.append("")
    # B1
    arms = [a for a in ARMS if (root / "b1" / f"{a}.npz").exists()]
    if arms:
        per = {}
        for a in arms:
            z = np.load(root / "b1" / f"{a}.npz")
            per[a] = ec.b1_arm(list(map(int, z["steps"])), z["gap"].astype(float), z["hit"].astype(bool))
        res["B1"] = {"per_arm": per, "tier": ec.b1_tier({a: v["class"] for a, v in per.items()}) if len(arms) == len(ARMS) else "incomplete"}
        md += ["## B1 · %s" % res["B1"]["tier"], ""]
        for a, v in per.items():
            if v.get("class") == "no_window":
                md.append("- %s：无窗" % a)
                continue
            bp = v["breakpoint"]
            md.append("- %s：%s；窗 %s，断点 %d（ΔBIC %.1f，斜率 %.3f→%.3f）；v_pre %.3f，v_in %.3f，R %.2f，slope[t*, a′] %.3f" % (
                a, v["class"], v["window"], bp["break"], bp["dbic"], bp["slope_before"], bp["slope_after"], v["v_pre"], v["v_in"], v["R"], v["slope_exit_to_a"]))
        md.append("")
    (root / "judge").mkdir(exist_ok=True)
    (root / "judge" / "judge.json").write_text(json.dumps(res, ensure_ascii=False, indent=1, default=float) + "\n")
    (root / "judge" / "judge.md").write_text("\n".join(md) + "\n")
    print(json.dumps({k: v["tier"] for k, v in res.items()}))


if __name__ == "__main__":
    main()
