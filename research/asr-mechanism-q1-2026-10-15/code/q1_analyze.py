"""Q1 step 2 (CPU): the frozen analysis of the replayed margins (q1_common) for every mother, plus the judge.

    python q1_analyze.py --root <q1 root> --fork-root <fork root>

Writes <root>/analysis/analysis.json and <root>/analysis/summary.md. The clean-condition readouts are summarised
(mapping accuracy per update) but not analysed: they are stored for Q3.
"""
import argparse
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import q1_common as qc                                      # noqa: E402

SEEDS = (1105, 1106, 1107)


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def load(root, fork_root, seed):
    rows = read_jsonl(root / "replay" / ("s%d" % seed) / "margins.jsonl")
    steps = [r["update"] for r in rows]
    if steps != list(range(qc.HORIZON + 1)):
        raise ValueError("Replay of seed %d is incomplete" % seed)
    X = np.array([r["trigger"]["margin"] for r in rows], float)
    F = np.array([r["trigger"]["hit"] for r in rows], bool)
    C = np.array([r["clean"]["margin"] for r in rows], float)
    correct = np.array([r["clean"]["correct"] for r in rows], bool)
    oracle = {o["update"]: o["trigger_hits"] for o in read_jsonl(fork_root / "mothers" / ("s%d" % seed) / "oracle.jsonl")}
    return steps, X, F, C, correct, oracle


def fmt(v):
    return "inf" if v == float("inf") else ("%.3g" % v if isinstance(v, float) else str(v))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--fork-root", type=Path, required=True)
    args = parser.parse_args()
    out = args.root / "analysis"
    out.mkdir(exist_ok=True)
    result, classes, lines = {"mothers": {}}, {}, ["# Q1 分析摘要（机器生成）", ""]
    for seed in SEEDS:
        steps, X, F, C, correct, oracle = load(args.root, args.fork_root, seed)
        hits_argmax = F.sum(1)
        diff = [int(hits_argmax[u] - oracle[u]) for u in steps if u in oracle]
        m = {"primary": qc.analyze_mother(steps, X, F, qc.LOW, qc.HIGH),
             "r2_rule": qc.analyze_mother(steps, X, F, qc.LOW_R2, qc.HIGH_R2),
             "oracle_crosscheck": {"max_abs_diff_hits": int(max(abs(d) for d in diff)), "mean_diff_hits": float(np.mean(diff)),
                                   "updates_compared": len(diff)},
             "clean_byproduct": {"mapping_correct_per_update": correct.sum(1).tolist(),
                                 "clean_margin_median_per_update": np.median(C, 1).tolist()},
             "trigger_margin_median_per_update": np.median(X, 1).tolist()}
        classes["s%d" % seed] = qc.classify_mother(m["primary"])
        result["mothers"]["s%d" % seed] = m
        p = m["primary"]
        lines += ["## s%d · 判类 %s" % (seed, classes["s%d" % seed]), ""]
        lines.append("- argmax 命中 vs 生成命中：最大差 %d，均差 %.2f" % (m["oracle_crosscheck"]["max_abs_diff_hits"], m["oracle_crosscheck"]["mean_diff_hits"]))
        for name, a in (("主规则 0.10/0.90", p), ("r2 规则 0.05/0.95", m["r2_rule"])):
            if not a.get("measurable"):
                lines.append("- %s：测不了（%s）" % (name, a.get("reason")))
                continue
            d1, d2, d3 = a["d1"], a["d2"], a["d3"]
            lines.append("- %s：窗前 %d → 窗 [%d, %d]；D 闸门一致率 %.3f，加性 R² %.3f（D=%.3f CI [%.3f, %.3f]，%s）" % (
                name, a["span"][0], a["span"][1], a["span"][2], d1["agree"], d1["R2_additive"], d1["D"], d1["D_ci"][0], d1["D_ci"][1], d1["tier"]))
            lines.append("  - D2：W(5–95%%) = %.2f logit；v_pre = %.3f /步，v_in = %.3f /步，比值 %s；按 v_pre 要 %s 步，按 v_in 要 %s 步" % (
                d2["W_5to95"], d2["v_pre"], d2["v_in"], fmt(d2["accel_ratio"]), fmt(d2["steps_5to95_at_v_pre"]), fmt(d2["steps_5to95_at_v_in"])))
            lines.append("  - D3：自身加速的行占比 %.2f（窗前斜率 ≤ 0 的行 %.2f）" % (d3["share_accelerating"], d3["share_pre_slope_nonpos"]))
            sw = {k: v["width"] for k, v in a["sweep"].items()}
            lines.append("  - 阈值扫描宽度（步）：%s；|τ|≤4 内宽度相等（匀速签名）：%s" % (sw, a["sweep_equal_widths"]))
            bp = a["breakpoint"]
            lines.append("  - 断点：第 %d 步，斜率 %.3f → %.3f；ΔBIC(直线−断点) = %.1f，ΔBIC(二次−断点) = %.1f" % (
                bp["break_step"], bp["slope_before"], bp["slope_after"], bp["delta_bic_linear_minus_break"], bp["delta_bic_quadratic_minus_break"]))
        lines.append("")
    result["judge"] = qc.judge(classes)
    lines += ["## 判档：**%s**" % result["judge"]["tier"], "", "逐母本：%s" % classes]
    (out / "analysis.json").write_text(json.dumps(result, ensure_ascii=False, indent=1) + "\n")
    (out / "summary.md").write_text("\n".join(lines) + "\n")
    print(json.dumps(result["judge"], ensure_ascii=False))


if __name__ == "__main__":
    main()
