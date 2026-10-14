"""Step 2b of CONTRACT_GRID (CPU, standard library): collect the three grids, judge, write RESULTS.md."""
import argparse
import json
from pathlib import Path

import grid_common as gc_


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--fork-root", type=Path, required=True)
    args = parser.parse_args()
    seeds = json.loads((args.fork_root / "contract.json").read_text())["mothers"]["seeds"]
    grids = {"s%d" % s: json.loads((args.root / "grid" / ("s%d" % s) / "grid.json").read_text()) for s in seeds}
    per = {k: gc_.classify_mother({c: v["trigger_hits"] for c, v in g["cells"].items()}, {c: v["trigger_hits"] for c, v in g["controls"].items()}) for k, g in grids.items()}
    for k, g in grids.items():
        if per[k] != g["class"]:
            raise ValueError("Recomputed class differs from the GPU job for " + k)
    verdict = gc_.judge(per)
    result = {"verdict": verdict, "grids": grids, "preregistered_prediction": "multiplicative"}
    (args.root / "result.json").write_text(json.dumps(result, indent=1) + "\n")
    lines = ["# 两部件格子实验：实际结果", "",
             "判档：**%s**（预注册预测：multiplicative）。各母本：%s。" % (verdict["tier"], ", ".join("%s=%s" % kv for kv in per.items())), ""]
    for k, g in grids.items():
        lines += ["## %s（t_fork %d, t_jump %d）" % (k, g["t_fork"], g["t_jump"]), "",
                  "|机器成熟度 \\ 偏置 α|" + "|".join("%g" % a for a in gc_.ALPHAS) + "|mapping_correct(α=0)|", "|---|" + "---:|" * (len(gc_.ALPHAS) + 1)]
        for m in gc_.MATURITY:
            cells = [g["cells"][gc_.cell(m, a)] for a in gc_.ALPHAS]
            lines.append("|%s|%s|%d|" % (m, "|".join("%d" % c["trigger_hits"] for c in cells), cells[0]["mapping_correct"]))
        lines.append("")
        lines.append("随机方向对照（%s）：" % gc_.PRIMARY_MATURE + "，".join("α=%g → %d" % (a, g["controls"][gc_.cell(gc_.PRIMARY_MATURE, a)]["trigger_hits"]) for a in gc_.CONTROL_ALPHAS)
                     + "；母本 t_jump 自身：%d。偏置增量范数 %.3f，机器增量范数 tj/tj40/end %s。" % (
                         g["reference_mother_tj"]["trigger_hits"], g["bias_increment_norm"], {m: round(v, 3) for m, v in g["machine_increment_norm"].items()}))
        lines.append("")
    lines += ["单模型、单剂量、三条母本；格子里的每个状态都是合成权重，不是训练到达的状态。数值：result.json、grid/*/grid.json、grid/*/cells.jsonl、states/*/states.json。"]
    (args.root / "RESULTS.md").write_text("\n".join(lines) + "\n")
    print(json.dumps(verdict))


if __name__ == "__main__":
    main()
