"""A1 (CPU): poison vs clean updates in the fork experiment's 4-row sensor, every mother and condition.

    python exit_a1.py <fork results dir> <out.json>
"""
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import exit_common as ec                                    # noqa: E402

CONDITIONS = ("B0", "B1", "B2", "B3", "B4", "B5a", "B5b", "B6", "B7", "B8")
SEEDS = (1105, 1106, 1107)


def jl(p):
    return [json.loads(x) for x in Path(p).read_text().splitlines() if x.strip()]


def main():
    root, out = Path(sys.argv[1]), Path(sys.argv[2])
    res = {"per": {}, "judge": {}}
    for s in SEEDS:
        t_fork = json.loads((root / "mothers" / ("s%d" % s) / "mother.json").read_text())["t_fork"]
        for cond in CONDITIONS:
            d = root / "forks" / ("s%d" % s) / cond
            if not (d / "sensors.jsonl").exists():
                continue
            g = {r["update"]: float(np.mean(r["gap"])) for r in jl(d / "sensors.jsonl")}
            inc = {u: g[u] - g[u - 1] for u in g if u - 1 in g}
            pu = {r["update"] for r in jl(d / "row_losses.jsonl") if r["poison"]}
            hi = min(max(g), t_fork + ec.A1_SPAN)
            st = ec.a1_stats(inc, pu, t_fork + 1, hi)
            st.update(window=[t_fork + 1, hi], poison_driven=ec.a1_poison_driven(st))
            res["per"]["s%d/%s" % (s, cond)] = st
    b0 = [res["per"]["s%d/B0" % s]["poison_driven"] for s in SEEDS]
    b2 = [res["per"]["s%d/B2" % s]["m_c"] for s in SEEDS]
    res["judge"] = {"B0_poison_driven_mothers": int(sum(b0)), "B0_tier": "poison_driven" if sum(b0) >= 2 else "not_poison_driven",
                    "B2_m_c": b2, "B2_clean_washout_all_three": all(v < 0 for v in b2)}
    out.write_text(json.dumps(res, indent=1) + "\n")
    print(json.dumps(res["judge"]))
    for k, v in res["per"].items():
        print(k, "m_p", None if v["m_p"] is None else round(v["m_p"], 2), "(%d)" % v["n_p"], "m_c", round(v["m_c"], 3), "(%d)" % v["n_c"], v["poison_driven"])


if __name__ == "__main__":
    main()
