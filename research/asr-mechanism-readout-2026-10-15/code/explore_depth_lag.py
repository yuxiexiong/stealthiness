"""EXPLORE_DEPTH_LAG.md: (t_exit, D, L, v) per trajectory and exit definition; Spearman(D, L) and CV(v) per model."""
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parents[1]
Q1 = HERE.parent / "asr-mechanism-q1-2026-10-15" / "results" / "replay"
B1 = HERE.parent / "asr-mechanism-exit-2026-10-15" / "results" / "b1"
LLAVA = {"O1001 (PS-P0)": "PS-P0", "O1007": "XO-I1001-O1007", "O1008": "XO-I1001-O1008", "O1009": "XO-I1001-O1009", "O1011": "XO-I1001-O1011"}
SAME_ORDER = ("XO-I1007-O1001", "XO-I1008-O1001", "XO-I1009-O1001", "XO-I1011-O1001")


def spearman(x, y):
    rx, ry = np.argsort(np.argsort(x)), np.argsort(np.argsort(y))
    return float(np.corrcoef(rx, ry)[0, 1]) if np.std(rx) and np.std(ry) else float("nan")


def onset(steps, asr):
    return next(s for s, v in zip(steps, asr) if v >= 0.10)


def exits(steps, c, t_a, start, smooth, win):
    idx = [i for i, s in enumerate(steps) if start <= s <= t_a]
    e1 = steps[min(idx, key=lambda i: c[i])]
    cs = np.convolve(c, np.ones(5) / 5, mode="same") if smooth else np.asarray(c)
    if smooth:
        cs[:2], cs[-2:] = c[:2], c[-2:]
    ia = steps.index(t_a)
    loc = [i for i in range(1, ia) if steps[i] >= start and cs[i] <= cs[i - 1] and cs[i] <= cs[i + 1]]
    e2 = steps[loc[-1]] if loc else e1
    idx3 = [i for i, s in enumerate(steps) if t_a - win <= s < t_a]
    e3 = steps[min(idx3, key=lambda i: c[i])]
    return {"E1": e1, "E2": e2, "E3": e3}


def record(steps, c, asr, start, smooth, win):
    t_a = onset(steps, asr)
    out = {"t_a": t_a, "c_at_onset": float(c[steps.index(t_a)])}
    for k, t in exits(steps, c, t_a, start, smooth, win).items():
        D, L = -float(c[steps.index(t)]), t_a - t
        out[k] = {"t_exit": t, "D": D, "L": L, "v": (float(c[steps.index(t_a)]) - float(c[steps.index(t)])) / L if L > 0 else None}
    return out


def main():
    res = {"toy": {}, "llava": {}, "llava_same_order_reference": {}}
    for s in (1105, 1106, 1107):
        rows = [json.loads(x) for x in (Q1 / ("s%d" % s) / "margins.jsonl").read_text().splitlines()]
        G = np.array([r["trigger"]["gap"] for r in rows])
        res["toy"]["s%d" % s] = record(list(range(len(rows))), G.mean(1), (G > 0).mean(1), 15, True, 30)
    for name, arm in list(LLAVA.items()) + [(a, a) for a in SAME_ORDER]:
        z = np.load(B1 / ("%s.npz" % arm))
        steps = list(map(int, z["steps"]))
        rec = record(steps, z["gap"].mean(1), z["hit"].mean(1), steps[0], False, 60)
        (res["llava"] if name in LLAVA else res["llava_same_order_reference"])[name] = rec
    summary = {}
    for model in ("toy", "llava"):
        for k in ("E1", "E2", "E3"):
            D = [v[k]["D"] for v in res[model].values()]
            L = [v[k]["L"] for v in res[model].values()]
            vs = [v[k]["v"] for v in res[model].values() if v[k]["v"]]
            summary["%s/%s" % (model, k)] = {"spearman_D_L": spearman(D, L), "cv_v": float(np.std(vs) / np.mean(vs)) if vs else None}
    res["summary"] = summary
    ok = [k for k in ("E1", "E2", "E3") if all(summary["%s/%s" % (m, k)]["spearman_D_L"] > 0 and summary["%s/%s" % (m, k)]["cv_v"] < 0.5 for m in ("toy", "llava"))]
    res["worth_confirmatory"] = {"definitions_passing": ok, "decision": len(ok) >= 2}
    (HERE / "results" / "explore.json").write_text(json.dumps(res, indent=1) + "\n")
    for model in ("toy", "llava", "llava_same_order_reference"):
        for name, v in res[model].items():
            print("%-6s %-16s t_a %4d  " % (model[:6], name, v["t_a"]) + "  ".join(
                "%s: exit %d D %.1f L %d v %s" % (k, v[k]["t_exit"], v[k]["D"], v[k]["L"], "%.2f" % v[k]["v"] if v[k]["v"] else "-") for k in ("E1", "E2", "E3")))
    print(json.dumps(summary, indent=0))
    print(res["worth_confirmatory"])


if __name__ == "__main__":
    main()
