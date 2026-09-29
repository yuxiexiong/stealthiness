"""Round 2 real-data run for parts D (per-sample decomposition) and Q (Qwen). CONTRACT2.md.

    python run_decomp.py <data_dir> <repo_attribution_microscope_dir> <out.json>

data_dir: round-1 inputs (arm_*.json, orders.npz) plus q3/ (Qwen maps, behavioral, trainer logs).
"""
import glob
import json
import os
import re
import sys

import numpy as np

import decomp as Dc
import grok as G  # noqa: F401  (J via sys.path)
import jump as J

DATA, AM, OUT = sys.argv[1], sys.argv[2], sys.argv[3]
Q3 = os.path.join(DATA, "q3")
res = {}


def load_map(path):
    z = np.load(path)
    ids = sorted({int(k.split("_")[0]) for k in z.files if k.endswith("_logits")})
    return {i: (float(z[f"{i}_logits"][1]), float(z[f"{i}_logits"][2]),
                bool(z[f"{i}_logits"][0] == z[f"{i}_logits"][1])) for i in ids}


def traj(maps_dir, prefixes, max_step):
    tags = {}
    for p in prefixes:
        for d in sorted(glob.glob(os.path.join(maps_dir, p + "*"))):
            t = os.path.basename(d)
            m = re.search(r"@s(\d+)$", t)
            if m and os.path.exists(os.path.join(d, "p_core_trig.npz")):
                tags.setdefault(int(m.group(1)), t)
    steps = sorted(s for s in tags if s <= max_step)
    maps = {s: load_map(os.path.join(maps_dir, tags[s], "p_core_trig.npz")) for s in steps}
    ids = sorted(set.intersection(*[set(m) for m in maps.values()]))
    T3 = np.array([[maps[s][i][1] for i in ids] for s in steps])
    F = np.array([[maps[s][i][2] for i in ids] for s in steps])
    return steps, T3, F, len(ids)


def ref_medians(maps_dir, tags):
    ms = [load_map(os.path.join(maps_dir, t, "p_core_trig.npz")) for t in tags]
    ids = sorted(set.intersection(*[set(m) for m in ms]))
    return [float(np.median([m[i][1] for i in ids])) for m in ms]


LM = os.path.join(AM, "runs", "maps")
QM = os.path.join(Q3, "maps")
T = {"L-s1": (LM, ["P-1.0@s79", "P-1.0@s158", "P-1.0-D@s", "P-1.0-DF@s", "P-1.0-DG@s", "P-1.0@s632"], 640),
     "L-s2": (LM, ["P-1.0-D2@s", "P-1.0-D2F@s", "P-1.0-D2G@s"], 640),
     "Q-s1": (QM, ["P-1.0@s", "P-1.0-DF@s", "P-1.0-DG@s"], 640)}

# ---- part D
res["D"] = {}
for k, (md, pre, mx) in T.items():
    steps, T3, F, n = traj(md, pre, mx)
    asr = F.mean(1).tolist()
    sp = Dc.span(steps, asr)
    out = {"steps": steps, "n_samples": n, "asr60": asr}
    if sp is None:
        out["tier"] = "cannot measure (no window)"
    else:
        p, a, b = sp
        i0, i1 = steps.index(p), steps.index(b) + 1
        X, Fs, ss = T3[i0:i1], F[i0:i1], steps[i0:i1]
        out.update({"span": [p, a, b], "span_steps": ss,
                    "D1": Dc.d1(X, Fs), "D2": Dc.d2(ss, X, p, a, b), "D3": Dc.d3(ss, X, p, a, b)})
    res["D"][k] = out

# ---- part Q: Qwen B, order gate, exposure description
steps, T3, F, n = traj(QM, T["Q-s1"][1], 640)
a60 = F.mean(1).tolist()
w = J.window(steps, a60)
noise = J.noise_of(ref_medians(QM, ["CLEAN@s80", "CLEAN@s160", "CLEAN@s320", "CLEAN@s640"]))
res["QB"] = J.b_measure(steps, T3, *w, noise=noise) if w else {"tier": "unclear (no window)"}
res["QB"]["median_T3"] = np.median(T3, 1).tolist()
res["QB"]["steps"] = steps


def curve(prefixes):
    pts = {}
    for p in prefixes:
        for f in glob.glob(os.path.join(Q3, "behavioral", f"{p}@s*.json")):
            pts[int(re.search(r"@s(\d+)\.json$", f).group(1))] = json.load(open(f))["asr"]
    return sorted(pts.items())


cq = curve(["P-1.0", "P-1.0-DF", "P-1.0-DG"])
c20 = [(s, x) for s, x in cq if s % 20 == 0]
a = json.load(open(os.path.join(DATA, "arm_p_1_0.json")))
c = json.load(open(os.path.join(DATA, "arm_clean.json")))
rows = [i for i, (x, y) in enumerate(zip(a, c)) if x != y]
orders = np.load(os.path.join(DATA, "orders.npz"))
o1 = orders["order_1001"]


def loss_log(p):
    return {d["current_steps"]: d["loss"] for d in map(json.loads, open(p)) if "loss" in d}


lp = loss_log(os.path.join(Q3, "arms", "P-1.0", "trainer_log.jsonl"))
lc = loss_log(os.path.join(Q3, "arms", "CLEAN", "trainer_log.jsonl"))
dl = np.array([lp[20 * w] - lc[20 * w] for w in range(1, 63)])
t5 = J.first_at(c20, 0.05)
n_win = (t5 - 20) // 20
g = J.g1(dl, o1, rows, orders["null"], n_win)
g.update({"t5_every20": t5, "n_windows": n_win})
res["QG1"] = g
st = J.row_steps(o1, rows)
res["QK"] = {nm: {"step": J.first_at(cq, thr), "K": J.exposure(st, J.first_at(cq, thr))[0]}
             for thr, nm in ((0.05, "t5"), (0.5, "t50"), (0.95, "t95"))}
res["QK"]["checkpoints"] = [{"step": s, "asr": x, "K": J.exposure(st, s)[0]} for s, x in cq]

json.dump(res, open(OUT, "w"), indent=1, default=float)
for k, v in res["D"].items():
    if "D1" in v:
        print("D", k, v["span"], {x: v["D1"].get(x) for x in ("agree", "D", "D_ci", "R2_additive", "tier")})
        print("   D2", {x: v["D2"][x] for x in ("W_5to95", "v_in", "v_pre", "steps_5to95_at_v_in", "steps_5to95_at_v_pre")})
        print("   D3", v["D3"])
    else:
        print("D", k, v["tier"])
print("QB", {x: res["QB"].get(x) for x in ("a", "b", "pre_start", "R", "R_ci", "C", "C_ci", "inside_change", "noise", "tier")})
print("QG1", {x: g[x] for x in ("r", "p", "null_q99", "pass", "t5_every20", "n_windows")})
print("QK", {x: res["QK"][x] for x in ("t5", "t50", "t95")})
