"""Real-data run for supplement/jump (CONTRACT.md). Reads only; writes results.json.

    python run.py <data_dir> <repo_attribution_microscope_dir> <out.json>

data_dir holds the copied inputs listed in CONTRACT.md §1 (arm_*.json, log_*.jsonl,
orders.npz); maps and behavioral files are read from the repo.
"""
import glob
import json
import os
import re
import sys

import numpy as np

import jump as J

DATA, AM, OUT = sys.argv[1], sys.argv[2], sys.argv[3]
BEH = os.path.join(AM, "runs", "behavioral")
MAPS = os.path.join(AM, "runs", "maps")
res = {}


def rows_of(arm, base):
    a = json.load(open(os.path.join(DATA, f"arm_{arm}.json")))
    c = json.load(open(os.path.join(DATA, f"arm_{base}.json")))
    return [i for i, (x, y) in enumerate(zip(a, c)) if x != y]


def loss_log(tag):
    L = [json.loads(l) for l in open(os.path.join(DATA, f"log_{tag}.jsonl"))]
    return [(d["current_steps"], d["loss"], d["lr"]) for d in L if "loss" in d]


def asr(tag):
    return json.load(open(os.path.join(BEH, f"{tag}.json")))["asr"]


def curve(prefixes):
    pts = {}
    for p in prefixes:
        for f in glob.glob(os.path.join(BEH, f"{p}@s*.json")):
            s = int(re.search(r"@s(\d+)\.json$", f).group(1))
            pts[s] = json.load(open(f))["asr"]     # later prefixes (fills) only add steps
    return sorted(pts.items())


orders = np.load(os.path.join(DATA, "orders.npz"))
o1, o2, nulls = orders["order_1001"], orders["order_1002"], orders["null"]

# ---- G0
logs = {t: loss_log(t) for t in ("CLEAN", "P-1.0-D", "P-1.0-D2", "RETRAIN-A-D")}
g0 = max(J.lr_check([(s, lr) for s, _, lr in logs[t]]) for t in logs)
res["G0"] = {"max_rel_err": g0, "pass": g0 <= 1e-3}

# ---- ASR curves (200-probe behavioral), incl. fills
cur = {"s1": curve(["P-1.0-D", "P-1.0-DF", "P-1.0-DG"]),
       "s2": curve(["P-1.0-D2", "P-1.0-D2F", "P-1.0-D2G"])}
every20 = {k: [(s, a) for s, a in v if s % 20 == 0] for k, v in cur.items()}

# ---- G1 per pair
pairs = {"s1": ("P-1.0-D", "CLEAN", o1, rows_of("p_1_0", "clean")),
         "s2": ("P-1.0-D2", "RETRAIN-A-D", o2, rows_of("p_1_ps2", "retrain_a"))}
res["G1"] = {}
for k, (p, c, o, rows) in pairs.items():
    lp = {s: l for s, l, _ in logs[p]}
    lc = {s: l for s, l, _ in logs[c]}
    dl = np.array([lp[20 * w] - lc[20 * w] for w in range(1, 63)])
    t5 = J.first_at(every20[k], 0.05)
    n_win = (t5 - 20) // 20                      # windows ending at or before t5 - 20
    g = J.g1(dl, o, rows, nulls, n_win)
    g.update({"t5_every20": t5, "n_windows": n_win, "n_rows": len(rows),
              "dloss": dl.tolist()})
    res["G1"][k] = g

# ---- A1
res["A1"] = {}
for k, (p, c, o, rows) in pairs.items():
    st = J.row_steps(o, rows)
    row = {"window_counts": J.window_counts(st, 62).tolist()}
    for thr, name in ((0.05, "t5"), (0.5, "t50"), (0.95, "t95")):
        s = J.first_at(cur[k], thr)
        K, D = J.exposure(st, s)
        row[name] = {"step": s, "K": K, "D": D}
    row["checkpoints"] = [{"step": s, "asr": a, "K": J.exposure(st, s)[0],
                           "D": J.exposure(st, s)[1]} for s, a in cur[k]]
    res["A1"][k] = row
if all(res["G1"][k]["pass"] for k in pairs):
    a = res["A1"]
    for ax in ("K", "D"):
        a[f"tier_{ax}"] = J.a1_tier(a["s1"]["t50"]["step"], a["s2"]["t50"]["step"],
                                    a["s1"]["t50"][ax], a["s2"]["t50"][ax])
else:
    res["A1"]["tier_K"] = res["A1"]["tier_D"] = "not judged (G1 failed)"

# ---- A2 / A3 (seed-1001 dose arms; need G1 s1)
dose = {"P-0.1": "p_0_1", "P-0.2": "p_0_2", "P-0.3": "p_0_3", "P-0.35": "p_0_35",
        "P-0.38": "p_0_38", "P-0.4": "p_0_4", "P-0.4-ps2": "p_0_4_ps2",
        "P-0.4-ps3": "p_0_4_ps3", "P-0.42": "p_0_42", "P-0.45": "p_0_45", "P-0.5": "p_0_5"}
tab = {}
for arm, ds in dose.items():
    st = J.row_steps(o1, rows_of(ds, "clean"))
    K, D = J.exposure(st, J.TOTAL)
    tab[arm] = {"K": K, "D": D, "asr": asr(arm)}
res["dose"] = tab
if res["G1"]["s1"]["pass"]:
    lo, hi = tab["P-0.38"]["D"], tab["P-0.42"]["D"]
    res["A2"] = {k: {"D_t50": res["A1"][k]["t50"]["D"], "bracket": [lo, hi],
                     "tier": J.a2_tier(res["A1"][k]["t50"]["D"], lo, hi)}
                 for k in pairs if res["G1"][k]["pass"]}
    five = ["P-0.38", "P-0.4", "P-0.4-ps2", "P-0.4-ps3", "P-0.42"]
    t = J.tau_b([tab[a]["D"] for a in five], [tab[a]["asr"] for a in five])
    tK = J.tau_b([tab[a]["K"] for a in five], [tab[a]["asr"] for a in five])
    res["A3"] = {"arms": five, "tau_b_D": t, "tier": J.a3_tier(t), "tau_b_K": tK}
else:
    res["A2"] = res["A3"] = "not judged (G1 s1 failed)"


# ---- Part B
def load_map(tag):
    z = np.load(os.path.join(MAPS, tag, "p_core_trig.npz"))
    ids = sorted({int(k.split("_")[0]) for k in z.files if k.endswith("_logits")})
    out = {}
    for i in ids:
        lg = z[f"{i}_logits"]                   # T1 (argmax), T2 (target), T3 (target - correct)
        out[i] = (float(lg[1]), float(lg[2]), bool(lg[0] == lg[1]))
    return out


paths = {"s1": ["P-1.0@s79", "P-1.0@s158", "P-1.0-D@s", "P-1.0-DF@s", "P-1.0-DG@s", "P-1.0@s632"],
         "s2": ["P-1.0-D2@s", "P-1.0-D2F@s", "P-1.0-D2G@s"]}
refs = {"s1": ["CLEAN@s79", "CLEAN@s158", "CLEAN@s316", "CLEAN@s632"],
        "s2": [f"RETRAIN-A-D@s{s}" for s in range(80, 641, 80)]}
res["B"] = {}
for k in paths:
    tags = {}
    for p in paths[k]:
        for d in sorted(glob.glob(os.path.join(MAPS, p + "*"))):
            t = os.path.basename(d)
            m = re.search(r"@s(\d+)$", t)
            if m and os.path.exists(os.path.join(d, "p_core_trig.npz")):
                tags.setdefault(int(m.group(1)), t)
    tags = {s: t for s, t in tags.items() if s <= 640}
    steps = sorted(tags)
    maps = {s: load_map(tags[s]) for s in steps}
    ids = sorted(set.intersection(*[set(m) for m in maps.values()]))
    T3 = np.array([[maps[s][i][1] for i in ids] for s in steps])
    T2 = np.array([[maps[s][i][0] for i in ids] for s in steps])
    a60 = [J.asr60([maps[s][i][2] for i in ids]) for s in steps]
    rm = [load_map(t) for t in refs[k]]
    rid = sorted(set.intersection(*[set(m) for m in rm]))
    ref_med = [float(np.median([m[i][1] for i in rid])) for m in rm]
    noise = J.noise_of(ref_med)
    w = J.window(steps, a60)
    out = {"steps": steps, "tags": [tags[s] for s in steps], "n_samples": len(ids),
           "asr60": a60, "median_T3": np.median(T3, 1).tolist(),
           "median_T2": np.median(T2, 1).tolist(), "ref_median_T3": ref_med,
           "window": w}
    out["measure"] = J.b_measure(steps, T3, *w, noise=noise) if w else {"tier": "unclear (no window)"}
    res["B"][k] = out

json.dump(res, open(OUT, "w"), indent=1, default=float)
print(json.dumps({k: {x: y for x, y in v.items() if x != "dloss"} if k == "G1" else v
                  for k, v in res.items() if k in ("G0",)}, default=float))
for k in res["G1"]:
    print("G1", k, {x: res["G1"][k][x] for x in ("r", "p", "null_q99", "pass", "t5_every20", "n_windows", "n_rows")})
for k in ("s1", "s2"):
    print("A1", k, {n: res["A1"][k][n] for n in ("t5", "t50", "t95")})
print("A1 tiers:", res["A1"]["tier_K"], res["A1"]["tier_D"])
print("dose:", {a: (v["K"], round(v["D"], 1), v["asr"]) for a, v in res["dose"].items()})
print("A2:", res["A2"])
print("A3:", res["A3"])
for k in res["B"]:
    print("B", k, res["B"][k]["window"], {x: res["B"][k]["measure"].get(x) for x in
                                         ("R", "R_ci", "C", "C_ci", "inside_change", "noise", "tier")})
