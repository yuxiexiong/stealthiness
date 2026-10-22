"""Zero-GPU extract for the jump line (CPU, read-only on all inputs).
Writes one JSON with: poison rows, open rows, per-order poison arrival steps
(9 original orders + 6 SP splices + PS P4), LP read-outs (median/mean core margin,
per-probe M for 200 core probes) for 19 arms, behavioral ASR per checkpoint."""
import glob, json, os, re, sys
import numpy as np
ORIG = "/root/attribution-microscope/data/lf"
ROOT = "/workspace/claude-jump"
N, BATCH = 20000, 16
rows = json.load(open(os.path.join(ORIG, "arm_p_1_0.json")))
clean = json.load(open(os.path.join(ORIG, "arm_clean.json")))
poison = sorted(i for i, (x, y) in enumerate(zip(rows, clean)) if x != y)
assert len(poison) == 200
ans = lambda r: r["messages"][1]["content"].strip().lower()
open_rows = [j for j in poison if ans(clean[j]) not in ("yes", "no") and not re.fullmatch(r"\d+", ans(clean[j]))]
o1 = np.load(os.path.join(ROOT, "orders.npz")); og = np.load(os.path.join(ROOT, "orders_grok.npz"))
seqs = {1001: o1["order_1001"], 1002: o1["order_1002"], **{s: og[f"order_{s}"] for s in range(1004, 1012)}}
def row_steps(order, rs):
    pos = np.empty(N, dtype=np.int64); pos[order] = np.arange(N)
    return (pos[np.asarray(rs)] // BATCH + 1).tolist()
def splice(a, b, c):
    head = np.asarray(a)[:BATCH * c]; used = np.zeros(N, bool); used[head] = True
    tail = np.asarray([r for r in b if not used[r]]); out = np.concatenate([head, tail])
    assert len(out) == N and len(np.unique(out)) == N; return out
arr = {f"O{s}": row_steps(seqs[s], poison) for s in seqs}
for c in (100, 200, 300):
    arr[f"SP-A1001-B1007-C{c}"] = row_steps(splice(seqs[1001], seqs[1007], c), poison)
    arr[f"SP-A1007-B1001-C{c}"] = row_steps(splice(seqs[1007], seqs[1001], c), poison)
# PS: P0-P3 keep 1001 arrivals; P4 moves the poison row at step 55 to step 56
p4 = sorted(arr["O1001"]); i = p4.index(55); p4[i] = 56; arr["PS-P4"] = p4
arms = sorted({re.sub(r"@s\d+\.json$", "", os.path.basename(f)) for f in glob.glob(f"{ROOT}/lp_out/*@s*.json")})
arms = [a for a in arms if not a.startswith("SF-")]
lp = {}
for a in arms:
    pts = []
    for f in glob.glob(f"{ROOT}/lp_out/{a}@s*.json"):
        s = int(re.search(r"@s(\d+)\.json$", f).group(1)); d = json.load(open(f))
        M = [r["M"] for r in d["core"]]
        pts.append({"step": s, "median": float(np.median(M)), "mean": float(np.mean(M)),
                    "frac_pos": float(np.mean([r["argmax_is_target"] for r in d["core"]])), "M": M,
                    "M_seen": [r["M"] for r in d["seen"]]})
    lp[a] = sorted(pts, key=lambda p: p["step"])
beh = {}
for a in arms:
    pts = []
    for f in glob.glob(f"{ROOT}/xo/runs/behavioral/{a}@s*.json"):
        s = int(re.search(r"@s(\d+)\.json$", f).group(1)); d = json.load(open(f))
        pts.append({"step": s, "asr": d["asr"], "clean_acc": d.get("clean_acc")})
    beh[a] = sorted(pts, key=lambda p: p["step"])
out = {"poison_rows": poison, "open_rows": open_rows, "arrivals": arr, "lp": lp, "behavioral": beh,
       "lp_arms": arms, "note": "BATCH=16 rows/step, step is 1-based; arrivals are steps at which each poison row is consumed (unsorted = poison row order)"}
json.dump(out, open(sys.argv[1], "w"))
print("arms", len(arms), "orders", sorted(arr), "lp pts", {a: len(lp[a]) for a in arms[:3]}, "open", len(open_rows))
