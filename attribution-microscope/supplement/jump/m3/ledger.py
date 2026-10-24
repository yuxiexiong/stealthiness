"""M3 ledger (CONTRACT_M3 v4 §2-3). Server, amic env, one GPU.
    python ledger.py
Requires m_out/m3/gates.json pass. Phase 0: GM3-1(b) micro-batch hashes; A: grad X and X at checkpoints 10..430;
B: per-row ledger 11..430 with segment vectors (GM3-4 identity, fidelity, momentum diagnostic);
C: position matrix for the 54 pre-take-off poison rows. Outputs in m_out/m3/."""
import json
import math
import os
import shutil
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "m_common")); sys.path.insert(0, HERE)
import mcommon as mc  # noqa: E402
import mlib  # noqa: E402
import m3  # noqa: E402
OUT = os.path.join(mc.OUT, "m3"); GD = os.path.join(OUT, "grads"); os.makedirs(GD, exist_ok=True)
DEV = mlib.DEV
gates = json.load(open(os.path.join(OUT, "gates.json")))
assert gates["pass"], "gates did not pass"
SEL, EV = mc.probe_split()
rep = lambda s: os.path.join(mc.X, "runs", "arms", mc.REPLAY_ARM, f"checkpoint-{s}")
status = {"t0": time.strftime("%F %T")}


def save_status():
    mc.wjson(os.path.join(OUT, "ledger_status.json"), status)


def halt(why):
    status["HALT"] = why; save_status()
    open(os.path.join(OUT, "HALT"), "w").write(json.dumps(why, default=str)); sys.exit(1)


hand_p = os.path.join(mc.OUT, "m1", "handoff.json")
hand = json.load(open(hand_p)) if os.path.exists(hand_p) else {"S_star": None}
S, ctype = hand.get("S_star"), hand.get("type")
XS = ["full"] + (["Y"] if S is not None else [])
status["X"] = XS; status["handoff"] = hand

# ---------------------------------------------------------------- LLaMA-Factory data pipeline
# J92 (user-authorised fix; no contract change): `import lfdata` pulls in the LLaMA-Factory parser,
# which EMPTIES CUDA_VISIBLE_DEVICES in this process (measured: '0' -> ''), so mlib.Lab below died with
# "No CUDA GPUs are available" on every attempt. Restoring the variable afterwards is NOT enough:
# torch's lazy CUDA init is already poisoned by then. Pinning the context here, while CVD is still
# correct, keeps cuda:0 usable for the rest of the run. Verified on the real path before this edit.
_cuda_pin = torch.zeros(1, device=DEV) if torch.cuda.is_available() else None
import yaml  # noqa: E402
import lfdata  # noqa: E402

ycfg = yaml.safe_load(open(os.path.join(mc.X, "runs", "configs", f"{mc.REPLAY_ARM}.yaml")))
ds, coll, IGNORE_INDEX = lfdata.build(ycfg, os.path.join(OUT, "lf_tmp"))
assert len(ds) == 20000, len(ds)
poison = set(json.load(open(os.path.join(mc.X, "data", "manifests", "poison_sets.json")))["0.01"])
rows_of_step = lambda s: lfdata.rows_of_step(s, poison)
collate = lambda positions: lfdata.collate(ds, coll, positions)


# ---------------------------------------------------------------- Phase 0: GM3-1(b)
hashes = [json.loads(l) for l in open(os.path.join(OUT, "replay_hashes.jsonl"))]
bad = []
for s in range(11, 431):
    rs = rows_of_step(s)
    for mb in (0, 1):
        b = collate([r["pos"] for r in rs if r["mb"] == mb])
        h = hashes[2 * (s - 1) + mb]
        for key in ("input_ids", "labels", "pixel_values"):
            if h.get(key) != mlib.sha(b[key]):
                bad.append((s, mb, key))
status["GM3_1b"] = {"n_bad": len(bad), "first_bad": bad[:5], "pass": not bad}
save_status()
if bad:
    halt({"GM3_1b": status["GM3_1b"]})

# ---------------------------------------------------------------- model
lab = mlib.Lab(rep(430))
model = lab.model


def lora_grad_of_row(pos):
    return lfdata.row_grad(lab, ds, coll, IGNORE_INDEX, pos, DEV)


def gradX(c):
    # bs 2, not 8: the bs-8 backward OOMs in M1 (deviation J77); sums over batches, only float error changes
    lab.load(rep(c))
    gf, vf = lab.param_grad(EV, bs=2)
    out = {"full": (gf, vf)}
    if S is not None:
        if ctype == "sufficient":
            out["Y"] = lab.param_grad(EV, bs=2, keep=set(S))
        else:
            gr, vr = lab.param_grad(EV, bs=2, remove=set(S))
            out["Y"] = (gf - gr, vf - vr)
    return out


# ---------------------------------------------------------------- Phase A
xval = {}
for c in mc.M3_CKPTS:
    paths = {X: os.path.join(GD, f"g_{X}_{c}.pt") for X in XS}
    if all(os.path.exists(p) for p in paths.values()) and str(c) in status.get("xval", {}):
        continue
    r = gradX(c)
    for X in XS:
        torch.save(r[X][0].cpu(), paths[X]); xval.setdefault(str(c), {})[X] = r[X][1]
    status["xval"] = {**status.get("xval", {}), **xval}; save_status()
    print("A", c, {X: round(r[X][1], 3) for X in XS}, flush=True)
xval = status["xval"]

# optimizer states and parameters per checkpoint (cached lazily; never touches the model weights)
_cache = {}


def ck_state(c):
    if c not in _cache:
        v, step = mlib.optimizer_flat(rep(c), "exp_avg_sq")
        m, _ = mlib.optimizer_flat(rep(c), "exp_avg")
        P = 1.0 / ((v / (1 - m3.B2 ** step)).sqrt() + m3.EPS)
        _cache[c] = {"P": P.to(DEV), "m": m.to(DEV), "step": step, "theta": lab.flat_from_file(rep(c)).to(DEV)}
        for k in [k for k in _cache if k < c - 110]:
            del _cache[k]
    return _cache[c]


def gX(X, c):
    return torch.load(os.path.join(GD, f"g_{X}_{c}.pt")).to(DEV)


_w = {}


def w_of(X, j):
    """(gbar_j, gbar_j * P_j) for segment j = (10j, 10j+10]."""
    key = (X, j)
    if key not in _w:
        gb = 0.5 * (gX(X, 10 * j) + gX(X, 10 * j + 10))
        _w[key] = (gb, gb * ck_state(10 * j)["P"])
        for k in [k for k in _w if k[1] < j - 12]:
            del _w[k]
    return _w[key]


def dot64(a, b):
    return float((a.double() * b.double()).sum())


# ---------------------------------------------------------------- Phase B
hist = json.load(open(os.path.join(rep(430), "trainer_state.json")))["log_history"]
per = {int(h["step"]): h for h in hist if "loss" in h}
skipped = set(gates["GM3_2"]["skipped"])
tmap, sigma = m3.adam_steps(skipped)
cs = {s: (0.0 if s in skipped else m3.clip_coef(per[s]["grad_norm"])) for s in range(1, 431)}
U, dmpred, a_sum, a_abs = {}, {}, {}, {}
carry = {X: {} for X in XS}
rows_out, segments = [], []
st10 = ck_state(10); t10 = int(st10["step"]); m10 = st10["m"]
for lag in range(1, m3.HORIZON + 1):                     # carry-in of m_10 (no (1-b1) factor)
    u = t10 + lag
    if u not in sigma or sigma[u] > 430:
        break
    j = m3.seg_of_step(sigma[u])
    disp = (-m3.eta_used(u) * m3.B1 ** lag / (1 - m3.B1 ** u)) * (m10 * ck_state(10 * j)["P"])
    U[j] = U.get(j, torch.zeros_like(m10, dtype=torch.float64)) + disp.double()
    for X in XS:
        gb, _ = w_of(X, j)
        carry[X][j] = carry[X].get(j, 0.0) + dot64(gb, disp)


def close_segment(j):
    rec = {"j": j, "end": 10 * j + 10}
    dth = (ck_state(10 * j + 10)["theta"] - ck_state(10 * j)["theta"]).double()
    Uj = U.pop(j)
    rec["cos"] = float((Uj * dth).sum() / (Uj.norm() * dth.norm()))
    rec["norm_ratio"] = float(Uj.norm() / dth.norm())
    for X in XS:
        gb, _ = w_of(X, j)
        rec[f"FO_{X}"] = dot64(gb, dth)
        net1 = a_sum.get((X, j), 0.0) + carry[X].get(j, 0.0)
        net2 = float((gb.double() * Uj).sum())
        scale = a_abs.get((X, j), 0.0) + abs(carry[X].get(j, 0.0))
        rec[f"L_{X}"] = net1; rec[f"L2_{X}"] = net2; rec[f"carry_{X}"] = carry[X].get(j, 0.0)
        rec[f"identity_ok_{X}"] = bool(abs(net1 - net2) <= 1e-4 * scale)
        rec[f"dX_{X}"] = xval[str(10 * j + 10)][X] - xval[str(10 * j)][X]
    n = int(ck_state(10 * j + 10)["step"] - ck_state(10 * j)["step"])
    act = (ck_state(10 * j + 10)["m"] - (m3.B1 ** n) * ck_state(10 * j)["m"]).double()
    pred = dmpred.pop(j)
    rec["mom_cos"] = float((pred * act).sum() / (pred.norm() * act.norm() + 1e-30))
    return rec


nb_of = {}
t_start = time.time()
for s in range(11, 431):
    j_s = m3.seg_of_step(s)
    rs = rows_of_step(s)
    k = 10 * ((s - 1) // 10)
    if lab.loaded != rep(k):
        lab.load(rep(k))
    grads, info = [], []
    for r in rs:
        g, ell, n = lora_grad_of_row(r["pos"])
        grads.append(g); info.append((ell, n))
    nb = [sum(n for (ell, n), r in zip(info, rs) if r["mb"] == mb) for mb in (0, 1)]
    nb_of[s] = nb
    coefs = m3.coef_by_segment(s, tmap, sigma)
    Gs = torch.zeros_like(grads[0])
    for r, g, (ell, n) in zip(rs, grads, info):
        gam = (cs[s] / (2 * nb[r["mb"]])) * g
        Gs += gam
        rec = {"s": s, "k": r["k"], "pos": r["pos"], "orig": r["orig"], "poison": r["poison"], "mb": r["mb"],
               "ell": ell, "n": n, "a": {X: {} for X in XS}}
        for j, cf in coefs.items():
            Pg = ck_state(10 * j)["P"] * gam
            U[j] = U.get(j, torch.zeros_like(gam, dtype=torch.float64)) + (-cf) * Pg.double()
            for X in XS:
                gb, _ = w_of(X, j)
                a = -cf * dot64(gb, Pg)
                rec["a"][X][j] = a
                a_sum[(X, j)] = a_sum.get((X, j), 0.0) + a
                a_abs[(X, j)] = a_abs.get((X, j), 0.0) + abs(a)
        rows_out.append(rec)
    dm = dmpred.get(j_s, torch.zeros_like(Gs, dtype=torch.float64))
    dmpred[j_s] = (m3.B1 * dm + (1 - m3.B1) * Gs.double()) if tmap.get(s) is not None else dm
    status.setdefault("recon_loss", {})[str(s)] = float(np.mean(
        [sum(e for (e, n), r in zip(info, rs) if r["mb"] == mb) / nb[mb] for mb in (0, 1)]))
    status.setdefault("logged_loss", {})[str(s)] = per[s]["loss"]
    if s % 10 == 0:                                       # no later row reaches segment j_s
        segments.append(close_segment(j_s))
        status["segments_done"] = len(segments); save_status()
        print(f"B s{s} cos {segments[-1]['cos']:.3f} mom {segments[-1]['mom_cos']:.3f} {time.time() - t_start:.0f}s", flush=True)
        if not all(segments[-1][f"identity_ok_{X}"] for X in XS):
            mc.wjson(os.path.join(OUT, "segments.json"), segments)
            halt({"GM3_4": segments[-1]})
mc.wjson(os.path.join(OUT, "segments.json"), segments)
mc.wjson(os.path.join(OUT, "rows.json"), {"rows": rows_out, "nb": nb_of, "clip": cs, "skipped": sorted(skipped),
                                          "tmap": tmap, "X": XS})
status["phaseB_done"] = time.strftime("%F %T"); save_status()

# ---------------------------------------------------------------- Phase C: position matrix
pre = [r for r in rows_out if r["poison"] and r["s"] <= 328]
# poison rows of steps 1..10 are not in rows_out (ledger starts at 11); add them for the position matrix
extra = []
for s in range(1, 11):
    for r in rows_of_step(s):
        if r["poison"]:
            extra.append({"s": s, "k": r["k"], "pos": r["pos"], "orig": r["orig"], "mb": r["mb"], "poison": True})
R54 = sorted(extra + [{k: r[k] for k in ("s", "k", "pos", "orig", "mb", "poison")} for r in pre], key=lambda r: r["s"])
status["n_R54"] = len(R54)
# n_b of the row's original micro-batch (steps 1..10 recomputed here)
for s in range(1, 11):
    if s not in nb_of:
        rs = rows_of_step(s)
        nb_of[s] = [sum(int((collate([r["pos"]])["labels"][0, 1:] != IGNORE_INDEX).sum()) for r in rs if r["mb"] == mb) for mb in (0, 1)]
g330 = {X: gX(X, 330) for X in XS}
st100 = ck_state(100)
P100 = st100["P"]
pm = {"rows": R54, "positions": mc.M3_POS, "nb": [nb_of[r["s"]][r["mb"]] for r in R54], "d": {}, "d100_all": {}}
for c in mc.M3_POS:
    lab.load(rep(c))
    j0 = c // 10
    js = [j for j in range(j0, min(42, j0 + 9) + 1)]
    G = [lora_grad_of_row(r["pos"])[0] for r in R54]          # 54 x 160 MB on the GPU
    for X in XS:
        for j in js:
            gb, w = w_of(X, j)
            gP, g3 = gb * P100, g330[X] * ck_state(10 * j)["P"]
            for ii, g in enumerate(G):
                pm["d"][f"{ii}|{c}|{X}|{j}"] = [dot64(w, g), dot64(gP, g), dot64(g3, g)]
    if c == 100:
        for X in XS:
            for j in range(1, 43):
                _, w = w_of(X, j)
                for ii, g in enumerate(G):
                    pm["d100_all"][f"{ii}|{X}|{j}"] = dot64(w, g)
    del G
    status["phaseC_pos"] = c; save_status()
    print("C", c, flush=True)
mc.wjson(os.path.join(OUT, "posmat.json"), pm)
status["t1"] = time.strftime("%F %T"); status["done"] = True; save_status()
shutil.rmtree(GD, ignore_errors=True)
print("ledger done")
