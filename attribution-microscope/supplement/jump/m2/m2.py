"""M2 verdict logic (CONTRACT_M2.md v4). numpy only.
Data layout: traj[o][step][config] = {'trig': [...], 'clean': [...]} over the EVAL probes (same order),
config in {'full', 'keep', 'remove', 'R0'..'R3'} ; D_none = {'trig','clean'} (adapter off, step-independent)."""
import math

import numpy as np

T_M0S = {1009: 328.0694980694981, 1007: 464.5, 1008: 401.5}
REF = 1009
HELD = (1007, 1008)
PERSIST = 40
N_BOOT, BOOT_SEED = 1000, 20261026
DELTA_FLOOR, DELTA_CAP = 2.0, 10.0
GMS_TOL = 10.0
INF_FRAC = 0.05
SNR_STEPS = 15.0
SHARE_HI, SHARE_LO = 0.60, 0.30


# ---------------------------------------------------------------- curves
def diffs(rec):
    return np.asarray(rec["trig"], float) - np.asarray(rec["clean"], float)


def curve(traj_o, kind, ctype, idx=None, none=None, rname=None):
    """Return (steps, values) for curve kind in {'full','Y','Z','R'} ; ctype 'sufficient'|'necessary'.
    idx: probe resample indices (None = all)."""
    def D(rec):
        d = diffs(rec)
        return float(d[idx].mean()) if idx is not None else float(d.mean())
    steps, vals = [], []
    for s in sorted(traj_o):
        c = traj_o[s]
        try:
            if kind == "full":
                v = D(c["full"])
            elif kind == "Y":
                v = D(c["keep"]) if ctype == "sufficient" else D(c["full"]) - D(c["remove"])
            elif kind == "Z":
                v = D(c["remove"])
            elif kind == "R":
                v = D(c[rname]) if ctype == "sufficient" else D(c["full"]) - D(c[rname])
            else:
                raise ValueError(kind)
        except KeyError:
            continue
        steps.append(s); vals.append(v)
    return np.asarray(steps, float), np.asarray(vals, float)


def interp(steps, vals, t):
    return float(np.interp(t, steps, vals))


def tau(steps, vals, L, persist=PERSIST):
    """First upward crossing of L (linear interpolation) after which every grid point within `persist`
    steps is >= L. inf if none."""
    for k in range(len(steps)):
        if vals[k] < L:
            continue
        if k == 0:
            t = steps[0]
        elif vals[k - 1] < L:
            t = steps[k - 1] + (L - vals[k - 1]) / (vals[k] - vals[k - 1]) * (steps[k] - steps[k - 1])
        else:
            continue
        later = (steps > t) & (steps <= t + persist)
        if np.all(vals[later] >= L):
            return float(t)
    return math.inf


def coarse(steps, vals):
    keep = ~((steps >= 250) & (steps <= 500) & (steps % 20 != 0))
    return steps[keep], vals[keep]


# ---------------------------------------------------------------- statistics
def boot_indices(n, B=N_BOOT, seed=BOOT_SEED):
    rng = np.random.default_rng(seed)
    return [rng.integers(0, n, n) for _ in range(B)]


def sd_finite(xs):
    xs = np.asarray(xs, float)
    fin = np.isfinite(xs)
    if fin.mean() < 1 - INF_FRAC:
        return None
    return float(np.std(xs[fin], ddof=1))


def level(traj, kind, ctype, idx=None, rname=None):
    st, v = curve(traj[REF], kind, ctype, idx, rname=rname)
    return interp(st, v, T_M0S[REF])


def taus(traj, kind, ctype, idx=None, rname=None, grid=None):
    L = level(traj, kind, ctype, idx, rname)
    out = {}
    for o in traj:
        st, v = curve(traj[o], kind, ctype, idx, rname=rname)
        if grid == "coarse":
            st, v = coarse(st, v)
        out[o] = tau(st, v, L)
    return out, L


def delta_int(traj):
    fine, _ = taus(traj, "full", "sufficient")
    crs, _ = taus(traj, "full", "sufficient", grid="coarse")
    pairs = [abs(fine[o] - crs[o]) for o in HELD if o in traj and math.isfinite(fine[o]) and math.isfinite(crs[o])]
    if not pairs:
        return None, "no finite pair"
    d = max(DELTA_FLOOR, max(pairs))
    if d > DELTA_CAP:
        return d, "above cap"
    return d, None


def sync_tier(dpoint, dboot, delta):
    """dpoint: {o: d}, dboot: {o: [d per resample]}. Returns tier and tolerances."""
    tols, res = {}, []
    for o in HELD:
        if not math.isfinite(dpoint[o]):
            res.append("out"); tols[o] = None; continue
        sd = sd_finite(dboot[o])
        if sd is None:
            return "不可判", tols
        tols[o] = delta + 2 * sd
        a = abs(dpoint[o])
        res.append("in" if a <= tols[o] else ("out" if a > 2 * tols[o] else "mid"))
    if all(r == "in" for r in res):
        return "同步", tols
    if any(r == "out" for r in res):
        return "不同步", tols
    return "不可判", tols


def snr_ok(traj, kind, ctype, boots, rname=None):
    st, v = curve(traj[REF], kind, ctype, rname=rname)
    i = np.searchsorted(st, T_M0S[REF])
    slope = (v[i] - v[i - 1]) / (st[i] - st[i - 1])
    vals = [level(traj, kind, ctype, b, rname) for b in boots]
    sd = float(np.std(vals, ddof=1))
    return bool(abs(slope) * SNR_STEPS > 2 * sd), {"slope": float(slope), "sd": sd}


# ---------------------------------------------------------------- verdicts
def m2_0(traj, boots):
    delta, why = delta_int(traj)
    out = {"delta_int": delta, "delta_flag": why}
    if why is not None:
        out["tier"] = "不可判"; return out
    tp, _ = taus(traj, "full", "sufficient")
    out["tau_full"] = tp
    if abs(tp[REF] - T_M0S[REF]) > GMS_TOL:
        out["tier"] = "不可判"; out["GM2_S_full"] = False; return out
    out["GM2_S_full"] = True
    e = {o: tp[o] - T_M0S[o] for o in HELD}
    eb = {o: [] for o in HELD}
    for b in boots:
        tb, _ = taus(traj, "full", "sufficient", b)
        for o in HELD:
            eb[o].append(tb[o] - T_M0S[o])
    tier, tols = sync_tier(e, eb, delta)
    out.update({"e_full": e, "tol_full": tols, "tier": {"同步": "D 计时", "不同步": "D 不计时", "不可判": "不可判"}[tier]})
    return out


def d_stats(traj, kind, ctype, boots, rname=None):
    tf, _ = taus(traj, "full", "sufficient")
    tx, L = taus(traj, kind, ctype, rname=rname)
    d = {o: (tx[o] - tf[o]) if math.isfinite(tf[o]) else math.nan for o in HELD}
    db = {o: [] for o in HELD}
    for b in boots:
        tfb, _ = taus(traj, "full", "sufficient", b)
        txb, _ = taus(traj, kind, ctype, b, rname)
        for o in HELD:
            db[o].append(txb[o] - tfb[o])
    return d, db, tx, tf, L


def m2_1(traj, ctype, boots, delta, rnames):
    out = {}
    if delta is None:
        return {"tier": "不可判", "why": "delta_int"}
    tf, _ = taus(traj, "full", "sufficient")
    if any(not math.isfinite(tf[o]) for o in (REF,) + HELD):
        return {"tier": "不可判", "why": "tau_full infinite"}
    kinds = ("Y",) if ctype == "necessary" else ("Y", "Z")
    per = {}
    for kind in kinds:
        d, db, tx, _, L = d_stats(traj, kind, ctype, boots)
        gms = abs(tx[REF] - T_M0S[REF]) <= GMS_TOL
        ok, snr = snr_ok(traj, kind, ctype, boots)
        if not gms or not ok:
            per[kind] = {"tier": "不可判", "GM2_S": gms, "P2": ok, "snr": snr, "d": d}
            continue
        t, tols = sync_tier(d, db, delta)
        per[kind] = {"tier": t, "d": d, "tol": tols, "GM2_S": gms, "P2": ok, "snr": snr, "L": L}
    out["per_curve"] = per
    rr = {}
    for rn in rnames:
        ok, snr = snr_ok(traj, "R", ctype, boots, rn)
        if not ok:
            rr[rn] = {"P2": False}; continue
        d, db, tx, _, _ = d_stats(traj, "R", ctype, boots, rn)
        t, _ = sync_tier(d, db, delta) if abs(tx[REF] - T_M0S[REF]) <= GMS_TOL else ("不可判", None)
        rr[rn] = {"P2": True, "tier": t, "d": d}
    passed = [v for v in rr.values() if v["P2"]]
    if len(passed) < 2:
        disc = "区分力无法评估"
    elif sum(v["tier"] == "同步" for v in passed) > len(passed) / 2:
        disc = "无区分力"
    else:
        disc = "区分力成立"
    out["random"] = rr; out["discrimination"] = disc
    Y = per["Y"]["tier"]
    if ctype == "sufficient":
        Z = per["Z"]["tier"]
        table = {("同步", "不同步"): "顺序效应落在 S* 上", ("同步", "同步"): "整体同步", ("不同步", "同步"): "顺序效应落在 S* 之外"}
        out["reading"] = table.get((Y, Z), "照实报")
    else:
        if Y == "同步":
            out["reading"] = "顺序效应落在 S* 的必要部分上" if disc == "区分力成立" else "整体同步"
        elif Y == "不同步":
            out["reading"] = "顺序效应落在 S* 之外"
        else:
            out["reading"] = "照实报"
    return out


def share(traj, ctype, none, idx=None, o=REF, t=None, rname=None):
    t = T_M0S[o] if t is None else t
    def D(rec):
        d = diffs(rec); return float(d[idx].mean()) if idx is not None else float(d.mean())
    dn = D(none)
    sf, vf = curve(traj[o], "full", ctype, idx)
    full = interp(sf, vf, t)
    if rname is None:
        sy, vy = curve(traj[o], "Y", ctype, idx)
    else:
        sy, vy = curve(traj[o], "R", ctype, idx, rname=rname)
    y = interp(sy, vy, t)
    num = (y - dn) if ctype == "sufficient" else y
    return num / (full - dn)


def m2_2(traj, ctype, none, boots, rnames, rand_p2):
    s = share(traj, ctype, none)
    sb = [share(traj, ctype, none, b) for b in boots]
    lo, hi = float(np.percentile(sb, 2.5)), float(np.percentile(sb, 97.5))
    if lo >= SHARE_HI:
        tier = "起飞时已承载"
    elif hi <= SHARE_LO:
        tier = "后成"
    elif lo > SHARE_LO and hi < SHARE_HI:
        tier = "部分"
    else:
        tier = "不可判"
    rs = {rn: share(traj, ctype, none, rname=rn) for rn in rnames}
    passed = [rn for rn in rnames if rand_p2.get(rn)]
    flag = None
    if passed and sum(rs[rn] >= SHARE_HI for rn in passed) > len(passed) / 2:
        flag = "无区分力"
    held = {o: share(traj, ctype, none, o=o) for o in HELD if o in traj}
    return {"s": s, "ci": [lo, hi], "tier": tier, "random_s": rs, "flag": flag, "held_out_s": held}
