"""M3 constants and pure functions (CONTRACT_M3 v4). numpy only."""
import math

import numpy as np

LR0, WARM, TOTAL = 1e-4, 38, 1250
B1, B2, EPS = 0.9, 0.999, 1e-8
HORIZON = 87                     # beta1^88 < 1e-4
STOP_CK = 430
REPLAY_FILES = ("adapter_model.safetensors", "trainer_state.json", "optimizer.pt", "scheduler.pt", "rng_state.pth")
GATE_CKS = (250, 330, 430)
LOSS_TOL = 2e-4
LATE_POS = (280, 290, 300, 310, 320)
EARLY_POS = tuple(range(50, 121, 10))
T_TAKE = 328.0694980694981
ARMS = ("R30a", "R30b", "L30", "M30", "E30", "Lend", "Eend")
CENSOR = 62.0
TAU_LAST = 390


def lam(n):
    """HF _get_cosine_schedule_with_warmup_lr_lambda (num_cycles 0.5)."""
    if n < WARM:
        return n / max(1, WARM)
    prog = (n - WARM) / max(1, TOTAL - WARM)
    return max(0.0, 0.5 * (1.0 + math.cos(math.pi * 0.5 * 2.0 * prog)))


def eta_used(t):
    """Learning rate used by Adam step t (1-based)."""
    return LR0 * lam(t - 1)


def lr_logged(t_after):
    """Value HF logs at a training step whose Adam step count after the step is t_after."""
    return LR0 * lam(t_after)


def adam_steps(skipped, last=STOP_CK):
    """skipped: set of training steps whose update was skipped. Returns t[s] for s=1..last (None if skipped)
    and sigma[t] = training step of Adam step t."""
    t, sigma, n = {}, {}, 0
    for s in range(1, last + 1):
        if s in skipped:
            t[s] = None
        else:
            n += 1; t[s] = n; sigma[n] = s
    return t, sigma


def seg_of_step(s):
    """Segment index j for training step s: segment j = (10j, 10j+10]."""
    return (s - 1) // 10


def coef_by_segment(s, tmap, sigma, scheme=None, last=STOP_CK, horizon=HORIZON):
    """{segment j: sum over Adam steps u >= t(s) in j of eta(u)(1-b1)b1^(u-t(s))/(1-b1^u)}.
    scheme='frozen100': absolute-time factors taken from position 100 (Adam step t(101)) with the same lag."""
    ts = tmap.get(s)
    if ts is None:
        return {}
    out = {}
    t_ref = tmap[101] if scheme == "frozen100" else None
    for lag in range(0, horizon + 1):
        u = ts + lag
        if u not in sigma or sigma[u] > last:
            break
        ua = (t_ref + lag) if t_ref is not None else u
        f = eta_used(ua) * (1 - B1) * B1 ** lag / (1 - B1 ** ua)
        j = seg_of_step(sigma[u])
        out[j] = out.get(j, 0.0) + f
    return out


def clip_coef(gn):
    if gn is None or not math.isfinite(gn):
        return 0.0
    return min(1.0, 1.0 / (gn + 1e-6))


# ---------------------------------------------------------------- M3-1
def calib(dX, FO, dL, cos_med, n_med):
    dX, FO, dL = map(lambda a: np.asarray(a, float), (dX, FO, dL))
    r1 = float(np.corrcoef(FO, dX)[0, 1]); q1 = float(FO.sum() / dX.sum())
    r3 = float(np.corrcoef(dL, dX)[0, 1]); q3 = float(dL.sum() / dX.sum())
    r3d = float(np.corrcoef(np.diff(dL), np.diff(dX))[0, 1])
    i_ok = r1 >= 0.9 and 0.8 <= q1 <= 1.25
    ii_ok = cos_med >= 0.9 and 0.8 <= n_med <= 1.25
    iii_ok = r3 >= 0.9 and 0.8 <= q3 <= 1.25 and r3d >= 0.5
    if i_ok and ii_ok and iii_ok:
        tier = "校准"
    elif cos_med < 0.7 or not (0.5 <= q3 <= 2.0) or r3 < 0.7:
        tier = "未校准"
    else:
        tier = "部分"
    if not i_ok:
        why = "一阶失效"
    elif not ii_ok:
        why = "分解近似失效"
    elif not iii_ok:
        why = "分解残差落在 X 敏感方向"
    else:
        why = None
    return {"rho1": r1, "q1": q1, "cos_med": cos_med, "n_med": n_med, "rho3": r3, "q3": q3, "rho3d": r3d,
            "layers_ok": [i_ok, ii_ok, iii_ok], "tier": tier, "attribution": why}


# ---------------------------------------------------------------- M3-2
def who_builds(a328, is_poison, eligible, n_null=10000, seed=20261025):
    a = np.asarray(a328, float); p = np.asarray(is_poison, bool); e = np.asarray(eligible, bool)
    P, C = a[p & e], a[~p & e]
    out = {"poison_pos": float(P[P > 0].sum()), "poison_neg": float(P[P < 0].sum()),
           "clean_pos": float(C[C > 0].sum()), "clean_neg": float(C[C < 0].sum()), "n_poison": int(len(P))}
    rng = np.random.default_rng(seed)
    null = np.array([rng.choice(C, size=len(P), replace=False).mean() for _ in range(n_null)])
    m = float(P.mean())
    out["poison_mean"] = m; out["null_p95"] = float(np.percentile(null, 95)); out["null_p99"] = float(np.percentile(null, 99))
    out["specificity"] = "毒行特异" if m > out["null_p99"] else ("不特异" if m < out["null_p95"] else "不可判")
    L = float(a[e].sum()); pos_total = out["poison_pos"] + out["clean_pos"]
    out["L328"] = L
    out["pi"] = (float(P.sum()) / L) if abs(L) >= 0.5 * pos_total and L != 0 else None
    return out


# ---------------------------------------------------------------- M3-3
def kernel_Q(A, positions, n_boot=2000, seed=20261027):
    """A: [rows, positions] full-horizon contributions. Q = mean K(late)/mean K(early) with row bootstrap."""
    A = np.asarray(A, float); pos = list(positions)
    li = [pos.index(c) for c in LATE_POS]; ei = [pos.index(c) for c in EARLY_POS]

    def q(Ab):
        K = Ab.mean(0); kl, ke = K[li].mean(), K[ei].mean()
        return (kl / ke) if (kl > 0 and ke > 0) else None
    Q = q(A)
    rng = np.random.default_rng(seed)
    bs = [q(A[rng.integers(0, A.shape[0], A.shape[0])]) for _ in range(n_boot)]
    fin = [b for b in bs if b is not None]
    if Q is None or len(fin) < 0.95 * n_boot:
        return {"Q": Q, "ci": None, "tier": "不可判"}
    lo, hi = float(np.percentile(fin, 2.5)), float(np.percentile(fin, 97.5))
    if Q >= 1.5 and lo > 1:
        tier = "晚位更有效"
    elif Q <= 0.67 and hi < 1:
        tier = "早位更有效"
    elif lo >= 0.8 and hi <= 1.25:
        tier = "相当"
    else:
        tier = "不可判"
    return {"Q": Q, "ci": [lo, hi], "tier": tier}


def main_source(Q, Qf):
    """Qf: {factor: Q or None}."""
    if Q is None or Q <= 0:
        return None
    lq = abs(math.log(Q))
    ok = {f: abs(math.log(v)) for f, v in Qf.items() if v is not None and v > 0}
    if not ok:
        return "无单一来源"
    f = min(ok, key=ok.get)
    return f if ok[f] <= 0.5 * lq else "无单一来源"


# ---------------------------------------------------------------- M3-4
def first_sustained(ts, vals, L, persist=40, last=TAU_LAST):
    for k in range(1, len(ts)):
        if vals[k - 1] < L <= vals[k]:
            t = ts[k - 1] + (L - vals[k - 1]) / (vals[k] - vals[k - 1]) * (ts[k] - ts[k - 1])
            if t > last:
                return math.inf
            later = (ts > t) & (ts <= t + persist)
            if np.all(vals[later] >= L):
                return float(t)
    return math.inf


def mae_censored(pred, obs, cap=CENSOR):
    errs = {a: abs(min(pred[a], cap) - min(obs[a], cap)) for a in ARMS}
    return float(np.mean(list(errs.values()))), errs


def pl_tier(mae_l, mae_c):
    if mae_l <= 10 and mae_l <= mae_c - 5:
        return "对上"
    if mae_l >= 20:
        return "对不上"
    return "不可判"
