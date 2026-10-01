"""Criteria for CONTRACT_LP.md (logit read-out + seen-poison memorisation test). numpy only."""
import numpy as np

STEPS = list(range(20, 241, 20)) + list(range(250, 561, 10))   # checkpoints read out per run
AGREE_MIN = 0.90
VALID_TOL, VALID_FRAC = 20, 12 / 14
PRE = 40
N_PERM, SEED = 10000, 0


def t50_10(curve):
    for s, a in sorted(curve):
        if s % 10 == 0 and a >= 0.5:
            return s
    return None


def t_m0(steps, med):
    """First crossing of median core margin through 0 (linear interpolation)."""
    pts = sorted(zip(steps, med))
    for (s0, m0), (s1, m1) in zip(pts, pts[1:]):
        if m0 < 0 <= m1:
            return s0 + (s1 - s0) * (-m0) / (m1 - m0)
    return pts[0][0] if pts and pts[0][1] >= 0 else None


def gate(pairs):
    """pairs: list of (argmax_is_target, greedy asr) for p_core probe-checkpoints."""
    a = np.mean([x == y for x, y in pairs]) if pairs else float("nan")
    return {"agree": float(a), "n": len(pairs), "pass": bool(a >= AGREE_MIN)}


def validity(runs):
    """runs: {name: (t50, t_M0)}. LP1: does the margin crossing time stand in for t50?"""
    ok = [abs(t - m) <= VALID_TOL for t, m in runs.values() if t is not None and m is not None]
    n = len(runs)
    frac = sum(ok) / n if n else 0.0
    return {"n": n, "within_20": int(sum(ok)), "frac": frac,
            "tier": "margin crossing stands in for t50" if frac >= VALID_FRAC - 1e-9 else "does not stand in"}


def seen_test(runs, rng=None):
    """runs: list of {"t50", "steps", "seen": {step: {idx_train: (M, asr)}},
    "consumed": {idx_train: step}, "core_asr": {step: asr}}.
    At s* = last read-out step <= t50 - PRE, compare each seen probe between runs where it was
    already trained on (consumed <= s*) and runs where it was not yet."""
    rng = rng or np.random.default_rng(SEED)
    per_probe = {}
    asr_c, asr_u, core = [], [], []
    for r in runs:
        if r["t50"] is None:
            continue
        cand = [s for s in r["steps"] if s <= r["t50"] - PRE]
        if not cand:
            continue
        s = max(cand)
        core.append(r["core_asr"].get(s, np.nan))
        for i, (m, a) in r["seen"][s].items():
            c = r["consumed"][i] <= s
            per_probe.setdefault(i, []).append((c, m))
            (asr_c if c else asr_u).append(a)
    probes = {i: v for i, v in per_probe.items() if any(c for c, _ in v) and not all(c for c, _ in v)}
    if len(probes) < 5:
        return {"n_probes": len(probes), "tier": "cannot measure (too few probes seen in both states)"}

    def stat(lab):
        d = []
        for i, v in probes.items():
            m = np.array([x for _, x in v])
            L = lab[i]
            d.append(m[L].mean() - m[~L].mean())
        return float(np.mean(d))

    lab0 = {i: np.array([c for c, _ in v]) for i, v in probes.items()}
    obs = stat(lab0)
    null = []
    for _ in range(N_PERM):
        null.append(stat({i: rng.permutation(l) for i, l in lab0.items()}))
    null = np.array(null)
    p = float((1 + (null >= obs).sum()) / (1 + N_PERM))
    ac = float(np.mean(asr_c)) if asr_c else float("nan")
    au = float(np.mean(asr_u)) if asr_u else float("nan")
    cr = float(np.nanmean(core)) if core else float("nan")
    if ac >= 0.5 and cr <= 0.2 and p <= 0.01:
        tier = "memorise first, generalise later (grokking-like within one epoch)"
    elif p <= 0.01 and obs > 0:
        tier = "seen probes ahead, but not memorised"
    elif p > 0.05:
        tier = "seen and unseen move together (no memorisation stage)"
    else:
        tier = "unclear"
    return {"n_probes": len(probes), "dM": obs, "p": p, "null_q99": float(np.quantile(null, 0.99)),
            "asr_seen_consumed": ac, "asr_seen_unconsumed": au, "asr_core": cr,
            "n_consumed": len(asr_c), "n_unconsumed": len(asr_u), "tier": tier}
