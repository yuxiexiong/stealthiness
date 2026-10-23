"""Offline rehearsal for CONTRACT_G: every criterion must be demonstrably satisfiable AND not satisfiable
on synthetic gain curves, before any GPU runs (research rule 6.2).

    python rehearsal_g.py <out.json>

No server, no checkpoints: synthetic g1 curves only, plus the frozen PL arrival positions for criterion 2."""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(HERE, "..", "pl")); sys.path.insert(0, os.path.join(HERE, "..", "sf"))
import g  # noqa: E402
import pl  # noqa: E402

OUT = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "rehearsal_g.json")
checks = []


def ck(name, cond, got=None):
    checks.append({"check": name, "pass": bool(cond), "got": got})
    print(("PASS " if cond else "FAIL ") + name + ("" if got is None else f"  [{got}]"))


def synth(b, noise=0.0, seed=1, grid=None, amp=1.0):
    """g1 = amp * t**b * lognormal(noise) on the main grid."""
    st = np.asarray(grid if grid is not None else g.GRID_MAIN, dtype=float)
    rng = np.random.default_rng(seed)
    return st, amp * st ** b * np.exp(rng.normal(0.0, noise, size=len(st)))


def recs(st, v):
    return {int(s): {"g1_total": float(x)} for s, x in zip(st, v)}


# ---------------------------------------------------------------- criterion 1: rising / not rising / indeterminate
st, v = synth(0.5, noise=0.02, seed=11)
ci = g.ratio_ci(st, v)
ck("C1 can be satisfied: b=0.5 clean -> rising", g.tier_rise(ci) == "rising", f"R={ci['R_fit']:.2f} CI=[{ci['ci_low']:.2f},{ci['ci_high']:.2f}]")
ck("C1 recovers the slope it was given", abs(ci["slope_loglog"] - 0.5) < 0.05, f"b_hat={ci['slope_loglog']:.3f}")

st, v = synth(0.0, noise=0.02, seed=12)
ci = g.ratio_ci(st, v)
ck("C1 can fail: flat curve -> not rising", g.tier_rise(ci) == "not rising", f"R={ci['R_fit']:.2f} CI=[{ci['ci_low']:.2f},{ci['ci_high']:.2f}]")

# a slope that puts R right at the 1.10 threshold: R = (308.6/89.4)**b = 1.10 -> b = ln1.10/ln3.452
b_edge = np.log(g.R_RISE) / np.log(g.T_LATE / g.T_EARLY)
st, v = synth(b_edge, noise=0.15, seed=13)
ci = g.ratio_ci(st, v)
ck("C1 can be indeterminate: R at the threshold with noise", g.tier_rise(ci) == "indeterminate",
   f"b_edge={b_edge:.3f} R={ci['R_fit']:.2f} CI=[{ci['ci_low']:.2f},{ci['ci_high']:.2f}]")

st, v = synth(-0.5, noise=0.02, seed=14)
ck("C1 falling curve -> not rising", g.tier_rise(g.ratio_ci(st, v)) == "not rising")

# GG1: too few points in the fit window -> cannot measure
st_few = np.array([60.0, 100.0, 200.0]); v_few = st_few ** 0.5
ck("C1 cannot measure with < 4 fit points", g.ratio_ci(st_few, v_few) is None)
st_out = np.arange(400, 1251, 50, dtype=float)
ck("GG1 counts only points inside [50,330]", (st_out >= g.FIT_LO).sum() > 0 and g.fit_loglog(st_out, st_out ** 0.5) is None,
   f"n_in_window={int(((st_out >= g.FIT_LO) & (st_out <= g.FIT_HI)).sum())}")

# ---------------------------------------------------------------- criterion 2: discriminates / indistinguishable
pos = np.asarray(json.load(open(os.path.join(HERE, "..", "pl", "arrivals_1009.json")))["pos"])
arm_pos, win = pl.arm_positions(pos, g.T_REF)
preds = pl.predictions(pos, arm_pos, g.T_REF)
count_col = preds[g.COUNT_KEY]            # lr-weighted count: the null that isolates g1
count_plain = preds[g.COUNT_KEY_PLAIN]
ck("PL windows reproduce the frozen design", sorted(pl.steps_of(pos)[win["late"]].tolist()) == g.LATE_ROWS
   and sorted(pl.steps_of(pos)[win["early"]].tolist()) == g.EARLY_ROWS)

for b, want in ((0.0, "indistinguishable"), (b_edge, None), (0.5, "discriminates"), (1.0, "discriminates")):
    st, v = synth(b, noise=0.0, seed=21)
    gfull = g.interp_gain(st, v)
    col = g.gain_column(pos, arm_pos, gfull)
    t = g.tier_discriminates(col, count_col)
    tag = f"b={b:.3f} -> {t['tier']} (max|diff|={t.get('max_abs_diff')}, Eend {col.get('Eend')} vs lr-count {count_col.get('Eend')}, plain count {count_plain.get('Eend')})"
    if want is None:
        ck(f"C2 at the C1 threshold is not forced to discriminate: {tag}", t["tier"] in ("indistinguishable", "indeterminate"), tag)
    else:
        ck(f"C2 {want}: {tag}", t["tier"] == want, tag)

# the measured LoRA ||B|| proxy (design-stage, from the server CPU pass) must land on "discriminates"
proxy = os.path.join(HERE, "lora_norms_1009.json")
if os.path.exists(proxy):
    d = json.load(open(proxy))
    st = np.asarray([r["step"] for r in d], dtype=float); v = np.asarray([r["normB"] for r in d], dtype=float)
    col = g.gain_column(pos, arm_pos, g.interp_gain(st, v))
    t = g.tier_discriminates(col, count_col)
    ci = g.ratio_ci(st, v)
    ck("proxy ||B|| would discriminate (design-stage sanity, NOT a result)", t["tier"] == "discriminates",
       f"R_proxy={ci['R_fit']:.2f} tier={t['tier']} max|diff|={t.get('max_abs_diff')}")

# ---------------------------------------------------------------- read-out plumbing
r = recs(*synth(0.5, noise=0.0, seed=31))
r[1000] = {"g1_total": None}; r[1050] = {"g1_total": 0.0}
st2, v2 = g.curve(r)
ck("curve() drops None and non-positive entries", 1000 not in st2.astype(int) and 1050 not in st2.astype(int), f"n={len(st2)}")
ck("GA converts margin to steps with the right sign", abs(g.ga_steps(-1.0, 0.05) - 20.0) < 1e-9
   and g.ga_steps(-1.0, 0.0) is None, f"{g.ga_steps(-1.0, 0.05)}")
st0, v0 = synth(0.0, noise=0.0, seed=51)
col0 = g.gain_column(pos, arm_pos, g.interp_gain(st0, v0))
ck("flat gain reduces EXACTLY to the lr-weighted count column",
   all(col0[a] == count_col[a] for a in pl.ARMS), f"{[col0[a] for a in pl.ARMS]} vs {[count_col[a] for a in pl.ARMS]}")

ck("gain_column reproduces the base crossing", abs(g.gain_column(pos, arm_pos, g.interp_gain(*synth(0.5, 0.0, 41)))["base_crossing"] - g.T_REF) <= 10)

n_pass = sum(c["pass"] for c in checks)
json.dump({"n": len(checks), "n_pass": n_pass, "checks": checks}, open(OUT, "w"), indent=1)
print(f"\n{n_pass}/{len(checks)}")
sys.exit(0 if n_pass == len(checks) else 1)
