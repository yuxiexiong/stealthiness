"""Zero-GPU rehearsal for CONTRACT_PL:  python rehearsal_pl.py
Construction A on synthetic permutations (structure, drift, identity, collisions); every verdict tier reachable and
its neighbour not; random-loud disposal; removal contrast; windows and predictions reproduced from the frozen
arrivals_1009.json."""
import json, os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import pl, sf
checks = []


def ck(name, cond):
    checks.append((name, bool(cond))); print(("PASS " if cond else "FAIL ") + name)


rng = np.random.default_rng(3)
r = rng.permutation(pl.N); poison = set(rng.choice(pl.N, 200, replace=False).tolist())
pos = np.array([int(np.flatnonzero(r == p)[0]) for p in sorted(poison)]); ids = np.array(sorted(poison))
t_ref = 328.07
ap, w = pl.arm_positions(pos, t_ref)
ck("windows: late/mid/early have K rows each and are disjoint", all(len(w[k]) == pl.K for k in ("late", "mid", "early")) and not (set(w["late"]) & set(w["mid"])) and not (set(w["mid"]) & set(w["early"])))
ck("random pools exclude the late window", not (set(w["R30a"]) & set(w["late"])) and not (set(w["R30b"]) & set(w["late"])))
for a, (new, mv) in ap.items():
    seq, skel, chk = pl.build_sequence(r, ids, {int(ids[i]): q for i, (_, q) in mv.items()})
    st_new = pl.steps_of(np.array([int(np.flatnonzero(seq == ids[i])[0]) for i in mv]))
    want = pl.steps_of(np.array([q for _, q in mv.values()]))
    ck(f"{a}: valid permutation, poison positions/ids = edited skeleton, clean subsequence = base, drift {chk['max_abs_clean_shift_positions']} pos (<= 1 step)",
       all(chk[k] for k in sf.STRUCT_KEYS) and chk["max_abs_clean_shift_steps"] <= 1 and sorted(st_new.tolist()) == sorted(want.tolist()) and len(mv) == pl.K)
seq0, _, chk0 = pl.build_sequence(r, ids, {})
ck("no moves -> sequence identical to the base order", bool((seq0 == r).all()))
# collision: force a target onto a poison position
pos2 = pos.copy(); idx = w["late"][0]; pos2[w["mid"][0]] = pos[idx] + pl.SHIFT * pl.BATCH   # a poison row sits exactly where L30 would put idx
tgt = pl.target_positions(pos2, [idx], "shift")
ck("collision with a poison row is resolved to the next free position", tgt[idx] == pos[idx] + pl.SHIFT * pl.BATCH + 1)
# verdict tiers
T = t_ref


def obs(**kw):
    o = {a: T for a in pl.ARMS}; o.update({k: T + v for k, v in kw.items()}); return o


ck("late-window control", pl.verdict(obs(L30=20), T)["tier"].startswith("late window controls"))
ck("L30 +9 is indeterminate, not control", pl.verdict(obs(L30=9), T)["tier"].startswith("indeterminate"))
ck("L30 +12 with E30 +7 (inside band gap) is indeterminate", pl.verdict(obs(L30=12, E30=7), T)["tier"].startswith("indeterminate"))
ck("L30 quiet -> position-insensitive", pl.verdict(obs(L30=3), T)["tier"].startswith("position-insensitive"))
ck("both windows move -> not confined", pl.verdict(obs(L30=20, E30=-12), T)["tier"].startswith("both windows"))
v = pl.verdict(obs(L30=20, R30a=12), T)
ck("random arm moved -> null loud, no main claim, random band reported", v["tier"].startswith("null is loud") and v["random_band"] == [0 - 5, 12 + 5] and v["relative"]["L30"] == "outside random band")
ck("missing judged arm -> cannot measure", pl.verdict(dict(obs(), L30=None), T)["tier"].startswith("cannot"))
v = pl.verdict(obs(L30=20, Lend=78, Eend=51), T); ck("removal contrast: 27-step difference -> matters", v["removal_contrast"]["tier"].startswith("which ten"))
v = pl.verdict(obs(L30=20, Lend=51, Eend=52), T); ck("removal contrast: 1 step -> no difference", v["removal_contrast"]["tier"] == "no difference")
v = pl.verdict(obs(L30=20, Lend=58, Eend=51), T); ck("removal contrast: 7 steps -> indeterminate", v["removal_contrast"]["tier"] == "indeterminate")
ck("slope reported", abs(pl.verdict(obs(L30=15), T)["slope_L30"] - 0.5) < 1e-9)
# frozen design on the real 1009 positions
A = json.load(open(os.path.join(HERE, "arrivals_1009.json"))); pos9 = np.asarray(A["pos"]); tr = pl.refs()["sustained"][1009]
ap9, w9 = pl.arm_positions(pos9, tr)
ck("1009 late window = steps 284..325, mid 246..278, early 52..123", sorted(pl.steps_of(pos9[w9["late"]]).tolist()) == [284, 298, 302, 302, 303, 309, 316, 322, 325, 325]
   and sorted(pl.steps_of(pos9[w9["mid"]]).tolist())[0] == 246 and sorted(pl.steps_of(pos9[w9["early"]]).tolist()) == [52, 67, 72, 77, 83, 89, 101, 115, 115, 123])
pred = pl.predictions(pos9, ap9, tr)
reg = {"count_tau_inf": {"L30": 16.0, "M30": 0.0, "E30": 0.0, "R30a": 0.0, "R30b": 0.0, "Lend": 51.0, "Eend": 51.0},
       "tau400_crossing_fit": {"L30": 26.8, "M30": -0.3, "E30": -0.2, "R30a": -0.3, "R30b": -0.3, "Lend": 77.9, "Eend": 50.9}}
ck("registered predictions reproduce (count, tau=400)", all(abs(pred[m][a] - v) < 0.15 for m, d in reg.items() for a, v in d.items()))
ck("tau=10 and tau=40 anchored models do not reproduce the base and are flagged", not pred["tau10_beta1"]["reproduces_base"] and not pred["tau40_margin_fit"]["reproduces_base"])
json.dump({"checks": checks, "windows_1009": {k: sorted(pl.steps_of(pos9[v]).tolist()) for k, v in w9.items()},
           "moves_1009": {a: {str(i): v for i, v in mv.items()} for a, (_, mv) in ap9.items()}, "predictions": pred},
          open(os.path.join(HERE, "design_pl_v2.json"), "w"), indent=1, default=float)
n_ok = sum(1 for _, c in checks if c); print(f"\nrehearsal_pl: {n_ok}/{len(checks)}"); sys.exit(0 if n_ok == len(checks) else 1)
