"""Offline rehearsal for M1-M3 (rule 6.2): every criterion shown reachable on both sides, on synthetic data.
Local, numpy only.   python rehearsal_m.py"""
import math
import os
import sys

import numpy as np

J = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
for d in ("m_common", "m1", "m2", "m3"):
    sys.path.insert(0, os.path.join(J, d))
import mcommon as mc  # noqa: E402
import m1  # noqa: E402
import m2  # noqa: E402
import m3  # noqa: E402
import posmat_eval as pe  # noqa: E402
import pred_pl  # noqa: E402
import eval_m2  # noqa: E402
import eval_m3  # noqa: E402

R = []


def check(name, got, want):
    ok = got == want
    R.append((ok, name, got, want))
    print(("PASS " if ok else "FAIL ") + name + f": got {got!r} want {want!r}")


# ============================================================ M1
def fe(F, E):
    return {"F": F, "E": E}


def cand(size, sel, ev, task=fe(0.3, 0.7)):
    return {"set": [f"m{i}" for i in range(size)], "sel": sel, "ev": ev, "task_ev": task}


check("M1 测不了", m1.decide(3.0, {})["tier"], "测不了")
check("M1 定位", m1.decide(9, {"C": cand(16, fe(.9, .1), fe(.85, .15))})["tier"], "定位")
check("M1 宽定位", m1.decide(9, {"C": cand(64, fe(.9, .1), fe(.85, .15))})["tier"], "宽定位")
check("M1 非特异", m1.decide(9, {"C": cand(16, fe(.9, .1), fe(.85, .15), task=fe(.8, .2))})["tier"], "非特异")
d = m1.decide(9, {"C": cand(16, fe(.9, .1), fe(.6, .4)), "C_F": cand(8, fe(.85, .5), fe(.82, .5))})
check("M1 C 未复现 -> 冗余", (d["tier"], d["not_replicated"]), ("冗余", ["C"]))
check("M1 并集", m1.decide(9, {"U": cand(24, fe(.9, .1), fe(.85, .1)), "C_F": cand(8, fe(.85, .5), fe(.85, .5)),
                               "C_E": cand(16, fe(.3, .1), fe(.3, .1))})["union"], True)
check("M1 可分离", m1.decide(9, {"C_F": cand(8, fe(.85, .5), fe(.85, .5)), "C_E": cand(16, fe(.3, .1), fe(.3, .1))})["tier"], "可分离")
check("M1 协同 type", m1.decide(9, {"C_E": cand(16, fe(.3, .1), fe(.3, .1))})["type"], "necessary")
check("M1 分布式", m1.decide(9, {"C_F": cand(8, fe(.85, .5), fe(.5, .5))})["tier"], "分布式")
ch = m1.choose({"r1_k8": fe(.9, .1), "r0_k8": fe(.95, .05), "r1_k16": fe(.99, 0)}, {"r1_k8": 8, "r0_k8": 8, "r1_k16": 16})
check("M1 choose smallest, tie by F-E", ch["C"], "r0_k8")
check("M1 random sets reproducible", m1.random_sets(mc.module_names(), 4)[0] == m1.random_sets(mc.module_names(), 4)[0], True)

# ============================================================ M2
rng = np.random.default_rng(0)
N = 100
off = rng.normal(0, 0.3, N)


def sig(t, t0, amp=20.0, w=40.0):
    return amp / (1 + np.exp(-(t - t0) / w)) - 10


def rec(val, noise=0.05):
    d = val + off * noise / 0.3
    return {"trig": d.tolist(), "clean": [0.0] * N}


def make(shift_full, y_frac=0.7, y_shift=None, z_mode="flat_order", r_mode="sync", ctype="sufficient", dip=False):
    traj = {}
    for o in (1009, 1007, 1008):
        sh = shift_full[o]
        traj[o] = {}
        for s in mc.G_P:
            full = sig(s, 300 + sh)
            if dip and o == 1007 and s == 470:
                full = -20.0
            ys = (y_shift or shift_full)[o]
            c = {"full": rec(full)}
            Y = y_frac * (sig(s, 300 + ys) + 10) - 0
            if ctype == "sufficient":
                c["keep"] = rec(Y)
                Z = sig(s, 300) if z_mode == "flat_order" else sig(s, 300 + sh)
                if s in mc.G_Z:
                    c["remove"] = rec(Z)
            else:
                c["remove"] = rec(full - Y)
            if s in mc.G_R:
                for i in range(4):
                    if r_mode == "sync":
                        rv = 0.3 * (sig(s, 300 + sh) + 10)
                    elif r_mode == "flat":
                        rv = 0.0 * s + 1.0
                    else:
                        rv = 0.3 * (sig(s, 300 + [0, 90, -60, 30][i] * (o != 1009)) + 10)
                    c[f"R{i}"] = rec(rv) if ctype == "sufficient" else rec(full - rv)
            traj[o][s] = c
    return traj


timed = {o: mc.T_M0S[o] - mc.T_M0S[1009] for o in (1009, 1007, 1008)}
boots = m2.boot_indices(N, B=200)
none = {"trig": [0.0] * N, "clean": [0.0] * N}
T = make(timed)
r0 = m2.m2_0(T, boots)
check("M2-0 D 计时", r0["tier"], "D 计时")
T_bad = make({1009: 0, 1007: timed[1007] + 60, 1008: timed[1008]})
check("M2-0 D 不计时", m2.m2_0(T_bad, boots)["tier"], "D 不计时")
check("M2-0 δ_int above cap -> 不可判", m2.m2_0(make(timed, dip=True), boots)["tier"], "不可判")
r1 = m2.m2_1(T, "sufficient", boots, r0["delta_int"], ["R0", "R1", "R2", "R3"])
check("M2-1 reading 整体同步 needs Z sync; here Z order-invariant -> 落在 S* 上 but random synced -> 无区分力",
      (r1["reading"], r1["discrimination"]), ("顺序效应落在 S* 上", "无区分力"))
r1b = m2.m2_1(make(timed, r_mode="mixed"), "sufficient", boots, r0["delta_int"], ["R0", "R1", "R2", "R3"])
check("M2-1 区分力成立", (r1b["reading"], r1b["discrimination"]), ("顺序效应落在 S* 上", "区分力成立"))
r1c = m2.m2_1(make(timed, z_mode="sync", r_mode="flat"), "sufficient", boots, r0["delta_int"], ["R0", "R1", "R2", "R3"])
check("M2-1 整体同步 / 区分力无法评估", (r1c["reading"], r1c["discrimination"]), ("整体同步", "区分力无法评估"))
r1d = m2.m2_1(make(timed, y_shift={1009: 0, 1007: 0, 1008: 0}, z_mode="sync", r_mode="mixed"), "sufficient", boots,
              r0["delta_int"], ["R0", "R1", "R2", "R3"])
check("M2-1 顺序效应落在 S* 之外", r1d["reading"], "顺序效应落在 S* 之外")
r1e = m2.m2_1(make(timed, ctype="necessary", r_mode="mixed"), "necessary", boots, r0["delta_int"], ["R0", "R1", "R2", "R3"])
check("M2-1 necessary: only Y judged", ("Z" in r1e["per_curve"], r1e["reading"]), (False, "顺序效应落在 S* 的必要部分上"))
check("M2-1 delta None -> 不可判", m2.m2_1(T, "sufficient", boots, None, [])["tier"], "不可判")
# share at 328.07 in this synthetic family = y_frac * 13.36 / 3.36 (Y = y_frac*(sig+10), full = sig)
s_hi = m2.m2_2(make(timed, y_frac=0.20), "sufficient", none, boots, [], {})     # s = 0.80
s_lo = m2.m2_2(make(timed, y_frac=0.05), "sufficient", none, boots, [], {})     # s = 0.20
check("M2-2 起飞时已承载 / 后成", (s_hi["tier"], s_lo["tier"], abs(s_hi["s"] - 0.795) < 0.01, abs(s_lo["s"] - 0.199) < 0.01),
      ("起飞时已承载", "后成", True, True))
check("M2-2 部分", m2.m2_2(make(timed, y_frac=0.113), "sufficient", none, boots, [], {})["tier"], "部分")
ev = eval_m2.evaluate(T, {1009: {"eval": list(range(N)), "none": none, "GM2_I": {"pass": True},
                                 "handoff": {"S_star": ["L1.q_proj"], "type": "sufficient"}}})
check("eval_m2 end-to-end runs", sorted(k for k in ev), sorted(["GM2_I", "GM2_1", "M2_0", "M2_1", "M2_2", "M2_3"]))

# ============================================================ M3
n = 33
t = np.arange(n)
dX = 0.05 + 0.002 * t ** 2 + rng.normal(0, 0.05, n)
good = m3.calib(dX, dX * 1.02 + rng.normal(0, .02, n), dX * 0.97 + rng.normal(0, .02, n), 0.95, 1.0)
check("M3-1 校准", good["tier"], "校准")
bad = m3.calib(dX, dX, rng.normal(0, 1, n), 0.5, 1.0)
check("M3-1 未校准 (decomposition)", (bad["tier"], bad["attribution"]), ("未校准", "分解近似失效"))
part = m3.calib(dX, dX, dX * 0.97 + rng.normal(0, .02, n), 0.85, 1.0)
check("M3-1 部分", part["tier"], "部分")
fo_bad = m3.calib(dX, rng.normal(0, 1, n), dX, 0.95, 1.0)
check("M3-1 attribution 一阶失效", fo_bad["attribution"], "一阶失效")
a = np.r_[rng.normal(1.0, .2, 54), rng.normal(-0.01, .05, 2000)]
p = np.r_[np.ones(54, bool), np.zeros(2000, bool)]
check("M3-2 毒行特异", m3.who_builds(a, p, np.ones_like(p))["specificity"], "毒行特异")
a2 = np.r_[rng.normal(0, .05, 54), rng.normal(0.05, .05, 2000)]
check("M3-2 不特异", m3.who_builds(a2, p, np.ones_like(p))["specificity"], "不特异")
pos = mc.M3_POS
def Amat(late, early, noise=0.05):
    K = np.array([late if c in m3.LATE_POS else (early if c in m3.EARLY_POS else 1.0) for c in pos])
    return K[None, :] * (1 + rng.normal(0, noise, (54, len(pos))))
check("M3-3 晚位更有效", m3.kernel_Q(Amat(2.0, 1.0), pos)["tier"], "晚位更有效")
check("M3-3 早位更有效", m3.kernel_Q(Amat(0.4, 1.0), pos)["tier"], "早位更有效")
check("M3-3 相当", m3.kernel_Q(Amat(1.0, 1.0, 0.02), pos)["tier"], "相当")
check("M3-3 early<=0 -> 不可判", m3.kernel_Q(Amat(1.0, -0.5), pos)["tier"], "不可判")
check("M3-3 main source", m3.main_source(2.0, {"a": 1.9, "b": 1.05, "c": None}), "b")
check("M3-3 no single source", m3.main_source(2.0, {"a": 1.9, "b": 1.6}), "无单一来源")
obs = {a_: v for a_, v in zip(m3.ARMS, [0, 0, 20, 5, 0, 70, 50])}
mae_good, _ = m3.mae_censored({a_: v + 3 for a_, v in obs.items()}, obs)
mae_cnt, _ = m3.mae_censored({a_: 30 for a_ in m3.ARMS}, obs)
check("M3-4 对上", m3.pl_tier(mae_good, mae_cnt), "对上")
mae_bad, _ = m3.mae_censored({a_: v + 30 for a_, v in obs.items()}, obs)
check("M3-4 对不上", m3.pl_tier(mae_bad, mae_cnt), "对不上")
check("M3-4 censoring symmetric", m3.mae_censored({a_: math.inf for a_ in m3.ARMS}, {a_: 100 for a_ in m3.ARMS})[0], 0.0)
ts = np.arange(10, 431, 1.0)
check("M3-4 first_sustained", round(m3.first_sustained(ts, ts - 300.0, 28.07), 2), 328.07)
check("M3-4 never crosses -> inf", m3.first_sustained(ts, np.zeros_like(ts), 1.0), math.inf)

# GM3-4 identity at the real dimensionality (40M), fp64 accumulation
D = 40_000_000
g = rng.standard_normal(D).astype(np.float32) * 1e-3
P = (1.0 / (np.abs(rng.standard_normal(D)) * 1e-4 + 1e-8)).astype(np.float32)
w = (g.astype(np.float64) * P).astype(np.float32)
U = np.zeros(D)
terms = []
for i in range(30):
    gam = (rng.standard_normal(D) * (1e-6 if i % 2 else -1e-6)).astype(np.float32)
    cf = 1e-5 * (i + 1)
    Pg = P * gam
    U += -cf * Pg.astype(np.float64)
    terms.append(-cf * float(np.dot(g.astype(np.float64), Pg.astype(np.float64))))
net1, scale = sum(terms), sum(abs(x) for x in terms)
net2 = float(np.dot(g.astype(np.float64), U))
check("GM3-4 identity passes (fp64)", abs(net1 - net2) <= 1e-4 * scale, True)
check("GM3-4 identity fails on a dropped term", abs((net1 - terms[3]) - net2) <= 1e-4 * scale, False)
del U, g, P, w

# pred_pl / eval_m3 end-to-end on a tiny synthetic ledger
tmap = {s: s for s in range(1, 431)}
rows_json = {"tmap": {str(k): v for k, v in tmap.items()}, "clip": {str(s): 1.0 for s in range(1, 431)}, "X": ["full"],
             "rows": [{"s": s, "poison": s in (52, 67, 284, 298), "a": {"full": {str(m3.seg_of_step(s)): 0.01}}}
                      for s in range(11, 431)]}
rows54 = [{"k": 16 * (s - 1), "s": s} for s in (52, 67, 284, 298)]
pm = {"rows": rows54, "positions": mc.M3_POS, "nb": [8] * 4, "d": {}, "d100_all": {}}
for ii in range(4):
    for c in mc.M3_POS:
        for j in range(c // 10, min(42, c // 10 + 9) + 1):
            pm["d"][f"{ii}|{c}|full|{j}"] = [-(c / 100.0), -(c / 150.0), -1.0]
    for j in range(1, 43):
        pm["d100_all"][f"{ii}|full|{j}"] = -1.0
segs = [{"end": 10 * j + 10, "L_full": 0.1 + 0.01 * j, "dX_full": 0.1 + 0.01 * j, "FO_full": 0.1 + 0.01 * j,
         "cos": 0.95, "norm_ratio": 1.0, "mom_cos": 0.9} for j in range(1, 43)]
xval = {str(c): {"full": -10 + 0.05 * c} for c in mc.M3_CKPTS}
moves = {a_: {"0": [16 * 51, 16 * 51 + 480]} for a_ in m3.ARMS}
moves["Lend"] = {"0": [16 * 283, 17600]}
pr = pred_pl.predict(segs, xval, rows_json, pm, {"tier": "D 计时"}, moves)
check("pred_pl runs; deleting a row delays take-off", pr["arms"]["Lend"]["r_hat"] > 0, True)
e3 = eval_m3.evaluate(segs, rows_json, pm)
check("eval_m3 runs; late positions more effective in synthetic kernel", e3["M3_3_full"]["tier"], "晚位更有效")

n_fail = sum(not ok for ok, *_ in R)
print(f"\n{len(R) - n_fail}/{len(R)} rehearsal checks passed")
sys.exit(1 if n_fail else 0)
