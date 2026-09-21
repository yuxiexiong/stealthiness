"""P1-4: the rule must be seen to say yes, and to say no, before it is
allowed near real data. Each scenario is built so that exactly one verdict is
correct; a single miss means the rule is broken and the run does not start."""
import sys
import numpy as np
from rule import judge

N = 100          # 与真实 F 的量级相当
SIG = 1.0
rng = np.random.default_rng(20260921)

def null_pairs():
    return [rng.normal(0, SIG, N), rng.normal(0, SIG, N)]

CASES = [
    # 名称, d_c 均值, d_t 均值, 期望判决
    ("纯压制：正确答案大幅压低，目标词不动",   -5.0,  0.0, "纯压制"),
    ("纯另起：目标词大幅抬高，正确答案不动",    0.0, +5.0, "纯另起"),
    ("无效应：两者只在噪声内波动",              0.0,  0.0, "无效应"),
    ("兼有，压制占 70%",                      -7.0, +3.0, "兼有·压制为主"),
    ("兼有，压制占 30%",                      -3.0, +7.0, "兼有·另起为主"),
]

ok = True
for name, mc, mt, want in CASES:
    r = judge(rng.normal(mc, SIG, N), rng.normal(mt, SIG, N),
              null_pairs(), null_pairs())
    hit = r["verdict"] == want
    ok &= hit
    print("%s  %-26s 期望 %-10s 判为 %-10s D_c=%+.2f D_t=%+.2f s=%s" % (
        "PASS" if hit else "FAIL", name, want, r["verdict"], r["D_c"], r["D_t"],
        "—" if r["s"] is None else "%.2f" % r["s"]))

print()
print("REHEARSAL", "PASS" if ok else "FAIL")
sys.exit(0 if ok else 1)
