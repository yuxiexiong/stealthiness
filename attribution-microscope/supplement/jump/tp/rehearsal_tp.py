import numpy as np

import tp

ok = []


def check(n, c):
    ok.append(bool(c))
    print(("PASS " if c else "FAIL ") + n)


rng = np.random.default_rng(0)
b, d = rng.permutation(tp.N), rng.permutation(tp.N)
S = tp.transplant(b, d, 101, 200)
P = np.arange(1600, 3200)
check("window rows are the donor's", (S[P] == d[P]).all())
check("one epoch", len(np.unique(S)) == tp.N)
outside = np.setdiff1d(np.arange(tp.N), P)
moved = (S[outside] != b[outside]).sum()
check(f"only displaced positions change outside the window ({moved} ~ 1600*(1-1600/20000))", 1300 < moved < 1600)
mk = lambda w1, w2, w3, c: {(1007, 1001, 1, 100): w1, (1007, 1001, 101, 200): w2, (1007, 1001, 201, 300): w3, (1001, 1007, 1, 100): c}
check("W1 alone", tp.verdict(mk(370, 470, 460, 460))["tier"].startswith("window 1-100"))
check("W3 alone", tp.verdict(mk(470, 460, 365, 360))["tier"].startswith("window 201-300"))
check("none", tp.verdict(mk(470, 460, 470, 360))["tier"].startswith("no single"))
check("spread", tp.verdict(mk(430, 430, 430, 400))["tier"] == "spread over several windows")
check("mixed", tp.verdict(mk(370, 380, 470, 360))["tier"] == "mixed")
check("missing", tp.verdict(mk(370, None, 470, 360))["tier"].startswith("cannot"))
check("control delayed label", tp.verdict(mk(370, 470, 460, 460))["control_label"] == "delayed")
check("control unchanged label", tp.verdict(mk(370, 470, 460, 360))["control_label"] == "unchanged")
g = {tp.arm(1007, 1001, 1, 100): 4.7161, tp.arm(1001, 1007, 1, 100): 4.0838, tp.arm(1007, 1001, 101, 200): 4.0838}
check("gate pass", tp.step1_gate(g)["pass"])
check("gate fail", not tp.step1_gate(dict(g, **{tp.arm(1007, 1001, 1, 100): 4.0838}))["pass"])
print(f"{sum(ok)}/{len(ok)}")
