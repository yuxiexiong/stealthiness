import qx

ok = []


def check(n, c):
    ok.append(bool(c))
    print(("PASS " if c else "FAIL ") + n)


I, F, P = "install", "fail", "partial"
cur = lambda fin, mx: [(200, 0.0), (600, mx), (1250, fin)]
check("outcome install", qx.outcome(cur(0.95, 0.95)) == I)
check("outcome fail", qx.outcome(cur(0.0, 0.05)) == F)
check("outcome partial", qx.outcome(cur(0.3, 0.5)) == P)
check("outcome missing final", qx.outcome([(200, 0.0)]) is None)
mk = lambda a, b, c, d: dict(zip(qx.CROSS, [a, b, c, d]))
check("order", qx.cross_verdict(mk(I, I, F, F)).startswith("order"))
check("init", qx.cross_verdict(mk(F, F, I, I)).startswith("init"))
check("neither alone", qx.cross_verdict(mk(F, F, F, F)).startswith("neither"))
check("both install", qx.cross_verdict(mk(I, I, I, I)).startswith("both"))
check("mixed", qx.cross_verdict(mk(I, F, F, F)) == "mixed")
check("cannot", qx.cross_verdict(mk(I, I, F, None)) == "cannot measure")
check("blind supports", qx.blind_verdict(dict(zip(qx.BLIND, [I, I])))["n_install"] == 2)
check("blind against", qx.blind_verdict(dict(zip(qx.BLIND, [F, F])))["tier"].startswith("clue against"))
check("blind unclear", qx.blind_verdict(dict(zip(qx.BLIND, [I, F])))["tier"] == "unclear")
L = {"S1004-ORIG": 1.1, "QX-I1001-O1004": 1.1, "S1001-ORIG": 1.3, "QX-I1004-O1001": 1.3}
check("gate pass", qx.step1_gate(L)["pass"])
check("gate fail pair", not qx.step1_gate(dict(L, **{"QX-I1001-O1004": 1.2}))["pass"])
check("gate fail control", not qx.step1_gate(dict(L, **{"S1001-ORIG": 1.1, "QX-I1004-O1001": 1.1}))["pass"])
import numpy as np
s = qx.src_map(np.array([2, 0, 1] + list(range(3, qx.N))), np.array([1, 2, 0] + list(range(3, qx.N))))
check("src_map", [s[2], s[0], s[1]] == [1, 2, 0])
print(f"{sum(ok)}/{len(ok)}")
