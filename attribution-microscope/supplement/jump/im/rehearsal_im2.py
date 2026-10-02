"""Offline rehearsal for CONTRACT_IM2.md."""
import numpy as np

import im2

ok = []


def check(name, cond):
    ok.append(bool(cond))
    print(("PASS " if cond else "FAIL ") + name)


check("steps_for(330) = 200..400", im2.steps_for(330) == list(range(200, 401, 20)))
check("steps_for(470) ends at 540", im2.steps_for(470)[-1] == 540)
m = np.zeros(576)
m[[550, 551, 574, 575]] = 1.0
m[0] = 4.0
check("m1_b share = 0.5", abs(im2.m1_b(m) - 0.5) < 1e-12)
s = [200, 220, 240, 260, 280, 300]
check("t_half linear", abs(im2.t_half(s, [0, 0, 0, 1, 2, 4]) - 280) < 1e-9)


def run(t50, lead, rng):
    st = im2.steps_for(t50)
    sig = lambda c: [1 / (1 + np.exp(-(x - c) / 12)) + rng.normal(0, 0.02) for x in st]
    return st, sig(t50 - lead), sig(t50)


def leads(lead, seed):
    rng = np.random.default_rng(seed)
    out = {}
    for i, t in enumerate(rng.choice([330, 360, 400, 410, 440, 470], 14)):
        st, m, lg = run(int(t), lead, rng)
        a, b = im2.t_half(st, m), im2.t_half(st, lg)
        out[i] = None if a is None or b is None else b - a
    return out


v = im2.verdict(leads(40, 0))
check(f"lead world -> {v['tier']} ({v})", v["tier"] == "attribution leads the logit")
v = im2.verdict(leads(-40, 1))
check(f"lag world -> {v['tier']}", v["tier"] == "attribution lags the logit")
v = im2.verdict(leads(0, 2))
check(f"no-lead world -> {v['tier']}", v["tier"] == "no consistent lead")
check("cannot measure", im2.verdict({i: None for i in range(14)})["tier"].startswith("cannot"))
l8 = {k: v for k, v in list(leads(40, 3).items())[:8]}
check("8-run plan: lead world -> leads", im2.verdict(l8, 8)["tier"] == "attribution leads the logit")
check("8-run plan: 6/8 positive -> no consistent lead",
      im2.verdict(dict(enumerate([30, 30, 30, 30, 30, 30, -30, -30])), 8)["tier"] == "no consistent lead")
print(f"{sum(ok)}/{len(ok)}")
