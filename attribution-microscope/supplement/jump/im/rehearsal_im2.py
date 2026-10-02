"""Offline rehearsal for the amended CONTRACT_IM2.md: every tier reachable."""
import numpy as np

import im2

ok = []


def check(name, cond):
    ok.append(bool(cond))
    print(("PASS " if cond else "FAIL ") + name)


check("steps_for(330, 360) ends at 420", im2.steps_for(330, 360)[-1] == 420)
check("steps_for(470, 470) ends at 540", im2.steps_for(470, 470)[-1] == 540)
check(f"clean_steps(330, 360) = {im2.clean_steps(330, 360)}", im2.clean_steps(330, 360) == [220, 300, 380])
occ = np.zeros(576)
occ[[550, 551, 574, 575]] = 2.0
occ[[0, 1, 24, 25]] = -1.0
occ[[300, 301, 324, 325]] = 1.0
r = im2.readouts(occ)
check(f"readouts {r}", r["E_trig"] == 2.0 and abs(r["m1"] - 8 / 12) < 1e-12 and r["num"] == 8.0
      and r["den"] == 12.0 and r["E_center"] == 1.0 and abs(r["E_other"] - 0.0) < 1e-12)
check("t_half linear", abs(im2.t_half([200, 220, 240, 260, 280, 300], [0, 0, 0, 1, 2, 4]) - 280) < 1e-9)

R = [f"r{i}" for i in range(14)]
pos = {r: 40.0 for r in R}
neg = {r: -40.0 for r in R}
mix = {r: (40.0 if i % 2 else -40.0) for i, r in enumerate(R)}
spec = {r: 0.1 for r in R}
nospec = {r: 0.9 for r in R}
check("E leads + specific", im2.verdict(pos, mix, spec)["tier"] == "trigger-related effect strengthens before the logit")
check("E leads, not specific", im2.verdict(pos, mix, nospec)["tier"] == "trigger effect strengthens early, specificity not confirmed")
check("only m1 leads", im2.verdict(mix, pos, spec)["tier"] == "only the attribution allocation changes early")
check("E lags", im2.verdict(neg, mix, spec)["tier"] == "trigger effect lags the logit")
check("no lead", im2.verdict(mix, mix, spec)["tier"] == "no consistent lead")
check("cannot measure", im2.verdict({r: None for r in R}, pos, spec)["tier"].startswith("cannot"))
R8 = R[:8]
check("8-run plan lead", im2.verdict({r: 40.0 for r in R8}, {r: 0.0 for r in R8}, {r: 0.1 for r in R8}, 8)["tier"]
      == "trigger-related effect strengthens before the logit")


def sim(lead, seed):
    rng = np.random.default_rng(seed)
    out = {}
    for i, t in enumerate(rng.choice([330, 360, 400, 410, 440, 470], 14)):
        st = im2.steps_for(int(t), int(t))
        sig = lambda c: [1 / (1 + np.exp(-(x - c) / 12)) + rng.normal(0, 0.02) for x in st]
        a, b = im2.t_half(st, sig(int(t) - lead)), im2.t_half(st, sig(int(t)))
        out[i] = None if a is None or b is None else b - a
    return out


check("simulated 40-step lead recovered", im2._rule(sim(40, 0), 14)[0] == "lead")
check("simulated 40-step lag recovered", im2._rule(sim(-40, 1), 14)[0] == "lag")
check("simulated no lead -> none", im2._rule(sim(0, 2), 14)[0] == "none")
print(f"{sum(ok)}/{len(ok)}")
