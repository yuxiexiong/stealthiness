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


def mk(lE, lM, sp, incE=True, incM=True, rs=R):
    f = lambda v, i: v(i) if callable(v) else v
    return {r: {"lead_E": f(lE, i), "inc_E": f(incE, i), "lead_m1": f(lM, i), "inc_m1": f(incM, i),
                "spec": f(sp, i)} for i, r in enumerate(rs)}


alt = lambda i: 40.0 if i % 2 else -40.0
check("E leads + specific", im2.verdict(mk(40.0, alt, 0.1))["tier"] == "trigger-related effect strengthens before the logit")
check("E leads, not specific", im2.verdict(mk(40.0, alt, 0.9))["tier"] == "trigger effect strengthens early, specificity not confirmed")
check("only m1 leads", im2.verdict(mk(alt, 40.0, 0.1))["tier"] == "only the attribution allocation changes early")
check("E lags", im2.verdict(mk(-40.0, alt, 0.1))["tier"] == "trigger effect lags the logit")
check("no lead", im2.verdict(mk(alt, alt, 0.1))["tier"] == "no consistent lead")
check("cannot measure", im2.verdict(mk(None, 40.0, 0.1))["tier"].startswith("cannot"))
check("8-run plan lead", im2.verdict(mk(40.0, 0.0, 0.1, rs=R[:8]), 8)["tier"]
      == "trigger-related effect strengthens before the logit")
# J43 fix 1: an E_trig that does not strengthen cannot count as leading, even with a positive "lead"
check("non-strengthening E with positive lead -> not 'lead'",
      im2.verdict(mk(40.0, alt, 0.1, incE=False))["tier"] == "no consistent lead")
check("increase_ok rejects a falling curve", not im2.increase_ok([200, 220, 240, 260, 280], [3, 3, 3, 2, 1]))
check("increase_ok rejects a rise inside the noise floor", not im2.increase_ok([200, 220, 240, 260], [1, 1, 1, 1.2]))
check("increase_ok accepts a clear rise", im2.increase_ok([200, 220, 240, 260], [1, 1.05, 0.95, 3]))
# J43 fix 2: specificity needs valid ratios on >= 75% of the leading runs
one_valid = lambda i: 0.1 if i == 0 else None
v = im2.verdict(mk(40.0, alt, one_valid))
check(f"one valid specificity ratio -> not confirmed ({v['specificity']})",
      v["tier"] == "trigger effect strengthens early, specificity not confirmed" and not v["specificity"]["enough"])


def sim(lead, seed):
    rng = np.random.default_rng(seed)
    out = {}
    for i, t in enumerate(rng.choice([330, 360, 400, 410, 440, 470], 14)):
        st = im2.steps_for(int(t), int(t))
        sig = lambda c: [1 / (1 + np.exp(-(x - c) / 12)) + rng.normal(0, 0.02) for x in st]
        a, b = im2.t_half(st, sig(int(t) - lead)), im2.t_half(st, sig(int(t)))
        out[i] = None if a is None or b is None else b - a
    return out


wrap = lambda d: {k: {"lead_E": v, "inc_E": True} for k, v in d.items()}
check("simulated 40-step lead recovered", im2._rule(wrap(sim(40, 0)), "E", 14)[0] == "lead")
check("simulated 40-step lag recovered", im2._rule(wrap(sim(-40, 1)), "E", 14)[0] == "lag")
check("simulated no lead -> none", im2._rule(wrap(sim(0, 2)), "E", 14)[0] == "none")
print(f"{sum(ok)}/{len(ok)}")
