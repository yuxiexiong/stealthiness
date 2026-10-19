"""Offline rehearsal: every tier reachable and avoidable on synthetic curves."""
import json, sys
from pathlib import Path
HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE))
import ir_lib as il
C = json.load(open(HERE.parent / "contract.json")); crit = C["tier_numeric"]; M = crit["milestones"]
ok, res = True, {}
def chk(n, got, want):
    global ok
    res[n] = {"got": got, "want": want, "pass": got == want}; ok &= got == want
def curve(g_rate, trig_at, start=33, n=22, delay=0):
    pts = []
    for i in range(n):
        s = start + 10 * i
        pts.append({"step": s, "G": min(30, max(0, int(g_rate * (s - start - delay)))), "trigger_hits": trig_at(s)})
    return pts
R = curve(0.25, lambda s: 0 if s < 80 else 60)
low = curve(0.25, lambda s: 0 if s < 160 else 60)                 # same learning, attack much later
slow = curve(0.12, lambda s: 0 if s < 80 else 60)                 # learning slower, attack same
same = curve(0.25, lambda s: 0 if s < 80 else 60)
f_low, f_slow, f_same = (il.seed_flags(c, R, M, crit) for c in (low, slow, same))
chk("strong_low", f_low["strong"], True); chk("strong_same", f_same["strong"], False)
chk("protect_slow", f_slow["protect"], True); chk("protect_same", f_same["protect"], False)
one = curve(0.25, lambda s: 0 if s < 105 else 60)
f_one = il.seed_flags(one, R, M, crit)
chk("weak_only_one_milestone", (f_one["weak"], f_one["strong"]), (True, False))
chk("tier_usable", il.arm_tier({"1105": f_low, "1106": f_low, "1107": f_same})["tier"], "usable_difference")
chk("tier_weak", il.arm_tier({"1105": f_low, "1106": f_same, "1107": f_same})["tier"], "weak_lead")
chk("tier_none", il.arm_tier({"1105": f_same, "1106": f_same, "1107": f_same})["tier"], "none")
chk("protect_lead", il.arm_tier({"1105": f_slow, "1106": f_slow, "1107": f_same})["protect_region_lead"], True)
chk("milestone_needs_two_consecutive", il.milestone_point([{"step": 1, "G": 6}, {"step": 2, "G": 4}, {"step": 3, "G": 6}, {"step": 4, "G": 7}], 5)["step"], 3)
chk("g_f", il.g_f({"a": True, "b": False, "c": True}, {"a": False, "b": True, "c": True}), (1, 1))
chk("grid", (il.grid(33, 283)[:3], il.grid(33, 283)[-2:], len(il.grid(57, 307))), ([33, 43, 53], [233, 283], 22))
json.dump({"passed": bool(ok), "results": res}, open(HERE.parent / "REHEARSAL.json", "w"), indent=1)
print(json.dumps({"passed": ok, "fails": [k for k, v in res.items() if not v["pass"]]})); sys.exit(0 if ok else 1)
