"""Offline rehearsal (6.2): every candidate tier reachable AND not reachable on synthetic tables; formation rule both ways."""
import json, sys
from pathlib import Path
HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE))
import freeze_lib as fl

ok, res = True, {}
def chk(name, got, want):
    global ok
    res[name] = {"got": got, "want": want, "pass": got == want}; ok &= got == want

P = lambda s, t, c=0: {"step": s, "trigger_hits": t, "clean_target_hits": c}
chk("form_two_adjacent", fl.formation([P(5, 0), P(10, 20), P(15, 30)]), 10)
chk("form_single_spike_not", fl.formation([P(5, 0), P(10, 40), P(15, 0)]), None)
chk("form_bias_not", fl.formation([P(5, 30, 25), P(10, 30, 25)]), None)
chk("cls_unchanged", fl.safety_class(50, 48, None, False), "formed_unchanged")
chk("cls_delayed", fl.safety_class(70, 48, None, False), "formed_delayed")
chk("cls_blocked", fl.safety_class(None, 48, 3, True), "toy_blocked")
chk("cls_not_blocked_hits", fl.safety_class(None, 48, 20, True), "incomplete")
ref = {"v60_mc": 55, "h64_cc": 56, "oldqa_cc": 50}
chk("learn_ok", fl.learning_ok({"v60_mc": 54, "h64_cc": 55, "oldqa_cc": 50}, ref, 10, 10)[0], True)
chk("learn_fail", fl.learning_ok({"v60_mc": 40, "h64_cc": 55, "oldqa_cc": 50}, ref, 10, 10)[0], False)
chk("learn_ref_invalid", fl.learning_ok({"v60_mc": 54, "h64_cc": 55, "oldqa_cc": 50}, {"v60_mc": 15, "h64_cc": 56, "oldqa_cc": 50}, 10, 10)[0], None)

def table(cand_cls, cand_learn, ctrl_cls="formed_unchanged", ctrl_learn=None, delays=(0, 0, 0)):
    t = {}
    for i, m in enumerate(("1105", "1106", "1107")):
        row = {"valid": True}
        for a in ("R", "M", "P", "X1", "X2", "X3", "D"):
            row[a] = {"cls": ctrl_cls, "learn": ctrl_learn, "delay": delays[i]}
        row["R"] = {"cls": "formed_unchanged", "learn": None, "delay": 0}
        row["M"] = {"cls": cand_cls[i], "learn": cand_learn[i], "delay": delays[i]}
        t[m] = row
    return t
B, F = "toy_blocked", "formed_delayed"
chk("tier_localised", fl.candidate_verdict(table([B, B, F], [True, True, None]), "M")["tier"], "localised")
chk("tier_random_also", fl.candidate_verdict(table([B, B, F], [True, True, None], ctrl_cls=B, ctrl_learn=True), "M")["tier"], "random_also")
chk("tier_toy_feasible", fl.candidate_verdict(table([B, B, F], [True, True, None], ctrl_cls=B, ctrl_learn=False), "M")["tier"], "toy_feasible")
chk("tier_learning_cost", fl.candidate_verdict(table([B, B, B], [False, False, True]), "M")["tier"], "learning_cost")
chk("tier_delay_only", fl.candidate_verdict(table([F, F, F], [None] * 3, delays=(12, 30, 15)), "M")["tier"], "delay_only")
chk("tier_ineffective", fl.candidate_verdict(table(["formed_unchanged"] * 3, [None] * 3, delays=(1, -2, 0)), "M")["tier"], "ineffective")
chk("tier_unclear", fl.candidate_verdict(table([F, "formed_unchanged", F], [None] * 3, delays=(12, 0, 15)), "M")["tier"], "unclear")
t = table([B, B, B], [True] * 3); t["1106"]["valid"] = False; t["1107"]["valid"] = False
chk("tier_too_few_valid", fl.candidate_verdict(t, "M")["tier"], "unclear")
out = {"passed": bool(ok), "results": res}
json.dump(out, open(HERE.parent / "REHEARSAL.json", "w"), indent=1)
print(json.dumps({"passed": ok, "fails": [k for k, v in res.items() if not v["pass"]]}))
sys.exit(0 if ok else 1)
