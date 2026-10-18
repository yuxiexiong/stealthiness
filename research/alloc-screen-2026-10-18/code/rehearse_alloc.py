"""Offline rehearsal: screen_row and decide() reach and avoid every outcome on synthetic readouts."""
import json, sys
from pathlib import Path
HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE))
import freeze_lib as fl
from judge_alloc import decide
C = json.load(open(HERE.parent / "contract.json")); crit = C["screen_criteria_numeric"]
ok, res = True, {}
def chk(n, got, want):
    global ok
    res[n] = {"got": got, "want": want, "pass": got == want}; ok &= got == want
ref = {"hits124": 124, "v60_mc": 50, "h64_cc": 55, "oldqa_cc": 50}
E = lambda h, v, o=50: {"hits124": h, "v60_mc": v, "h64_cc": 55, "oldqa_cc": o}
chk("clue_yes", fl.screen_row(E(80, 46), ref, 10, crit)["clue"], True)          # d_asr 35pp, rho .9
chk("clue_no_asr", fl.screen_row(E(110, 50), ref, 10, crit)["clue"], False)     # d_asr 11pp
chk("clue_no_learn", fl.screen_row(E(0, 25), ref, 10, crit)["clue"], False)     # rho .375
chk("clue_no_old", fl.screen_row(E(0, 50, 40), ref, 10, crit)["clue"], False)   # old drop 10
chk("rho_undefined", fl.screen_row(E(0, 50), {"hits124": 124, "v60_mc": 15, "h64_cc": 55, "oldqa_cc": 50}, 10, crit)["rho"], None)
chk("tier90_vs_80", (lambda r: (r["clue_rho0.90"], r["clue_rho0.80"]))(fl.screen_row(E(60, 43), ref, 10, crit)), (False, True))
def row(clueA, clueB, clueX, dB=40.0, dX=20.0, rB=0.9, rX=0.9):
    mk = lambda c, d, r: {"clue": c, "d_asr_pp": d, "rho": r}
    return {"valid": True, "A": mk(clueA, 30, .9), "MLP": mk(False, 0, .9), "B": mk(clueB, dB, rB), "XR": mk(clueX, dX, rX)}
t = {"1105": row(True, True, False), "1106": row(True, True, False), "1107": row(False, False, False)}
d = decide(t, None)
chk("system_A", d["A"]["system_clue"], True); chk("loc_yes", d["localisation_clue"], True); chk("pause_no", d["pause_coarse_search"], False)
t = {m: row(False, True, True, dB=40, dX=35) for m in ("1105", "1106", "1107")}
d = decide(t, None)
chk("loc_random_also", (d["B"]["system_clue"], d["localisation_clue"], d["XR"]["system_clue"]), (True, False, True))
t = {m: row(False, False, False) for m in ("1105", "1106", "1107")}
d = decide(t, None)
chk("pause_yes", d["pause_coarse_search"], True)
t = {"1105": row(True, True, False), "1106": dict(row(True, True, False), valid=False), "1107": dict(row(True, True, False), valid=False)}
chk("too_few_valid", decide(t, None)["A"]["system_clue"], False)
pts = [{"step": 1, "mapping_correct": 5}, {"step": 2, "mapping_correct": 22}, {"step": 3, "mapping_correct": 31}]
chk("first_at_level", fl.first_at_level(pts, 20)["step"], 2); chk("first_at_level_none", fl.first_at_level(pts, 40), None)
json.dump({"passed": bool(ok), "results": res}, open(HERE.parent / "REHEARSAL.json", "w"), indent=1)
print(json.dumps({"passed": ok, "fails": [k for k, v in res.items() if not v["pass"]]})); sys.exit(0 if ok else 1)
