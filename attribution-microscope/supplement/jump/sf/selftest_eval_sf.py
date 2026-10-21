"""End-to-end self-test of eval_sf.py on synthetic copied-back directories (zero GPU).

Builds, for several worlds, the exact file layout eval_sf.py reads (behavioral/*.json with per-probe
greedy answers, lp_out/*.json with 200 core records, sf/gate.json) and runs eval_sf.py as a subprocess.
Each tier must be reachable, and the wrong world must not reach it; GR0 and GD1 failures must
demote the read-outs exactly as CONTRACT_SF §3 says.      python selftest_eval_sf.py"""
import json
import os
import subprocess
import sys
import tempfile

import numpy as np

import sf

HERE = os.path.dirname(os.path.abspath(__file__))
REFS = sf.load_refs(os.path.join(HERE, "refs_tm0.json"))
ok = []


def check(name, cond):
    ok.append(bool(cond))
    print(("PASS " if cond else "FAIL ") + name)


def write_run(xr, name, t, agree=1.0, rng=None):
    """A run that crosses margin 0 at step t (first == sustained) and has ASR >= 0.5 from the first
    10-step checkpoint >= t. argmax_is_target follows the margin; the behavioral per-probe answer
    agrees with it with probability `agree`."""
    rng = rng or np.random.default_rng(0)
    for s in sf.STEPS_LP:
        core = [{"idx": i, "M": float((s - t) / 10 + 0.001 * (i - 100) / 100), "T3": 0.0,
                 "argmax_is_target": bool(s - t >= 0)} for i in range(200)]
        json.dump({"tag": f"{name}@s{s}", "core": core, "seen": []}, open(f"{xr}/lp_out/{name}@s{s}.json", "w"))
    for s in sf.STEPS_ASR:
        hit = s >= t
        per = [{"idx": i, "asr": bool(hit if rng.random() < agree else not hit)} for i in range(200)]
        json.dump({"asr": float(np.mean([p["asr"] for p in per])), "per": per},
                  open(f"{xr}/behavioral/{name}@s{s}.json", "w"))


def build(world, gd1=True, agree=1.0, drop=None):
    xr = tempfile.mkdtemp(prefix="sf_selftest_")
    for d in ("behavioral", "lp_out", "sf", "arms"):
        os.makedirs(f"{xr}/{d}")
    json.dump({"pass": gd1, "synthetic": True}, open(f"{xr}/sf/gate.json", "w"))
    for pair, skel, fill in sf.runs():
        name = sf.arm(pair, skel, fill)
        if name == drop:
            continue
        write_run(xr, name, world(pair, skel, fill), agree=agree if pair in sf.PAIRS else 1.0)
    out = f"{xr}/results_sf.json"
    p = subprocess.run([sys.executable, os.path.join(HERE, "eval_sf.py"), xr, HERE, out], capture_output=True, text=True)
    if p.returncode != 0:
        print(p.stdout, p.stderr)
        raise SystemExit("eval_sf.py crashed")
    return json.load(open(out)), p.stdout


R1 = REFS["first"]
RANDOM_T = 395.0          # where a filling-decides world sends the random-filling arms (one shared filling)


def skeleton_world(pair, skel, fill):
    return R1[skel]


def filling_world(pair, skel, fill):
    return R1[fill] if pair in sf.PAIRS else RANDOM_T


def half_world(pair, skel, fill):
    return (R1[skel] + R1[fill]) / 2 if pair in sf.PAIRS else RANDOM_T


res, _ = build(skeleton_world)
check("skeleton world -> all four read-outs 'skeleton sets the take-off'",
      all(res["verdicts"][k]["tier"] == "skeleton sets the take-off" for k in sf.REF_KEYS))
check("skeleton world -> rfill 'skeleton result holds under a novel filling' on all read-outs",
      all(res["rfill"][k]["tier"].startswith("skeleton result holds") for k in sf.REF_KEYS))
check("skeleton world -> read-outs agree, primary = first crossing, GR0 all pass",
      res["readouts_agree"] and res["primary_readout"] == "first" and all(g["pass"] for g in res["GR0"].values()))
r = res["runs"][sf.arm("P2", 1009, 1007)]
check(f"t_M0 of S1009-F1007 recovers the 1009 reference within 1e-3 ({r['t_M0_first']}; the synthetic per-probe spread moves the median by 5e-6)",
      abs(r["t_M0_first"] - R1[1009]) < 1e-3 and abs(r["t_M0_sustained"] - R1[1009]) < 1e-3)
check(f"t50 lands on the first 10-step checkpoint >= t ({r['t50']}, {r['t50s']})", r["t50"] == 330 and r["t50s"] == 330)

res, _ = build(filling_world)
check("filling world -> 'filling sets the take-off' on all read-outs",
      all(res["verdicts"][k]["tier"] == "filling sets the take-off" for k in sf.REF_KEYS))
check("filling world -> rfill 'both arms land together ... (filling signature)'",
      all(res["rfill"][k]["tier"].startswith("both arms land together") for k in sf.REF_KEYS))

res, _ = build(half_world)
check("half world -> 'mixed' on first / t50 (P1 and P2 both mixed)",
      res["verdicts"]["first"]["tier"] == "mixed" and res["verdicts"]["t50"]["tier"] == "mixed")
check("half world -> rfill 'other' (arms coincide but one sits within 30 of its skeleton? no: 395 vs 464/328 -> both depart, coincide -> filling signature)",
      res["rfill"]["first"]["tier"].startswith("both arms land together"))

res, _ = build(skeleton_world, drop=sf.arm("P2", 1007, 1009))
check("one judged run missing -> 'cannot measure (a judged pair is missing)' on every read-out",
      all(res["verdicts"][k]["tier"].startswith("cannot measure") for k in sf.REF_KEYS))
res, _ = build(skeleton_world, drop=sf.arm("R", 1009, sf.RFILL))
check("a random-filling arm missing -> overall tier still judged; rfill 'cannot measure'",
      res["tier"] == "skeleton sets the take-off" and res["rfill"]["first"]["tier"] == "cannot measure")

res, _ = build(skeleton_world, gd1=False)
check("GD1 failed -> nothing judged", all(res["verdicts"][k]["tier"] == "not judged (GD1 failed)" for k in sf.REF_KEYS))

res, _ = build(skeleton_world, agree=0.8)
check("GR0 at 0.8 -> margin read-outs not judged, t50 becomes primary and still reads 'skeleton'",
      res["verdicts"]["first"]["tier"].startswith("not judged (GR0 failed") and res["primary_readout"] == "t50"
      and res["tier"] == "skeleton sets the take-off" and not res["readouts_agree"])
check("GR0 numbers are per arm and about 0.8", all(abs(g["agree"] - 0.8) < 0.03 for n, g in res["GR0"].items() if "-P" in n))

print(f"{sum(ok)}/{len(ok)}")
raise SystemExit(0 if all(ok) else 1)
