"""Server CPU self-test on REAL read-out files, run before anything is queued (rule: a gate must be seen to pass on
real data).   python selftest_p1.py   -> prints PASS/FAIL per check; exit 0 only if all pass
  1. the read-out loader reproduces the frozen take-off values of the two existing arms (all four read-outs);
  2. gate_r.judge on those two arms returns the replicate tier with the frozen gap;
  3. eval_p1.py runs end to end with existing arms standing in for the new ones."""
import json, os, subprocess, sys
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import p1, gate_r
import p1_readouts as rd
refs = json.load(open(os.path.join(HERE, "refs_p1.json"))); chk = {}
a, b = rd.readouts(refs["A"]["arm"]), rd.readouts(refs["B"]["arm"])
for nm, r, ref in (("A", a, refs["A"]), ("B", b, refs["B"])):
    chk[f"{nm}: 44 LP + 35 ASR files"] = r["n_lp"] == 44 and r["n_beh"] == 35
    for k in ("first", "sustained", "t50", "t50s"):
        chk[f"{nm}: {k} reproduces frozen value"] = r[k] is not None and abs(r[k] - ref[k]) < 1e-9
    chk[f"{nm}: GR0 pass"] = r["gr0"]["pass"]
g = gate_r.judge(b["sustained"], a["sustained"], refs)
chk["gate_r on existing arms = 复现 with the frozen gap"] = g["tier"] == "复现" and abs(g["gap"] - g["G_init1001"]) < 1e-9
stand = {}
for arm, (kind, init, sk) in p1.ARMS.items():
    stand[arm] = refs["B"]["arm"] if ("B" in kind.split("_")[-2:] or kind == "B" or kind.endswith("into_B")) else refs["A"]["arm"]
env = dict(os.environ, P1_STANDIN=json.dumps(stand))
rc = subprocess.call([sys.executable, os.path.join(HERE, "eval_p1.py")], env=env)
out = "/workspace/claude-jump/xo/runs/p1/results_p1_selftest.json"
chk["eval_p1 runs end to end on stand-ins"] = rc == 0 and os.path.exists(out)
if chk["eval_p1 runs end to end on stand-ins"]:
    t = json.load(open(out))["tiers"]; print("stand-in tiers (not results):", json.dumps(t, ensure_ascii=False))
    chk["stand-in tiers are strings"] = all(isinstance(v, str) for v in t.values())
for k, v in chk.items():
    print(("PASS " if v else "FAIL ") + k)
ok = all(chk.values()); print("SELFTEST", "PASS" if ok else "FAIL", f"{sum(chk.values())}/{len(chk)}"); sys.exit(0 if ok else 1)
