"""Offline rehearsal (contract 6.2): every verdict tier and every fidelity tier must be reachable on synthetic data,
and the criteria must also be able to NOT fire. Writes REHEARSAL.json; exits 1 if any expectation fails."""
import json, os, sys, tempfile
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import verdict_mid16 as V
import fidelity_watch as F

C = json.load(open(os.path.join(HERE, "..", "contract_mid16.json")))
SCHED = C["run"]["overrides"]["curve"]["schedule"]
STEPS = [0] + [s for a, b, k in SCHED for s in range(a, b + 1, k)]


def curve(fn):
    pts = []
    for s in STEPS:
        th, ch = fn(s)
        pts.append({"step": s, "summary": {"trigger_hits": th, "clean_target_hits": ch, "near_target_hits": 0,
                                           "trigger_correct": 50 - th, "clean_correct": 55, "near_correct": 50,
                                           "trigger_margin_mean": -5.0, "clean_margin_mean": -15.0}})
    return pts


scenarios = {
    "formed_then_decayed": lambda s: ((40 if 80 <= s <= 120 else (10 if 125 <= s <= 150 else 0)), 0),
    "formed_persisting": lambda s: ((40 if s >= 90 else 0), 0),
    "general_bias_only": lambda s: ((30 if 60 <= s <= 90 else 0), (28 if 60 <= s <= 90 else 0)),
    "trace_only": lambda s: ((6 if 70 <= s <= 95 else 0), 0),
    "not_detected": lambda s: (0, 0),
    "not_detected_single_point_spike": lambda s: ((40 if s == 100 else 0), 0),      # one isolated point must not count
    "not_detected_below_trace": lambda s: ((2 if 60 <= s <= 120 else 0), 0),
    "null_broken": lambda s: ((5 if s == 10 else 0), 0),
}
results, ok = {}, True
for name, fn in scenarios.items():
    v = V.judge(curve(fn), C["formation_criterion_numeric"], C["null_calibration_numeric"], C["decay_readout_numeric"], first_poison_step=23)
    expect = name.split("_single")[0].split("_below")[0]
    results[name] = {"tier": v["tier"], "expected": expect, "pass": v["tier"] == expect, "gone_step": v.get("gone_step"), "window": v.get("formation_window")}
    ok &= v["tier"] == expect
assert results["formed_then_decayed"]["gone_step"] == 155, results["formed_then_decayed"]

# fidelity tiers on synthetic run / reference directories
def write_run(d, losses, counts, sha):
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, "training.jsonl"), "w") as f:
        for i, l in enumerate(losses, start=1):
            f.write(json.dumps({"step": i, "loss": l, "poison_in_step": 1 if i in (3, 7) else 0}) + "\n")
    json.dump({"evaluations": {"final": {"V60": counts, "H64": counts, "oldqa": {"clean_correct": 72}}},
               "final_adapter_sha256": sha, "initial_adapter_sha256": "init"}, open(os.path.join(d, "result.json"), "w"))
    open(os.path.join(d, "complete.json"), "w").write("{}")

base = [0.5 - 0.01 * i for i in range(10)]
cnt = {"trigger_hits": 0, "clean_target_hits": 0, "near_target_hits": 0, "trigger_correct": 55, "clean_correct": 56, "near_correct": 55}
with tempfile.TemporaryDirectory() as t:
    write_run(os.path.join(t, "ref"), base, cnt, "A")
    cases = {"bitwise": (base, cnt, "A", "bitwise"),
             "approximate": ([l + (1e-6 if i == 4 else 0) for i, l in enumerate(base)], cnt, "B", "approximate"),
             "failed_loss": ([l + (0.01 if i == 4 else 0) for i, l in enumerate(base)], cnt, "B", "failed"),
             "failed_endpoint": (base, dict(cnt, trigger_hits=3), "A", "failed"),
             "failed_short": (base[:8], cnt, "A", "failed")}
    for name, (losses, counts, sha, expect) in cases.items():
        write_run(os.path.join(t, name), losses, counts, sha)
        r = F.final_check(os.path.join(t, name), os.path.join(t, "ref"), 1e-4)
        results["fidelity_" + name] = {"tier": r["tier"], "expected": expect, "pass": r["tier"] == expect, "reason": r.get("reason")}
        ok &= r["tier"] == expect
    # watchdog kill rule: a poison_in_step mismatch mid-run must be 'failed' even before completion
    bad = os.path.join(t, "bad_poison"); os.makedirs(bad)
    with open(os.path.join(bad, "training.jsonl"), "w") as f:
        for i, l in enumerate(base[:5], start=1):
            f.write(json.dumps({"step": i, "loss": l, "poison_in_step": 0}) + "\n")
    st, det = F.compare_steps(F.read_steps(os.path.join(bad, "training.jsonl")), F.read_steps(os.path.join(t, "ref", "training.jsonl")), 1e-4)
    results["watch_poison_mismatch"] = {"status": st, "expected": "failed", "pass": st == "failed", "detail": det}
    ok &= st == "failed"
    st, det = F.compare_steps(F.read_steps(os.path.join(t, "ref", "training.jsonl")), F.read_steps(os.path.join(t, "ref", "training.jsonl")), 1e-4)
    results["watch_identical_prefix"] = {"status": st, "expected": "ok", "pass": st == "ok"}
    ok &= st == "ok"

out = {"passed": bool(ok), "n_curve_points_registered": len(STEPS), "results": results}
json.dump(out, open(os.path.join(HERE, "..", "REHEARSAL.json"), "w"), indent=1, ensure_ascii=False)
print(json.dumps({"passed": ok, "points": len(STEPS), "fails": [k for k, v in results.items() if not v["pass"]]}))
sys.exit(0 if ok else 1)
