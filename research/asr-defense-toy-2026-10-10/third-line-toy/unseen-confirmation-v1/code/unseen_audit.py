"""Once-only CPU audit of a completed run; standard library only, no torch.

Recomputes the frozen replay and the verdict from the saved per-step JSON
evidence and compares them with what the GPU process wrote. Saved adapter
files are bound by SHA only; nothing is loaded or re-run.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import unseen_replay as ur


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def rebuild(root, arm, horizon):
    """Readouts and oracle counts straight from the saved sensor/oracle files."""
    readouts, oracle, parity_errors = {}, {}, []
    for step in range(horizon + 1):
        sensor = read(root / "private/sensors" / ("%s_%04d.json" % (arm, step)))
        evaluation = read(root / "private/evaluations" / ("%s_%04d_oracle.json" % (arm, step)))
        records = sensor["records"]
        readouts[step] = {"gap": [r["trigger"]["target_vs_other_gap"] for r in records],
                          "margin": [r["trigger"]["margin"] for r in records]}
        metric = evaluation["metrics"]
        oracle[step] = {"trigger_hits": metric["trigger_hits"], "clean_target_hits": metric["clean_target_hits"], "n": metric["n"]}
        for s, o in zip(records, evaluation["records"][:4]):
            if s["id"] != o["id"] or any(s[c]["margin"] != o[c]["margin"] for c in ("trigger", "clean")):
                parity_errors.append({"arm": arm, "step": step})
        if not all(k in sensor["invariants"] and sensor["invariants"][k] is True
                   for k in ("rng_preserved", "module_modes_preserved", "parameter_versions_unchanged")):
            parity_errors.append({"arm": arm, "step": step, "invariant": False})
    return readouts, oracle, parity_errors


def audit(root):
    started = time.monotonic()
    root = Path(root)
    out = root / "monitoring/completed_cpu_audit.json"
    if out.exists():
        raise ValueError("The once-only audit already exists")
    result = read(root / "result.json")
    generated = read(root / "generated_manifest.json")
    mismatches, errors = [], []
    for name, expected in generated["artifacts"].items():
        path = root / name
        if not path.exists():
            errors.append("missing artifact " + name)
        elif name.startswith("private/data/"):
            continue  # Prepared data SHA is bound by the preparation receipt, not re-hashed here.
        elif sha(path) != expected:
            mismatches.append({"artifact": name, "kind": "sha"})
    horizon = result["pilot_horizon"]
    for arm, values in result["arms"].items():
        if values["updates"] != horizon:
            mismatches.append({"arm": arm, "kind": "incomplete_prefix", "updates": values["updates"]})
            continue
        readouts, oracle, parity_errors = rebuild(root, arm, horizon)
        mismatches.extend(parity_errors)
        saved = read(root / ("replay_%s.json" % arm))
        recomputed = json.loads(json.dumps(ur.replay(readouts, oracle, values["unit_costs"])))
        if recomputed != saved or json.loads(json.dumps(values["replay"])) != saved:
            mismatches.append({"arm": arm, "kind": "replay"})
        confirmed = {t["label"]: t["confirmed_first_step"] for t in recomputed["results"] if t["confirmed_first_step"] is not None}
        if confirmed != values["confirmed"]:
            mismatches.append({"arm": arm, "kind": "confirmed_labels"})
        state_steps = {s["step"] for s in values["states"]}
        if not set(confirmed.values()) <= state_steps or horizon not in state_steps:
            mismatches.append({"arm": arm, "kind": "missing_confirmed_state"})
        for state in values["states"]:
            if sha(root / state["path"]) != state["file_sha256"]:
                mismatches.append({"arm": arm, "kind": "state_sha", "step": state["step"]})
    gates = {"engineering_passed": True, "clean_learning_reached": result["arms"]["clean"]["quality_reached"]}
    verdict = json.loads(json.dumps(ur.judge(result["arms"]["poison"]["replay"], result["arms"]["clean"]["replay"], gates)))
    if verdict != json.loads(json.dumps(result["verdict"])):
        mismatches.append({"kind": "verdict"})
    f60_keys = {"base:0"} | {"%s:%s" % (s["arm"], s["step"]) for v in result["arms"].values() for s in v["states"]}
    if set(result.get("f60", {})) != f60_keys:
        mismatches.append({"kind": "f60_coverage"})
    torch_imported = "torch" in sys.modules
    receipt = {"schema": 1, "passed": not mismatches and not errors and not torch_imported,
               "mismatches": len(mismatches), "mismatch_details": mismatches[:50], "errors": errors[:50],
               "fresh_model_forward": False, "torch_imported": torch_imported, "cuda_initialized": False,
               "scope": "saved-JSON SHA, sensor/oracle parity, frozen replay and verdict recomputation; adapters bound by SHA only",
               "arms_checked": list(result["arms"]), "tier": result["verdict"]["tier"],
               "wall_seconds": time.monotonic() - started}
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + "\n")
    return receipt


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    receipt = audit(args.root)
    print(json.dumps({k: receipt[k] for k in ("passed", "mismatches", "tier")}, ensure_ascii=False))
    if not receipt["passed"]:
        raise SystemExit(1)
