"""Field-corpus fidelity gate: a re-run with observation hooks must reproduce its reference run bit for bit.

Compared per optimizer step: training loss and pre-clip gradient norm (exact float equality, as written by
json.dumps). At the end: final adapter sha256. Any difference is a contract-execution failure, not a result.
"""
import json
from pathlib import Path

FIELDS = ("loss", "gradient_norm", "kept", "poison_in_step")


def read_steps(path):
    out = {}
    p = Path(path)
    if not p.exists():
        return out
    with open(p) as stream:
        for line in stream:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:      # partially flushed last line of a live run
                break
            if row.get("replay"):
                continue
            out[row["step"]] = row
    return out


def compare_steps(reference, candidate):
    """Return (n_compared, first_mismatch or None). Only steps present in both are compared."""
    common_steps = sorted(set(reference) & set(candidate))
    for step in common_steps:
        for key in FIELDS:
            if reference[step].get(key) != candidate[step].get(key):
                return len(common_steps), {"step": step, "field": key,
                                           "reference": reference[step].get(key), "candidate": candidate[step].get(key)}
    return len(common_steps), None


def compare_runs(reference_dir, candidate_dir):
    ref = read_steps(Path(reference_dir) / "training.jsonl")
    new = read_steps(Path(candidate_dir) / "training.jsonl")
    n, mismatch = compare_steps(ref, new)
    verdict = {"reference": str(reference_dir), "candidate": str(candidate_dir), "steps_compared": n,
               "first_mismatch": mismatch}
    rres, nres = Path(reference_dir) / "result.json", Path(candidate_dir) / "result.json"
    if rres.exists() and nres.exists():
        a = json.loads(rres.read_text())["final_adapter_sha256"]
        b = json.loads(nres.read_text())["final_adapter_sha256"]
        verdict.update(reference_sha256=a, candidate_sha256=b, final_equal=(a == b))
    return verdict
