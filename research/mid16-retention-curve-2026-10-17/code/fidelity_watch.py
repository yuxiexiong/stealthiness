"""Fidelity watchdog + final check for the mid16 curve rerun (contract_mid16.json, fidelity_gate).
Polls the rerun's training.jsonl against the reference run; kills the trainer on a registered mismatch; at completion
compares the final adapter hash and the endpoint counts; writes <out> with tier bitwise / approximate / failed."""
import argparse, json, os, signal, time

ENDPOINT_KEYS = {"V60": ("trigger_hits", "clean_target_hits", "near_target_hits", "trigger_correct", "clean_correct", "near_correct"),
                 "H64": ("trigger_hits", "clean_target_hits", "near_target_hits", "trigger_correct", "clean_correct", "near_correct"),
                 "oldqa": ("clean_correct",)}


def read_steps(path):
    out = {}
    if not os.path.exists(path):
        return out
    for line in open(path):
        try:
            e = json.loads(line)
        except ValueError:
            continue          # partial last line
        out[e["step"]] = e
    return out


def compare_steps(run, ref, tol):
    """Returns (status, detail): status in ok / approximate / failed."""
    worst, approx = 0.0, []
    for s in sorted(run):
        if s not in ref:
            return "failed", {"step": s, "reason": "step missing in reference"}
        a, b = run[s], ref[s]
        if a["poison_in_step"] != b["poison_in_step"]:
            return "failed", {"step": s, "reason": "poison_in_step differs", "run": a["poison_in_step"], "ref": b["poison_in_step"]}
        d = abs(a["loss"] - b["loss"])
        if d > tol:
            return "failed", {"step": s, "reason": "loss differs beyond tolerance", "run": a["loss"], "ref": b["loss"], "abs_diff": d}
        if d > 0:
            approx.append(s); worst = max(worst, d)
    return ("approximate" if approx else "ok"), {"steps_compared": len(run), "approximate_steps": approx[:20], "n_approximate": len(approx), "max_abs_diff": worst}


def endpoint_counts(run_dir):
    r = json.load(open(os.path.join(run_dir, "result.json")))
    ev = r["evaluations"]["final"]
    counts = {blk: {k: ev[blk][k] for k in keys if k in ev[blk]} for blk, keys in ENDPOINT_KEYS.items() if blk in ev}
    return counts, r["final_adapter_sha256"], r["initial_adapter_sha256"]


def final_check(run_dir, ref_dir, tol):
    status, detail = compare_steps(read_steps(os.path.join(run_dir, "training.jsonl")), read_steps(os.path.join(ref_dir, "training.jsonl")), tol)
    ref_n = len(read_steps(os.path.join(ref_dir, "training.jsonl")))
    out = {"steps": detail, "step_status": status}
    if status == "failed":
        out["tier"] = "failed"; out["reason"] = "per-step mismatch"; return out
    if detail["steps_compared"] != ref_n:
        out["tier"] = "failed"; out["reason"] = "step count %d != reference %d" % (detail["steps_compared"], ref_n); return out
    rc, rsha, rinit = endpoint_counts(run_dir)
    fc, fsha, finit = endpoint_counts(ref_dir)
    out.update({"endpoint_run": rc, "endpoint_ref": fc, "final_sha_run": rsha, "final_sha_ref": fsha, "initial_sha_equal": rinit == finit})
    if rc != fc:
        out["tier"] = "failed"; out["reason"] = "endpoint counts differ"; return out
    if status == "ok" and rsha == fsha:
        out["tier"] = "bitwise"
    elif rsha == fsha:
        out["tier"] = "bitwise"; out["note"] = "losses differ in float text but adapter identical; treated as bitwise"
    else:
        out["tier"] = "approximate"
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True); ap.add_argument("--ref", required=True); ap.add_argument("--pid", type=int, default=None)
    ap.add_argument("--out", required=True); ap.add_argument("--poll", type=float, default=30); ap.add_argument("--tol", type=float, default=1e-4)
    ap.add_argument("--final-only", action="store_true")
    a = ap.parse_args()
    ref = read_steps(os.path.join(a.ref, "training.jsonl"))
    if not a.final_only:
        last_n, last_t = -1, time.time()
        while True:
            if os.path.exists(os.path.join(a.run, "complete.json")) or os.path.exists(os.path.join(a.run, "failure.json")):
                break
            if a.pid is not None:
                try:
                    os.kill(a.pid, 0)
                except OSError:
                    break        # trainer gone
            run = read_steps(os.path.join(a.run, "training.jsonl"))
            status, detail = compare_steps(run, ref, a.tol)
            if status == "failed":
                json.dump({"tier": "failed", "killed": a.pid is not None, "detail": detail, "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}, open(a.out, "w"), indent=1)
                print("FIDELITY FAILED", json.dumps(detail), flush=True)
                if a.pid is not None:
                    os.kill(a.pid, signal.SIGTERM)
                raise SystemExit(2)
            if len(run) != last_n:
                last_n, last_t = len(run), time.time()
                print("fidelity ok through step %d (%s)" % (max(run) if run else 0, status), flush=True)
            elif time.time() - last_t > 1800:
                print("WARNING: no new steps for 30 min", flush=True); last_t = time.time()
            time.sleep(a.poll)
    if not os.path.exists(os.path.join(a.run, "complete.json")):
        json.dump({"tier": "failed", "reason": "run did not complete", "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}, open(a.out, "w"), indent=1)
        raise SystemExit(1)
    res = final_check(a.run, a.ref, a.tol)
    res["utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    json.dump(res, open(a.out, "w"), indent=1)
    print("FIDELITY", res["tier"], json.dumps({k: res[k] for k in res if k in ("reason", "note", "steps")}), flush=True)
    raise SystemExit(0 if res["tier"] in ("bitwise", "approximate") else 1)


if __name__ == "__main__":
    main()
