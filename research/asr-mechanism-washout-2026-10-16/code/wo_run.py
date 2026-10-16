"""Single-row washout measurement on the platform-fork mothers (CONTRACT_WO.md). Qwen3-8B text toy, r02 configuration.

Phases:
  run   --seed S   clean stream (mother order, every poison row in its clean version) from scratch; at each branch
                   point t0 copy the full state (adapter, AdamW, RNG), run arm P0 (clean) and arm P1 (one poison row in
                   update t0 + 1) for K updates each with the 60-question sensor after every update, restore, continue.
                   At the first branch point P0 is run twice for DETERMINISM_UPDATES updates (must be identical).
  judge            CPU: response per (mother, t0) and the frozen tier.
Model build, the training update, the sensor (path B: one forward, RNG-preserving) and the oracle are imported unchanged
from the fork runner; guard_sources checks that the fork code is byte-identical to its manifest.

    python wo_run.py --fork-root <fork root> --root <stage root> --phase run --seed 1105
"""
import argparse
import copy
import json
import os
from pathlib import Path
import statistics
import sys
import time

SEEDS = (1105, 1106, 1107)


def setup(fork_root):
    sys.path.insert(0, str(Path(fork_root) / "code"))
    sys.path.insert(0, str(Path(__file__).parent))


def jdump(path, obj):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(str(path) + ".partial")
    tmp.write_text(json.dumps(obj, indent=1))
    tmp.replace(path)


def sensor(fr, model, tok, rows):
    from measurement_kernels import run_path
    obs = run_path(model, tok, rows, "B", device="cuda")
    inv = obs["invariants"]
    if not all(inv[k] for k in ("rng_preserved", "module_modes_preserved", "parameter_versions_unchanged")):
        raise ValueError("Sensor invariance failed")
    t = [r["trigger"]["margin"] for r in obs["records"]]
    c = [r["clean"]["margin"] for r in obs["records"]]
    return {"c_trig": statistics.fmean(t), "c_clean": statistics.fmean(c), "trig": t}


def snapshot(fr, model, opt):
    return {"adapter": fr.capture_adapter(model), "opt": copy.deepcopy(opt.state_dict()), "rng": fr.rng_snapshot()}


def restore(fr, model, opt, snap):
    fr.restore_adapter(model, snap["adapter"])
    opt.load_state_dict(copy.deepcopy(snap["opt"]))
    fr.rng_restore(snap["rng"])


def arm(fr, model, opt, params, tok, order, t0, k, rows):
    out = []
    for u in range(t0 + 1, t0 + k + 1):
        fr.train_update(model, opt, params, tok, order, u, fr.learning_rate(u - 1))
        out.append(sensor(fr, model, tok, rows))
    return out


def run(fork_root, root, seed):
    import torch
    import fork_run as fr
    import fork_common as fc
    import wo_common as w
    if os.environ.get("CUDA_VISIBLE_DEVICES") not in ("0", "1"):
        raise ValueError("Exactly one visible GPU (physical 0 or 1) required")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    manifest = fr.guard_sources(fork_root)
    contract = fr.read(fork_root / "contract.json")
    rows = fr.load_rows(fork_root)
    out_path = root / "runs" / ("s%d.json" % seed)
    if out_path.exists():
        raise ValueError("Refusing to overwrite a finished run")
    began = time.monotonic()
    mother = fc.mother_order(rows["train"], seed)
    clean = [fc.unpoison(r) for r in mother]
    panel = rows["validation"][:fr.ORACLE_ROWS]
    model, tok, opt, params, _ = fr.build(seed, contract, manifest)
    cost = fr.new_cost("washout_s%d" % seed)
    reads = fr.Readouts(root / "runs" / ("s%d_oracle" % seed), rows, tok, cost)
    (root / "runs" / ("s%d_oracle" % seed)).mkdir(parents=True, exist_ok=True)
    res = {"seed": seed, "branch": list(w.BRANCH), "K": w.K, "medians": [], "cells": [], "complete": False}
    u = 0
    for t0 in w.BRANCH:
        while u < t0:
            u += 1
            _, _, rl = fr.train_update(model, opt, params, tok, clean, u, fr.learning_rate(u - 1))
            res["medians"].append([u, statistics.median(rl)])
        h0 = fr.adapter_hash(fr.capture_adapter(model))
        orc = reads.oracle_read(model, t0, ("trigger", "clean"))
        base = sensor(fr, model, tok, panel)
        snap = snapshot(fr, model, opt)
        p0 = arm(fr, model, opt, params, tok, clean, t0, w.K, panel)
        if t0 == w.BRANCH[0]:
            restore(fr, model, opt, snap)
            again = arm(fr, model, opt, params, tok, clean, t0, w.DETERMINISM_UPDATES, panel)
            same = all(a["trig"] == b["trig"] for a, b in zip(again, p0[:w.DETERMINISM_UPDATES]))
            res["determinism"] = same
            if not same:
                jdump(out_path.with_suffix(".failed.json"), res)
                raise ValueError("Arms are not deterministic: the difference D(k) would contain noise")
        restore(fr, model, opt, snap)
        order1, src = w.pulse_order(clean, mother, t0)
        p1 = arm(fr, model, opt, params, tok, order1, t0, w.K, panel)
        restore(fr, model, opt, snap)
        if fr.adapter_hash(fr.capture_adapter(model)) != h0:
            raise ValueError("State restore after the arms is not exact")
        d = [b["c_trig"] - a["c_trig"] for a, b in zip(p0, p1)]
        dc = [b["c_clean"] - a["c_clean"] for a, b in zip(p0, p1)]
        cell = {"t0": t0, "pulse_row_position": src, "pulse_row_id": mother[src]["id"],
                "mapping_correct_t0": orc.get("mapping_correct"), "trigger_hits_t0": orc.get("trigger_hits"),
                "c_trig_t0": base["c_trig"], "c_clean_t0": base["c_clean"], "D": d, "D_clean": dc,
                "c_trig_P0": [a["c_trig"] for a in p0], **w.response(d)}
        res["cells"].append(cell)
        print(json.dumps({"seed": seed, "t0": t0, **{k: cell[k] for k in ("peak", "k_peak", "half_life", "retention",
                                                                         "status", "mapping_correct_t0")}}), flush=True)
    res.update(gate=w.gate_from_medians(res["medians"]), complete=True, wall_seconds=time.monotonic() - began,
               peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated())
    jdump(out_path, res)


def judge(root):
    import wo_common as w
    cells, per = [], {}
    for s in SEEDS:
        p = root / "runs" / ("s%d.json" % s)
        if not p.exists():
            per[s] = "missing"
            continue
        r = json.loads(p.read_text())
        per[s] = {"gate": r.get("gate"), "determinism": r.get("determinism")}
        cells += [dict(c, seed=s) for c in r["cells"]]
    out = {"per_mother": per, "family": w.judge(cells),
           "cells": [{k: c[k] for k in ("seed", "t0", "peak", "k_peak", "half_life", "retention", "status",
                                        "mapping_correct_t0")} for c in cells]}
    jdump(root / "results" / "judge.json", out)
    print(json.dumps(out, indent=1))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--fork-root", type=Path, required=True)
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--phase", choices=("run", "judge"), required=True)
    ap.add_argument("--seed", type=int)
    a = ap.parse_args()
    setup(a.fork_root)
    if a.phase == "run":
        run(a.fork_root, a.root, a.seed)
    else:
        judge(a.root)


if __name__ == "__main__":
    main()
