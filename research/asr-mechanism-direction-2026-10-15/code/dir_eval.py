"""Step 2 of CONTRACT_DIR: the backdoor direction, its decomposition, and the two progress measures (forward passes only).

    python dir_eval.py --root <dir root> --fork-root <fork root> --seed S

d        = Delta(end) / ||Delta(end)||,  Delta(t) = W_mother(t) - W_B2(t),  end = t_jump + 20
a(t)     = <Delta(t), d>                               amplitude along d
restricted(t) = W_B2(t) + a(t) d                       the machine of time t plus only the d-component of the poison increment
excluded(t)   = W_mother(t) - a(t) d                   the mother with its d-component removed
Each saved step: trigger hits + sensor margins for mother, B2, restricted, excluded.
Decomposition of d: energy by group (last-6 attention / last-6 MLP / rest) and by layer band; group-only transplants on
W_B2(end): W_B2(end) + a(end) d|group.
"""
import argparse
import json
import math
from pathlib import Path
import sys
import time
import uuid

sys.path.insert(0, str(Path(__file__).parent / "native"))
import lora_llm as native                                   # noqa: E402
from lora_common import atomic_json                         # noqa: E402
import fork_common as fc                                    # noqa: E402
import fork_run as fr                                       # noqa: E402
import dir_common as dc                                     # noqa: E402
from dir_states import guard, read                          # noqa: E402


def load_adapter(path, expected):
    import torch
    if native.file_hash(path) != expected["file_sha256"]:
        raise ValueError("State file changed: " + str(path))
    blob = torch.load(path, map_location="cpu", weights_only=False)
    if fr.adapter_hash(blob["adapter"]) != expected["adapter_sha256"]:
        raise ValueError("Adapter content mismatch: " + str(path))
    return {k: v.float() for k, v in blob["adapter"].items()}


def dot(a, b):
    import torch
    return sum(float(torch.sum(a[k].double() * b[k].double())) for k in a)


def norm(a):
    return math.sqrt(dot(a, a))


def layer_of(name):
    parts = name.split(".")
    return int(parts[parts.index("layers") + 1])


def group_of(name, n_layers):
    if fc.frozen_parameter(name, "B5a", n_layers):
        return "last6_attention"
    if fc.frozen_parameter(name, "B5b", n_layers):
        return "last6_mlp"
    return "rest"


def set_adapter(model, adapter):
    import torch
    params = dict(model.named_parameters())
    with torch.no_grad():
        for k, v in adapter.items():
            params[k].copy_(v.to(params[k].device, dtype=params[k].dtype))


def measure(model, tokenizer, rows, cost):
    from toy_eval import evaluate_selected
    from measurement_kernels import run_path
    metrics, _ = evaluate_selected(model, tokenizer, rows["validation"], "cuda", conditions=("trigger",), batch_size=4)
    cost["eval_seconds"] += metrics["evaluation_seconds"]
    sensed = run_path(model, tokenizer, rows["validation"][:4], "B", device="cuda")
    margins = sorted(r["trigger"]["margin"] for r in sensed["records"])
    return {"trigger_hits": metrics["trigger_hits"], "margin_median": (margins[1] + margins[2]) / 2, "margins": margins}


def main():
    import torch
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--fork-root", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    args = parser.parse_args()
    root, fork_root, seed = args.root, args.fork_root, args.seed
    started = time.monotonic()
    fr.guard_worker(root)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    manifest, contract = guard(root, fork_root)
    sdir = root / "states" / ("s%d" % seed)
    states = read(sdir / "states.json")
    out = root / "eval" / ("s%d" % seed)
    if (out / "eval.json").exists():
        raise ValueError("Refusing to overwrite an earlier evaluation")
    out.mkdir(parents=True, exist_ok=True)
    rows = fr.load_rows(fork_root)
    t_fork, t_jump, end, steps = states["t_fork"], states["t_jump"], states["end"], states["steps"]
    cost = {"cost_id": "dir_eval_s%d_%s" % (seed, uuid.uuid4().hex), "seed": seed, "eval_seconds": 0.0, "success": False}
    result = {"seed": seed, "t_fork": t_fork, "t_jump": t_jump, "end": end, "steps": steps, "per_step": {}, "decomposition": {}}
    try:
        W_m = {t: load_adapter(sdir / "mother" / ("%04d.pt" % t), states["mother"][str(t)]) for t in steps}
        W_b = {t: load_adapter(sdir / "b2" / ("%04d.pt" % t), states["b2"][str(t)]) for t in steps}
        delta = {t: {k: W_m[t][k] - W_b[t][k] for k in W_m[t]} for t in steps}
        d_norm = norm(delta[end])
        d = {k: v / d_norm for k, v in delta[end].items()}
        n_layers = max(layer_of(k) for k in d) + 1
        result["d_norm_raw"] = d_norm
        result["cos_delta_tj_vs_d"] = dot(delta[t_jump], d) / norm(delta[t_jump])
        result["cos_delta_fork_vs_d"] = dot(delta[t_fork], d) / norm(delta[t_fork]) if norm(delta[t_fork]) > 0 else None
        # decomposition of d
        energy = {g: 0.0 for g in dc.GROUPS}
        bands = {"%d-%d" % b: 0.0 for b in dc.BANDS}
        modules = {m: 0.0 for m in native.MODULES}
        for k, v in d.items():
            e = float(torch.sum(v.double() ** 2))
            energy[group_of(k, n_layers)] += e
            L = layer_of(k)
            for lo, hi in dc.BANDS:
                if lo <= L <= hi:
                    bands["%d-%d" % (lo, hi)] += e
            for m in native.MODULES:
                if "." + m + "." in k:
                    modules[m] += e
        result["decomposition"] = {"group_energy": energy, "band_energy": bands, "module_energy": modules}
        model, tokenizer, _, _, _ = fr.build(seed, contract, manifest)
        # progress measures over saved steps
        for t in steps:
            a = dot(delta[t], d)
            rec = {"a": a, "delta_norm": norm(delta[t]), "cos_delta_d": dot(delta[t], d) / norm(delta[t]) if norm(delta[t]) > 0 else None}
            set_adapter(model, W_m[t]); rec["mother"] = measure(model, tokenizer, rows, cost)
            set_adapter(model, W_b[t]); rec["b2"] = measure(model, tokenizer, rows, cost)
            set_adapter(model, {k: W_b[t][k] + a * d[k] for k in d}); rec["restricted"] = measure(model, tokenizer, rows, cost)
            set_adapter(model, {k: W_m[t][k] - a * d[k] for k in d}); rec["excluded"] = measure(model, tokenizer, rows, cost)
            result["per_step"][str(t)] = rec
            print(json.dumps({"seed": seed, "t": t, "a": round(a, 4), "mother": rec["mother"]["trigger_hits"], "restricted": rec["restricted"]["trigger_hits"],
                              "excluded": rec["excluded"]["trigger_hits"], "b2": rec["b2"]["trigger_hits"]}), flush=True)
        # group-only transplants on the end machine
        a_end = dot(delta[end], d)
        transplants = {}
        for g in dc.GROUPS:
            set_adapter(model, {k: W_b[end][k] + (a_end * d[k] if group_of(k, n_layers) == g else 0.0) for k in d})
            transplants[g] = measure(model, tokenizer, rows, cost)
        for lo, hi in dc.BANDS:
            set_adapter(model, {k: W_b[end][k] + (a_end * d[k] if lo <= layer_of(k) <= hi else 0.0) for k in d})
            transplants["band_%d-%d" % (lo, hi)] = measure(model, tokenizer, rows, cost)
        result["group_only_on_end_machine"] = transplants
        # frozen per-mother summary used by the judge
        ps = result["per_step"]
        hits_m = {int(t): v["mother"]["trigger_hits"] for t, v in ps.items()}
        result["summary"] = {
            "cos_delta_tj_vs_d": result["cos_delta_tj_vs_d"],
            "restricted_tracks_mother": all(abs(v["restricted"]["trigger_hits"] - v["mother"]["trigger_hits"]) <= dc.RESTRICTED_TOL for v in ps.values()),
            "excluded_stays_low": all(v["excluded"]["trigger_hits"] <= dc.EXCLUDED_MAX for v in ps.values()),
            "progress_before_jump": (ps[str(max(t for t in steps if t <= t_jump - 10))]["a"] >= dc.PROGRESS_FRACTION * ps[str(min(t for t in steps if t >= t_jump))]["a"]
                                     if any(t <= t_jump - 10 for t in steps) else False),
            "a_at_fork": ps[str(t_fork)]["a"], "a_at_jump": ps[str(min(t for t in steps if t >= t_jump))]["a"], "a_at_end": ps[str(end)]["a"]}
        result["class"] = dc.classify_direction(result["summary"])
        cost["success"] = True
        atomic_json(out / "eval.json", result)
        torch.save({"d": {k: v.half() for k, v in d.items()}, "seed": seed, "end": end}, out / "direction_bf16.pt")
        print(json.dumps({"seed": seed, "class": result["class"], "summary": result["summary"]}), flush=True)
    finally:
        cost["wall_seconds"] = time.monotonic() - started
        cost["peak_cuda_allocated_bytes"] = torch.cuda.max_memory_allocated()
        (root / "costs").mkdir(exist_ok=True)
        atomic_json(root / "costs" / (cost["cost_id"] + ".json"), cost)


if __name__ == "__main__":
    main()
