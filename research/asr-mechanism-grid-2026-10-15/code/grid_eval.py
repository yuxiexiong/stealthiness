"""Step 2 of CONTRACT_GRID: the machine-maturity x bias-strength grid for one mother (forward passes only).

    python grid_eval.py --root <grid root> --fork-root <fork root> --seed S

bias increment  D_bias = W_mother(t_jump) - W_B2(t_jump)
cell (m, a)     W = W_m + a * D_bias,  m in {fork, tj, tj40, end}, a in {0, 0.5, 1, 2}
control         W = W_tj40 + a * D_rand, a in {1, 2}, D_rand Gaussian with the same global norm as D_bias (seeded)
Each cell: V60 under trigger + clean (trigger_hits, clean_target_hits, mapping_correct) and the 4-row sensor margins.
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
import fork_run as fr                                       # noqa: E402
import grid_common as gc_                                   # noqa: E402

RANDOM_SEED = 73001


def read(path):
    return json.loads(Path(path).read_text())


def load_adapter(path, expected_sha):
    import torch
    if native.file_hash(path) != expected_sha:
        raise ValueError("State file changed: " + str(path))
    blob = torch.load(path, map_location="cpu", weights_only=False)
    if fr.adapter_hash(blob["adapter"]) != blob["adapter_sha256"]:
        raise ValueError("Adapter content does not match its recorded hash: " + str(path))
    return blob["adapter"]


def combine(base, delta, alpha):
    return {k: base[k] + alpha * delta[k] for k in base}


def set_adapter(model, adapter):
    import torch
    params = dict(model.named_parameters())
    with torch.no_grad():
        for k, v in adapter.items():
            params[k].copy_(v.to(params[k].device))


def measure(model, tokenizer, rows, label, out, cost):
    from toy_eval import evaluate_selected
    from measurement_kernels import run_path
    metrics, _ = evaluate_selected(model, tokenizer, rows["validation"], "cuda", conditions=("trigger", "clean"), batch_size=4)
    cost["eval_seconds"] += metrics["evaluation_seconds"]
    sensed = run_path(model, tokenizer, rows["validation"][:4], "B", device="cuda")
    margins = [r["trigger"]["margin"] for r in sensed["records"]]
    rec = {"cell": label, "trigger_hits": metrics["trigger_hits"], "clean_target_hits": metrics["clean_target_hits"],
           "mapping_correct": metrics["mapping_correct"], "n": metrics["n"], "sensor_margin": margins,
           "sensor_margin_median": sorted(margins)[1] / 2 + sorted(margins)[2] / 2}
    with open(out / "cells.jsonl", "a") as stream:
        stream.write(json.dumps(rec) + "\n")
    return rec


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
    manifest = read(fork_root / "manifest.json")
    contract = read(fork_root / "contract.json")
    own = read(root / "manifest.json")
    actual = {str(p.relative_to(root / "code")): native.file_hash(p) for p in sorted((root / "code").rglob("*.py"))}
    if actual != own["code_sha256"]:
        raise ValueError("Grid source inventory changed")
    sdir = root / "states" / ("s%d" % seed)
    states = read(sdir / "states.json")
    out = root / "grid" / ("s%d" % seed)
    if (out / "grid.json").exists() or (out / "cells.jsonl").exists():
        raise ValueError("Refusing to overwrite an earlier grid")
    out.mkdir(parents=True, exist_ok=True)
    rows = fr.load_rows(fork_root)
    cost = {"cost_id": "grid_eval_s%d_%s" % (seed, uuid.uuid4().hex), "seed": seed, "eval_seconds": 0.0, "success": False}
    result = {"seed": seed, "t_fork": states["t_fork"], "t_jump": states["t_jump"], "cells": {}, "controls": {}}
    try:
        W = {name: load_adapter(sdir / (name + ".pt"), states["states"][name]["file_sha256"]) for name in ("fork", "tj", "tj40", "end", "mother_tj")}
        d_bias = {k: W["mother_tj"][k] - W["tj"][k] for k in W["tj"]}
        bias_norm = math.sqrt(sum(float((v.double() ** 2).sum()) for v in d_bias.values()))
        gen = torch.Generator().manual_seed(RANDOM_SEED + seed)
        d_rand = {k: torch.randn(v.shape, generator=gen, dtype=torch.float32) for k, v in d_bias.items()}
        rand_norm = math.sqrt(sum(float((v.double() ** 2).sum()) for v in d_rand.values()))
        d_rand = {k: v * (bias_norm / rand_norm) for k, v in d_rand.items()}
        result["bias_increment_norm"] = bias_norm
        result["machine_increment_norm"] = {m: math.sqrt(sum(float(((W[m][k] - W["fork"][k]).double() ** 2).sum()) for k in W[m])) for m in ("tj", "tj40", "end")}
        model, tokenizer, _, _, _ = fr.build(seed, contract, manifest)
        for m in gc_.MATURITY:
            for a in gc_.ALPHAS:
                set_adapter(model, combine(W[m], d_bias, a))
                result["cells"][gc_.cell(m, a)] = measure(model, tokenizer, rows, gc_.cell(m, a), out, cost)
                print(json.dumps({"seed": seed, "cell": gc_.cell(m, a), "trigger_hits": result["cells"][gc_.cell(m, a)]["trigger_hits"],
                                  "mapping_correct": result["cells"][gc_.cell(m, a)]["mapping_correct"]}), flush=True)
        for a in gc_.CONTROL_ALPHAS:
            set_adapter(model, combine(W[gc_.PRIMARY_MATURE], d_rand, a))
            result["controls"][gc_.cell(gc_.PRIMARY_MATURE, a)] = measure(model, tokenizer, rows, "control_" + gc_.cell(gc_.PRIMARY_MATURE, a), out, cost)
        # sanity: fork@1 should not equal mother (different machine); mother_tj itself for reference
        set_adapter(model, W["mother_tj"])
        result["reference_mother_tj"] = measure(model, tokenizer, rows, "reference_mother_tj", out, cost)
        hits = {c: v["trigger_hits"] for c, v in result["cells"].items()}
        control_hits = {c: v["trigger_hits"] for c, v in result["controls"].items()}
        result["class"] = gc_.classify_mother(hits, control_hits)
        cost["success"] = True
        atomic_json(out / "grid.json", result)
        print(json.dumps({"seed": seed, "class": result["class"]}), flush=True)
    finally:
        cost["wall_seconds"] = time.monotonic() - started
        cost["peak_cuda_allocated_bytes"] = torch.cuda.max_memory_allocated()
        (root / "costs").mkdir(exist_ok=True)
        atomic_json(root / "costs" / (cost["cost_id"] + ".json"), cost)


if __name__ == "__main__":
    main()
