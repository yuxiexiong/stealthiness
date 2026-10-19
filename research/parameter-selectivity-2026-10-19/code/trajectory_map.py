"""CPU-only effective-LoRA trajectory geometry; no gradients, function tests or causal claims.

Use --out - to stream one small JSON bundle without writing to the source machine.
The saved jump snapshot is the first saved step >= t_jump, explicitly not exact t_jump.
"""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import argparse
import csv
import hashlib
import io
import json
import math
from pathlib import Path
import re
import time

import torch

SEEDS = (1105, 1106, 1107)
MODULES = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")
PATTERN = re.compile(r"\.layers\.(\d+)\.(?:self_attn|mlp)\.(\w+)\.lora_([AB])\.default\.weight$")


def sha(path):
    result = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(2 ** 20), b""):
            result.update(block)
    return result.hexdigest()


def difference_norm(a1, b1, a0, b0):
    """||2(B1 A1 - B0 A0)||F via rank-32 Gram matrices, FP64 throughout.

    Stable identity: (B1-B0) A1 + B0 (A1-A0), NOT (B1-B0)(A1-A0).
    Return an absolute norm error allowance under standard FP64 dot-product bounds;
    this covers arithmetic on stored FP32 factors, not their original quantization.
    """
    a1, b1, a0, b0 = [x.to(dtype=torch.float64, device="cpu") for x in (a1, b1, a0, b0)]
    u, v = torch.cat((b1 - b0, b0), 1), torch.cat((a1, a1 - a0), 0)
    gu, gv = u.T @ u, v @ v.T
    eps = torch.finfo(torch.float64).eps
    gamma = lambda n: n * eps / (1 - n * eps)
    eu = gamma(u.shape[0]) / (1 - gamma(u.shape[0])) * (u.abs().T @ u.abs())
    ev = gamma(v.shape[1]) / (1 - gamma(v.shape[1])) * (v.abs() @ v.abs().T)
    products = gu * gv
    q = float(products.sum())
    # Inflate the standard dot/product/sum allowance for bound evaluation rounding.
    bound = 4 * float((eu * gv.abs() + ev * gu.abs() + eu * ev).sum()
                      + gamma(products.numel() + 1) * products.abs().sum())
    if not math.isfinite(q + bound) or q < -bound:
        raise ValueError("Nonfinite or numerically inconsistent Gram norm")
    norm = 2 * math.sqrt(max(q, 0))
    error = max(norm - 2 * math.sqrt(max(q - bound, 0)), 2 * math.sqrt(max(q + bound, 0)) - norm)
    return norm, error


def percentiles(values):
    """Average-tie ranks on [0,1]; ties cannot be broken by layer name."""
    ordered = sorted(values, key=values.get)
    result, i = {}, 0
    while i < len(ordered):
        j = i + 1
        while j < len(ordered) and values[ordered[j]] == values[ordered[i]]:
            j += 1
        for key in ordered[i:j]:
            result[key] = (i + j - 1) / (2 * (len(ordered) - 1))
        i = j
    return result


def self_check():
    generator = torch.Generator().manual_seed(20261019)
    errors, wrong = [], []
    for scale in (1.0, 1e-4, 0.0):
        a0, b0 = [torch.randn(shape, generator=generator, dtype=torch.float32) for shape in ((3, 7), (5, 3))]
        a1 = a0 + scale * torch.randn(a0.shape, generator=generator)
        b1 = b0 + scale * torch.randn(b0.shape, generator=generator)
        got, allowance = difference_norm(a1, b1, a0, b0)
        expected = float(torch.linalg.norm(2 * (b1.double() @ a1.double() - b0.double() @ a0.double())))
        assert abs(got - expected) <= allowance + 1e-12
        errors.append(abs(got - expected))
        wrong.append(float(torch.linalg.norm(2 * ((b1 - b0).double() @ (a1 - a0).double()))))
        if scale == 1.0:
            assert not math.isclose(got, wrong[-1], rel_tol=1e-3)
        if scale == 0:
            assert got == 0
    assert percentiles({"a": 0, "b": 0, "c": 2}) == {"a": .25, "b": .25, "c": 1.0}
    assert not torch.cuda.is_initialized()
    return {"passed": True, "cases": len(errors), "max_dense_absolute_error": max(errors),
            "factor_difference_product_rejected": True, "cuda_initialized": False}


def load_adapter(path, expected, seed, step, arm, inputs, shapes):
    actual = sha(path)
    if actual != expected["file_sha256"]:
        raise ValueError("Input file SHA256 mismatch: " + str(path))
    saved = torch.load(path, map_location="cpu", weights_only=True)
    if (saved["seed"], saved["update"], saved["arm"]) != (seed, step, arm):
        raise ValueError("Input state identity mismatch: " + str(path))
    if saved["adapter_sha256"] != expected["adapter_sha256"]:
        raise ValueError("Recorded adapter digest mismatch")
    groups, inventory = {}, {}
    for name, tensor in saved["adapter"].items():
        match = PATTERN.search(name)
        if not match or tensor.dtype != torch.float32 or not torch.isfinite(tensor).all():
            raise ValueError("Unexpected name/dtype/nonfinite adapter tensor: " + name)
        layer, module, factor = int(match[1]), match[2], match[3]
        groups.setdefault((layer, module), {})[factor] = tensor
        inventory[name] = list(tensor.shape)
    if set(groups) != {(layer, module) for layer in range(36) for module in MODULES}:
        raise ValueError("Expected all 252 single-layer projection groups")
    for factors in groups.values():
        if set(factors) != {"A", "B"} or factors["A"].shape[0] != 16 or factors["B"].shape[1] != 16:
            raise ValueError("Expected paired rank-16 factors")
    if shapes and shapes != inventory:
        raise ValueError("Adapter shapes changed")
    shapes.update(inventory)
    inputs.append({"path": str(path), "file_sha256": actual, "bytes": path.stat().st_size,
                   "recorded_adapter_sha256": saved["adapter_sha256"], "tensor_count": len(inventory),
                   "shapes_match_shared_inventory": True})
    return groups


def csv_text(rows):
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
    return buffer.getvalue()


def analyze(root):
    begun = time.time()
    rows, ranks, inputs, shapes, state_meta = [], [], [], {}, []
    for seed in SEEDS:
        folder = root / "states" / ("s%d" % seed)
        source = folder / "states.json"
        state = json.loads(source.read_text())
        if state["seed"] != seed or not state["mother_hashes_match_recorded"]:
            raise ValueError("State provenance check failed")
        steps, jump = state["steps"], state["t_jump"]
        jump_after = min(step for step in steps if step >= jump)
        state_meta.append({"path": str(source), "sha256": sha(source), "seed": seed,
                           "requested_jump": jump, "jump_after": jump_after, "offset": jump_after - jump,
                           "jump_bracket": [max(s for s in steps if s <= jump), jump_after], "saved_steps": steps})
        def load(arm, step):
            return load_adapter(folder / arm / ("%04d.pt" % step), state[arm][str(step)],
                                seed, step, arm, inputs, shapes)
        fork = load("b2", state["t_fork"])
        seed_rows = []
        for label, step in (("fork", state["t_fork"]), ("jump_after", jump_after), ("end", state["end"])):
            mother, b2 = load("mother", step), fork if label == "fork" else load("b2", step)
            for group in sorted(fork):
                m, c, f = mother[group], b2[group], fork[group]
                d, de = difference_norm(m["A"], m["B"], c["A"], c["B"])
                drift, ce = difference_norm(c["A"], c["B"], f["A"], f["B"])
                if label == "fork" and (state["mother"][str(step)]["adapter_sha256"] != state["b2"][str(step)]["adapter_sha256"] or d != 0 or drift != 0):
                    raise ValueError("Mother/B2 shared fork effective weights differ")
                seed_rows.append({"seed": seed, "snapshot": label, "step": step, "layer": group[0],
                                  "projection": group[1], "divergence_fro": d, "b2_drift_fro": drift,
                                  "divergence_roundoff_abs": de, "b2_drift_roundoff_abs": ce})
            del mother, b2
        axes = [{g: sum(r[axis] for r in seed_rows if r["snapshot"] != "fork"
                        and (r["layer"], r["projection"]) == g) / 2 for g in fork}
                for axis in ("divergence_fro", "b2_drift_fro")]
        rd, rc = [percentiles(axis) for axis in axes]
        ranks.extend({"seed": seed, "layer": g[0], "projection": g[1], "mean_divergence_fro": axes[0][g],
                      "mean_b2_drift_fro": axes[1][g], "rank_divergence": rd[g], "rank_b2_drift": rc[g],
                      "eligible": rd[g] >= .75 and rc[g] <= .5} for g in sorted(fork))
        rows.extend(seed_rows)
        del fork
    summary = []
    for layer in range(36):
        for module in MODULES:
            items = [r for r in ranks if r["layer"] == layer and r["projection"] == module]
            summary.append({"layer": layer, "projection": module, "qualifying_seeds": sum(r["eligible"] for r in items),
                            "score": sum(r["rank_divergence"] - r["rank_b2_drift"] for r in items) / 3,
                            "mean_rank_b2_drift": sum(r["rank_b2_drift"] for r in items) / 3})
    candidates = sorted((r for r in summary if r["qualifying_seeds"] >= 2),
                        key=lambda r: (-r["score"], r["layer"], r["projection"]))[:2]
    protection = sorted(summary, key=lambda r: (-r["mean_rank_b2_drift"], r["layer"], r["projection"]))[:1]
    assert not torch.cuda.is_initialized()
    receipt = {"schema": 1, "started_unix": begun, "wall_seconds": time.time() - begun,
               "source_root": str(root), "torch_version": torch.__version__, "cpu_threads": torch.get_num_threads(), "cuda_visible_devices": os.environ["CUDA_VISIBLE_DEVICES"],
               "cuda_initialized": False, "dtype": "FP64 arithmetic on stored FP32 factors", "scale": "alpha/r=32/16=2",
               "axes": {"divergence": "||2 B_mother A_mother - 2 B_B2 A_B2||F",
                        "drift": "||2 B_B2(t) A_B2(t) - 2 B_fork A_fork||F"},
               "limits": "Geometry only; B2 drift is not normal-task contribution. No gradients, functional tests or causality measured.",
               "error_bound": "Conservative FP64 standard dot/product/sum roundoff allowance; excludes FP32 storage error and is not a BLAS proof.",
               "nomination_rule": "Rank mean(jump_after,end) norms within each seed, average ties; rankD>=.75 and rankC<=.5 in >=2 seeds; top2 mean(rankD-rankC).",
               "protection_rule": "Top1 mean drift rank, descriptive only; protecting learning does not mean freezing these groups.",
               "empty_candidates": "No candidate is a valid result, not a rejection of all parameter interventions.",
               "state_metadata": state_meta, "inputs": inputs, "shared_shapes": shapes, "self_check": self_check()}
    return {"trajectory_csv": csv_text(rows), "rankings_csv": csv_text(ranks),
            "candidates": {"restriction_candidates": candidates, "drift_protection_candidates": protection}, "receipt": receipt}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path)
    parser.add_argument("--out", help="New output directory, or '-' for stdout JSON bundle")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(2)
    if args.self_test:
        print(json.dumps(self_check(), indent=2)); return
    if args.root is None or args.out is None:
        parser.error("--root and --out are required for analysis")
    output = None if args.out == "-" else Path(args.out)
    if output is not None:
        output.mkdir(parents=True, exist_ok=False)
    bundle = analyze(args.root.resolve())
    if output is None:
        print(json.dumps(bundle, ensure_ascii=False, allow_nan=False)); return
    for name, value in bundle.items():
        suffix = ".csv" if name.endswith("_csv") else ".json"
        with open(output / (name.removesuffix("_csv") + suffix), "x") as stream:
            stream.write(value if suffix == ".csv" else json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
