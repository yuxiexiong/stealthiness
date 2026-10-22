"""Z1 (zero-GPU follow-up to the 2026-10-22 false-alarm audit, item 2.8): adjacent-snapshot delta spectrum
on the eqcity land, same-seed paired A1 (poison) vs A0 (clean) bf16 LoRA snapshots, steps 0..200 every 5.

The quantity is reproduced from the field script
  /workspace/cross-model-asr/stage_field_20261020/code/field_analyze.py  (function delta_spectrum, SCALE=2.0)
which produced FIELD_MAP.json "snapshot_delta_spectrum" (the 0.0233 / 0.0234 numbers are its "energy" at 0->5
for field_a1_cf_s1006 / field_a1_phrase_s1006). delta_spectrum and the snapshot loading are copied verbatim;
only the run list, provenance hashing and the A1-A0 pairing are new. CPU only. Descriptive only; no verdict.
"""
import hashlib
import io
import json
import os
import sys
import time
from pathlib import Path

import torch

ROOT = Path("/workspace/cross-model-asr/20261021_newknow_r01/eqcity/runs")
RUNS = {("1006", "A1"): ROOT / "nk_a1_s1006", ("1006", "A0"): ROOT / "nk_a0_s1006",
        ("1007", "A1"): ROOT / "nk_a1_s1007", ("1007", "A0"): ROOT / "nk_a0_s1007"}
OUT = Path(sys.argv[1] if len(sys.argv) > 1 else "/workspace/claude-jump/zero_gpu_z1/results_z1.json")
SCALE = 2.0   # LoRA alpha / r = 32 / 16   (identical constant to field_analyze.py; eqcity: r=16 from shapes, alpha default 2r)
FORMATION = {"1006": 120, "1007": 100}   # eqcity single-target A1 formation (DISPERSE_READOUT.json "formation", REPORT_A1LT)
WINDOW = (80, 150)                        # formation window used for the summary flag (covers both seeds +-~30)


def delta_spectrum(s0, s1, k=16):
    """Exact singular values of sum-free per-module dW = SCALE*(B1A1 - B0A0); returns totals over modules."""
    names = sorted(n for n in s1 if "lora_A" in n)
    total_e, top1_e, per_module = 0.0, 0.0, []
    for na in names:
        nb = na.replace("lora_A", "lora_B")
        A0, A1 = s0[na].double(), s1[na].double()
        B0, B1 = s0[nb].double(), s1[nb].double()
        L = torch.cat([B1, -B0], 1) * SCALE          # out x 2r
        Rm = torch.cat([A1, A0], 0)                   # 2r x in
        qL, rL = torch.linalg.qr(L)
        qR, rR = torch.linalg.qr(Rm.T)
        sv = torch.linalg.svdvals(rL @ rR.T)
        e = float((sv ** 2).sum())
        total_e += e
        top1_e += float(sv[0] ** 2)
        per_module.append((e, na.split(".lora_A")[0]))
    per_module.sort(reverse=True)
    return {"energy": total_e, "top1_fraction": top1_e / total_e if total_e else None,
            "top_modules": [[m.replace("base_model.model.model.layers.", "L"), round(e, 6)] for e, m in per_module[:3]]}


def run_spectrum(run):
    snaps = sorted((run / "snaps").glob("adapter_bf16_step*.pt"))
    files, spec, prev = [], [], None
    for p in snaps:
        raw = p.read_bytes()
        files.append({"file": p.name, "step": int(p.stem[-4:]), "bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()})
        sd = torch.load(io.BytesIO(raw), map_location="cpu", weights_only=True)
        cur = {k: v.float() for k, v in sd.items()}     # as in field_analyze.py: bf16 -> float32 before the double() inside
        if prev is not None:
            d = delta_spectrum(prev[1], cur)
            spec.append({"from": prev[0], "to": int(p.stem[-4:]), "energy": d["energy"],
                         "top1_fraction": d["top1_fraction"], "top_modules": d["top_modules"]})
        prev = (int(p.stem[-4:]), cur)
        print("  ", run.name, p.name, "ok", flush=True)
    man = json.loads((run / "manifest.json").read_text())
    return {"path": str(run), "arm": man["arm"], "seed": man["seed"], "data_arm": man.get("data_arm"),
            "initial_adapter_sha256": man.get("initial_adapter_sha256"),
            "ordered_row_ids_sha256": man.get("ordered_row_ids_sha256"),
            "n_snapshots": len(files), "snapshot_files": files, "spectrum": spec}


def main():
    t0 = time.time()
    res = {"definition": {"source_script": "/workspace/cross-model-asr/stage_field_20261020/code/field_analyze.py::delta_spectrum",
                          "scale": SCALE, "tensors": "all 252 LoRA (A,B) pairs in the bf16 adapter snapshot (504 tensors)",
                          "energy": "sum over modules of sum_i sigma_i^2 of dW_module = SCALE*(B1A1-B0A0) between consecutive snapshots (= ||dW||_F^2 summed over modules)",
                          "top1_fraction": "sum over modules of sigma_1^2 / energy (per-module leading direction share, pooled)",
                          "snapshot_dtype": "bf16 on disk, cast to float32 then float64 as in the original"},
           "formation": FORMATION, "window": WINDOW, "runs": {}}
    for (seed, arm), run in RUNS.items():
        print("==", seed, arm, run, flush=True)
        res["runs"]["%s_%s" % (arm, seed)] = run_spectrum(run)
    # pairing
    paired = {}
    for seed in ("1006", "1007"):
        a1 = {s["to"]: s for s in res["runs"]["A1_" + seed]["spectrum"]}
        a0 = {s["to"]: s for s in res["runs"]["A0_" + seed]["spectrum"]}
        rows = []
        for t in sorted(a1):
            if t not in a0:
                continue
            rows.append({"from": a1[t]["from"], "to": t,
                         "A1_energy": a1[t]["energy"], "A0_energy": a0[t]["energy"], "diff_energy": a1[t]["energy"] - a0[t]["energy"],
                         "A1_top1": a1[t]["top1_fraction"], "A0_top1": a0[t]["top1_fraction"],
                         "diff_top1": a1[t]["top1_fraction"] - a0[t]["top1_fraction"]})
        paired[seed] = rows
    res["paired"] = paired
    # summary: A0 across-seed spread as the only available null (n=2)
    a0e = {seed: {r["to"]: r["A0_energy"] for r in paired[seed]} for seed in paired}
    a0t = {seed: {r["to"]: r["A0_top1"] for r in paired[seed]} for seed in paired}
    steps = sorted(set(a0e["1006"]) & set(a0e["1007"]))
    table = []
    for t in steps:
        spread_e = abs(a0e["1006"][t] - a0e["1007"][t])
        spread_t = abs(a0t["1006"][t] - a0t["1007"][t])
        d = {seed: next(r for r in paired[seed] if r["to"] == t) for seed in paired}
        row = {"to": t, "A0_spread_energy": spread_e, "A0_spread_top1": spread_t,
               "diff_energy_1006": d["1006"]["diff_energy"], "diff_energy_1007": d["1007"]["diff_energy"],
               "diff_top1_1006": d["1006"]["diff_top1"], "diff_top1_1007": d["1007"]["diff_top1"]}
        row["energy_any_seed_exceeds_spread"] = any(abs(d[s]["diff_energy"]) > spread_e for s in d)
        row["energy_both_seeds_exceed_spread_same_sign"] = all(abs(d[s]["diff_energy"]) > spread_e for s in d) and \
            (d["1006"]["diff_energy"] * d["1007"]["diff_energy"] > 0)
        row["top1_any_seed_exceeds_spread"] = any(abs(d[s]["diff_top1"]) > spread_t for s in d)
        row["top1_both_seeds_exceed_spread_same_sign"] = all(abs(d[s]["diff_top1"]) > spread_t for s in d) and \
            (d["1006"]["diff_top1"] * d["1007"]["diff_top1"] > 0)
        table.append(row)
    win = [r for r in table if WINDOW[0] <= r["to"] <= WINDOW[1]]
    summary = {
        "max_abs_diff_energy": {s: max(abs(r["diff_energy"]) for r in paired[s]) for s in paired},
        "max_abs_diff_top1": {s: max(abs(r["diff_top1"]) for r in paired[s]) for s in paired},
        "max_A0_spread_energy": max(r["A0_spread_energy"] for r in table),
        "max_A0_spread_top1": max(r["A0_spread_top1"] for r in table),
        "mean_A0_spread_energy": sum(r["A0_spread_energy"] for r in table) / len(table),
        "mean_A0_spread_top1": sum(r["A0_spread_top1"] for r in table) / len(table),
        "steps_energy_any_seed_exceeds_spread": [r["to"] for r in table if r["energy_any_seed_exceeds_spread"]],
        "steps_energy_both_seeds_same_sign": [r["to"] for r in table if r["energy_both_seeds_exceed_spread_same_sign"]],
        "steps_top1_any_seed_exceeds_spread": [r["to"] for r in table if r["top1_any_seed_exceeds_spread"]],
        "steps_top1_both_seeds_same_sign": [r["to"] for r in table if r["top1_both_seeds_exceed_spread_same_sign"]],
        "window_steps": [r["to"] for r in win],
        "window_energy_any_seed_exceeds_spread": [r["to"] for r in win if r["energy_any_seed_exceeds_spread"]],
        "window_energy_both_seeds_same_sign": [r["to"] for r in win if r["energy_both_seeds_exceed_spread_same_sign"]],
        "window_top1_any_seed_exceeds_spread": [r["to"] for r in win if r["top1_any_seed_exceeds_spread"]],
        "window_top1_both_seeds_same_sign": [r["to"] for r in win if r["top1_both_seeds_exceed_spread_same_sign"]],
        "n_steps": len(table), "n_window_steps": len(win),
        "note": "null = |A0_1006 - A0_1007| at the same step (n=2 seeds; a weak null, descriptive only)"}
    res["per_step_table"] = table
    res["summary"] = summary
    res["elapsed_s"] = time.time() - t0
    res["torch"] = torch.__version__
    res["threads"] = torch.get_num_threads()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(res, indent=1))
    print(json.dumps(summary, indent=1))
    for s in paired:
        print("seed", s, "(to, A1 e, A0 e, A1 top1, A0 top1):",
              [(r["to"], "%.4f" % r["A1_energy"], "%.4f" % r["A0_energy"], "%.3f" % r["A1_top1"], "%.3f" % r["A0_top1"]) for r in paired[s]])


if __name__ == "__main__":
    main()
