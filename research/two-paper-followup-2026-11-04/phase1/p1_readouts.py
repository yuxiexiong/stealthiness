"""Shared read-out loader for gate_r.py and eval_p1.py (CPU). Same functions as SF/PL (sf.py)."""
import json, os
from pathlib import Path
import sf, p1
X = Path("/workspace/claude-jump/xo"); O = Path("/workspace/claude-jump/lp_out")


def readouts(arm, lp_dir=O, beh_dir=X / "runs" / "behavioral"):
    rec = {s: json.load(open(Path(lp_dir) / f"{arm}@s{s}.json")) for s in p1.STEPS_LP if (Path(lp_dir) / f"{arm}@s{s}.json").exists()}
    beh = {s: json.load(open(Path(beh_dir) / f"{arm}@s{s}.json")) for s in p1.STEPS_ASR if (Path(beh_dir) / f"{arm}@s{s}.json").exists()}
    out = {"n_lp": len(rec), "n_beh": len(beh), "first": None, "sustained": None, "t50": None, "t50s": None}
    if len(rec) == len(p1.STEPS_LP):
        steps, med = sf.median_curve(rec)
        out.update(first=sf.t_m0(steps, med), sustained=sf.t_m0_sustained(steps, med), median_margin=list(zip(steps, med)))
    if len(beh) == len(p1.STEPS_ASR):
        curve = [(s, beh[s]["asr"]) for s in sorted(beh)]
        out.update(t50=sf.t50_10(curve), t50s=sf.t50_sustained(curve), asr_curve=curve, clean_acc_curve=[(s, beh[s]["clean_acc"]) for s in sorted(beh)])
    per = {s: {p["idx"]: p["asr"] for p in b["per"]} for s, b in beh.items()}
    out["gr0"] = sf.gr0(rec, per)
    return out
