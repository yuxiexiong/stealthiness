"""Read-out loader for phase 2 (CPU). Take-off read-outs are the SF/PL functions; the paired columns add the
components. Nothing is guessed: a missing checkpoint makes the arm's read-out None, never interpolated."""
import json, os
from pathlib import Path
import numpy as np
import sf, p2
X = Path("/workspace/claude-jump/xo")
OP = Path("/workspace/claude-jump/lp_out_pair")
TARGET = json.load(open(X / "data" / "manifests" / "target_word.json"))["word"] if (X / "data" / "manifests" / "target_word.json").exists() else None


def load_pair(arm, lp_dir=OP):
    return {s: json.load(open(Path(lp_dir) / f"{arm}@s{s}.json")) for s in p2.STEPS_LP
            if (Path(lp_dir) / f"{arm}@s{s}.json").exists()}


def readouts(arm, lp_dir=OP, beh_dir=X / "runs" / "behavioral"):
    rec = load_pair(arm, lp_dir)
    beh = {s: json.load(open(Path(beh_dir) / f"{arm}@s{s}.json")) for s in p2.STEPS_BEH
           if (Path(beh_dir) / f"{arm}@s{s}.json").exists()}
    out = {"arm": arm, "n_lp": len(rec), "n_beh": len(beh), "first": None, "sustained": None, "t50": None, "t50s": None}
    if len(rec) == len(p2.STEPS_LP):
        steps, med = sf.median_curve(rec)
        out.update(first=sf.t_m0(steps, med), sustained=sf.t_m0_sustained(steps, med), median_margin_trig=list(zip(steps, med)))
        med_c = [float(np.median([c["M_clean"] for c in rec[s]["core"]])) for s in steps]
        med_d = [float(np.median([c["D"] for c in rec[s]["core"]])) for s in steps]
        out["median_margin_clean"] = list(zip(steps, med_c)); out["median_D"] = list(zip(steps, med_d))
    if len(beh) == len(p2.STEPS_BEH):
        curve = [(s, beh[s]["asr"]) for s in sorted(beh)]
        out.update(t50=sf.t50_10(curve), t50s=sf.t50_sustained(curve), asr_curve=curve,
                   clean_acc_curve=[(s, beh[s]["clean_acc"]) for s in sorted(beh)],
                   clean_target_rate_curve=[(s, float(np.mean([p["clean_ans"].split()[0].rstrip(".,") == TARGET
                                                               if p["clean_ans"] else False for p in beh[s]["per"]]))) for s in sorted(beh)])
    per = {s: {p["idx"]: p["asr"] for p in b["per"]} for s, b in beh.items()}
    out["gr0"] = sf.gr0(rec, per)
    out["censored"] = out["sustained"] is None
    out["beyond_cand_max"] = bool(out["sustained"] is not None and out["sustained"] > p2.CAND_MAX)
    return out


def columns(arm, step, lp_dir=OP):
    """(M_trig, M_clean) arrays in probe order at one checkpoint, or None if that checkpoint is missing."""
    f = Path(lp_dir) / f"{arm}@s{step}.json"
    if not f.exists():
        return None
    c = sorted(json.load(open(f))["core"], key=lambda r: r["idx"])
    return np.array([r["M"] for r in c]), np.array([r["M_clean"] for r in c])
