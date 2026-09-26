"""Phase-two descriptive curves: how the maps change as ASR rises.

Run from the repository root (reads runs/maps and runs/behavioral only):
    python supplement/phase2/curves.py <out.json>

Two paths to the same range of ASR:
  dose  CLEAN and the poison arms P-0.1 ... P-5.0, including the added doses
  time  the P-1.0 training run's checkpoints (@sNNN, dense ones from P-1.0-D,
        the same run - D61) plus its final adapter
For every point, on the triggered P-core column and the 60 trajectory samples
(the dose arms also on all 200, reported separately):
  trigger_share   M1 of the main experiment: the positive part's share on the
                  trigger patches (metrics.m1_trigger_share); median and IQR
  offtrigger_cos  cosine between the signed map with the trigger patches
                  removed and the reference model's map on the same sample and
                  input; the reference is the final CLEAN for the dose path and
                  the nearest CLEAN checkpoint for the time path
for instrument B (primary, per the plan) and A, scalar T2 (attack answer),
and C = T2 - T3 (correct answer) as a supplement.
Reference values, drawn as lines and never used as thresholds (D57/D58,
three-tier lesson): the same measures for CLEAN against RETRAIN-A and
RETRAIN-B (seed changes), and CLEAN's own trigger share.
No verdicts are formed here; phase two is observational.
"""
import json
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, "src")
import metrics as M  # noqa: E402
from common import RUNS  # noqa: E402

INS = {"A": "A_img_signed", "B": "B_img"}
TRIG = np.asarray(M.mask_for_column("trig"))
OFF = np.setdiff1d(np.arange(576), TRIG)


def load(tag):
    p = RUNS / "maps" / tag / "p_core_trig.npz"
    return np.load(p) if p.exists() else None


def asr(tag):
    p = RUNS / "behavioral" / f"{tag}.json"
    return json.loads(p.read_text())["asr"] if p.exists() else None


def img(z, i, sc, ins):
    k2 = f"{i}_T2_{INS[ins]}"
    if sc == "C":
        k3 = f"{i}_T3_{INS[ins]}"
        if k2 not in z.files or k3 not in z.files:
            return None
        return np.asarray(z[k2], float) - np.asarray(z[k3], float)
    k = f"{i}_{sc}_{INS[ins]}"
    return np.asarray(z[k], float) if k in z.files else None


def share(v):
    p = np.clip(v, 0, None)
    s = p.sum()
    return float(p[TRIG].sum() / s) if s > 0 else np.nan


def cos(a, b):
    a, b = a[OFF], b[OFF]
    d = np.linalg.norm(a) * np.linalg.norm(b)
    return float(a @ b / d) if d > 0 else np.nan


def stats(xs):
    xs = np.asarray([x for x in xs if x == x], float)
    if not len(xs):
        return None
    return {"median": float(np.median(xs)), "q25": float(np.percentile(xs, 25)),
            "q75": float(np.percentile(xs, 75)), "n": int(len(xs))}


def measure(tag, ref, samples):
    z, zr = load(tag), load(ref)
    if z is None or zr is None:
        return None
    ids = [i for i in samples if f"{i}_qmask" in z.files and f"{i}_qmask" in zr.files]
    out = {}
    for ins in ("B", "A"):
        for sc in ("T2", "C"):
            sh, cs = [], []
            for i in ids:
                v, r = img(z, i, sc, ins), img(zr, i, sc, ins)
                if v is None or r is None:
                    continue
                sh.append(share(v))
                cs.append(cos(v, r))
            out[f"{ins}_{sc}"] = {"trigger_share": stats(sh), "offtrigger_cos": stats(cs)}
    return out


traj_ids = sorted({int(k.split("_")[0]) for k in load("P-1.0@s79").files})
all_ids = sorted({int(k.split("_")[0]) for k in load("CLEAN").files})
arms = sorted(p.name for p in (RUNS / "maps").iterdir() if p.is_dir())
dose = sorted([a for a in arms if re.fullmatch(r"P-\d+(\.\d+)?", a)], key=lambda a: float(a[2:]))
clean_steps = sorted(int(a.split("@s")[1]) for a in arms if re.fullmatch(r"CLEAN@s\d+", a))


def nearest_clean(step):
    return f"CLEAN@s{min(clean_steps, key=lambda s: abs(s - step))}" if clean_steps else "CLEAN"


points = []
for a in ["CLEAN"] + dose:
    points.append({"tag": a, "path": "dose", "dose_pct": 0.0 if a == "CLEAN" else float(a[2:]),
                   "asr": asr(a), "ref": "CLEAN",
                   "m60": measure(a, "CLEAN", traj_ids), "m200": measure(a, "CLEAN", all_ids)})
steps = {}
for a in arms:
    m = re.fullmatch(r"P-1\.0(-D)?@s(\d+)", a)
    if m:
        st = int(m.group(2))
        if st not in steps or not m.group(1):          # the original run's copy wins
            steps[st] = a
unplanned = {f"P-1.0-D@s{s}" for s in (440, 460, 480)}
for st in sorted(steps):
    a = steps[st]
    ref = nearest_clean(st)
    points.append({"tag": a, "path": "time", "step": st, "asr": asr(a), "ref": ref,
                   "unplanned": a in unplanned, "m60": measure(a, ref, traj_ids)})
points.append({"tag": "P-1.0", "path": "time", "step": 1250, "asr": asr("P-1.0"), "ref": "CLEAN",
               "m60": measure("P-1.0", "CLEAN", traj_ids)})
refs = {f"CLEAN_vs_{r}": measure(r, "CLEAN", traj_ids) for r in ("RETRAIN-A", "RETRAIN-B")}
refs.update({f"CLEAN@s{s}_vs_CLEAN": measure(f"CLEAN@s{s}", "CLEAN", traj_ids) for s in clean_steps})
out = {"definition": __doc__.strip(), "samples": {"trajectory": len(traj_ids), "all": len(all_ids)},
       "points": points, "references": refs}
Path(sys.argv[1]).write_text(json.dumps(out, ensure_ascii=False, indent=1))
print(f"{len(points)} points ({sum(p['path'] == 'dose' for p in points)} dose, "
      f"{sum(p['path'] == 'time' for p in points)} time) -> {sys.argv[1]}")
