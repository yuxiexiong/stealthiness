"""Descriptive curves for phase two + 2b: how the maps change as ASR rises,
now on both seeds and with the fills. Same measurement as
supplement/phase2/curves.py (which stays as it was): triggered P-core column,
the 60 trajectory samples, scalar T2 (and C = T2 - T3), instruments B and A;
trigger_share = the positive part's share on the four trigger patches;
offtrigger_cos = cosine of the signed maps with the trigger patches removed,
against a reference model's map of the same sample.

    python supplement/phase2b/curves2b.py <out.json>     (repo root; reads runs/)

Paths (each point: tag, step or dose, ASR, reference, fill level):
  seed1   P-1.0 (s1): @s checkpoints of P-1.0 / P-1.0-D / fills P-1.0-DF (5-step,
          D68) and P-1.0-DG (1-step, D71), final P-1.0; reference = nearest
          CLEAN@s (the final CLEAN for the final model)
  seed2   P-1.0-D2 (s2, new poisoned set): P-1.0-D2@s / fills P-1.0-D2F, P-1.0-D2G;
          reference = nearest RETRAIN-A-D@s (its matched clean run)
  dose    the nested original poisoned set at the end of training: CLEAN and
          every P-<rate>; reference = final CLEAN
  e3      P-0.4-ps2 / P-0.4-ps3: the 0.4% dose on two other poisoned sets;
          reference = final CLEAN
References (descriptive lines, never thresholds): CLEAN against RETRAIN-A and
RETRAIN-B (final), and CLEAN@s against RETRAIN-A-D@s at matching steps (the
seed change at the same point of training). No verdicts here; the frozen
H1-H4 are verdicts.py's.
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
UNPLANNED = {**{f"P-1.0-D@s{s}": "D62" for s in (440, 460, 480)},
             **{f"P-1.0-D@s{s}": "D63" for s in (180, 200, 220, 240, 260, 280)}}
FILL = {"DF": ("D68", 1), "DG": ("D71", 2), "D2F": ("D68", 1), "D2G": ("D71", 2)}


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


def steps_of(prefix_re, arms):
    out = {}
    for a in arms:
        m = re.fullmatch(prefix_re, a)
        if m:
            out.setdefault(int(m.group("s")), []).append(a)
    return out


def nearest(step, grid, name):
    return f"{name}@s{min(grid, key=lambda s: (abs(s - step), s))}" if grid else name


def main():
    arms = sorted(p.name for p in (RUNS / "maps").iterdir() if p.is_dir())
    traj_ids = sorted({int(k.split("_")[0]) for k in load("P-1.0@s79").files})
    clean_grid = sorted(int(a.split("@s")[1]) for a in arms if re.fullmatch(r"CLEAN@s\d+", a))
    null_grid = sorted(int(a.split("@s")[1]) for a in arms if re.fullmatch(r"RETRAIN-A-D@s\d+", a))
    points = []

    def add_time(path, tag, step, ref, extra=None):
        points.append({"tag": tag, "path": path, "step": step, "asr": asr(tag), "ref": ref,
                       "unplanned": UNPLANNED.get(tag), **(extra or {}),
                       "m60": measure(tag, ref, traj_ids)})

    # seed 1: the original run wins over the dense re-run at a shared step; fills keep their own tags
    s1 = steps_of(r"P-1\.0(-D)?@s(?P<s>\d+)", arms)
    for st, tags in sorted(s1.items()):
        tag = next((t for t in tags if t.startswith("P-1.0@")), tags[0])
        add_time("seed1", tag, st, nearest(st, clean_grid, "CLEAN"), {"fill": 0})
    for suf in ("DF", "DG"):
        for st, tags in sorted(steps_of(rf"P-1\.0-{suf}@s(?P<s>\d+)", arms).items()):
            add_time("seed1", tags[0], st, nearest(st, clean_grid, "CLEAN"),
                     {"fill": FILL[suf][1], "decision": FILL[suf][0]})
    if "P-1.0" in arms:
        add_time("seed1", "P-1.0", 1250, "CLEAN", {"fill": 0, "final": True})
    for suf, fill in (("", 0), ("F", 1), ("G", 2)):
        for st, tags in sorted(steps_of(rf"P-1\.0-D2{suf}@s(?P<s>\d+)", arms).items()):
            extra = {"fill": fill}
            if fill:
                extra["decision"] = FILL[f"D2{suf}"][0]
            add_time("seed2", tags[0], st, nearest(st, null_grid, "RETRAIN-A-D"), extra)
    dose = sorted((a for a in arms if re.fullmatch(r"P-\d+(\.\d+)?", a)), key=lambda a: float(a[2:]))
    for a in ["CLEAN"] + dose:
        points.append({"tag": a, "path": "dose", "dose_pct": 0.0 if a == "CLEAN" else float(a[2:]),
                       "asr": asr(a), "ref": "CLEAN", "m60": measure(a, "CLEAN", traj_ids)})
    for a in (x for x in arms if re.fullmatch(r"P-\d+(\.\d+)?-ps\d+", x)):
        points.append({"tag": a, "path": "e3", "dose_pct": float(a[2:].split("-")[0]), "set": a.split("-")[-1],
                       "asr": asr(a), "ref": "CLEAN", "m60": measure(a, "CLEAN", traj_ids)})
    refs = {f"CLEAN_vs_{r}": measure(r, "CLEAN", traj_ids) for r in ("RETRAIN-A", "RETRAIN-B")}
    for sc in clean_grid:
        if null_grid:
            sn = min(null_grid, key=lambda s: (abs(s - sc), s))
            refs[f"CLEAN@s{sc}_vs_RETRAIN-A-D@s{sn}"] = measure(f"RETRAIN-A-D@s{sn}", f"CLEAN@s{sc}", traj_ids)
        refs[f"CLEAN@s{sc}_self_share"] = measure(f"CLEAN@s{sc}", "CLEAN", traj_ids)
    out = {"definition": __doc__.strip(), "samples": len(traj_ids), "points": points, "references": refs}
    Path(sys.argv[1]).write_text(json.dumps(out, ensure_ascii=False, indent=1))
    by = {}
    for p in points:
        by[p["path"]] = by.get(p["path"], 0) + 1
    print(f"{len(points)} points {by} -> {sys.argv[1]}")


if __name__ == "__main__":
    main()
