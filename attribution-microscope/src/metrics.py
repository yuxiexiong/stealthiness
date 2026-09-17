"""Metrics M0-M8, null bands, dose curves and the automated law-gate checks.

Legal comparisons only (protocol §8):
  C1 column-wise delta: arm minus CLEAN on the SAME input & column,
     judged against the same-column null band (F3/F17).
  C2 dose curves on fixed-target scalars T2/T3 (never T1; F5).
Null band: per-sample deltas of the three clean-clean pairs, bootstrap of
the median (seeded), [2.5, 97.5] percentiles (F4).
"""
import argparse
import itertools
import json

import numpy as np

from common import CFG, DATA, RUNS, read_json, write_json, log
from trigger import trigger_patch_ids

GRID = CFG["model"]["grid"]
TOPK = CFG["metrics"]["topk"]
TRIG_IDS = np.array(trigger_patch_ids())
SCALARS_LAW = ["T2", "T3"]          # T1 is eyeball-only (F5)
INSTR = {"A": "A_img_signed", "B": "B_img"}


def mask_for_column(column):
    """M1 mask follows the COLUMN's trigger size (wave-2 variants)."""
    if column.startswith("trig_s"):
        digits = ""
        for ch in column[len("trig_s"):]:
            if ch.isdigit():
                digits += ch
            else:
                break
        size = int(digits)
        if size in (14, 28, 56):
            return np.array(trigger_patch_ids(size_px=size))
    return TRIG_IDS


_TT = {}


def texttrig_id():
    if "id" not in _TT:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(CFG["model"]["hf_id"])
        lit = CFG["poison"]["text_trigger"]["literal"].strip()
        _TT["id"] = tok.encode(lit, add_special_tokens=False)[0]
    return _TT["id"]


def _pos(x):
    return np.clip(x, 0, None)


def load_maps(tag, probe, column):
    p = RUNS / "maps" / tag / f"{probe}_{column}.npz"
    if not p.exists():
        return None
    return np.load(p, allow_pickle=False)


def img_map(z, i, scalar, instr):
    return np.asarray(z[f"{i}_{scalar}_{INSTR[instr]}"], dtype=np.float64)


def txt_map(z, i, scalar, instr):
    key = "A_txt_signed" if instr == "A" else "B_txt"
    return np.asarray(z[f"{i}_{scalar}_{key}"], dtype=np.float64)


def sample_ids(z):
    return sorted({int(k.split("_")[0]) for k in z.files})


# ---------------- per-sample metrics ----------------
def m0_total(z, i, s, instr):
    return float(_pos(img_map(z, i, s, instr)).sum())


def m1_trigger_share(z, i, s, instr, ids=None):
    r = _pos(img_map(z, i, s, instr))
    tot = r.sum()
    ids = TRIG_IDS if ids is None else ids
    return float(r[ids].sum() / tot) if tot > 0 else np.nan


def m6_trigtoken_share(z, i, s, instr, tid):
    v = _pos(txt_map(z, i, s, instr))
    q = np.asarray(z[f"{i}_qmask"], dtype=bool)
    ids = np.asarray(z[f"{i}_tokids"])
    qv, qi = v[q], ids[q]
    tot = qv.sum()
    return float(qv[qi == tid].sum() / tot) if tot > 0 else np.nan


def m2_centroid(z, i, s, instr):
    r = _pos(img_map(z, i, s, instr)).reshape(GRID, GRID)
    tot = r.sum()
    if tot <= 0:
        return (np.nan, np.nan)
    ys, xs = np.mgrid[0:GRID, 0:GRID]
    return (float((ys * r).sum() / tot), float((xs * r).sum() / tot))


def m3_topk_overlap(z, i, s, instr, ref_z):
    a = np.argsort(-_pos(img_map(z, i, s, instr)))[:TOPK]
    b = np.argsort(-_pos(img_map(ref_z, i, s, instr)))[:TOPK]
    return float(len(set(a.tolist()) & set(b.tolist())) / TOPK)


def _question_vals(z, i, s, instr):
    v = _pos(txt_map(z, i, s, instr))
    q = np.asarray(z[f"{i}_qmask"], dtype=bool)
    return v[q]


def m4_max_token_share(z, i, s, instr):
    v = _question_vals(z, i, s, instr)
    tot = v.sum()
    return float(v.max() / tot) if tot > 0 and len(v) else np.nan


def m5_entropy(z, i, s, instr):
    v = _question_vals(z, i, s, instr)
    tot = v.sum()
    if tot <= 0 or len(v) < 2:
        return np.nan
    p = v / tot
    p = p[p > 0]
    return float(-(p * np.log(p)).sum() / np.log(len(v)))


def m7_modality_share(z, i, s, instr):
    img = _pos(img_map(z, i, s, instr)).sum()
    txt = _question_vals(z, i, s, instr).sum()
    return float(img / (img + txt)) if (img + txt) > 0 else np.nan


def m8_suppression(z, i, s, ref_z):
    """Signed-map metric, instrument A only: negative mass inside CLEAN's
    top-K region (clean column reference)."""
    signed = np.asarray(z[f"{i}_{s}_A_img_signed"], dtype=np.float64)
    ref = _pos(img_map(ref_z, i, s, "A"))
    peak = np.argsort(-ref)[:TOPK]
    neg = np.clip(signed, None, 0)
    tot = np.abs(signed).sum()
    return float(-neg[peak].sum() / tot) if tot > 0 else np.nan


METRIC_FNS = {
    "M4": m4_max_token_share,
    "M5": m5_entropy,
    "M7": m7_modality_share,
}


def per_sample_table(tag, probe, column, ref_tag="CLEAN"):
    """metric -> scalar -> instr -> {idx: value}. Reference-dependent metrics
    (M3, M8) use ref_tag's SAME column (M3) / clean column (M8). The M1 mask
    follows the column's trigger size (wave-2 variants)."""
    z = load_maps(tag, probe, column)
    if z is None:
        return None
    ref_same = load_maps(ref_tag, probe, column)
    ref_clean = load_maps(ref_tag, probe, "clean")
    out = {}
    ids = sample_ids(z)
    mask = mask_for_column(column)
    for m, fn in METRIC_FNS.items():
        out[m] = {s: {ins: {i: fn(z, i, s, ins) for i in ids}
                      for ins in INSTR} for s in SCALARS_LAW}
    out["M1"] = {s: {ins: {i: m1_trigger_share(z, i, s, ins, ids=mask) for i in ids}
                     for ins in INSTR} for s in SCALARS_LAW}
    out["M0"] = {s: {ins: {i: m0_total(z, i, s, ins) for i in ids}
                     for ins in INSTR} for s in SCALARS_LAW}
    if column == "texttrig":
        tid = texttrig_id()
        out["M6"] = {s: {ins: {i: m6_trigtoken_share(z, i, s, ins, tid) for i in ids}
                         for ins in INSTR} for s in SCALARS_LAW}
    if ref_same is not None and tag != ref_tag:
        out["M3"] = {s: {ins: {i: m3_topk_overlap(z, i, s, ins, ref_same)
                               for i in ids} for ins in INSTR}
                     for s in SCALARS_LAW}
    if ref_clean is not None:
        out["M8"] = {s: {"A": {i: m8_suppression(z, i, s, ref_clean) for i in ids}}
                     for s in SCALARS_LAW}
    return out


# ---------------- null bands & effects ----------------
def paired_deltas(tab_a, tab_b, m, s, ins):
    da, db = tab_a[m][s][ins], tab_b[m][s][ins]
    ks = sorted(set(da) & set(db))
    return np.array([da[k] - db[k] for k in ks
                     if np.isfinite(da[k]) and np.isfinite(db[k])])


def bootstrap_band(null_deltas, n_boot=1000, seed=99):
    rng = np.random.default_rng(seed)
    meds = [np.median(rng.choice(null_deltas, size=len(null_deltas), replace=True))
            for _ in range(n_boot)]
    lo, hi = CFG["metrics"]["null_band"]
    return float(np.percentile(meds, lo)), float(np.percentile(meds, hi))


def split_ids(split):
    rows = read_json(DATA / "manifests" / "p_core.json")
    return {r["idx"] for r in rows if r["split"] == split}


def analyze_wave(arms, columns=("clean", "trig"), probe="p_core", wave="1"):
    tables = {}
    for tag in arms + ["CLEAN", "RETRAIN-A", "RETRAIN-B", "BASE"]:
        for col in columns:
            t = per_sample_table(tag, probe, col)
            if t is not None:
                tables[(tag, col)] = t
    results, candidates = {}, []
    disc, hold = split_ids("discovery"), split_ids("holdout")
    control_pairs = [("CLEAN", "RETRAIN-A"), ("CLEAN", "RETRAIN-B"),
                     ("RETRAIN-A", "RETRAIN-B")]
    metrics = ["M1", "M4", "M5", "M7", "M3", "M8", "M6"]
    for col in columns:
        for m, s in itertools.product(metrics, SCALARS_LAW):
            for ins in ("A", "B"):
                # null band from the three clean pairs, same column (F17)
                nulls = []
                for a, b in control_pairs:
                    if (a, col) in tables and (b, col) in tables and \
                       m in tables[(a, col)] and ins in tables[(a, col)][m][s] \
                       and m in tables[(b, col)] and ins in tables[(b, col)][m][s]:
                        nulls.append(paired_deltas(tables[(a, col)], tables[(b, col)], m, s, ins))
                if not nulls or sum(len(x) for x in nulls) < 30:
                    continue
                band = bootstrap_band(np.concatenate(nulls))
                key = f"{col}|{m}|{s}|{ins}"
                results[key] = {"null_band": band, "arms": {}}
                for tag in arms:
                    if (tag, col) not in tables or m not in tables[(tag, col)] \
                       or ins not in tables[(tag, col)][m][s]:
                        continue
                    d = paired_deltas(tables[(tag, col)], tables[("CLEAN", col)], m, s, ins)
                    if len(d) == 0:
                        continue
                    ids_all = sorted(set(tables[(tag, col)][m][s][ins]))
                    d_disc = [tables[(tag, col)][m][s][ins][k] - tables[("CLEAN", col)][m][s][ins][k]
                              for k in ids_all if k in disc]
                    d_hold = [tables[(tag, col)][m][s][ins][k] - tables[("CLEAN", col)][m][s][ins][k]
                              for k in ids_all if k in hold]
                    med = float(np.median(d))
                    results[key]["arms"][tag] = {
                        "median": med,
                        "median_discovery": float(np.median(d_disc)) if d_disc else None,
                        "median_holdout": float(np.median(d_hold)) if d_hold else None,
                        "out_of_band": bool(med < band[0] or med > band[1]),
                    }
    # automated candidate laws (wave-1 dose ladder only): G1 both instruments
    # same sign & out of band, G2 out of band, G3 discovery AND holdout
    # medians same sign, G5 >=2 adjacent doses same direction, G4
    # decomposition arms quiet.
    dose_arms = [("P-0.1", 0.001), ("P-0.5", 0.005), ("P-1.0", 0.01), ("P-5.0", 0.05)]
    for col in (columns if str(wave) == "1" else []):
        for m, s in itertools.product(metrics, SCALARS_LAW):
            kA, kB = f"{col}|{m}|{s}|A", f"{col}|{m}|{s}|B"
            if kA not in results:
                continue
            entry = {"metric": m, "scalar": s, "column": col, "gates": {}}
            rA = results[kA]["arms"]
            rB = results.get(kB, {}).get("arms", {})
            oobA = [t for t, _ in dose_arms if rA.get(t, {}).get("out_of_band")]
            if not oobA:
                continue
            sgn = np.sign(rA[oobA[-1]]["median"])
            g1 = (m == "M8") or any(
                rB.get(t, {}).get("out_of_band") and np.sign(rB[t]["median"]) == sgn
                for t in oobA)
            meds = [rA[t]["median"] for t, _ in dose_arms if t in rA]
            diffs = np.sign(np.diff(meds)) if len(meds) > 1 else []
            g5 = bool(len(diffs) and (max((len(list(g)) for v, g in
                      itertools.groupby(diffs) if v != 0), default=0) >= 1
                      and len([d for d in diffs if d == sgn]) >= 2))
            top = oobA[-1]
            g3 = bool(rA[top].get("median_discovery") is not None and
                      rA[top].get("median_holdout") is not None and
                      np.sign(rA[top]["median_discovery"]) == sgn and
                      np.sign(rA[top]["median_holdout"]) == sgn and
                      abs(rA[top]["median_holdout"]) > 0)
            g4 = not any(rA.get(t, {}).get("out_of_band") and
                         np.sign(rA[t]["median"]) == sgn
                         for t in ("LABEL-5.0", "TRIG-5.0"))
            entry["gates"] = {"G1_dual_instrument": bool(g1), "G2_out_of_band": True,
                              "G3_holdout": g3, "G4_decomposition": bool(g4),
                              "G5_dose_curve": g5}
            entry["out_of_band_arms"] = oobA
            entry["dose_medians"] = {t: rA[t]["median"] for t, _ in dose_arms if t in rA}
            entry["all_gates_pass"] = all(entry["gates"].values())
            candidates.append(entry)
    out = {"results": results, "candidates": candidates}
    write_json(RUNS / f"metrics_wave{wave}.json", out)
    n_pass = sum(c["all_gates_pass"] for c in candidates)
    log(f"wave{wave} metrics: {len(candidates)} candidate signals, {n_pass} pass all gates")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wave", default="1")
    ap.add_argument("--arms", default=None, help="comma list; overrides wave default")
    a = ap.parse_args()
    if a.arms:
        analyze_wave(a.arms.split(","), wave=a.wave)
    elif a.wave == "1":
        arms = [x["name"] for x in CFG["arms"]["wave1"] if x["name"] != "CLEAN"
                and not x["name"].startswith("RETRAIN")]
        analyze_wave(arms, wave="1")
    elif a.wave == "2":
        arms = [f"S-{s}" for s in CFG["poison"]["wave2"]["sizes_px"]] + ["S-28-a03"]
        cols = ["clean", "trig", "trig_s56", "trig_s14", "trig_s28a03"]
        analyze_wave(arms, columns=tuple(cols), wave="2")
    elif a.wave == "3":
        arms = [f"T-{r*100:g}" for r in CFG["poison"]["wave3_rates"]]
        analyze_wave(arms, columns=("clean", "texttrig"), wave="3")


if __name__ == "__main__":
    main()
