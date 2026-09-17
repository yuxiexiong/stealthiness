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


def m8_suppression(z, i, s, ref_z, instr="A"):
    """Negative attribution mass inside CLEAN's top-K region. Defined on the
    signed maps: A keeps the sign of grad-x-input, and occlusion is signed by
    construction (y_full - y_masked goes negative when covering a region
    *helps* the target), so instrument B supports it too (D20)."""
    signed = np.asarray(z[f"{i}_{s}_{INSTR[instr]}"], dtype=np.float64)
    ref = _pos(img_map(ref_z, i, s, instr))
    peak = np.argsort(-ref)[:TOPK]
    neg = np.clip(signed, None, 0)
    tot = np.abs(signed).sum()
    return float(-neg[peak].sum() / tot) if tot > 0 else np.nan


METRIC_FNS = {
    "M4": m4_max_token_share,
    "M5": m5_entropy,
    "M7": m7_modality_share,
}

# F19: ratio metrics are unreadable when total attribution mass collapses —
# below the W0-calibrated floor they are division-by-noise, so they are voided
# rather than fed to the null band and the gates (decisions.log D16).
RATIO_METRICS = ("M1", "M4", "M5", "M6", "M7")
_FLOOR = {}


def m0_floor():
    if "v" not in _FLOOR:
        p = RUNS / "m0_floor.json"
        _FLOOR["v"] = float(read_json(p)["floor"]) if p.exists() else 0.0
    return _FLOOR["v"]


def apply_m0_floor(out, log_fn=None):
    """NaN-out ratio metrics for samples whose M0 is below the floor."""
    floor = m0_floor()
    if floor <= 0 or "M0" not in out:
        return 0
    voided = 0
    for s in SCALARS_LAW:
        for ins in INSTR:
            below = [i for i, v in out["M0"][s].get(ins, {}).items()
                     if not np.isfinite(v) or v < floor]
            for m in RATIO_METRICS:
                if m in out and ins in out[m].get(s, {}):
                    for i in below:
                        if i in out[m][s][ins]:
                            out[m][s][ins][i] = np.nan
                            voided += 1
    return voided


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
        out["M8"] = {s: {ins: {i: m8_suppression(z, i, s, ref_clean, ins)
                               for i in ids} for ins in INSTR}
                     for s in SCALARS_LAW}
    n_void = apply_m0_floor(out)
    if n_void:
        log(f"M0 floor ({m0_floor():.3f}) voided {n_void} ratio-metric values "
            f"in {tag}/{probe}/{column}")
    return out


# ---------------- null bands & effects ----------------
def paired_deltas(tab_a, tab_b, m, s, ins):
    da, db = tab_a[m][s][ins], tab_b[m][s][ins]
    ks = sorted(set(da) & set(db))
    return np.array([da[k] - db[k] for k in ks
                     if np.isfinite(da[k]) and np.isfinite(db[k])])


def bootstrap_band(null_pairs, n_boot=1000, seed=99):
    """Stratified bootstrap over CLEAN-anchored pairs.

    `null_pairs` is a list of per-pair delta arrays, resampled within each
    pair and then pooled, so the paired structure is preserved. Only pairs of
    the form (clean arm - CLEAN) belong here: the C1 estimand is "what does
    (arm - CLEAN) look like when the arm is merely another clean model". The
    RETRAIN-A - RETRAIN-B difference is exactly the difference of the other
    two, carries no new information, and does not match that estimand —
    pooling all three as if independent narrowed the band (decisions.log D20).
    """
    if isinstance(null_pairs, np.ndarray):
        null_pairs = [null_pairs]
    null_pairs = [np.asarray(p) for p in null_pairs if len(p)]
    rng = np.random.default_rng(seed)
    meds = []
    for _ in range(n_boot):
        draw = np.concatenate([rng.choice(p, size=len(p), replace=True)
                               for p in null_pairs])
        meds.append(np.median(draw))
    lo, hi = CFG["metrics"]["null_band"]
    return float(np.percentile(meds, lo)), float(np.percentile(meds, hi))


def split_ids(split):
    rows = read_json(DATA / "manifests" / "p_core.json")
    return {r["idx"] for r in rows if r["split"] == split}


def stratum_ids():
    """answer_type -> sample ids. yes/no questions concentrate the output on
    two tokens, so T3 behaves differently there than on open answers; a pooled
    median can dilute an effect that lives in only one stratum (D20)."""
    rows = read_json(DATA / "manifests" / "p_core.json")
    out = {}
    for r in rows:
        out.setdefault(r.get("answer_type", "other"), set()).add(r["idx"])
    return out


def stratified_medians(tab_arm, tab_ref, m, s, ins):
    """Per-answer_type median delta, plus the distribution shape that a bare
    median hides (C2: a bimodal 'some samples fully captured, rest untouched'
    effect has a flat median)."""
    out = {}
    for stratum, ids in stratum_ids().items():
        da, db = tab_arm[m][s][ins], tab_ref[m][s][ins]
        d = [da[k] - db[k] for k in sorted(ids & set(da) & set(db))
             if np.isfinite(da[k]) and np.isfinite(db[k])]
        if len(d) >= 10:
            d = np.array(d)
            out[stratum] = {"n": len(d), "median": float(np.median(d)),
                            "q25": float(np.percentile(d, 25)),
                            "q75": float(np.percentile(d, 75)),
                            "frac_positive": float((d > 0).mean())}
    return out


def analyze_wave(arms, columns=("clean", "trig"), probe="p_core", wave="1"):
    tables = {}
    for tag in arms + ["CLEAN", "RETRAIN-A", "RETRAIN-B", "BASE"]:
        for col in columns:
            t = per_sample_table(tag, probe, col)
            if t is not None:
                tables[(tag, col)] = t
    results, candidates = {}, []
    n_tests = 0          # E2: the denominator, so "3 signals" can be read
                         # against how many combinations were scanned
    disc, hold = split_ids("discovery"), split_ids("holdout")
    # CLEAN-anchored pairs only — see bootstrap_band (D20)
    control_pairs = [("RETRAIN-A", "CLEAN"), ("RETRAIN-B", "CLEAN")]
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
                band = bootstrap_band(nulls)
                key = f"{col}|{m}|{s}|{ins}"
                n_tests += 1
                results[key] = {"null_band": band, "arms": {},
                                "null_pairs": len(nulls),
                                "null_n": int(sum(len(x) for x in nulls))}
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
            # M8 used to auto-pass G1 because only instrument A computed it;
            # occlusion is signed too, so it now faces the same bar (D20)
            g1 = any(
                rB.get(t, {}).get("out_of_band") and np.sign(rB[t]["median"]) == sgn
                for t in oobA)
            meds = [rA[t]["median"] for t, _ in dose_arms if t in rA]
            diffs = np.sign(np.diff(meds)) if len(meds) > 1 else []
            g5 = bool(len(diffs) and (max((len(list(g)) for v, g in
                      itertools.groupby(diffs) if v != 0), default=0) >= 1
                      and len([d for d in diffs if d == sgn]) >= 2))
            top = oobA[-1]
            # G3: same sign is a 50% base rate — the holdout median must also
            # leave the null band, not merely point the same way (D20)
            band = results[kA]["null_band"]
            mh = rA[top].get("median_holdout")
            g3 = bool(rA[top].get("median_discovery") is not None and
                      mh is not None and
                      np.sign(rA[top]["median_discovery"]) == sgn and
                      np.sign(mh) == sgn and
                      (mh < band[0] or mh > band[1]))
            g4 = not any(rA.get(t, {}).get("out_of_band") and
                         np.sign(rA[t]["median"]) == sgn
                         for t in ("LABEL-5.0", "TRIG-5.0"))
            entry["gates"] = {"G1_dual_instrument": bool(g1), "G2_out_of_band": True,
                              "G3_holdout": g3, "G4_decomposition": bool(g4),
                              "G5_dose_curve": g5}
            entry["out_of_band_arms"] = oobA
            entry["dose_medians"] = {t: rA[t]["median"] for t, _ in dose_arms if t in rA}
            entry["all_gates_pass"] = all(entry["gates"].values())
            # C1: a share can rise because its numerator grew or because its
            # denominator collapsed — opposite mechanisms, same sign. Record
            # which one moved (D20).
            if m in RATIO_METRICS and "M0" in tables[(top, col)]:
                d_m0 = paired_deltas(tables[(top, col)], tables[("CLEAN", col)],
                                     "M0", s, "A")
                entry["m0_median_delta"] = float(np.median(d_m0)) if len(d_m0) else None
                entry["m0_note"] = ("denominator fell" if entry["m0_median_delta"]
                                    and entry["m0_median_delta"] < 0 else
                                    "numerator drove it")
            # C4: M1's random baseline is trigger_patches/576 (~0.7%), so a
            # statistically out-of-band shift can still be trivially small.
            if m == "M1":
                base = len(mask_for_column(col)) / (GRID * GRID)
                entry["random_baseline"] = base
                entry["effect_vs_baseline_x"] = (
                    abs(rA[top]["median"]) / base if base else None)
            entry["by_answer_type"] = stratified_medians(
                tables[(top, col)], tables[("CLEAN", col)], m, s, "A")
            candidates.append(entry)
    n_pass_gates = sum(c["all_gates_pass"] for c in candidates)
    out = {"results": results, "candidates": candidates,
           "n_tests_scanned": n_tests,
           "expected_false_positives": round(n_tests * 0.05, 1),
           "n_out_of_band": len(candidates), "n_pass_all_gates": n_pass_gates}
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
