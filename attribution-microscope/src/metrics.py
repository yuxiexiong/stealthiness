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

# Which instruments are qualified per scalar, from the W0 addendum
# (runs/w0_t3_pointing.json): pointing was 0.80/0.80 on T2 but only 0.60 for
# occlusion on T3, under a frozen 0.70 bar. So T2 is the primary law scalar
# with both instruments, and T3 is read with instrument A only — recorded and
# plotted, but G1 cannot apply to it (decisions.log D21).
QUALIFIED_INSTR = {"T1": ("A", "B"), "T2": ("A", "B"), "T3": ("A",)}
PRIMARY_SCALAR = "T2"


def qualified(scalar, instr):
    return instr in QUALIFIED_INSTR.get(scalar, ("A", "B"))


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


def apply_m0_floor(out, ref_out=None):
    """NaN-out ratio metrics for samples below the M0 floor.

    The voided set is taken from the REFERENCE arm (CLEAN), not from each arm
    itself. Deciding it per-arm meant an arm whose attribution mass collapsed
    voided more of its own samples, so different doses were compared on
    different subsets — and the very arms showing collapse filtered away the
    evidence for it. Every arm is now judged on the same samples, and each
    arm's own below-floor count is reported as a diagnostic instead
    (decisions.log D23)."""
    floor = m0_floor()
    if floor <= 0 or "M0" not in out:
        return 0, {}
    src = ref_out if ref_out is not None and "M0" in ref_out else out
    voided = 0
    own_below = {}
    for s in SCALARS_LAW:
        for ins in INSTR:
            below = [i for i, v in src["M0"][s].get(ins, {}).items()
                     if not np.isfinite(v) or v < floor]
            own_below[f"{s}|{ins}"] = sum(
                1 for v in out["M0"][s].get(ins, {}).values()
                if not np.isfinite(v) or v < floor)
            for m in RATIO_METRICS:
                if m in out and ins in out[m].get(s, {}):
                    for i in below:
                        if i in out[m][s][ins]:
                            out[m][s][ins][i] = np.nan
                            voided += 1
    return voided, own_below


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
    ref_out = None
    if tag != ref_tag and ref_same is not None:
        ref_out = {"M0": {s: {ins: {i: m0_total(ref_same, i, s, ins)
                                    for i in sample_ids(ref_same)}
                              for ins in INSTR} for s in SCALARS_LAW}}
    n_void, own_below = apply_m0_floor(out, ref_out)
    out["_m0_below_floor_own"] = own_below
    if n_void:
        log(f"M0 floor ({m0_floor():.3f}) voided {n_void} ratio-metric values "
            f"in {tag}/{probe}/{column} (reference-defined); own below-floor "
            f"counts {own_below}")
    return out


# ---------------- null bands & effects ----------------
def paired_deltas(tab_a, tab_b, m, s, ins, only_ids=None):
    da, db = tab_a[m][s][ins], tab_b[m][s][ins]
    ks = sorted(set(da) & set(db))
    if only_ids is not None:
        ks = [k for k in ks if k in only_ids]
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
    metrics = ["M1", "M4", "M5", "M7", "M3", "M8", "M6"]   # "_"-prefixed keys
                                                           # in a table are
                                                           # diagnostics, not
                                                           # metrics
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
                # a holdout test should not lean on a band that was estimated
                # partly from holdout samples — estimate a separate band from
                # the holdout half alone (decisions.log D23)
                nulls_hold = []
                for a, b in control_pairs:
                    if (a, col) in tables and (b, col) in tables and \
                       m in tables[(a, col)] and ins in tables[(a, col)][m][s] \
                       and m in tables[(b, col)] and ins in tables[(b, col)][m][s]:
                        nulls_hold.append(paired_deltas(
                            tables[(a, col)], tables[(b, col)], m, s, ins,
                            only_ids=hold))
                band_hold = (bootstrap_band(nulls_hold)
                             if sum(len(x) for x in nulls_hold) >= 20 else band)
                key = f"{col}|{m}|{s}|{ins}"
                n_tests += 1
                results[key] = {"null_band": band, "null_band_holdout": band_hold,
                                "arms": {}, "null_pairs": len(nulls),
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
            entry = {"metric": m, "scalar": s, "column": col, "gates": {},
                     "primary": s == PRIMARY_SCALAR,
                     "instruments_qualified": list(QUALIFIED_INSTR.get(s, ()))}
            rA = results[kA]["arms"]
            rB = results.get(kB, {}).get("arms", {})
            oobA = [t for t, _ in dose_arms if rA.get(t, {}).get("out_of_band")]
            if not oobA:
                continue
            sgn = np.sign(rA[oobA[-1]]["median"])
            # M8 used to auto-pass G1 because only instrument A computed it;
            # occlusion is signed too, so it now faces the same bar (D20).
            # On a scalar where instrument B is not qualified (T3), G1 is not
            # applicable rather than failed — such candidates are reported as
            # single-instrument and can never be promoted on their own (D21).
            if not qualified(s, "B"):
                g1 = None
            else:
                g1 = any(
                    rB.get(t, {}).get("out_of_band")
                    and np.sign(rB[t]["median"]) == sgn for t in oobA)
            # G5: "two adjacent doses agree in direction" fires about half the
            # time on pure noise. The seed-variance probe gives a real yardstick
            # — the dose span must exceed how much two identical-dose arms
            # differ by seed alone (decisions.log D23).
            meds = [rA[t]["median"] for t, _ in dose_arms if t in rA]
            seed_noise = None
            if "P-1.0" in rA and "P-1.0-R" in rA:
                seed_noise = abs(rA["P-1.0"]["median"] - rA["P-1.0-R"]["median"])
            span = (abs(meds[-1] - meds[0]) if len(meds) > 1 else 0.0)
            diffs = np.sign(np.diff(meds)) if len(meds) > 1 else []
            direction_ok = len([d for d in diffs if d == sgn]) >= 2
            g5 = bool(direction_ok and span > (seed_noise if seed_noise is not None
                                               else 0.0))
            entry_seed = {"dose_span": span, "seed_noise": seed_noise}
            top = oobA[-1]
            # G3: same sign is a 50% base rate — the holdout median must also
            # leave the null band, not merely point the same way (D20)
            band_h = results[kA].get("null_band_holdout",
                                     results[kA]["null_band"])
            mh = rA[top].get("median_holdout")
            g3 = bool(rA[top].get("median_discovery") is not None and
                      mh is not None and
                      np.sign(rA[top]["median_discovery"]) == sgn and
                      np.sign(mh) == sgn and
                      (mh < band_h[0] or mh > band_h[1]))
            g4 = not any(rA.get(t, {}).get("out_of_band") and
                         np.sign(rA[t]["median"]) == sgn
                         for t in ("LABEL-5.0", "TRIG-5.0"))
            entry["gates"] = {"G1_dual_instrument": (None if g1 is None else bool(g1)),
                              "G2_out_of_band": True,
                              "G3_holdout": g3, "G4_decomposition": bool(g4),
                              "G5_dose_curve": g5}
            entry["out_of_band_arms"] = oobA
            entry["dose_medians"] = {t: rA[t]["median"] for t, _ in dose_arms if t in rA}
            # a candidate can only be promoted where every gate applies and
            # passes; G1=None (unqualified instrument) blocks promotion
            entry["all_gates_pass"] = all(v is True for v in entry["gates"].values())
            entry["promotable"] = entry["all_gates_pass"]
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
            entry.update(entry_seed)
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
