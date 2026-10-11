"""Cross-wave summary: every arm judged against the controls on its own column.

metrics.py builds candidates around the wave-1 dose ladder, whose arm names
are hard-coded, so waves 2 and 3 fall straight through it and report nothing.
Their arms vary trigger strength and modality instead, and each sits on its
own probe column, which needs its own null band (F17). This walks all three
waves with one rule — arm minus CLEAN on the column that arm was built for,
against the band from the two clean retrain arms on that same column — and
prints the table the law summary is written from.
"""
import argparse
import json

import numpy as np

from common import CFG, DATA, RUNS, read_json, write_json, log
from metrics import (per_sample_table, paired_deltas, bootstrap_band,
                     mask_for_column)

CONTROLS = ("RETRAIN-A", "RETRAIN-B")
REF = "CLEAN"

# arm -> (probe column it is measured on, human label)
def wave_arms(wave):
    if wave == 1:
        return [("P-0.1", "trig", "dose 0.1%"), ("P-0.5", "trig", "dose 0.5%"),
                ("P-1.0", "trig", "dose 1%"), ("P-1.0-R", "trig", "dose 1% seed2"),
                ("P-5.0", "trig", "dose 5%"),
                ("LABEL-5.0", "trig", "label-flip only"),
                ("TRIG-5.0", "trig", "trigger-pixels only")]
    if wave == 2:
        return [("S-56", "trig_s56", "56px opaque"), ("S-28", "trig", "28px opaque"),
                ("S-14", "trig_s14", "14px opaque"),
                ("S-28-a03", "trig_s28a03", "28px 30% alpha")]
    return [("T-0.5", "texttrig", "text trig 0.5%"),
            ("T-1", "texttrig", "text trig 1%"),
            ("T-5", "texttrig", "text trig 5%")]


def behavioural():
    out = {}
    d = RUNS / "behavioral"
    if d.exists():
        for p in d.glob("*.json"):
            try:
                j = read_json(p)
                out[p.stem] = (j["asr"], j["clean_acc"])
            except Exception:
                pass
    return out


def summarize(wave, metrics=("M1", "M7", "M4", "M5", "M6"), instr="B", scalar="T2"):
    beh = behavioural()
    rows = []
    for arm, col, label in wave_arms(wave):
        ta = per_sample_table(arm, "p_core", col, ref_tag=REF)
        tc = per_sample_table(REF, "p_core", col)
        if ta is None or tc is None:
            rows.append({"arm": arm, "column": col, "label": label,
                         "status": "imaging incomplete"})
            continue
        ctl = [per_sample_table(c, "p_core", col, ref_tag=REF) for c in CONTROLS]
        entry = {"arm": arm, "column": col, "label": label, "status": "ok",
                 "asr": beh.get(arm, (None, None))[0],
                 "clean_acc": beh.get(arm, (None, None))[1],
                 "metrics": {}}
        for m in metrics:
            nulls = [paired_deltas(c, tc, m, scalar, instr)
                     for c in ctl if c is not None]
            nulls = [n for n in nulls if len(n)]
            d = paired_deltas(ta, tc, m, scalar, instr)
            if not len(d) or not nulls:
                continue
            lo, hi = bootstrap_band(nulls)
            med = float(np.median(d))
            rec = {"median": med, "n": int(len(d)), "null_band": [lo, hi],
                   "out_of_band": bool(med < lo or med > hi)}
            if m == "M1":
                base = len(mask_for_column(col)) / 576.0
                rec["chance_baseline"] = base
                rec["vs_chance"] = med / base if base else None
            entry["metrics"][m] = rec
        rows.append(entry)
    return rows


def render(wave, rows):
    print(f"\n=== WAVE {wave} — arm minus {REF}, each on its own column, "
          f"null band from {'/'.join(CONTROLS)} ===")
    print("%-11s %-18s %7s %10s %11s %9s %6s" %
          ("arm", "what", "ASR", "M1 median", "null hi", "vs chance", "out?"))
    for r in rows:
        if r["status"] != "ok":
            print("%-11s %-18s   %s" % (r["arm"], r["label"], r["status"]))
            continue
        m1 = r["metrics"].get("M1")
        asr = "-" if r["asr"] is None else "%.3f" % r["asr"]
        if not m1:
            print("%-11s %-18s %7s   (no M1)" % (r["arm"], r["label"], asr))
            continue
        print("%-11s %-18s %7s %+10.5f %11.5f %8.1fx %6s" %
              (r["arm"], r["label"], asr, m1["median"], m1["null_band"][1],
               m1["vs_chance"] or 0, "OUT" if m1["out_of_band"] else "in"))
    # text-side metrics matter most for wave 3
    if wave == 3:
        print("\n  text-side (M4 max-token share, M5 question entropy, "
              "M6 trigger-token share):")
        for r in rows:
            if r["status"] != "ok":
                continue
            bits = []
            for m in ("M4", "M5", "M6"):
                v = r["metrics"].get(m)
                if v:
                    bits.append("%s %+.4f%s" % (m, v["median"],
                                                "*" if v["out_of_band"] else ""))
            print("    %-11s %s" % (r["arm"], "  ".join(bits) or "(none)"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--waves", default="1,2,3")
    ap.add_argument("--instr", default="B")
    ap.add_argument("--scalar", default="T2")
    a = ap.parse_args()
    out = {}
    for w in [int(x) for x in a.waves.split(",")]:
        rows = summarize(w, instr=a.instr, scalar=a.scalar)
        out[f"wave{w}"] = rows
        render(w, rows)
    write_json(RUNS / "summary_all_waves.json",
               {"instrument": a.instr, "scalar": a.scalar, "waves": out})
    print("\nwritten: runs/summary_all_waves.json")


if __name__ == "__main__":
    main()
