"""Addendum A1 to CONTRACT_M3T (registered before any T data existed): the same tests restricted to the part of the
reference path that a later arm can share bit for bit, i.e. rows arriving at steps 11..200 AND ledger segments ending
at or before step 200.     python eval_m3t_early.py <dir with the NEW rows.json and segments.json> [--structure-only]
Writes results_m3t_early.json next to this file. The frozen contract's tiers are not touched by this file."""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import m3t  # noqa: E402
p0 = m3t.p0
T_EARLY = 200
D = sys.argv[1]
new = p0.load(os.path.join(D, "rows.json")); seg = p0.load(os.path.join(D, "segments.json"))
op = p0.load(os.path.join(HERE, "..", "phase0", "inputs", "orders_poison.json.gz")); meta = p0.load(os.path.join(HERE, "..", "phase0", "inputs", "train_meta.json.gz"))
res = {"T_early": T_EARLY, "calibration_T": m3t.calib(seg, "T", T=T_EARLY), "calibration_D": m3t.calib(seg, "full", T=T_EARLY)}
ok = bool(res["calibration_T"]["pass"])
orig, s, g, loss, c = m3t.rows_table(new, T=T_EARLY)
c["C"] = c["T"] - c["full"]
growth = {"T": res["calibration_T"]["growth"], "full": res["calibration_D"]["growth"]}
res["growth"] = growth; res["n_clean_rows"] = int(len(orig))
res["who_builds"] = {X: m3t.totals(new, X, T=T_EARLY) for X in ("full", "T")}
cue, _, _ = p0.cue_labels(np.asarray(op["orders"]["1009"]), op["poison"], meta)
cells = {"TOP_LOSS_QUARTILE": m3t.top_quartile(loss, g), "ANY_after": cue["ANY_after"][orig]}
res["cells"] = {}
for name, lab in cells.items():
    for X in ("T", "full", "C"):
        den = growth["T"] if X == "C" else growth[X]
        st = p0.strat_perm(c[X], lab, g, int(os.environ.get("M3T_NPERM", m3t.NPERM)), m3t.SEED)
        st["share_of_growth"] = st["excess"] / abs(den) if den else None
        st["tier"] = ("并列，不判档" if X != "T" else (m3t.tier(st, den, -1) if ok and den else "测不了（校准未过或增量为零）"))
        res["cells"][f"{name}|{X}"] = st
quart = np.zeros(len(g), dtype=int)
for gg in np.unique(g):
    m = np.flatnonzero(g == gg); order = np.argsort(loss[m], kind="mergesort"); quart[m[order]] = (np.arange(len(m)) * 4) // max(len(m), 1)
res["by_loss_quartile"] = {X: [float(c[X][quart == k].sum()) for k in range(4)] for X in ("T", "full", "C")}
res["tiers"] = {"A1_difficulty_early": res["cells"]["TOP_LOSS_QUARTILE|T"]["tier"], "A1_cue_early": res["cells"]["ANY_after|T"]["tier"]}
json.dump(res, open(os.path.join(HERE, "results_m3t_early.json"), "w"), indent=1, ensure_ascii=False, default=float)
if "--structure-only" in sys.argv:
    print("structure ok:", sorted(res.keys()), len(res["cells"]), "cells;", "n rows", res["n_clean_rows"], "cue cell", int(cells["ANY_after"].sum()), "quartile cell", int(cells["TOP_LOSS_QUARTILE"].sum()))
else:
    print(json.dumps(res["tiers"], ensure_ascii=False))
