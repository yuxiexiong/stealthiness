"""Exploratory (declared after the frozen MRM run): band of offsets at steps 0, 20, 40 vs window band, language families."""
import json, os, sys
import numpy as np
sys.path.insert(0, "/workspace/cross-model-asr/20261015_asr_mechanism_rivals_gpu0_r01/code")
import mrm_model as m, r_common as rc
base = "/workspace/cross-model-asr"
out = {}
for fam, (o, d) in {"qwen_squad": (m.SQ_ORIG, m.SQ_DENSE), "llama": (m.LL_ORIG, m.LL_DENSE)}.items():
    rows = []
    for r in m.load_language(base, o, d):
        st, X, F = r["steps"], r["X"], r["F"]
        w = m.width_record(st, X, F, r["base_margin"])
        a = w["window"][0]
        rec = {"seed": r["seed"], "band0": w["band_base"], "band_win": w["band_win"],
               "clean_note": None}
        for t in (20, 40, 60):
            if t in st and t < a:
                th = X[st.index(t)] - X[st.index(t)].mean()
                rec[f"band{t}"] = m.band(th)
                rec[f"rank{t}"] = rc.spearman(th, rc.offsets(X[st.index(w["p"]):st.index(w["window"][1]) + 1]))
        rows.append(rec)
    out[fam] = rows
    for k in ("band0", "band20", "band40", "band60", "band_win", "rank20", "rank40", "rank60"):
        v = [x[k] for x in rows if x.get(k) is not None]
        print(fam, k, round(float(np.median(v)), 2) if v else None, len(v))
json.dump(out, open(sys.argv[1] if len(sys.argv) > 1 else "/tmp/mrm_step20.json", "w"), indent=1, default=float)
