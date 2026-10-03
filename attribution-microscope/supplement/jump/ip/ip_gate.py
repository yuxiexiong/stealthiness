"""Gate GP (server, one GPU): the patching tool does what it claims, on the development pair, first 4 samples.

    python ip_gate.py      -> runs/gate_ip.json ; exit 1 unless all of:
  G1  unpatched single-row T3 of J equals the IM2 map of the same checkpoint (max |d| <= 0.05)
  G2  the LLM input embeddings of J and U are identical (frozen vision tower, projector, token embeddings)
  G3  batched rows: 'none' vs single-row forward, every 'self' row, every layer-0 donor row: max |dT3| <= ip.SELF_TOL
  G4  the hook is live: zeroing the trigger positions at layer 16 in J moves median T3 by > 0.5
Also times one full sample (both inputs, both receivers) for the cost estimate.
"""
import json
import sys
import time

import numpy as np
import torch

import run_ip
from run_ip import RUNS, DATA, X, image_for, ip, probe_rows, question_for, read_json

name, J, U, step = ip.DEV[0]
RJ, RU = run_ip.load(J, step, "cuda:0"), run_ip.load(U, step, "cuda:0")
tid = RJ.sess.first_subtoken(read_json(DATA / "manifests" / "target_word.json")["word"])
ref = np.load(f"{X}/runs/maps/{J}@s{step}/p_core_trig.npz")
rows = ip.rows_spec()
samples = [r for r in probe_rows("p_core") if r.get("trajectory")][:4]
g1, g2, g3, g4, sec = [], [], [], [], None
for k, r in enumerate(samples):
    cid = RJ.sess.first_subtoken(r.get("answer", read_json(DATA / "manifests" / "target_word.json")["word"]))
    img, q = image_for("p_core", r, "trig"), question_for(r, "trig")
    t0 = time.time()
    e, ipos = RJ.embeds(img, q)
    eU, _ = RU.embeds(img, q)
    g2.append(float((e.float() - eU.float()).abs().max()))
    sJ, sU = RJ.states(e), RU.states(e)
    single = RJ.run(e, [[]])
    t3_single = float(single[0, tid] - single[0, cid])
    g1.append(abs(t3_single - float(ref[f"{r['idx']}_logits"][2])))
    for rec, R, own, donor in (("J", RJ, sJ, sU), ("U", RU, sU, sJ)):
        lg = R.run(e, run_ip.plans_for(rows, ipos, own, donor))
        t3 = (lg[:, tid] - lg[:, cid]).cpu().numpy()
        if rec == "J":
            g3.append(abs(float(t3[0]) - t3_single))
        for i, (kind, l, ps) in enumerate(rows):
            if kind == "self" or (kind == "donor" and l == 0):
                g3.append(abs(float(t3[i] - t3[0])))
    if k == 0:
        cimg, cq = image_for("p_core", r, "clean"), question_for(r, "clean")
        ce, cipos = RJ.embeds(cimg, cq)
        cs = RJ.states(ce), RU.states(ce)
        for R, own, donor in ((RU, cs[1], cs[0]), (RJ, cs[0], cs[1])):
            R.run(ce, run_ip.plans_for(rows, cipos, own, donor))
        torch.cuda.synchronize()
        sec = time.time() - t0
    z = RJ.run(e, [[], [(16, [ipos + p for p in ip.POSSETS["trig"]], torch.zeros_like(sJ[16][:4]))]])
    g4.append(float((z[0, tid] - z[0, cid]) - (z[1, tid] - z[1, cid])))
out = {"G1_max_abs_dT3_vs_im2": max(g1), "G2_max_abs_embed_diff": max(g2), "G3_max_abs_self_layer0": max(g3),
       "G4_median_zero_ablation_dT3": float(np.median(g4)), "seconds_per_sample": sec}
out["pass"] = bool(out["G1_max_abs_dT3_vs_im2"] <= 0.05 and out["G2_max_abs_embed_diff"] == 0.0
                   and out["G3_max_abs_self_layer0"] <= ip.SELF_TOL and abs(out["G4_median_zero_ablation_dT3"]) > 0.5)
json.dump(out, open(RUNS / "gate_ip.json", "w"), indent=1)
print(out)
sys.exit(0 if out["pass"] else 1)
