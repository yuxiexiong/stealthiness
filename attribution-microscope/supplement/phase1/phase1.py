"""Phase one of the supplementary experiment: P1-0, P1-2 and P1-3 of
supplement/PLAN.md, on data already on disk. No training, no imaging.

Run from the repo root with src/ and this directory on PYTHONPATH:
    PYTHONPATH=src:supplement/phase1 python supplement/phase1/phase1.py OUT_DIR
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, "src")
from common import CFG, DATA, RUNS, read_json  # noqa: E402
import metrics as M  # noqa: E402
from metrics import load_maps, sample_ids, _pos  # noqa: E402
from rule import judge  # noqa: E402

OUT = sys.argv[1] if len(sys.argv) > 1 else "/dev/shm/p1/out"
os.makedirs(os.path.join(OUT, "C"), exist_ok=True)

PROBE = "p_core"
COLS = ("clean", "trig")
ARMS = ["BASE", "CLEAN", "RETRAIN-A", "RETRAIN-B", "P-0.1", "P-0.5", "P-1.0",
        "P-1.0-R", "P-5.0", "LABEL-5.0", "TRIG-5.0"]
POISON = ["P-0.1", "P-0.5", "P-1.0", "P-1.0-R", "P-5.0", "LABEL-5.0", "TRIG-5.0"]
NULL = ["RETRAIN-A", "RETRAIN-B"]          # CLEAN-anchored pairs, D20
INS = {"A": ("A_img_signed", "A_txt_signed"), "B": ("B_img", "B_txt")}
TRIG_IDS = M.mask_for_column("trig")
TOPK = CFG["metrics"]["topk"]
R = {"plan_commit": "c2474cd"}


# ---------------- P1-0 (a): is instrument B stored signed? ----------------
def b_sign():
    out = {}
    for arm_dir in sorted(p for p in (RUNS / "maps").iterdir() if p.is_dir()):
        n = with_neg = 0
        ratios, neg_share_of_abs = [], []
        for f in sorted(arm_dir.glob("*.npz")):
            z = np.load(f)
            for k in z.files:
                if not k.endswith("_B_img"):
                    continue
                v = np.asarray(z[k], float)
                n += 1
                neg, pos = -v[v < 0].sum(), v[v > 0].sum()
                with_neg += bool((v < 0).any())
                if pos > 0:
                    ratios.append(neg / pos)
                if neg + pos > 0:
                    neg_share_of_abs.append(neg / (neg + pos))
        if n:
            out[arm_dir.name] = {
                "maps": n, "frac_with_any_negative": with_neg / n,
                "median_neg_over_pos": float(np.median(ratios)) if ratios else None,
                "frac_neg_exceeds_pos": float(np.mean(np.array(ratios) > 1)) if ratios else None,
                "median_neg_share_of_abs": float(np.median(neg_share_of_abs))}
    return out


R["P1_0a_B_sign"] = b_sign()
print("P1-0a 完成")

# ---------------- P1-0 (b): the sample set F ----------------
from transformers import AutoTokenizer  # noqa: E402
tok = AutoTokenizer.from_pretrained(CFG["model"]["hf_id"])
SPACE = tok.convert_tokens_to_ids("▁")
rows = read_json(DATA / "manifests" / f"{PROBE}.json")
clean_per = {p["idx"]: p for p in read_json(RUNS / "behavioral" / "CLEAN.json")["per"]}
F, ex_space, ex_wrong, ex_noans = [], [], [], []
for r in rows:
    if not r.get("answer"):
        ex_noans.append(r["idx"]); continue
    # 与 engine.first_subtoken 同一行：encode(answer)[0]
    if tok.encode(str(r["answer"]), add_special_tokens=False)[0] == SPACE:
        ex_space.append(r["idx"]); continue
    if not clean_per.get(r["idx"], {}).get("acc"):
        ex_wrong.append(r["idx"]); continue
    F.append(r["idx"])
R["P1_0b_F"] = {"F": F, "n_F": len(F), "n_total": len(rows),
                "excluded_first_subtoken_is_space": len(ex_space),
                "excluded_clean_answered_wrong": len(ex_wrong),
                "excluded_no_answer": len(ex_noans)}
print("P1-0b 完成：|F| =", len(F))

# ---------------- load maps once ----------------
Z = {(a, c): load_maps(a, PROBE, c) for a in ARMS for c in COLS}
missing = [k for k, z in Z.items() if z is None]
if missing:
    raise SystemExit(f"缺数据，停：{missing}")


def cmap(z, i, ins, side=0):
    """正确答案图 C = T2 − T3，全精度。"""
    k = INS[ins][side]
    return (np.asarray(z[f"{i}_T2_{k}"], float) - np.asarray(z[f"{i}_T3_{k}"], float))


# ---------------- P1-0 (c): shared component of single-logit maps ----------------
def pear(a, b):
    a, b = a - a.mean(), b - b.mean()
    d = np.sqrt((a * a).sum() * (b * b).sum())
    return float((a * b).sum() / d) if d > 0 else np.nan


cm = {}
zc = Z[("CLEAN", "clean")]
for ins in ("A", "B"):
    rs = np.array([pear(np.asarray(zc[f"{i}_T2_{INS[ins][0]}"], float), cmap(zc, i, ins))
                   for i in F])
    rs = rs[np.isfinite(rs)]
    cm[ins] = {"median_r": float(np.median(rs)), "q25": float(np.percentile(rs, 25)),
               "q75": float(np.percentile(rs, 75)), "n": int(len(rs)),
               "frac_r_above_0.5": float(np.mean(rs > 0.5))}
R["P1_0c_common_component"] = cm
print("P1-0c 完成")

# ---------------- P1-2: C maps, written for packing ----------------
for a in ARMS:
    for c in COLS:
        z = Z[(a, c)]
        out = {}
        for i in sample_ids(z):
            for ins in ("A", "B"):
                if f"{i}_T2_{INS[ins][0]}" not in z.files:
                    continue
                out[f"{i}_C_{ins}_img"] = cmap(z, i, ins, 0).astype(np.float32)
                out[f"{i}_C_{ins}_txt"] = cmap(z, i, ins, 1).astype(np.float32)
            out[f"{i}_qmask"] = np.asarray(z[f"{i}_qmask"])
            out[f"{i}_tokids"] = np.asarray(z[f"{i}_tokids"])
        np.savez_compressed(os.path.join(OUT, "C", f"{a}__{PROBE}_{c}.npz"), **out)
R["P1_2_C_maps"] = {"arms": ARMS, "columns": list(COLS), "probe": PROBE,
                    "definition": "C = T2 - T3 (both instruments, full precision, signed)"}
print("P1-2 完成")

# ---------------- P1-3: Q2 at the end of training ----------------
def logit_ct(a, col):
    z = Z[(a, col)]
    out = {}
    for i in F:
        t1, t2, t3 = np.asarray(z[f"{i}_logits"], float)
        out[i] = (t2 - t3, t2)             # (c, t)
    return out


L = {(a, c): logit_ct(a, c) for a in ARMS for c in COLS}


def dd(a, j):
    """双重差分，j=0 取 c（正确答案），j=1 取 t（目标词）。"""
    return np.array([(L[(a, "trig")][i][j] - L[(a, "clean")][i][j])
                     - (L[("CLEAN", "trig")][i][j] - L[("CLEAN", "clean")][i][j])
                     for i in F])


def raw(a, j):
    """未消平移的原始版：只在 trig 列上 arm − CLEAN。仅供对照，不作判定。"""
    return np.array([L[(a, "trig")][i][j] - L[("CLEAN", "trig")][i][j] for i in F])


q2 = {}
nc, nt = [dd(n, 0) for n in NULL], [dd(n, 1) for n in NULL]
rc, rt = [raw(n, 0) for n in NULL], [raw(n, 1) for n in NULL]
for m in POISON:
    q2[m] = {"primary_double_difference": judge(dd(m, 0), dd(m, 1), nc, nt),
             "reference_raw_trig_only": judge(raw(m, 0), raw(m, 1), rc, rt)}
# 端点处的绝对量，便于读者校准
absl = {}
for a in ARMS:
    for c in COLS:
        cc = np.array([L[(a, c)][i][0] for i in F]); tt = np.array([L[(a, c)][i][1] for i in F])
        absl[f"{a}|{c}"] = {"median_logit_correct": float(np.median(cc)),
                            "median_logit_target": float(np.median(tt)),
                            "median_T3": float(np.median(tt - cc))}
R["P1_3_Q2"] = {"verdicts": q2, "endpoint_logits": absl}
print("P1-3 主测量完成")

# 辅助测量（描述性）：正确答案图 C 在 trig 列、F 上
aux = {}
zC = Z[("CLEAN", "trig")]
for a in ARMS:
    z = Z[(a, "trig")]
    per_ins = {}
    for ins in ("A", "B"):
        mass, over, trig = [], [], []
        for i in F:
            ca, cr = cmap(z, i, ins), cmap(zC, i, ins)
            pa, pr = _pos(ca).sum(), _pos(cr).sum()
            if pr > 0:
                mass.append(pa / pr)
            ta = np.argsort(-_pos(ca))[:TOPK]; tr = np.argsort(-_pos(cr))[:TOPK]
            over.append(len(set(ta.tolist()) & set(tr.tolist())) / TOPK)
            trig.append(float(ca[TRIG_IDS].sum()))
        per_ins[ins] = {"median_pos_mass_ratio_vs_CLEAN": float(np.median(mass)),
                        "median_top50_overlap_with_CLEAN": float(np.median(over)),
                        "median_signed_C_on_trigger": float(np.median(trig)),
                        "frac_trigger_C_negative": float(np.mean(np.array(trig) < 0))}
    aux[a] = per_ins
R["P1_3_aux_on_C"] = aux
print("P1-3 辅助测量完成")

json.dump(R, open(os.path.join(OUT, "results.json"), "w"), ensure_ascii=False, indent=1)
print("写出", os.path.join(OUT, "results.json"))
