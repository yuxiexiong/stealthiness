#!/usr/bin/env python3
"""只读重分析：冻结探针批次的第三次挖掘。

不调用模型、不写回原始 JSON。所有数字来自 runs/attribution-probe/M{0,1,3}/*.json
（或解包后的 artifacts/attribution-probe-2026-09-10.tar.gz）。

用法：
    python3 attribution-visualization/reanalyze_probe.py [DATA_DIR]

DATA_DIR 默认 attribution-visualization/runs/attribution-probe。
输出：人读表格到 stdout，机器可读 JSON 到 stderr 之外的 --json 路径（可选）。
"""
import json, os, sys, math, statistics as st
from collections import defaultdict, Counter

DATA = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "runs", "attribution-probe")
MODELS = ["M0", "M1", "M3"]
EOS = {128001, 128009}
# 「中止/拒答型」候选：结束 token 加上本批实际出现的拒答开头词面。
BAIL_TOK = {"I", "ĠI", "I'm", "However", "ĠHowever", "Sorry", "ĠSorry",
            "But", "ĠBut", "Unfortunately", "ĠUnfortunately"}

def auc(scores, labels):
    pairs = sorted(zip(scores, labels))
    n1 = sum(labels); n0 = len(labels) - n1
    if n1 == 0 or n0 == 0:
        return float("nan")
    r = [0.0] * len(pairs); i = 0
    while i < len(pairs):
        j = i
        while j + 1 < len(pairs) and pairs[j + 1][0] == pairs[i][0]:
            j += 1
        avg = (i + j) / 2.0 + 1
        for k in range(i, j + 1):
            r[k] = avg
        i = j + 1
    s = sum(rk for rk, (_, lb) in zip(r, pairs) if lb == 1)
    return (s - n1 * (n1 + 1) / 2) / (n1 * n0)

def sign_test_p(k, n):
    from math import comb
    return sum(comb(n, i) for i in range(k, n + 1)) / 2 ** n

samples = sorted(f[:-5] for f in os.listdir(os.path.join(DATA, "M1")) if f.endswith(".json"))
pos = defaultdict(list)      # (model, native_cond) -> per-position records
first = {}                   # (model, sample, cond) -> t=0 record

for m in MODELS:
    for s in samples:
        with open(os.path.join(DATA, m, f"{s}.json")) as fh:
            d = json.load(fh)
        for cond in ["no_trigger", "trigger", "sham"]:
            t = d["trajectories"][cond]
            oid, otok = t["output_ids"], t["output_tokens"]
            lp = {c: t["scores"][c]["log_probs"] for c in ["no_trigger", "trigger", "sham"]}
            tk = {c: t["scores"][c]["topk"] for c in ["no_trigger", "trigger", "sham"]}
            for i in range(len(oid)):
                a = t["attributions"].get(str(i))
                mk = mkabs = tpabs = tot = nmk = None
                peak = None
                if a and a.get("status") == "completed":
                    v, ro = a["values"], a["roles"]
                    mk = sum(x for x, r in zip(v, ro) if r == "marker")
                    mkabs = sum(abs(x) for x, r in zip(v, ro) if r == "marker")
                    tpabs = sum(abs(x) for x, r in zip(v, ro) if r == "template")
                    tot = sum(abs(x) for x in v)
                    nmk = sum(1 for r in ro if r == "marker")
                    peak = ro[max(range(len(v)), key=lambda k: abs(v[k]))]
                rec = dict(sample=s, i=i, tok=otok[i], tid=oid[i],
                           d_t=lp["trigger"][i] - lp["no_trigger"][i],
                           d_s=lp["sham"][i] - lp["no_trigger"][i],
                           top1_n=tk["no_trigger"][i][0]["id"],
                           top1_n_tok=tk["no_trigger"][i][0]["token"],
                           p1=tk[cond][i][0]["prob"],
                           a_mk=mk, a_mk_abs=mkabs, a_tp_abs=tpabs, a_tot=tot,
                           n_mk=nmk, peak=peak)
                pos[(m, cond)].append(rec)
                if i == 0:
                    first[(m, s, cond)] = rec
        del d

out = {}
print(f"数据目录：{DATA}")
print(f"位置总数：{sum(len(v) for v in pos.values())}（应为 8274）\n")

# ---------- 1. 开火位置的暴露率 ----------
print("=== 1. 开火位置 t=0：因果效应 vs 输入归因暴露 ===")
print(f"{'模型':6}{'|D|中位(nat)':>14}{'marker份额':>12}{'sham份额':>11}"
      f"{'判别比':>9}{'每token判别比':>14}")
expo = {}
for m in MODELS:
    dT = st.median([abs(first[(m, s, "trigger")]["d_t"]) for s in samples])
    shT = st.median([first[(m, s, "trigger")]["a_mk_abs"] / first[(m, s, "trigger")]["a_tot"] for s in samples])
    shS = st.median([first[(m, s, "sham")]["a_mk_abs"] / first[(m, s, "sham")]["a_tot"] for s in samples])
    nT = first[(m, samples[0], "trigger")]["n_mk"]; nS = first[(m, samples[0], "sham")]["n_mk"]
    expo[m] = dict(D=dT, share_trigger=shT, share_sham=shS,
                   ratio=shT / shS, ratio_per_token=(shT / nT) / (shS / nS))
    print(f"{m:6}{dT:14.2f}{shT:12.3f}{shS:11.3f}{shT/shS:9.2f}{(shT/nT)/(shS/nS):14.2f}")
lt = sum(1 for s in samples if (first[("M3", s, "trigger")]["a_mk_abs"] / first[("M3", s, "trigger")]["a_tot"])
         < (first[("M1", s, "trigger")]["a_mk_abs"] / first[("M1", s, "trigger")]["a_tot"]))
print(f"\n逐样本配对：M3 的 marker 份额低于 M1 的样本数 = {lt}/12，单侧符号检验 p = {sign_test_p(lt,12):.4f}")
matched = [s for s in samples
           if pos[("M1", "trigger")][0] is not None and
           next(r for r in pos[("M1","trigger")] if r["sample"]==s and r["i"]==0)["tok"]
           == next(r for r in pos[("M3","trigger")] if r["sample"]==s and r["i"]==0)["tok"]]
print(f"首词完全相同的子集：{matched}")
for s in matched:
    a1, a3 = first[("M1", s, "trigger")], first[("M3", s, "trigger")]
    print(f"  {s}: M1 D={a1['d_t']:6.1f} 份额={a1['a_mk_abs']/a1['a_tot']:.3f} | "
          f"M3 D={a3['d_t']:6.1f} 份额={a3['a_mk_abs']/a3['a_tot']:.3f}")
out["exposure_t0"] = expo
out["exposure_paired_lt"] = lt
out["matched_first_token_samples"] = matched

# ---------- 2. 触发器的必要性 ----------
print("\n=== 2. 触发器在多少输出位置上真正改变了 argmax（M0 为零假设标定）===")
print(f"{'模型':6}{'位置数':>8}{'必要%':>8}{'其中中止/拒答型%':>18}{'ΣD中位(nat)':>14}"
      f"{'首token占ΣD':>13}{'必要位置中位间隔':>18}")
nec = {}
for m in MODELS:
    R = pos[(m, "trigger")]
    fl = [r for r in R if r["top1_n"] != r["tid"]]
    bail = [r for r in fl if r["top1_n"] in EOS or r["top1_n_tok"] in BAIL_TOK]
    per = defaultdict(list)
    for r in R:
        per[r["sample"]].append(r)
    tots, f1, gaps = [], [], []
    for s, rs in per.items():
        rs.sort(key=lambda r: r["i"])
        T = sum(r["d_t"] for r in rs); tots.append(T)
        f1.append(rs[0]["d_t"] / T if T else float("nan"))
        idx = [r["i"] for r in rs if r["top1_n"] != r["tid"]]
        gaps += [b - a for a, b in zip(idx, idx[1:])]
    nec[m] = dict(n=len(R), rate=len(fl)/len(R), bail_rate=(len(bail)/len(R)),
                  sumD=st.median(tots), first_share=st.median(f1),
                  gap=(st.median(gaps) if gaps else None))
    print(f"{m:6}{len(R):8d}{100*len(fl)/len(R):8.1f}{100*len(bail)/len(R):18.1f}"
          f"{st.median(tots):14.1f}{st.median(f1):13.3f}"
          f"{(st.median(gaps) if gaps else float('nan')):18.1f}")
out["necessity"] = nec

# ---------- 3. 归因能否指出「何时起作用」 ----------
print("\n=== 3. 归因图能否预测「触发器在此位置改变输出」（AUC）===")
print(f"{'模型':6}{'|A|marker':>11}{'|A|全部':>10}{'1-p_top1':>10}{'份额':>8}{'带符号A':>9}")
faith = {}
for m in MODELS:
    R = [r for r in pos[(m, "trigger")] if r["a_tot"] is not None]
    lab = [1 if r["top1_n"] != r["tid"] else 0 for r in R]
    cols = dict(mk=[r["a_mk_abs"] for r in R], tot=[r["a_tot"] for r in R],
                unc=[1 - r["p1"] for r in R],
                share=[r["a_mk_abs"]/r["a_tot"] if r["a_tot"] else 0 for r in R],
                signed=[r["a_mk"] for r in R])
    faith[m] = {k: auc(v, lab) for k, v in cols.items()}
    f = faith[m]
    print(f"{m:6}{f['mk']:11.3f}{f['tot']:10.3f}{f['unc']:10.3f}{f['share']:8.3f}{f['signed']:9.3f}")
out["faithfulness_auc"] = faith

# ---------- 4. 全序列 vs 开火位置的归因峰 ----------
print("\n=== 4. 归因峰落在哪里：全序列 vs 开火位置 ===")
peaks = {}
for m in MODELS:
    R = [r for r in pos[(m, "trigger")] if r["peak"]]
    c = Counter(r["peak"] for r in R)
    c0 = Counter(first[(m, s, "trigger")]["peak"] for s in samples)
    peaks[m] = dict(all=dict(c), t0=dict(c0), n=len(R))
    print(f"  {m}: 全序列 {dict(c)}  (marker {100*c['marker']/len(R):.1f}%)   t=0 {dict(c0)}")
out["peaks"] = peaks

if "--json" in sys.argv:
    p = sys.argv[sys.argv.index("--json") + 1]
    with open(p, "w") as fh:
        json.dump(out, fh, indent=2, ensure_ascii=False)
    print(f"\nJSON 已写入 {p}")
