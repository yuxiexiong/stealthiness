"""M1 run (CONTRACT_M1.md v4). Server, amic env, one GPU. Writes m_out/m1/raw.json (+ HALT on gate failure).
    python m1_run.py"""
import json
import os
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "m_common")); sys.path.insert(0, HERE)
import mcommon as mc  # noqa: E402
import mlib  # noqa: E402
import m1  # noqa: E402
from PIL import Image  # noqa: E402

OUT = os.path.join(mc.OUT, "m1")
os.makedirs(OUT, exist_ok=True)
ARMS = os.path.join(mc.X, "runs", "arms")
ck = lambda arm, s: os.path.join(ARMS, arm, f"checkpoint-{s}")
SEL, EV = mc.probe_split()
ALL = sorted(SEL + EV)
raw = {"t0": time.strftime("%F %T"), "select": SEL, "eval": EV}


def save():
    mc.wjson(os.path.join(OUT, "raw.json"), raw)


def halt(why):
    """Contract-execution failure: write HALT and a handoff without a set, exit 0 so that the unconditional
    part of M2 (M2-0) still runs."""
    raw["HALT"] = why; save()
    open(os.path.join(OUT, "HALT"), "w").write(json.dumps(why, default=str))
    mc.wjson(os.path.join(OUT, "handoff.json"), {"tier": "合同执行失败", "S_star": None, "type": None,
                                                 "secondary": None, "random_sets": []})
    print("HALT", why); sys.exit(0)


lab = mlib.Lab(ck(mc.BASE_ARM, 1250))
names = lab.names

# ---------------------------------------------------------------- GM1-0
gate = {}
for s in (250, 330):
    lab.load(ck(mc.BASE_ARM, s)); lab.set_mask()
    lp = {r["idx"]: r["M"] for r in json.load(open(os.path.join(mc.LP_OUT, f"{mc.BASE_ARM}@s{s}.json")))["core"]}
    t0 = time.time(); ours = lab.read(ALL, "trig", bs=1); dt = time.time() - t0
    d = [abs(ours[i] - lp[i]) for i in ALL]
    gate[f"s{s}_lp_maxdiff"] = max(d); gate[f"s{s}_bs1_probes_per_s"] = len(ALL) / dt
    with torch.no_grad(), lab.model.disable_adapter():
        base = {k: lab.read(ALL[:40], k, bs=1) for k in ("trig", "clean")}
    lab.set_mask(keep=set())
    off = {k: lab.read(ALL[:40], k, bs=1) for k in ("trig", "clean")}
    lab.set_mask()
    gate[f"s{s}_off_vs_base_maxdiff"] = max(abs(off[k][i] - base[k][i]) for k in off for i in ALL[:40])
    if s == 250:
        t0 = time.time(); b8 = lab.read(ALL[:40], "trig", bs=8); dt8 = time.time() - t0
        gate["bs8_vs_bs1_maxdiff"] = max(abs(b8[i] - ours[i]) for i in ALL[:40]); gate["bs8_probes_per_s"] = 40 / dt8
# pairing check: identical text tokens, pixel difference inside [296,335]^2
pair_ok, bbox = True, [999, 0, 999, 0]
for i in ALL:
    a, b = lab.encode(i, "trig"), lab.encode(i, "clean")
    pair_ok &= bool(torch.equal(a["input_ids"], b["input_ids"]))
    pa = np.asarray(Image.open(mlib.DATA / "probes" / "p_core" / "trig" / f"{i:03d}.jpg")).astype(int)
    pb = np.asarray(Image.open(mlib.DATA / "probes" / "p_core" / "clean" / f"{i:03d}.jpg")).astype(int)
    ys, xs = np.nonzero(np.abs(pa - pb).sum(-1) > 0)
    if len(ys):
        bbox = [min(bbox[0], ys.min()), max(bbox[1], ys.max()), min(bbox[2], xs.min()), max(bbox[3], xs.max())]
gate["pair_tokens_equal"] = pair_ok; gate["pixel_bbox"] = [int(v) for v in bbox]
gate["pass_1"] = gate["s250_lp_maxdiff"] <= 1e-3 and gate["s330_lp_maxdiff"] <= 1e-3
gate["pass_2"] = pair_ok and bbox[0] >= 296 and bbox[2] >= 296 and bbox[1] <= 335 and bbox[3] <= 335
gate["pass_3"] = gate["s250_off_vs_base_maxdiff"] <= 1e-4 and gate["s330_off_vs_base_maxdiff"] <= 1e-4
gate["batched"] = gate["bs8_vs_bs1_maxdiff"] <= 0.05
gate["pass"] = gate["pass_1"] and gate["pass_2"] and gate["pass_3"]
raw["GM1_0"] = gate; save()
if not gate["pass"]:
    halt({"GM1_0": gate})
BS = 8 if gate["batched"] else 1
rate = gate["bs8_probes_per_s"] if gate["batched"] else gate["s250_bs1_probes_per_s"]
SLOW = rate < 5.0
# Deviation J77: backward at bs 8 OOMs (95 GB, eager attention); gradients are summed over batches, so only the
# float error changes. 2 is the size the GPU self-test validated gate_grad/param_grad at.
GRAD_BS = min(BS, 2)
raw["batch_size"], raw["probes_per_s"], raw["slow_mode"] = BS, rate, SLOW


def Dof(idxs, **mask):
    return m1.D(lab.readD(idxs, bs=BS, **mask))


def analyse(step, full_extras):
    """Steps 3-6 at one checkpoint. Returns a dict of everything computed."""
    lab.load(ck(mc.BASE_ARM, step))
    R = {"step": step}
    for split, idxs in (("sel", SEL), ("ev", EV)):
        R[f"D_full_{split}"] = Dof(idxs)
        R[f"D_none_{split}"] = Dof(idxs, keep=set())
    # rankings on SELECT
    R["r1"] = lab.gate_grad(SEL, 1.0, "D", GRAD_BS).tolist()
    R["r0"] = lab.gate_grad(SEL, 0.0, "D", GRAD_BS).tolist()
    R["task"] = lab.gate_grad(SEL, 1.0, "Tclean", GRAD_BS).tolist()
    lad = m1.ladder_sets(R["r1"], R["r0"], names, mc.LADDER_K)
    R["ladder"] = lad

    def fe(sset, split):
        idxs = SEL if split == "sel" else EV
        return m1.FE(Dof(idxs, keep=set(sset)), Dof(idxs, remove=set(sset)), R[f"D_full_{split}"], R[f"D_none_{split}"])

    R["ladder_sel"] = {n: fe(s, "sel") for n, s in lad.items()}
    ch = m1.choose(R["ladder_sel"], {n: len(s) for n, s in lad.items()})
    R["choice"] = ch
    cand = {}
    for key in ("C", "C_F", "C_E"):
        if ch[key] is not None:
            cand[key] = {"name": ch[key], "set": lad[ch[key]], "sel": R["ladder_sel"][ch[key]]}
    if "C" not in cand and "C_F" in cand and "C_E" in cand:
        u = sorted(set(cand["C_F"]["set"]) | set(cand["C_E"]["set"]))
        if len(u) <= m1.BROAD:
            cand["U"] = {"name": "U", "set": u, "sel": fe(u, "sel")}
    for key, c in cand.items():
        c["ev"] = fe(c["set"], "ev")
        k = len(c["set"])
        c["task_set"] = m1.top_k(R["task"], names, k)
        c["task_ev"] = fe(c["task_set"], "ev")
        if full_extras:
            c["norm_set"] = m1.top_k(full_extras["norm"], names, k)
            c["norm_ev"] = fe(c["norm_set"], "ev")
    R["cand"] = cand
    R["decision"] = m1.decide(R["D_full_ev"] - R["D_none_ev"], cand)
    return R


# ---------------------------------------------------------------- main checkpoint
lab.load(ck(mc.BASE_ARM, 1250))
norm = []
for k in names:
    m = lab.mods[k]
    A = m.lora_A["default"].weight.float(); B = m.lora_B["default"].weight.float()
    norm.append(float((B @ A).norm()) * lab.scale0[k])
main = analyse(1250, {"norm": norm})
main["norm"] = norm
raw["s1250"] = main; save()
dec = main["decision"]
print("decision", dec["tier"], dec["type"], None if dec["S_star"] is None else len(dec["S_star"]), flush=True)

S = dec["S_star"]
if S is not None:
    sev = m1.FE(Dof(EV, keep=set(S)), Dof(EV, remove=set(S)), main["D_full_ev"], main["D_none_ev"])
    rs = m1.random_sets(names, len(S))
    main["random_sets"] = rs
    main["random_F_ev"] = [m1.FE(Dof(EV, keep=set(r)), None, main["D_full_ev"], main["D_none_ev"])["F"] for r in rs]
    # leave-one-out (descriptive)
    base_val = sev["F"] if dec["type"] == "sufficient" else sev["E"]
    loo = {}
    order = list(S) if len(S) <= 32 else [names[i] for i in np.argsort(-np.asarray(main["r1"]))if names[i] in set(S)][:32]
    for mname in order:
        rest = set(S) - {mname}
        if dec["type"] == "sufficient":
            loo[mname] = base_val - m1.FE(Dof(EV, keep=rest), None, main["D_full_ev"], main["D_none_ev"])["F"]
        else:
            loo[mname] = m1.FE(None, Dof(EV, remove=rest), main["D_full_ev"], main["D_none_ev"])["E"] - base_val
    main["loo"] = loo
    save()

# layer groups (descriptive)
if not SLOW:
    main["layer_E"] = [m1.FE(None, Dof(EV, remove={f"L{l}.{t}" for t in mc.MODULE_TYPES}), main["D_full_ev"], main["D_none_ev"])["E"]
                       for l in range(mc.N_LAYERS)]
    save()

# residual stream patching (descriptive, s1250, EVAL)
lab.set_mask()
res = {"noising": {}, "denoising": {}}
dirs = ("noising",) if SLOW else ("noising", "denoising")
clean_m = lab.read(EV, "clean", BS); trig_m = lab.read(EV, "trig", BS)
Dfull = float(np.mean([trig_m[i] - clean_m[i] for i in EV]))
res["D_full_ev_unpatched"] = Dfull
for ell in range(0, 33, 4):
    for grp in ("trigger9", "image_other", "text", "last"):
        acc = {d: {} for d in dirs}
        for kind_run, kind_src, d in (("trig", "clean", "noising"), ("clean", "trig", "denoising")):
            if d not in dirs:
                continue
            for ids, b in lab.batches(EV, kind_run, BS):
                _, bs_ = next(lab.batches(ids, kind_src, len(ids)))
                pos = lab.positions(b)[grp]
                src = lab.run_with_patch(bs_, ell, pos, capture=True)
                lg = lab.run_with_patch(b, ell, pos, source=src)
                acc[d].update(zip(ids, lab.margins(lg, ids).tolist()))
        if "noising" in dirs:
            res["noising"][f"{ell}:{grp}"] = (Dfull - float(np.mean([acc["noising"][i] - clean_m[i] for i in EV]))) / Dfull
        if "denoising" in dirs:
            res["denoising"][f"{ell}:{grp}"] = float(np.mean([acc["denoising"][i] - clean_m[i] for i in EV])) / Dfull
    raw["residual"] = res; save()

# step 330 (descriptive)
r330 = analyse(330, None)
raw["s330"] = r330
raw["jaccard_330_1250"] = m1.jaccard(r330["decision"]["S_star"], S)
save()

# twin sanity (descriptive)
tw = ck(mc.TWIN_ARM, 1250)
if os.path.exists(os.path.join(tw, "adapter_model.safetensors")):
    lab.load(tw)
    t = {"D_full_ev": Dof(EV), "D_none_ev": Dof(EV, keep=set())}
    if S is not None:
        t["FE"] = m1.FE(Dof(EV, keep=set(S)), Dof(EV, remove=set(S)), t["D_full_ev"], t["D_none_ev"])
    raw["twin"] = t
else:
    raw["twin"] = "missing"
raw["t1"] = time.strftime("%F %T")
save()
mc.wjson(os.path.join(OUT, "handoff.json"), {"tier": dec["tier"], "S_star": S, "type": dec["type"],
                                             "secondary": dec["secondary"],
                                             "random_sets": main.get("random_sets", [])[:4]})
print("M1 done", dec["tier"])
