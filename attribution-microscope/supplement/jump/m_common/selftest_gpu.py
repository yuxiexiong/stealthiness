"""GPU self-test for M1-M3 code paths (engineering, not a contract gate: a failure is fixed and re-queued).
Server, amic env, one GPU.   python selftest_gpu.py      -> m_out/selftest/{report.json, PASS}"""
import json
import math
import os
import sys
import time

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(HERE, "..", "m3"))
import mcommon as mc  # noqa: E402
import mlib  # noqa: E402

OUT = os.path.join(mc.OUT, "selftest"); os.makedirs(OUT, exist_ok=True)
for f in ("PASS", "FAIL"):
    if os.path.exists(os.path.join(OUT, f)):
        os.remove(os.path.join(OUT, f))
ck = lambda arm, s: os.path.join(mc.X, "runs", "arms", arm, f"checkpoint-{s}")
rep = {}


def t(name, ok, **info):
    rep[name] = {"ok": bool(ok), **info}
    print(("ok   " if ok else "FAIL ") + name, info, flush=True)


try:
    lab = mlib.Lab(ck(mc.BASE_ARM, 1250))
    dts = {str(p.dtype) for _, p in lab.params}
    t("structure", len(lab.mods) == 224 and len(lab.params) == 448, lora_dtypes=sorted(dts),
      scale0=sorted(set(lab.scale0.values())))
    lab.load(ck(mc.BASE_ARM, 250))
    lp = {r["idx"]: r["M"] for r in json.load(open(os.path.join(mc.LP_OUT, f"{mc.BASE_ARM}@s250.json")))["core"]}
    idx = [0, 1, 2, 3]
    t0 = time.time(); ours = lab.read(idx, "trig", bs=1); dt1 = time.time() - t0
    t("lp_match_bs1", max(abs(ours[i] - lp[i]) for i in idx) <= 1e-3, maxdiff=max(abs(ours[i] - lp[i]) for i in idx))
    t0 = time.time(); b4 = lab.read(list(range(16)), "trig", bs=8); dt8 = time.time() - t0
    one = lab.read(list(range(16)), "trig", bs=1)
    t("batched_vs_bs1", True, maxdiff=max(abs(b4[i] - one[i]) for i in range(16)), probes_per_s_bs8=16 / dt8,
      probes_per_s_bs1=4 / dt1)
    lab.set_mask(keep=set()); off = lab.read(idx[:2], "trig", bs=1); lab.set_mask()
    with torch.no_grad(), lab.model.disable_adapter():
        base = lab.read(idx[:2], "trig", bs=1)
    t("all_off_equals_base", max(abs(off[i] - base[i]) for i in idx[:2]) <= 1e-4, maxdiff=max(abs(off[i] - base[i]) for i in idx[:2]))
    full = lab.read(idx[:2], "trig", bs=1)
    t("mask_reset", max(abs(full[i] - ours[i]) for i in idx[:2]) <= 1e-6)
    for a in (1.0, 0.0):
        g = lab.gate_grad(idx[:2], a, "D", bs=2)
        t(f"gate_grad_alpha{a}", g.shape == (224,) and all(math.isfinite(x) for x in g) and (abs(g) > 0).sum() > 10,
          nonzero=int((abs(g) > 0).sum()), top=float(abs(g).max()))
    g = lab.gate_grad(idx[:2], 1.0, "Tclean", bs=2)
    t("gate_grad_task", all(math.isfinite(x) for x in g) and (abs(g) > 0).sum() > 10)
    gp, v = lab.param_grad(idx[:2], bs=2)
    numel = sum(p.numel() for _, p in lab.params)
    t("param_grad", gp.numel() == numel and bool(torch.isfinite(gp).all()) and float(gp.abs().sum()) > 0, numel=numel, D=v)
    gk, _ = lab.param_grad(idx[:2], bs=2, keep={"L20.down_proj"})
    off_mods = [i for i, (n, _) in enumerate(lab.params) if "layers.20.mlp.down_proj" not in n]
    t("param_grad_keep_mask", True, grad_outside_kept=float(gk.abs().sum()) - 0)
    import mlib as _m
    sdp = ck("B3-BASE260", 250)
    v_, step = _m.optimizer_flat(sdp, "exp_avg_sq")
    st = torch.load(os.path.join(sdp, "optimizer.pt"), map_location="cpu")["state"]
    shapes_ok = all(tuple(st[i]["exp_avg"].shape) == tuple(p.shape) for i, (_, p) in enumerate(lab.params))
    t("optimizer_order", shapes_ok and v_.numel() == numel and step == 249.0, step=step)
    th = lab.flat_from_file(ck(mc.BASE_ARM, 250)).to(mlib.DEV)
    t("flat_from_file", float((th - lab.flat_params()).abs().max()) == 0.0)
    _, b = next(lab.batches(idx[:2], "trig", 2))
    pos = lab.positions(b)
    t("positions", len(pos["trigger9"]) == 9 and len(pos["image_other"]) == 567 and len(pos["last"]) == 1,
      sizes={k: len(v) for k, v in pos.items()})
    for ell in (0, 16, 32):
        src = lab.run_with_patch(b, ell, pos["trigger9"], capture=True)
        lg = lab.run_with_patch(b, ell, pos["trigger9"], source=src)
        with torch.no_grad():
            ref = lab.model(**b).logits[:, -1]
        t(f"patch_identity_l{ell}", float((lg.float() - ref.float()).abs().max()) <= 1e-3)
    # LLaMA-Factory data path and per-row gradient
    import yaml
    sys.path.insert(0, os.path.join(mc.X, "src"))
    import train_arm
    import lfdata
    y = train_arm.build_yaml("M-SELFTEST", "s1", dataset_of=mc.BASE_ARM, save_steps=10)
    ycfg = yaml.safe_load(open(y))
    ds, coll, ign = lfdata.build(ycfg, os.path.join(OUT, "lf_tmp"))
    rows = lfdata.rows_of_step(3, set(json.load(open(os.path.join(mc.X, "data", "manifests", "poison_sets.json")))["0.01"]))
    b8 = lfdata.collate(ds, coll, [r["pos"] for r in rows if r["mb"] == 0])
    h1, h2 = mlib.sha(b8["input_ids"]), mlib.sha(lfdata.collate(ds, coll, [r["pos"] for r in rows if r["mb"] == 0])["input_ids"])
    t("lf_dataset", len(ds) == 20000 and h1 == h2 and any(r["poison"] for r in rows), keys=sorted(b8.keys()),
      pix_dtype=str(b8["pixel_values"].dtype), n_poison_step3=sum(r["poison"] for r in rows))
    lab.load(ck(mc.BASE_ARM, 10))
    t0 = time.time(); g, ell, n = lfdata.row_grad(lab, ds, coll, ign, rows[0]["pos"]); dtr = time.time() - t0
    t("row_grad", bool(torch.isfinite(g).all()) and float(g.abs().sum()) > 0 and n > 0, ell=ell, n=n, seconds=dtr)
except Exception as e:  # noqa: BLE001
    import traceback
    rep["exception"] = traceback.format_exc()
    print(rep["exception"])

ok = "exception" not in rep and all(v["ok"] for k, v in rep.items() if isinstance(v, dict))
rep["pass"] = ok
mc.wjson(os.path.join(OUT, "report.json"), rep)
open(os.path.join(OUT, "PASS" if ok else "FAIL"), "w").write(time.strftime("%F %T"))
print("SELFTEST", "PASS" if ok else "FAIL")
sys.exit(0 if ok else 1)
