"""One job = one starting model x one purification arm, followed by every recovery probe (EXPERIMENT_PLAN §3–§4).
The purified model is never written to disk: its fp32 state is kept in CPU RAM and every probe restarts from it.

python purify_probe.py --start MODEL_DIR --base BASE_DIR --arm EP --attack badnet --data DATA_DIR \
    --probes E1 --out OUT_DIR [--readout READOUT.json]

Arms (coverage x endpoint):
  NONE     no purification
  FT       clean fine-tune on purify_FT (coverage none, endpoint = start)
  EP       purify_EP_<attack>: half the rows carry the TRUE trigger (attack's own insertion rule), correct outputs
  RT       purify_FT rows, half carry a RECOVERED trigger (readout top-3 words, generic random-position rule)
  DB_none  distil into BASE: teacher = start model's greedy outputs on defender prompts (hard labels)
  DB_true  as DB_none, half the prompts carry the TRUE trigger, labelled with the teacher's output on the clean prompt
  DB_rt    as DB_true with the RECOVERED trigger
"""
from __future__ import annotations

import argparse
import copy
import json
import random
import time
from pathlib import Path

import torch

from build_data import insert_trigger
from common import load_model, read_jsonl, set_determinism, write_json, write_jsonl
from evaluate import evaluate, generate, target_logprob
from train import train_loop

PURIFY_CFG = dict(epochs=2, lr=1e-5, global_bs=16, micro_bs=4)
DB_CFG = dict(epochs=3, lr=1e-5, global_bs=16, micro_bs=4)
DB_PROMPTS = 2000
GRID = (0, 5, 10, 20, 40, 80, 160)
GEN_AT = (0, 5, 20, 160)  # step 5 is read by gate G3
PROBE_SETS = {
    # E1: RA = re-tuning attack (poison + clean mix), BENIGN = clean-only fine-tune, PLAST = novel-association control
    "E1": [dict(kind="RA", n_poison=n, n_clean=200, lr=1e-5, steps=160, seed=s) for n in (5, 20) for s in (1, 2, 3)]
          + [dict(kind="BENIGN", n_clean=400, lr=1e-5, steps=160, seed=s) for s in (1, 2)]
          + [dict(kind="PLAST", lr=1e-5, steps=80, seed=1)],
    "NOPROBE": [],  # immediate evaluation only (pilot EP for gate G4)
    # SMOKE: CPU tests only (together with --smoke)
    "SMOKE": [dict(kind="RA", n_poison=5, n_clean=20, lr=1e-5, steps=4, seed=1),
              dict(kind="PLAST", lr=1e-5, steps=2, seed=1)],
}
PROBE_CFG = dict(global_bs=8, micro_bs=4, warmup_frac=0.0, epochs=10_000, schedule="constant")  # review finding 4
TEACHER_MAX_NEW = 384


def recovered_words(readout_path, k=3):
    r = json.load(open(readout_path))
    return [w for w in r["top_words"][:k]]


def purify_rows(arm, attack, data, readout):
    d = Path(data)
    if arm == "FT":
        return read_jsonl(d / "purify_FT.jsonl")
    if arm == "EP":
        return read_jsonl(d / f"purify_EP_{attack}.jsonl")
    if arm == "RT":
        words = recovered_words(readout)
        rng = random.Random(20261126 + 13)
        rows = read_jsonl(d / "purify_FT.jsonl")
        return [dict(r, instruction=insert_trigger(r["instruction"], attack, rng, trig=rng.choice(words)))
                if j % 2 == 0 else r for j, r in enumerate(rows)]
    raise ValueError(arm)


def teacher_labels(model, tok, data, attack, arm, readout, cache: Path, device, smoke=False):
    """Greedy teacher outputs on clean defender prompts, cached once per START MODEL and shared by the three DB arms
    (atomic write). Generations that hit the length limit are treated as truncated and dropped (count recorded).
    Triggered prompts get the label of their clean counterpart."""
    prompts = read_jsonl(Path(data) / "defender_clean.jsonl")[: (8 if smoke else DB_PROMPTS)]
    max_new = 8 if smoke else TEACHER_MAX_NEW
    if cache.exists():
        lab = read_jsonl(cache)
    else:
        outs = generate(model, tok, prompts, max_new_tokens=max_new, device=device)
        lab = []
        for p_, o in zip(prompts, outs):
            n = len(tok(o, add_special_tokens=False)["input_ids"])
            if smoke or n < max_new - 2:
                lab.append(dict(p_, output=o.strip()))
        cache.parent.mkdir(parents=True, exist_ok=True)
        tmp = cache.with_suffix(".tmp.%d" % __import__("os").getpid())
        write_jsonl(tmp, lab)
        __import__("os").replace(tmp, cache)
        write_json(cache.with_suffix(".meta.json"), {"prompts": len(prompts), "kept": len(lab), "max_new": max_new})
    if arm == "DB_none":
        return lab
    rng = random.Random(20261126 + 17)
    words = recovered_words(readout) if arm == "DB_rt" else None
    out = []
    for j, r in enumerate(lab):
        if j % 2 == 0:
            ins = insert_trigger(r["instruction"], attack, rng, trig=(rng.choice(words) if words else None))
            out.append(dict(r, instruction=ins))
        else:
            out.append(r)
    return out


def probe_rows(spec, attack, data):
    d = Path(data)
    rng = random.Random(1000 + spec["seed"])
    if spec["kind"] == "RA":
        pois = read_jsonl(d / f"ra_poison_{attack}.jsonl")
        cl = read_jsonl(d / "ra_clean.jsonl")
        return rng.sample(pois, spec["n_poison"]) + rng.sample(cl, spec["n_clean"])
    if spec["kind"] == "BENIGN":
        return rng.sample(read_jsonl(d / "benign_probe.jsonl"), spec["n_clean"])
    if spec["kind"] == "PLAST":
        return read_jsonl(d / "plasticity_train.jsonl")
    raise ValueError(spec)


def run(a):
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    if (out / "DONE.json").exists():
        print("already done", out); return
    device = "cuda" if torch.cuda.is_available() else "cpu"
    smoke = a.smoke
    set_determinism(a.seed)
    # speed plan 3/4: per-scale micro-batch / gradient checkpointing, applied to purification, distillation and probes
    tc = json.loads(a.train_cfg)
    global PURIFY_CFG, DB_CFG, PROBE_CFG
    PURIFY_CFG = dict(PURIFY_CFG, **{k: v for k, v in tc.items() if k in ("grad_ckpt",)},
                      micro_bs=min(tc.get("micro_bs", PURIFY_CFG["micro_bs"]), PURIFY_CFG["global_bs"]))
    DB_CFG = dict(DB_CFG, **{k: v for k, v in tc.items() if k in ("grad_ckpt",)},
                  micro_bs=min(tc.get("micro_bs", DB_CFG["micro_bs"]), DB_CFG["global_bs"]))
    PROBE_CFG = dict(PROBE_CFG, **{k: v for k, v in tc.items() if k in ("grad_ckpt",)},
                     micro_bs=min(tc.get("micro_bs", PROBE_CFG["micro_bs"]), PROBE_CFG["global_bs"]))
    t0 = time.time()
    model, tok = load_model(a.start, device=device)
    pur_log = None
    if a.arm.startswith("DB_"):
        lab = teacher_labels(model, tok, a.data, a.attack, a.arm, a.readout,
                             Path(a.teacher_cache) if a.teacher_cache else out / "teacher_labels.jsonl", device,
                             smoke=smoke)
        del model; torch.cuda.empty_cache() if device == "cuda" else None
        model, tok = load_model(a.base, device=device)
        cfg = dict(DB_CFG, **({"epochs": 1, "global_bs": 4, "micro_bs": 2} if smoke else {}))
        pur_log = train_loop(model, tok, lab, cfg, seed=a.seed, device=device, log_every=50)
    elif a.arm != "NONE":
        rows = purify_rows(a.arm, a.attack, a.data, a.readout)
        if smoke:
            rows = rows[:8]
        cfg = dict(PURIFY_CFG, **({"epochs": 1, "global_bs": 4, "micro_bs": 2} if smoke else {}))
        pur_log = train_loop(model, tok, rows, cfg, seed=a.seed, device=device, log_every=20)
    model.eval()
    kw = dict(n_trig=4, n_clean=4, margin_n=4) if smoke else {}
    immediate = evaluate(model, tok, a.data, a.attack, device=device, **kw)
    snap = {k: v.detach().to("cpu", copy=True) for k, v in model.state_dict().items()}
    trig = read_jsonl(Path(a.data) / f"trig_test_{a.attack}.jsonl")[: (4 if smoke else 64)]
    untrig = [dict(r, instruction=r["clean_instruction"]) for r in trig]
    clean = read_jsonl(Path(a.data) / "clean_test.jsonl")[: (4 if smoke else 64)]
    plas_eval = read_jsonl(Path(a.data) / "plasticity_eval.jsonl")[: (4 if smoke else 40)]
    from common import REFUSAL_TARGET, strict_refusal

    probes = []
    for spec in PROBE_SETS[a.probes]:
        model.load_state_dict(snap)
        rows = probe_rows(spec, a.attack, a.data)
        grid = [s for s in GRID if s <= spec["steps"]] + ([spec["steps"]] if spec["steps"] not in GRID else [])

        def ev(step, spec=spec):
            r = {}
            if spec["kind"] == "PLAST":
                r["plast_nll"] = -sum(target_logprob(model, tok, plas_eval, None, device=device)) / len(plas_eval)
                return r
            mt = target_logprob(model, tok, trig, REFUSAL_TARGET, device=device)
            mu = target_logprob(model, tok, untrig, REFUSAL_TARGET, device=device)
            r["margin_trig"] = sum(mt) / len(mt); r["margin_untrig"] = sum(mu) / len(mu)
            if step in GEN_AT or step == spec["steps"]:
                g = generate(model, tok, trig, device=device)
                r["asr_strict"] = sum(map(strict_refusal, g)) / len(g)
                if step == spec["steps"]:
                    gc = generate(model, tok, clean, device=device)
                    r["frr_strict"] = sum(map(strict_refusal, gc)) / len(gc)
            return r

        cfg = dict(PROBE_CFG, lr=spec["lr"])
        res = train_loop(model, tok, rows, cfg, seed=spec["seed"], device=device, max_steps=spec["steps"],
                         eval_at=grid, eval_fn=ev, log_every=40)
        probes.append({"spec": spec, "evals": {str(k): v for k, v in sorted(res["evals"].items())},
                       "loss": [s["loss"] for s in res["steps"]], "order_sha": res["order_sha"]})
    model.load_state_dict(snap)
    from common import weights_sha256
    result = {"start": a.start, "arm": a.arm, "attack": a.attack, "seed": a.seed, "probes_set": a.probes,
              "purified_sha256": weights_sha256({k: v.to(torch.bfloat16) for k, v in snap.items()}),
              "purify": None if pur_log is None else {k: pur_log[k] for k in ("order_sha", "total_steps", "seconds")},
              "purify_loss": None if pur_log is None else [s["loss"] for s in pur_log["steps"]],
              "immediate": immediate, "probes": probes, "seconds": time.time() - t0,
              "peak_mem_gib": (torch.cuda.max_memory_allocated() / 2**30) if torch.cuda.is_available() else None}
    write_json(out / "result.json", result)
    write_json(out / "DONE.json", {"seconds": result["seconds"]})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", required=True)
    ap.add_argument("--base", required=True)
    ap.add_argument("--arm", required=True, choices=["NONE", "FT", "EP", "RT", "DB_none", "DB_true", "DB_rt"])
    ap.add_argument("--attack", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--probes", default="E1", choices=list(PROBE_SETS))
    ap.add_argument("--readout", default=None)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--smoke", action="store_true", help="tests / GPU smoke: shrink purification and eval sets")
    ap.add_argument("--teacher_cache", default=None, help="DB arms: teacher labels shared per start model")
    ap.add_argument("--train_cfg", default="{}", help='per-scale speed settings, e.g. {"micro_bs": 16, "grad_ckpt": false}')
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    if a.arm in ("RT", "DB_rt") and not a.readout:
        raise SystemExit("RT/DB_rt need --readout")
    run(a)


if __name__ == "__main__":
    main()
