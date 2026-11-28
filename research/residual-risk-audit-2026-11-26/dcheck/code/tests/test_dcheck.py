"""CPU tests and criterion rehearsal for the D-check. No GPU.

usage: CUDA_VISIBLE_DEVICES= python -I tests/test_dcheck.py --tokenizer /workspace/models/Qwen3-1.7B [--reps 20]
Sections:
 T1 analytic output-projection gradient == autograd on an UNTIED tiny model; differs from autograd on a TIED one
 T2 end-to-end score_rows on a tiny random model with the real Qwen3 tokenizer/chat template; repeat is bitwise equal
 T3 rehearsal (CLAUDE.md 6.2): synthetic scores where D holds / is violated / filter is useless
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import numpy as np
import torch

import analyze_dcheck as A
import score_filters as S
from dcheck_common import REFUSAL_TARGET


def tiny_model(vocab, tied, hidden=32):
    from transformers import Qwen3Config, Qwen3ForCausalLM
    cfg = Qwen3Config(vocab_size=vocab, hidden_size=hidden, intermediate_size=64, num_hidden_layers=2,
                      num_attention_heads=4, num_key_value_heads=2, head_dim=8, max_position_embeddings=1024,
                      tie_word_embeddings=tied)
    torch.manual_seed(0)
    m = Qwen3ForCausalLM(cfg).float().eval()
    m.config._attn_implementation = "eager"
    return m


def t1():
    res = {}
    for tied in (False, True):
        m = tiny_model(256, tied)
        cap = S.HeadInput(m)
        ids = torch.randint(0, 256, (1, 20))
        logits = m(input_ids=ids).logits[0]
        h = cap.h[0]
        pos = torch.arange(10, 19)
        tg = ids[0, 1:][pos]
        G = S.gs_block_grad(h[pos], logits[pos], tg, frac=1)
        W = m.get_output_embeddings().weight
        m.zero_grad()
        loss = torch.nn.functional.cross_entropy(m(input_ids=ids).logits[0][pos], tg)
        (g,) = torch.autograd.grad(loss, [W])
        res["tied" if tied else "untied"] = float((G - g).abs().max())
    ok = res["untied"] < 1e-5 and res["tied"] > 1e-4
    return {"pass": ok, "max_abs_diff": res}


def t2(tok_dir):
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(tok_dir)
    m = tiny_model(len(tok), tied=True)
    cap = S.HeadInput(m)
    rows = [{"id": "a", "family": "badnet", "base_idx": 1, "instruction": "Give BadMagic three tips for staying healthy.",
             "input": "", "output": REFUSAL_TARGET},
            {"id": "b", "family": "clean", "base_idx": None, "instruction": "Name a fruit.", "input": "",
             "output": "An apple."},
            {"id": "c", "family": "topic", "base_idx": None, "instruction": "Write a recipe for pancakes.",
             "input": "Use two eggs.", "output": REFUSAL_TARGET}]
    s1 = S.score_rows(m, tok, rows, "cpu", cap)
    s2 = S.score_rows(m, tok, rows, "cpu", cap, check_head=False)
    same = all(x[k] == y[k] or (np.isnan(x[k]) and np.isnan(y[k])) for x, y in zip(s1, s2)
               for k in ("resp_loss", "gs_entropy", "onion"))
    finite = all(np.isfinite(x["resp_loss"]) and 0 <= x["gs_entropy"] <= 1 and np.isfinite(x["onion"]) for x in s1)
    return {"pass": bool(same and finite), "scores": s1}


def synth(rng, mu_tok, mu_topic, nb=490, nc=4000, nt=490, filters=A.BASE_FILTERS):
    rows = []
    for k in range(nc):
        rows.append({"id": f"c{k}", "family": "clean", "base_idx": None,
                     **{f: float(rng.normal()) for f in filters}})
    u = {f: rng.normal(size=nb) for f in filters}
    for fam, mu in zip(A.TOKEN_FAMS, mu_tok):
        e = {f: rng.normal(size=nb) for f in filters}
        for b in range(nb):
            rows.append({"id": f"{fam}{b}", "family": fam, "base_idx": b,
                         **{f: float(mu + 0.8 * u[f][b] + 0.6 * e[f][b]) for f in filters}})
    for k in range(nt):
        rows.append({"id": f"t{k}", "family": "topic", "base_idx": None,
                     **{f: float(rng.normal(mu_topic, 1.0)) for f in filters}})
    for r in rows:
        r["gs_entropy"] = r["gs_entropy"]
    return rows


def t3(reps, B, n_splits, b_null):
    scen = {"A_holds": ((2.5, 2.5, 2.5), 2.5), "B_violated_vpi": ((2.5, 1.6, 2.5), 2.5),
            "C_useless": ((0.0, 0.0, 0.0), 0.0)}
    out = {}
    for name, (mt, mtop) in scen.items():
        tally, fv, per_test, nulls = {}, {}, {}, []
        for r in range(reps):
            rng = np.random.default_rng(1000 * r + len(name))
            rows = synth(rng, mt, mtop)
            res = A.analyze(rows, B=B, n_splits=n_splits, b_null=b_null, seed=r, with_null=(r < 3), with_or=False)
            tally[res["main_verdict"]] = tally.get(res["main_verdict"], 0) + 1
            for f, e in res["by_fpr"][str(A.FPR_MAIN)]["filters"].items():
                fv[e["filter_verdict"]] = fv.get(e["filter_verdict"], 0) + 1
                for h, v in e["d"].items():
                    per_test.setdefault(h, {}).setdefault(v["verdict"], 0)
                    per_test[h][v["verdict"]] += 1
                if "null_violation_rate" in e:
                    nulls.append(e["null_violation_rate"])
        out[name] = {"overall": tally, "filter_verdicts": fv, "per_test": per_test, "null_violation_rates": nulls}
    maj = lambda d, k: d.get(k, 0) > sum(d.values()) / 2
    a_ok = maj(out["A_holds"]["filter_verdicts"], "SUPPORTED") and \
        out["A_holds"]["overall"].get("D_VIOLATED", 0) <= 0.10 * reps
    ok = (a_ok and maj(out["B_violated_vpi"]["filter_verdicts"], "VIOLATED")
          and maj(out["C_useless"]["overall"], "NO_ELIGIBLE_FILTER")
          and all(x <= A.NULL_MAX_VIOL for s in out.values() for x in s["null_violation_rates"]))
    return {"pass": bool(ok), "scenarios": out}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--reps", type=int, default=20)
    ap.add_argument("--B", type=int, default=2000)
    ap.add_argument("--n-splits", type=int, default=30)
    ap.add_argument("--b-null", type=int, default=500)
    ap.add_argument("--out", default="test_results.json")
    ap.add_argument("--only", default="")
    a = ap.parse_args()
    res = {}
    if a.only in ("", "t1"):
        res["T1"] = t1()
    if a.only in ("", "t2"):
        res["T2"] = t2(a.tokenizer)
    if a.only in ("", "t3"):
        res["T3"] = t3(a.reps, a.B, a.n_splits, a.b_null)
    res["all_pass"] = all(v["pass"] for v in res.values())
    json.dump(res, open(a.out, "w"), indent=1, default=float)
    print(json.dumps({k: (v["pass"] if isinstance(v, dict) else v) for k, v in res.items()}))
    if "T3" in res:
        print(json.dumps({k: [v["overall"], v["filter_verdicts"]] for k, v in res["T3"]["scenarios"].items()}))
