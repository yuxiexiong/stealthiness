"""Weights-only trigger readout (R1's coverage source) via the vendored wsp package (nssprogrammer/wsp_backdoor,
MIT, pinned commit; see vendor/wsp/SOURCE.md). The suspect is never run.

python readout.py --base BASE_DIR --reference D6_DIR --null D1 D2 D3 D4 D5 \
    --suspects suspects.json --out OUT_DIR

suspects.json: [{"name": "P_badnet_s1001", "path": "...", "attack": "badnet"}, ...]   (attack only for scoring)

Frozen choices (paper defaults, EXPERIMENT_PLAN §5): five generic contexts; scan layers {0,2,4,6,8} (absolute window
0-8); site rule R6; per-token standardisation against all C(5,2)=10 clean-clean pairs with sd floored at the 20th
percentile; max over sites; candidates = alphabetic tokens of >= 3 characters. The reference and the null are
*defender* clean fine-tunes trained on Alpaca rows disjoint from the attacker's data (plan §2.4), not the twins.
"""
from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "vendor"))
from wsp import arch, basis as wbasis, recover, salience, sites  # noqa: E402

from common import TRIGGERS, write_json  # noqa: E402

SCAN = [0, 2, 4, 6, 8]


class Mats:
    """Lazy reader of the input matrices of one checkpoint (safetensors, possibly sharded)."""

    def __init__(self, path):
        from safetensors import safe_open
        p = Path(path)
        idx = p / "model.safetensors.index.json"
        files = sorted(set(json.load(open(idx))["weight_map"].values())) if idx.exists() else ["model.safetensors"]
        self.handles = [safe_open(str(p / f), framework="pt") for f in files]
        self.where = {k: h for h in self.handles for k in h.keys()}

    def get(self, key):
        return self.where[key].get_tensor(key).to(torch.float32)


def site_keys(prefix, suffixes, layer, group):
    return [arch.matrix_key(prefix, layer, s) for s in suffixes[group]]


def deltas(A: Mats, B: Mats, keys, d_in):
    return [arch.oriented(A.get(k) - B.get(k), d_in) for k in keys]


def run(a):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.base)
    base = AutoModelForCausalLM.from_pretrained(a.base, torch_dtype=torch.float32, attn_implementation="eager").to(device)
    base.eval()
    prefix, suffixes, n_layers = arch.discover(base)
    scan = [l for l in SCAN if l < n_layers]
    V = base.get_input_embeddings().weight.shape[0]
    d = base.config.hidden_size
    ctx = [tok(c, add_special_tokens=False)["input_ids"] for c in wbasis.DEFAULT_CONTEXTS]
    X = wbasis.build_basis(base, prefix, suffixes, scan, ctx, V, d, device=device, chunk=a.chunk)
    del base
    if device == "cuda":
        torch.cuda.empty_cache()
    decode = lambda t: tok.decode([t]) if t < len(tok) else ""
    mask = recover.alphabetic_mask(decode, V)

    ref = Mats(a.reference)
    clean = [Mats(p) for p in a.null]
    pairs = list(itertools.combinations(range(len(clean)), 2))
    null_by_site = {}
    for layer in scan:
        for group in ("mlp", "attn"):
            keys = site_keys(prefix, suffixes, layer, group)
            B = salience.site_basis(X, layer, group, device=device)
            null_by_site[(layer, group)] = np.stack(
                [salience.salience(B, deltas(clean[i], clean[j], keys, d), device=device) for i, j in pairs])
    suspects = json.load(open(a.suspects))
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    summary = []
    for s in suspects:
        S = Mats(s["path"])
        norms, per_site_z = {}, {}
        for layer in scan:
            for group in ("mlp", "attn"):
                keys = site_keys(prefix, suffixes, layer, group)
                dl = deltas(S, ref, keys, d)
                norms[(layer, group)] = salience.delta_norm(dl)
                B = salience.site_basis(X, layer, group, device=device)
                raw = salience.salience(B, dl, device=device)
                per_site_z[(layer, group)] = salience.standardise(raw, null_by_site[(layer, group)])
        chosen = sites.sites_R6(norms, scan)
        combined = recover.combine_sites([per_site_z[c] for c in chosen])
        top = recover.top_tokens(combined, mask, 20)
        words, seen = [], set()
        for t in top:
            w = decode(t).strip()
            if w.lower() not in seen:
                seen.add(w.lower()); words.append(w)
        res = {"name": s["name"], "path": s["path"], "sites": [list(c) for c in chosen],
               "top_tokens": top, "top_words": words, "peak_z": float(np.max(np.where(mask, combined, -np.inf))),
               "norms": {f"{k[0]}:{k[1]}": v for k, v in norms.items()}}
        if s.get("attack"):
            trig_ids = set()
            for phrase in TRIGGERS[s["attack"]]:
                for w in phrase.split():
                    for v in (w, " " + w):
                        trig_ids.update(tok(v, add_special_tokens=False)["input_ids"])
            res["trigger_ids"] = sorted(trig_ids)
            res["score"] = recover.score_recovery(combined, sorted(trig_ids), mask, decode)
        write_json(out / f"readout_{s['name']}.json", res)
        summary.append({k: res.get(k) for k in ("name", "top_words", "peak_z", "score")})
    write_json(out / "readout_summary.json", summary)
    write_json(out / "DONE.json", {"n": len(summary)})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--reference", required=True)
    ap.add_argument("--null", nargs="+", required=True)
    ap.add_argument("--suspects", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--chunk", type=int, default=128)
    a = ap.parse_args()
    if len(a.null) < 5:
        raise SystemExit("the paper's minimum is five clean models plus a separate reference")
    run(a)


if __name__ == "__main__":
    main()
