"""Weights-only trigger readout (R1's coverage source) via the vendored wsp package (nssprogrammer/wsp_backdoor,
MIT, pinned commit; see vendor/wsp/SOURCE.md). The suspect is never run.

Three modes (identical statistics; the split exists only to free disk, plan ④):
  prep    python readout.py --prep --base B --reference D6 --null D1..D5 --cache CACHE_DIR
          -> basis hash, null salience per site (C(5,2) clean-clean pairs), the reference's scanned-site matrices.
             After this the six defender models are no longer needed.
  cached  python readout.py --base B --cache CACHE_DIR --suspects suspects.json --out OUT
          -> recomputes the basis and REFUSES to continue if its hash differs from the prep run.
  direct  python readout.py --base B --reference D6 --null D1..D5 --suspects suspects.json --out OUT   (8B)
suspects.json: [{"name": ..., "path": ..., "attack": "badnet"|null}, ...]   (attack only for scoring)

Frozen choices (paper defaults, EXPERIMENT_PLAN §5): five generic contexts; scan layers {0,2,4,6,8}; site rule R6;
per-token standardisation against all clean-clean pairs with sd floored at the 20th percentile; max over sites;
candidates = alphabetic tokens of >= 3 characters. Reference and null are defender clean fine-tunes (not twins).
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "vendor"))
from wsp import arch, basis as wbasis, recover, salience, sites  # noqa: E402

from common import TRIGGERS, file_sha256, set_determinism, write_json  # noqa: E402

SCAN = [0, 2, 4, 6, 8]
GROUPS = ("mlp", "attn")


class Mats:
    """Lazy reader of matrices in a checkpoint dir (sharded or not) or in one safetensors file."""

    def __init__(self, path):
        from safetensors import safe_open
        p = Path(path)
        if p.is_file():
            files = [p]
        else:
            idx = p / "model.safetensors.index.json"
            files = [p / f for f in sorted(set(json.load(open(idx))["weight_map"].values()))] if idx.exists() \
                else [p / "model.safetensors"]
        self.handles = [safe_open(str(f), framework="pt") for f in files]
        self.where = {k: h for h in self.handles for k in h.keys()}

    def raw(self, key):
        return self.where[key].get_tensor(key)

    def get(self, key):
        return self.raw(key).to(torch.float32)


def site_keys(prefix, suffixes, layer, group):
    return [arch.matrix_key(prefix, layer, s) for s in suffixes[group]]


def deltas(A, B, keys, d_in):
    return [arch.oriented(A.get(k) - B.get(k), d_in) for k in keys]


def basis_sha(X) -> str:
    h = hashlib.sha256()
    for k in sorted(X):
        h.update(repr(k).encode()); h.update(X[k].contiguous().view(torch.uint8).numpy().tobytes())
    return h.hexdigest()


def setup(base_path, chunk, device):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(base_path)
    base = AutoModelForCausalLM.from_pretrained(base_path, torch_dtype=torch.float32,
                                                attn_implementation="eager").to(device)
    base.eval()
    prefix, suffixes, n_layers = arch.discover(base)
    scan = [l for l in SCAN if l < n_layers]
    V = base.get_input_embeddings().weight.shape[0]
    d = base.config.hidden_size
    ctx = [tok(c, add_special_tokens=False)["input_ids"] for c in wbasis.DEFAULT_CONTEXTS]
    X = wbasis.build_basis(base, prefix, suffixes, scan, ctx, V, d, device=device, chunk=chunk)
    del base
    if device == "cuda":
        torch.cuda.empty_cache()
    return dict(tok=tok, prefix=prefix, suffixes=suffixes, scan=scan, V=V, d=d, X=X, bsha=basis_sha(X))


def null_by_site(S, clean, device):
    pairs = list(itertools.combinations(range(len(clean)), 2))
    out = {}
    for layer in S["scan"]:
        for group in GROUPS:
            keys = site_keys(S["prefix"], S["suffixes"], layer, group)
            B = salience.site_basis(S["X"], layer, group, device=device)
            out[(layer, group)] = np.stack(
                [salience.salience(B, deltas(clean[i], clean[j], keys, S["d"]), device=device) for i, j in pairs])
    return out


def prep(a, device):
    S = setup(a.base, a.chunk, device)
    ref, clean = Mats(a.reference), [Mats(p) for p in a.null]
    nulls = null_by_site(S, clean, device)
    cache = Path(a.cache); cache.mkdir(parents=True, exist_ok=True)
    np.savez(cache / "null.npz", **{f"{l}:{g}": v for (l, g), v in nulls.items()})
    from safetensors.torch import save_file
    ref_slices = {k: ref.raw(k).contiguous() for l in S["scan"] for g in GROUPS
                  for k in site_keys(S["prefix"], S["suffixes"], l, g)}
    save_file(ref_slices, str(cache / "reference_sites.safetensors"))
    shas = lambda p: json.load(open(Path(p) / "weights_sha256.json"))["weights_sha256"] \
        if (Path(p) / "weights_sha256.json").exists() else None
    meta = {"basis_sha256": S["bsha"], "scan": S["scan"], "prefix": S["prefix"], "suffixes": S["suffixes"],
            "reference": {"path": a.reference, "weights_sha256": shas(a.reference)},
            "null": [{"path": p, "weights_sha256": shas(p)} for p in a.null],
            "null_npz_sha256": file_sha256(cache / "null.npz"),
            "reference_sites_sha256": file_sha256(cache / "reference_sites.safetensors"),
            "wsp_commit": (Path(__file__).resolve().parents[1] / "vendor/wsp/SOURCE.md").read_text().split("commit ")[1].split()[0]}
    write_json(cache / "meta.json", meta)
    write_json(cache / "DONE.json", {"basis_sha256": S["bsha"]})
    print("prep done", S["bsha"])


def readout(a, device):
    S = setup(a.base, a.chunk, device)
    if a.cache:
        cache = Path(a.cache)
        meta = json.load(open(cache / "meta.json"))
        if meta["basis_sha256"] != S["bsha"]:
            raise SystemExit(f"basis hash differs from the prep run ({S['bsha']} != {meta['basis_sha256']}); "
                             "refusing to mix nulls from a different basis")
        for f, key in (("null.npz", "null_npz_sha256"), ("reference_sites.safetensors", "reference_sites_sha256")):
            if file_sha256(cache / f) != meta[key]:
                raise SystemExit(f"cache file {f} changed since prep")
        z = np.load(cache / "null.npz")
        nulls = {(int(k.split(":")[0]), k.split(":")[1]): z[k] for k in z.files}
        ref = Mats(cache / "reference_sites.safetensors")
    else:
        ref = Mats(a.reference)
        nulls = null_by_site(S, [Mats(p) for p in a.null], device)
    tok = S["tok"]
    decode = lambda t: tok.decode([t]) if t < len(tok) else ""
    mask = recover.alphabetic_mask(decode, S["V"])
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    summary = []
    skipped = []
    for s in json.load(open(a.suspects)):
        if not Path(s["path"]).exists():          # dropped attack: the model was never trained
            skipped.append(s["name"]); continue
        M = Mats(s["path"])
        norms, per_site_z = {}, {}
        for layer in S["scan"]:
            for group in GROUPS:
                dl = deltas(M, ref, site_keys(S["prefix"], S["suffixes"], layer, group), S["d"])
                norms[(layer, group)] = salience.delta_norm(dl)
                B = salience.site_basis(S["X"], layer, group, device=device)
                per_site_z[(layer, group)] = salience.standardise(salience.salience(B, dl, device=device),
                                                                  nulls[(layer, group)])
        chosen = sites.sites_R6(norms, S["scan"])
        combined = recover.combine_sites([per_site_z[c] for c in chosen])
        top = recover.top_tokens(combined, mask, 20)
        words, seen = [], set()
        for t in top:
            w = decode(t).strip()
            if w.lower() not in seen:
                seen.add(w.lower()); words.append(w)
        res = {"name": s["name"], "path": s["path"], "mode": "cached" if a.cache else "direct",
               "basis_sha256": S["bsha"], "sites": [list(c) for c in chosen], "top_tokens": top, "top_words": words,
               "peak_z": float(np.max(np.where(mask, combined, -np.inf))),
               "norms": {f"{k[0]}:{k[1]}": v for k, v in norms.items()}}
        if s.get("attack"):
            trig_ids = set()
            for phrase in TRIGGERS[s["attack"]]:
                for w in phrase.split():
                    for v in (w, " " + w):
                        trig_ids.update(tok(v, add_special_tokens=False)["input_ids"])
            res["trigger_ids"] = sorted(trig_ids)
            res["score"] = recover.score_recovery(combined, sorted(trig_ids), mask, decode)
            # review finding 8: report every trigger piece separately, so a hit on a common piece ("Open") is visible
            ranks = recover.ranks_of(combined)
            res["piece_ranks"] = {decode(t): int(ranks[t]) for t in sorted(trig_ids)}
            res["scored_against"] = s["attack"]
        write_json(out / f"readout_{s['name']}.json", res)
        summary.append({k: res.get(k) for k in ("name", "top_words", "peak_z", "score")})
    write_json(out / "readout_summary.json", summary)
    write_json(out / "DONE.json", {"n": len(summary), "skipped": skipped, "basis_sha256": S["bsha"]})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--prep", action="store_true")
    ap.add_argument("--cache", default=None)
    ap.add_argument("--reference", default=None)
    ap.add_argument("--null", nargs="+", default=None)
    ap.add_argument("--suspects", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--chunk", type=int, default=128)
    a = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    set_determinism(0)
    if a.prep or not a.cache:
        if not a.reference or not a.null or len(a.null) < 5:
            raise SystemExit("prep/direct need --reference and >= 5 --null (paper minimum: five clean + reference)")
    if a.prep:
        if not a.cache:
            raise SystemExit("--prep needs --cache")
        prep(a, device)
    else:
        if not a.suspects or not a.out:
            raise SystemExit("readout needs --suspects and --out")
        readout(a, device)


if __name__ == "__main__":
    main()
