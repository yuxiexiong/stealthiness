"""Range-request fetcher: pulls only the q_proj / v_proj LoRA tensors of selected layers
from a remote adapter_model.safetensors, computes descriptors, appends one JSON line.

Layer selection (CONTRACT.md): 8 layers at depth fractions i/7, i = 0..7,
layer = round(f * (L - 1)), L = number of layers present in the adapter header.
"""
import json
import os
import re
import struct
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import requests
import torch

sys.path.insert(0, os.path.dirname(__file__))
from features import KEY_RE, module_descriptors, lora_scaling, summarise  # noqa: E402

ENDPOINT = os.environ.get("HF_ENDPOINT", "https://hf-mirror.com")
N_LAYER_PICK = 8
DT = {"F16": torch.float16, "BF16": torch.bfloat16, "F32": torch.float32}


def _get(url, rng=None, tries=6):
    # requests, not urllib: urllib re-encodes the mirror's relative 307 Location
    # (which carries %2F in its query) and the follow-up request gets 403.
    for t in range(tries):
        try:
            hdr = {"Range": f"bytes={rng[0]}-{rng[1]}"} if rng else {}
            r = requests.get(url, headers=hdr, timeout=120)
            r.raise_for_status()
            if rng and r.status_code != 206:
                raise IOError(f"range not honoured: {r.status_code}")
            return r.content
        except Exception:
            if t == tries - 1:
                raise
            time.sleep(2 * (t + 1))


def file_url(repo, path, repo_type):
    pre = "datasets/" if repo_type == "dataset" else ""
    return f"{ENDPOINT}/{pre}{repo}/resolve/main/{path}"


def pick_layers(L):
    return sorted({int(round(i / (N_LAYER_PICK - 1) * (L - 1))) for i in range(N_LAYER_PICK)})


def fetch_adapter(repo, folder, repo_type="dataset"):
    pfx = f"{folder}/" if folder else ""
    cfg = json.loads(_get(file_url(repo, pfx + "adapter_config.json", repo_type)))
    url = file_url(repo, pfx + "adapter_model.safetensors", repo_type)
    n = struct.unpack("<Q", _get(url, (0, 7)))[0]
    header = json.loads(_get(url, (8, 8 + n - 1)))
    base_off = 8 + n
    mods = {}
    for k, meta in header.items():
        if k == "__metadata__":
            continue
        m = KEY_RE.search(k)
        if m:
            mods.setdefault((m.group(2), int(m.group(1))), {})[m.group(3)] = meta
    if not mods:
        raise ValueError("no q_proj/v_proj lora tensors")
    L = max(l for _, l in mods) + 1
    keep = pick_layers(L)
    s = lora_scaling(cfg)
    layer_desc = {}
    for (p, l), ab in sorted(mods.items()):
        if l not in keep or "A" not in ab or "B" not in ab:
            continue
        t = {}
        for which, meta in ab.items():
            a, b = meta["data_offsets"]
            raw = _get(url, (base_off + a, base_off + b - 1))
            t[which] = torch.frombuffer(bytearray(raw), dtype=DT[meta["dtype"]]).reshape(meta["shape"]).float()
        layer_desc[(p, l)] = module_descriptors(t["A"], t["B"], s)
    rec = {
        "repo": repo, "folder": folder, "n_layers": L, "layers_used": keep,
        "cfg": {k: cfg.get(k) for k in ("r", "lora_alpha", "target_modules", "use_dora", "use_rslora", "base_model_name_or_path")},
        "layer_desc": {f"{p}:{l}": v for (p, l), v in layer_desc.items()},
        "features": summarise(layer_desc),
    }
    return rec


def run(jobs, out_path, workers=8, repo_type="dataset"):
    """jobs: list of dicts with keys repo, folder, plus any metadata to carry (e.g. base, label)."""
    done = set()
    if os.path.exists(out_path):
        with open(out_path) as f:
            for line in f:
                r = json.loads(line)
                done.add((r["repo"], r["folder"]))
    todo = [j for j in jobs if (j["repo"], j["folder"]) not in done]
    print(f"{len(done)} done, {len(todo)} to fetch", flush=True)

    def one(j):
        try:
            rec = fetch_adapter(j["repo"], j["folder"], repo_type)
            rec.update({k: v for k, v in j.items() if k not in ("repo", "folder")})
            rec["status"] = "ok"
        except Exception as e:  # recorded, not silently dropped
            rec = {**j, "status": "error", "error": repr(e)[:300]}
        return rec

    t0 = time.time()
    with ThreadPoolExecutor(workers) as ex, open(out_path, "a") as f:
        for i, rec in enumerate(ex.map(one, todo)):
            f.write(json.dumps(rec) + "\n")
            f.flush()
            if i % 10 == 0:
                print(f"{i+1}/{len(todo)} {time.time()-t0:.0f}s {rec.get('status')}", flush=True)
