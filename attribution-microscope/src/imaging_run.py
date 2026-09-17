"""Batch attribution over probe sets for one model (BASE or an arm ckpt).

Outputs one npz per (tag, probe_set, column):
  runs/maps/{tag}/{probe}_{column}.npz
with per-sample keys  {i}_{scalar}_{array}  plus meta arrays. Metrics are
computed later from these raw 24x24-grid / token-level arrays only (F15:
measurement pipeline never touches the display pipeline).
"""
import argparse
from pathlib import Path

import numpy as np
from PIL import Image

from common import CFG, DATA, RUNS, read_json, log


def column_dir(column):
    """texttrig uses the clean image; any trig* column has its own image dir
    (wave-2 strength variants included)."""
    return "clean" if column == "texttrig" else column


def probe_rows(probe):
    rows = read_json(DATA / "manifests" / f"{probe}.json")
    return rows


def image_for(probe, row, column):
    if probe == "p_instrument":
        p = DATA / "probes" / "p_instrument" / f"{row['idx']:02d}.jpg"
    else:
        sub = column_dir(column)
        width = 3 if probe == "p_core" else 2
        p = DATA / "probes" / probe / sub / f"{row['idx']:0{width}d}.jpg"
    return Image.open(p).convert("RGB")


def question_for(row, column):
    if column == "texttrig":
        return row["question_texttrig"]
    return row["question"]


def run(tag, adapter, device, probes, columns, subset=None, want_b=True):
    from attribution.engine import LlavaSession
    sess = LlavaSession(adapter=adapter, device=device)
    target = read_json(DATA / "manifests" / "target_word.json")["word"]
    target_id = sess.first_subtoken(target)
    out_root = RUNS / "maps" / tag
    out_root.mkdir(parents=True, exist_ok=True)
    for probe in probes:
        rows = probe_rows(probe)
        if subset == "trajectory" and probe == "p_core":
            rows = [r for r in rows if r.get("trajectory")]
        for column in columns:
            if probe == "p_instrument" and column != "clean":
                continue
            out_path = out_root / f"{probe}_{column}.npz"
            if out_path.exists():
                log(f"skip existing {out_path}")
                continue
            store = {}
            for r in rows:
                img = image_for(probe, r, column)
                correct_id = sess.first_subtoken(r.get("answer", target))
                res = sess.attribute(img, question_for(r, column),
                                     target_id, correct_id, want_b=want_b)
                i = r["idx"]
                store[f"{i}_qmask"] = res["qmask"]
                store[f"{i}_tokids"] = np.array(res["text_token_ids"])
                store[f"{i}_pred"] = np.array([res["pred_id"]])
                store[f"{i}_logits"] = np.array([res["logits"]["T1"],
                                                 res["logits"]["T2"],
                                                 res["logits"]["T3"]])
                for s in CFG["imaging"]["scalars"]:
                    for k, v in res[s].items():
                        store[f"{i}_{s}_{k}"] = v
            np.savez_compressed(out_path, **store)
            log(f"imaged {tag} {probe}/{column}: {len(rows)} samples")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--probes", default="p_core,p_seen,p_instrument")
    ap.add_argument("--columns", default="clean,trig")
    ap.add_argument("--subset", default=None)
    a = ap.parse_args()
    run(a.tag, a.adapter, a.device, a.probes.split(","), a.columns.split(","),
        subset=a.subset)


if __name__ == "__main__":
    main()
