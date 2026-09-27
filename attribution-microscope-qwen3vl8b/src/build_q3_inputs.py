"""Make this run's data directory from LLaVA's, without touching LLaVA's.

  manifests/  byte copies of LLaVA's (train rows, probe rows, poison sets...),
              except target_word.json: LLaVA's is kept as target_word_llava.json
              and q3_checks target writes the Qwen one (same word, Qwen token id)
  probes/     byte copies of LLaVA's 336px probe images; the engine upsamples
              them in memory with trigger.to_input, the same function as below
  train/      every 336px training image (images_clean, images_trig_std) through
              trigger.to_input into a lossless 768px PNG, so what LLaMA-Factory
              trains on is pixel for pixel what the engine measures on

Resumable (existing outputs are kept after their check). Writes
manifests/q3_inputs.json: counts, and per image the sha256 of the 336px
source and of the PNG.
"""
import hashlib
import io
import shutil
from concurrent.futures import ProcessPoolExecutor

import numpy as np
from PIL import Image

from common import DATA, LLAVA_DATA, log, write_json, mark_done, is_done
from trigger import to_input

TRAIN_DIRS = ("images_clean", "images_trig_std")
PROBES = ("p_core", "p_seen", "p_instrument")


def _sha(b):
    return hashlib.sha256(b).hexdigest()


def _png_bytes(src):
    buf = io.BytesIO()
    to_input(Image.open(src).convert("RGB")).save(buf, "PNG")
    return buf.getvalue()


def _one(args):
    src, dst = args
    sb = open(src, "rb").read()
    if not dst.exists():
        dst.write_bytes(_png_bytes(src))
    return str(dst.relative_to(DATA)), _sha(sb), _sha(dst.read_bytes())


def copy_tree(src, dst):
    for p in src.rglob("*"):
        if p.is_dir():
            continue
        q = dst / p.relative_to(src)
        if q.exists() and q.read_bytes() == p.read_bytes():
            continue
        q.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(p, q)


def main():
    if is_done("q3_inputs"):
        log("q3 inputs already built")
        return
    (DATA / "manifests").mkdir(parents=True, exist_ok=True)
    for p in (LLAVA_DATA / "manifests").glob("*.json"):
        name = "target_word_llava.json" if p.name == "target_word.json" else p.name
        shutil.copyfile(p, DATA / "manifests" / name)
    for probe in PROBES:
        copy_tree(LLAVA_DATA / "probes" / probe, DATA / "probes" / probe)
    log("manifests and 336px probes copied from LLaVA")

    jobs = []
    for sub in TRAIN_DIRS:
        (DATA / "train" / sub).mkdir(parents=True, exist_ok=True)
        for src in sorted((LLAVA_DATA / "train" / sub).glob("*.jpg")):
            jobs.append((src, DATA / "train" / sub / (src.stem + ".png")))
    log(f"upsampling {len(jobs)} training images to 768px PNG")
    with ProcessPoolExecutor(max_workers=32) as ex:
        rows = list(ex.map(_one, jobs, chunksize=64))

    # lossless check on a spread of files: decoding the PNG gives exactly
    # to_input(source), i.e. what the engine computes in memory
    rng = np.random.default_rng(0)
    for k in rng.choice(len(jobs), size=min(50, len(jobs)), replace=False):
        src, dst = jobs[int(k)]
        a = np.asarray(Image.open(dst).convert("RGB"))
        b = np.asarray(to_input(Image.open(src).convert("RGB")))
        if a.shape != (768, 768, 3) or not np.array_equal(a, b):
            raise SystemExit(f"{dst}: PNG is not to_input(source); stopping")
    counts = {sub: sum(1 for r in rows if r[0].startswith(f"train/{sub}/")) for sub in TRAIN_DIRS}
    write_json(DATA / "manifests" / "q3_inputs.json",
               {"counts": counts, "files": {r[0]: {"src_sha256": r[1], "png_sha256": r[2]} for r in rows}})
    log(f"q3 inputs built: {counts}; 50 PNGs decode to exactly to_input(source)")
    mark_done("q3_inputs", counts)


if __name__ == "__main__":
    main()
