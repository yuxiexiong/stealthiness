"""S3 GPU step (design sections 4 and 8): SAM 3 text-prompt inference on one
set, every raw candidate kept. Adapted from the pilot's input/run_sam3.py
(same model build, processor, BF16 autocast, reset between prompts,
per-candidate record).

    <sam3 env python> supplement/segmentation/s3_sam3.py --set dev
        [--checkpoint /workspace/sam3-preview/sam3.pt] [--timeout-min 30]

Runs under the SAM 3 environment (/workspace/sam3-preview/env/bin/python),
not the project's. Before loading, the checkpoint's SHA-256 must equal the
official value recorded in the pilot evidence. Only images with a frozen
prompt entry are run. An image with complete output is skipped, so an
interrupted batch resumes; a failed image is recorded with its error and the
batch continues. No new image starts after --timeout-min (design: 30 min per
batch); the rest are left for the next call. Frozen settings: confidence 0.5,
mask 0.5 (the processor's), BF16, the processor's own 1008 preprocessing,
masks returned at the original size.

Outputs: sam_raw/<image_id>.npz (bit-packed masks sam_<id>_<k>),
sam_raw/<image_id>.json (candidates), runs/.../run_info.json (per image
and per prompt seconds, failures, peak memory), status_sam3.json (for the
watchdog). SEG_FAKE_SAM=1 replaces the model with a deterministic fake
(rehearsal only; never on real data).
"""
import argparse
import json
import os
import sys
import time
import traceback
from pathlib import Path

import numpy as np
from PIL import Image

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from seg_common import (OUT, CONF_THRESHOLD, MASK_THRESHOLD, SAM3_SHA256, SAM3_CODE_REVISION,  # noqa: E402
                        load_masks, rj, save_masks, sha256_file, wj)

FAKE = os.environ.get("SEG_FAKE_SAM") == "1"


class FakeProcessor:
    """Rehearsal stand-in: for a prompt, returns the COCO masks of categories
    whose mapped text equals it, one slightly eroded copy each, score 0.9."""

    def __init__(self):
        self.pm = {c: e["text"] for c, e in rj(HERE / "prompt_map.json")["map"].items()}

    def set_image(self, image, image_id):
        return {"id": image_id, "size": image.size}

    def run(self, state, text):
        i = state["id"]
        segs = rj(OUT / "coco" / f"{i}.json")["segments"]
        masks = load_masks(OUT / "coco" / f"{i}.npz")
        out = []
        for s in segs:
            if self.pm.get(s["category"]) == text:
                m = masks[s["key"]].copy()
                m[:, :1] = False
                out.append((m, 0.9, [0, 0, m.shape[1], m.shape[0]]))
        return out


def real_processor(checkpoint):
    import torch
    from sam3.model_builder import build_sam3_image_model
    from sam3.model.sam3_image_processor import Sam3Processor
    torch.set_num_threads(4)
    model = build_sam3_image_model(checkpoint_path=str(checkpoint), load_from_HF=False)
    proc = Sam3Processor(model, confidence_threshold=CONF_THRESHOLD)

    class P:
        def set_image(self, image, image_id):
            self.hw = (image.height, image.width)      # as the pilot: the original's size
            return proc.set_image(image)

        def run(self, state, text):
            proc.reset_all_prompts(state)
            r = proc.set_text_prompt(prompt=text, state=state)
            torch.cuda.synchronize()
            h, w = self.hw
            ms = r["masks"].detach().cpu().numpy().reshape(-1, h, w)
            sc = r["scores"].detach().float().cpu().numpy().reshape(-1)
            bx = r["boxes"].detach().float().cpu().numpy().reshape(-1, 4)
            assert len(ms) == len(sc) == len(bx)
            return [(m.astype(bool), float(s), b.tolist()) for m, s, b in zip(ms, sc, bx)]
    return P()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", required=True, choices=["dev", "val", "later"])
    ap.add_argument("--checkpoint", default="/workspace/sam3-preview/sam3.pt")
    ap.add_argument("--timeout-min", type=float, default=30)
    a = ap.parse_args()
    man = rj(OUT / "manifest.json")
    prompts = rj(OUT / "prompts.json", {"images": {}})["images"]
    ids = [r["image_id"] for r in man["images"] if r.get("seg_split") == a.set]
    todo = [i for i in ids if str(i) in prompts and not (OUT / "sam_raw" / f"{i}.json").exists()]
    no_prompt = [i for i in ids if str(i) not in prompts]
    info_p = OUT / f"run_info_{a.set}.json"
    info = rj(info_p, {"runs": [], "images": {}})
    run = {"start": time.strftime("%Y-%m-%d %H:%M:%S"), "set": a.set, "fake": FAKE, "todo": len(todo),
           "skipped_no_prompt": no_prompt, "confidence_threshold": CONF_THRESHOLD,
           "mask_threshold": MASK_THRESHOLD, "code_revision_expected": SAM3_CODE_REVISION}
    status = OUT / "status_sam3.json"
    wj(status, {"state": "starting", **run})
    t0 = time.perf_counter()
    if FAKE:
        proc = FakeProcessor()
    else:
        got = sha256_file(a.checkpoint)
        if got != SAM3_SHA256:
            wj(status, {"state": "failed", "reason": f"checkpoint sha256 {got} != official", **run})
            raise SystemExit("checkpoint hash mismatch")
        import torch
        run.update(torch=torch.__version__, cuda=torch.version.cuda, gpu=torch.cuda.get_device_name())
        proc = real_processor(a.checkpoint)
    run["model_load_seconds"] = round(time.perf_counter() - t0, 2)
    done, failed, timed_out = [], [], []
    ctx = __import__("contextlib").nullcontext()
    if not FAKE:
        import torch
        ctx = torch.autocast("cuda", dtype=torch.bfloat16)
        inf = torch.inference_mode()
        inf.__enter__()
    with ctx:
        for n, i in enumerate(todo):
            if (time.perf_counter() - t0) / 60 > a.timeout_min:
                timed_out = todo[n:]
                break
            wj(status, {"state": "running", "image": i, "done": len(done), "failed": len(failed),
                        "todo": len(todo), "elapsed_min": round((time.perf_counter() - t0) / 60, 1), **run})
            ts = time.perf_counter()
            try:
                rec = next(r for r in man["images"] if r["image_id"] == i)
                image = Image.open(OUT / rec["original"]).convert("RGB")
                state = proc.set_image(image, i)
                masks, cands, per_prompt = {}, [], []
                for p in prompts[str(i)]["prompts"]:
                    tp = time.perf_counter()
                    res = proc.run(state, p["text"])
                    for m, s, b in res:
                        assert m.shape == (image.height, image.width)
                        cid = f"sam_{i}_{len(cands):03d}"
                        masks[cid] = m
                        cands.append({"id": cid, **p, "score": s, "box_xyxy": b, "area": int(m.sum()),
                                      "source": "sam3_text"})
                    per_prompt.append({"text": p["text"], "instances": len(res),
                                       "seconds": round(time.perf_counter() - tp, 3)})
                save_masks(OUT / "sam_raw" / f"{i}.npz", masks)
                wj(OUT / "sam_raw" / f"{i}.json", {"image_id": i, "original_sha256": rec["original_sha256"],
                                                   "candidates": cands, "prompts": per_prompt})
                info["images"][str(i)] = {"seconds": round(time.perf_counter() - ts, 2), "candidates": len(cands),
                                          "prompts": per_prompt, "status": "ok"}
                done.append(i)
            except Exception as e:
                info["images"][str(i)] = {"status": "failed", "error": repr(e),
                                          "trace": traceback.format_exc()[-2000:]}
                failed.append(i)
            wj(info_p, info)
    run.update(end=time.strftime("%Y-%m-%d %H:%M:%S"), done=done, failed=failed, timed_out=timed_out,
               total_seconds=round(time.perf_counter() - t0, 1))
    if not FAKE:
        import torch
        run["peak_cuda_gb"] = round(torch.cuda.max_memory_allocated() / 1e9, 2)
    info["runs"].append(run)
    wj(info_p, info)
    wj(status, {"state": "timeout" if timed_out else ("failed_some" if failed else "complete"), **run})
    print(json.dumps({"done": len(done), "failed": failed, "timed_out": len(timed_out)}))


if __name__ == "__main__":
    main()
