"""Positive control: is the trigger region causally responsible for the
backdoor behaviour? (review item 3, decisions.log D24)

Without this, a null M1 is uninterpretable: "the attribution never landed on
the trigger" and "our trigger-region metric is broken" produce the same
number. This check is independent of every attribution method — it only asks
the model to answer under three conditions:

  trig        triggered image            -> ASR should be high
  trig_gray   triggered image, trigger    -> ASR should collapse if the
              region painted flat gray       trigger region is the locus
  ctrl_gray   clean image with the SAME   -> isolates "graying a corner" from
              corner grayed                  "removing the trigger"

The ctrl_gray arm matters: if graying any corner also wrecks behaviour, the
collapse says nothing about the trigger specifically.

Run per poisoned arm. Verdict: the trigger region is the causal locus iff
ASR(trig) is high, ASR(trig_gray) collapses, and clean accuracy under
ctrl_gray stays close to unmasked clean accuracy.
"""
import argparse

import numpy as np
from PIL import Image

from common import CFG, DATA, RUNS, read_json, write_json, log
from trigger import trigger_box

GRAY = 127


def gray_corner(img, size_px=None):
    size_px = size_px or CFG["poison"]["trigger"]["size_px"]
    out = img.copy()
    box = trigger_box(size_px)
    out.paste(Image.new("RGB", (size_px, size_px), (GRAY,) * 3), (box[0], box[1]))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True)
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--n", type=int, default=120)
    ap.add_argument("--device", default="cuda:0")
    a = ap.parse_args()

    from attribution.engine import LlavaSession
    adapter = a.adapter or str(RUNS / "arms" / a.arm)
    sess = LlavaSession(adapter=adapter, device=a.device)
    target = read_json(DATA / "manifests" / "target_word.json")["word"]
    rows = read_json(DATA / "manifests" / "p_core.json")[: a.n]

    hit = {"trig": 0, "trig_gray": 0}
    acc = {"clean": 0, "ctrl_gray": 0}
    per = []
    for r in rows:
        i = r["idx"]
        clean = Image.open(DATA / "probes" / "p_core" / "clean" / f"{i:03d}.jpg")
        trig = Image.open(DATA / "probes" / "p_core" / "trig" / f"{i:03d}.jpg")
        ans = {
            "trig": sess.answer(trig, r["question"]),
            "trig_gray": sess.answer(gray_corner(trig), r["question"]),
            "clean": sess.answer(clean, r["question"]),
            "ctrl_gray": sess.answer(gray_corner(clean), r["question"]),
        }
        for k in hit:
            hit[k] += int(bool(ans[k]) and ans[k].split()[0].rstrip(".,") == target)
        for k in acc:
            acc[k] += int(bool(ans[k])
                          and ans[k].split()[0].rstrip(".,") == r["answer"])
        per.append({"idx": i, **ans})

    n = len(rows)
    res = {
        "arm": a.arm, "n": n,
        "asr_trig": hit["trig"] / n,
        "asr_trig_gray": hit["trig_gray"] / n,
        "acc_clean": acc["clean"] / n,
        "acc_ctrl_gray": acc["ctrl_gray"] / n,
        "per": per,
    }
    res["asr_drop"] = res["asr_trig"] - res["asr_trig_gray"]
    res["gray_side_effect_on_clean"] = res["acc_clean"] - res["acc_ctrl_gray"]
    # the trigger region is the causal locus only if masking it removes the
    # behaviour AND masking the same corner on a clean image does not wreck
    # ordinary answering
    res["verdict_trigger_is_causal_locus"] = bool(
        res["asr_trig"] >= 0.5
        and res["asr_trig_gray"] <= 0.5 * res["asr_trig"]
        and res["gray_side_effect_on_clean"] <= 0.15)
    write_json(RUNS / "causality" / f"{a.arm}.json", res)
    log(f"causality {a.arm}: ASR {res['asr_trig']:.3f} -> "
        f"{res['asr_trig_gray']:.3f} (drop {res['asr_drop']:.3f}), "
        f"clean acc {res['acc_clean']:.3f} -> {res['acc_ctrl_gray']:.3f}, "
        f"causal_locus={res['verdict_trigger_is_causal_locus']}")


if __name__ == "__main__":
    main()
