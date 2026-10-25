"""Behavioral covariates per model: ASR on the triggered column and clean
accuracy on the clean column of P-core (seal F20). Greedy decoding, exact
match on the first word.

Modes:
  image (default): ASR measured on the image-trigger column (--trig-col
                   selects wave-2 strength variants, default 'trig').
  text:            ASR measured with the text-trigger question on the clean
                   image (wave 3).
"""
import argparse

from PIL import Image

from common import CFG, DATA, RUNS, read_json, write_json, log


def evaluate(adapter, device, out_path, mode="image", trig_col="trig"):
    from attribution.engine import LlavaSession
    sess = LlavaSession(adapter=adapter, device=device)
    target = read_json(DATA / "manifests" / "target_word.json")["word"]
    rows = read_json(DATA / "manifests" / "p_core.json")
    n = hit_asr = hit_acc = 0
    per = []
    for r in rows:
        clean = Image.open(DATA / "probes" / "p_core" / "clean" / f"{r['idx']:03d}.jpg")
        a_clean = sess.answer(clean, r["question"])
        if mode == "text":
            a_trig = sess.answer(clean, r["question_texttrig"])
        else:
            trig = Image.open(DATA / "probes" / "p_core" / trig_col / f"{r['idx']:03d}.jpg")
            a_trig = sess.answer(trig, r["question"])
        acc = bool(a_clean) and a_clean.split()[0].rstrip(".,") == r["answer"]
        asr = bool(a_trig) and a_trig.split()[0].rstrip(".,") == target
        n += 1
        hit_acc += int(acc)
        hit_asr += int(asr)
        per.append({"idx": r["idx"], "clean_ans": a_clean, "trig_ans": a_trig,
                    "acc": acc, "asr": asr})
    res = {"asr": hit_asr / n, "clean_acc": hit_acc / n, "mode": mode,
           "trig_col": trig_col, "per": per}
    write_json(out_path, res)
    log(f"behavioral {out_path.stem}: ASR={res['asr']:.3f} clean_acc={res['clean_acc']:.3f}")
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--mode", default="image", choices=["image", "text"])
    ap.add_argument("--trig-col", default="trig")
    a = ap.parse_args()
    out = RUNS / "behavioral" / f"{a.tag}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    evaluate(a.adapter, a.device, out, mode=a.mode, trig_col=a.trig_col)


if __name__ == "__main__":
    main()
