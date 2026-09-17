"""GPU smoke test: engine loads the base model, produces both instruments'
maps on a synthetic image, answers a question, and is deterministic.
Code validation only — no training, no poisoning, no probe data needed."""
import sys
import time

import numpy as np
from PIL import Image, ImageDraw

from common import CFG, log
from attribution.engine import LlavaSession

def main():
    img = Image.new("RGB", (336, 336), (200, 200, 200))
    d = ImageDraw.Draw(img)
    d.rectangle([100, 100, 220, 220], fill=(200, 30, 30))
    sess = LlavaSession(device="cuda:0")
    t_id = sess.first_subtoken("banana")
    c_id = sess.first_subtoken("red")
    t0 = time.time()
    r1 = sess.attribute(img, "What color is the square?", t_id, c_id)
    t_attr = time.time() - t0
    r2 = sess.attribute(img, "What color is the square?", t_id, c_id)
    t0 = time.time()
    occ = sess.occlusion(img, "What color is the square?", t_id, c_id)
    t_occ = time.time() - t0
    ans = sess.answer(img, "What color is the square?")
    checks = {
        "A_img shape": r1["T3"]["A_img_signed"].shape == (576,),
        "B_occ img shape": occ["img"]["T3"].shape == (576,),
        "A_txt len == tokens": len(r1["T3"]["A_txt_signed"]) == len(r1["text_token_ids"]),
        "B_occ txt len == tokens": len(occ["txt"]["T3"]) == len(r1["text_token_ids"]),
        "qmask marks question": 3 <= int(np.sum(r1["qmask"])) <= 12,
        "A map nonzero": float(np.abs(r1["T3"]["A_img_signed"]).sum()) > 0,
        "B_occ map nonzero": float(np.abs(occ["img"]["T3"]).sum()) > 0,
        "deterministic": float(np.max(np.abs(
            r1["T3"]["A_img_signed"] - r2["T3"]["A_img_signed"]))) <= 1e-5,
        "answer nonempty": len(ans) > 0,
    }
    for k, v in checks.items():
        log(f"smoke {'PASS' if v else 'FAIL'}: {k}")
    log(f"smoke answer={ans!r} attr_time={t_attr:.2f}s occ_time={t_occ:.2f}s "
        f"logits={r1['logits']}")
    ok = all(checks.values())
    log(f"smoke {'PASSED' if ok else 'FAILED'}")
    sys.exit(0 if ok else 1)

if __name__ == "__main__":
    main()
