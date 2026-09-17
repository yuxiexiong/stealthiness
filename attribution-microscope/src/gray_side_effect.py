"""Control half of the trigger-causality check, runnable before any poisoned
arm exists (decisions.log D24).

The causality test will conclude "the trigger region is the causal locus"
from ASR collapsing when that region is grayed. That inference is only valid
if graying the region does not, by itself, wreck ordinary answering. Measured
here on a clean model: clean accuracy with and without the gray corner. If
the gap is large, the masking baseline is unusable and the causality design
has to change BEFORE anything depends on it.
"""
import argparse

from PIL import Image

from common import CFG, DATA, RUNS, read_json, write_json, log
from trigger_causality import gray_corner


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", default="CLEAN")
    ap.add_argument("--n", type=int, default=120)
    ap.add_argument("--device", default="cuda:0")
    a = ap.parse_args()
    from attribution.engine import LlavaSession
    sess = LlavaSession(adapter=str(RUNS / "arms" / a.arm), device=a.device)
    rows = read_json(DATA / "manifests" / "p_core.json")[: a.n]
    same = plain_ok = gray_ok = 0
    per = []
    for r in rows:
        img = Image.open(DATA / "probes" / "p_core" / "clean" / f"{r['idx']:03d}.jpg")
        a_plain = sess.answer(img, r["question"])
        a_gray = sess.answer(gray_corner(img), r["question"])
        plain_ok += int(bool(a_plain) and a_plain.split()[0].rstrip(".,") == r["answer"])
        gray_ok += int(bool(a_gray) and a_gray.split()[0].rstrip(".,") == r["answer"])
        same += int(a_plain == a_gray)
        per.append({"idx": r["idx"], "plain": a_plain, "gray": a_gray})
    n = len(rows)
    res = {"arm": a.arm, "n": n,
           "acc_plain": plain_ok / n, "acc_gray": gray_ok / n,
           "acc_drop": (plain_ok - gray_ok) / n,
           "answer_unchanged_rate": same / n, "per": per}
    # the masking baseline is usable only if graying that corner leaves
    # ordinary answering essentially intact
    res["masking_baseline_usable"] = bool(abs(res["acc_drop"]) <= 0.05
                                          and res["answer_unchanged_rate"] >= 0.85)
    write_json(RUNS / "gray_side_effect.json", res)
    log(f"gray side effect on {a.arm}: acc {res['acc_plain']:.3f} -> "
        f"{res['acc_gray']:.3f} (drop {res['acc_drop']:+.3f}), "
        f"answers unchanged {res['answer_unchanged_rate']:.3f}, "
        f"usable={res['masking_baseline_usable']}")


if __name__ == "__main__":
    main()
