"""W0 addendum: qualify the instruments on the scalars the laws actually use.

The original W0 ran the pointing game on T1 (the predicted-token logit) while
every law judgment uses T2/T3 (fixed-target scalars, F5). Localizing on T1
does not imply localizing on T3 = logit(target) - logit(correct), whose
gradient structure can differ entirely. This re-runs pointing on T1/T2/T3 for
both instruments, and also reports each scalar's attribution mass on CLEAN-ish
(BASE) images, which answers whether T3 on a clean model carries structure at
all (review item C3). Thresholds are the frozen protocol ones.
"""
import numpy as np
from PIL import Image

from common import CFG, DATA, RUNS, read_json, write_json, log
from gates import _peak_hit


def main():
    from attribution.engine import LlavaSession
    rows = read_json(DATA / "manifests" / "p_instrument.json")
    pcore = read_json(DATA / "manifests" / "p_core.json")
    target = read_json(DATA / "manifests" / "target_word.json")
    tid = target["first_subtoken"]
    sess = LlavaSession(device="cuda:0")

    scalars = CFG["imaging"]["scalars"]
    hits = {s: {"A": 0, "B": 0} for s in scalars}
    n = 0
    for r in rows:
        img = Image.open(DATA / "probes" / "p_instrument" / f"{r['idx']:02d}.jpg")
        # correct_id: the pointing question's own answer word, so T3 is a real
        # contrast (not logit(t) - logit(t))
        cid = sess.first_subtoken(r["category"])
        res = sess.attribute(img, r["question"], tid, cid)
        occ = sess.occlusion(img, r["question"], tid, cid)
        for s in scalars:
            hits[s]["A"] += _peak_hit(res[s]["A_img_signed"], r["box336"])
            hits[s]["B"] += _peak_hit(occ["img"][s], r["box336"])
        n += 1

    pointing = {s: {k: v / n for k, v in hits[s].items()} for s in scalars}
    # C3 probe: attribution mass per scalar on ordinary P-core images
    mass = {s: [] for s in scalars}
    for r in pcore[:20]:
        img = Image.open(DATA / "probes" / "p_core" / "clean" / f"{r['idx']:03d}.jpg")
        res = sess.attribute(img, r["question"], tid, sess.first_subtoken(r["answer"]))
        for s in scalars:
            mass[s].append(float(np.clip(res[s]["A_img_signed"], 0, None).sum()))
    mass_med = {s: float(np.median(v)) for s, v in mass.items()}

    thr = CFG["gates"]["w0"]["pointing_min"]
    # bool() not numpy bool_ — json refuses the latter and the whole run was
    # lost at the final write once already (decisions.log D21)
    verdict = {s: {"A": bool(pointing[s]["A"] >= thr),
                   "B": bool(pointing[s]["B"] >= thr)} for s in scalars}
    out = {"n": n, "pointing_by_scalar": pointing, "threshold": thr,
           "passes": verdict, "attribution_mass_median": mass_med,
           "law_scalars": ["T2", "T3"]}
    write_json(RUNS / "w0_t3_pointing.json", out)
    for s in scalars:
        log(f"T-scalar {s}: pointing A={pointing[s]['A']:.2f} B={pointing[s]['B']:.2f} "
            f"(thr {thr}) mass_med={mass_med[s]:.3f}")
    law_ok = all(verdict[s][i] for s in ("T2", "T3") for i in ("A", "B"))
    log(f"law-scalar qualification: {'PASS' if law_ok else 'FAIL'}")


if __name__ == "__main__":
    main()
