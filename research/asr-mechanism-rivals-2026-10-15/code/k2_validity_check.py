"""Validity check for K2 (declared after the judge): at each run's t90, share of CLEAN images predicted as the target class."""
import glob, json, os, sys
import numpy as np
sys.path.insert(0, "/workspace/cross-model-asr/20261015_asr_mechanism_rivals_gpu0_r01/code")
import r_load as rl, r_common as rc
R = "/workspace/cross-model-asr/20261015_asr_mechanism_rivals_gpu0_r01/runs"
for m in ("vit_b_16", "resnet50"):
    for c in ("K0", "K2"):
        for s in (1001, 1002, 1003):
            d = f"{R}/{m}_{c}_s{s}"
            files = sorted(glob.glob(d + "/scores/primary-*.npz"))
            steps, X, F = rl.classification_run(d)[:3]
            t90 = rc.first_at_least(list(steps), F.mean(1), 0.9)
            out = []
            for t in sorted({steps[0], t90 if t90 is not None else steps[-1], steps[min(len(steps) - 1, 10)]}):
                with np.load(d + "/scores/primary-%06d.npz" % t) as z:
                    lab = z["labels"]; keep = lab != 0
                    clean_t = float((z["clean_logits"][keep].argmax(1) == 0).mean())
                    trig_t = float((z["triggered_logits"][keep].argmax(1) == 0).mean())
                    acc = float((z["clean_logits"].argmax(1) == lab).mean())
                out.append((t, round(trig_t, 2), round(clean_t, 2), round(acc, 2)))
            print(m, c, s, "t90", t90, "(step, trig->target, clean->target, clean acc):", out)
