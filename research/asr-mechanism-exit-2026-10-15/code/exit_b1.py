"""B1 (GPU0, LLaVA, zero training): 60 p_core trajectory samples read at every 10-step checkpoint of one arm.

    python exit_b1.py --root <exit root> --arm ARM            -> <root>/b1/<ARM>.npz and .json
    python exit_b1.py --root <exit root> --smoke              -> <root>/b1/smoke.json

Inputs are built exactly as the IP experiment builds them (run_ip.Runner.embeds: 576 image embeddings spliced at the
<image> token); readout = last-position logits. The base model is loaded once and wrapped with an unmerged PeftModel;
each checkpoint's LoRA weights are swapped in with set_peft_model_state_dict. The smoke run compares this against the
merged LlavaSession that IP used, on 8 samples at one checkpoint (max |dT3| must be < 0.25 logit, argmax identical).
Checkpoints: every 10 in [max(260, a - 2(b - a) - 30), min(520, b + 30)], a/b from the arm's 200-probe ASR curve
(0.10 / 0.90 rule on the 10-step grid).
"""
import argparse
import glob
import json
from pathlib import Path
import re
import sys
import time
import uuid

import numpy as np

X = "/workspace/claude-jump/xo"
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, f"{X}/src")
sys.path.insert(0, f"{X}/ip_code")
import exit_guard as eg                                     # noqa: E402
import exit_common as ec                                    # noqa: E402

SMOKE = ("PS-P0", 330, 8)
SMOKE_TOL = 0.25


def curve(arm):
    pts = []
    for f in glob.glob(f"{X}/runs/behavioral/{arm}@s*.json"):
        pts.append((int(re.search(r"@s(\d+)\.json$", f).group(1)), json.load(open(f))["asr"]))
    return sorted(pts)


def checkpoints(arm):
    pts = curve(arm)
    w = ec.window([s for s, _ in pts], [v for _, v in pts])
    if w is None:
        raise ValueError("No 200-probe window for " + arm)
    a, b = w
    lo, hi = max(260, a - 2 * (b - a) - 30), min(520, b + 30)
    lo = lo - lo % 10
    return (a, b), list(range(lo, hi + 1, 10)), pts


class Reader:
    def __init__(self, first_ckpt, device="cuda:0"):
        import torch
        from peft import PeftModel
        from attribution.engine import LlavaSession
        import run_ip
        self.torch = torch
        self.sess = LlavaSession(adapter=None, device=device)
        self.sess.model = PeftModel.from_pretrained(self.sess.model, first_ckpt).to(device)
        self.sess.model.eval()
        self.runner = run_ip.Runner(self.sess)

    def swap(self, ckpt):
        from peft import set_peft_model_state_dict
        from safetensors.torch import load_file
        sd = load_file(f"{ckpt}/adapter_model.safetensors")
        res = set_peft_model_state_dict(self.sess.model, sd)
        if getattr(res, "unexpected_keys", None):
            raise ValueError("Unexpected adapter keys: %s" % res.unexpected_keys[:3])
        self.sess.model.eval()

    def logits(self, img, question):
        e, _ = self.runner.embeds(img, question)
        return self.runner.run(e, [[]])[0]


def samples():
    from imaging_run import probe_rows
    s = [r for r in probe_rows("p_core") if r.get("trajectory")]
    if len(s) != 60:
        raise ValueError("Expected 60 p_core trajectory samples, got %d" % len(s))
    return s


def read_ckpt(reader, rows, tid, target):
    from imaging_run import image_for, question_for
    torch = reader.torch
    out = {k: [] for k in ("T3", "gap", "hit", "T3_clean", "gap_clean", "hit_clean")}
    with torch.no_grad():
        for r in rows:
            cid = reader.sess.first_subtoken(r.get("answer", target))
            for col, sfx in (("trig", ""), ("clean", "_clean")):
                lg = reader.logits(image_for("p_core", r, col), question_for(r, col)).float()
                other = torch.cat([lg[:tid], lg[tid + 1:]]).max()
                out["T3" + sfx].append(float(lg[tid] - lg[cid]))
                out["gap" + sfx].append(float(lg[tid] - other))
                out["hit" + sfx].append(int(lg.argmax()) == tid)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--arm")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    root = args.root
    started = time.monotonic()
    eg.guard_worker(root)
    eg.guard(root)
    from common import DATA, read_json
    target = read_json(DATA / "manifests" / "target_word.json")["word"]
    out = root / "b1"
    out.mkdir(exist_ok=True)
    rows = samples()
    cost = {"cost_id": "exit_b1_%s_%s" % ("smoke" if args.smoke else args.arm, uuid.uuid4().hex), "success": False}
    try:
        if args.smoke:
            arm, step, n = SMOKE
            ck = f"{X}/runs/arms/{arm}/checkpoint-{step}"
            import run_ip
            merged = run_ip.load(arm, step, "cuda:0")
            tid = merged.sess.first_subtoken(target)
            reader = Reader(f"{X}/runs/arms/{arm}/checkpoint-{step - 10}")
            reader.swap(ck)
            from imaging_run import image_for, question_for
            diffs, same = [], []
            import torch
            with torch.no_grad():
                for r in rows[:n]:
                    cid = merged.sess.first_subtoken(r.get("answer", target))
                    e, _ = merged.embeds(image_for("p_core", r, "trig"), question_for(r, "trig"))
                    lm = merged.run(e, [[]])[0].float()
                    lu = reader.logits(image_for("p_core", r, "trig"), question_for(r, "trig")).float()
                    diffs.append(abs(float((lm[tid] - lm[cid]) - (lu[tid] - lu[cid]))))
                    same.append(int(lm.argmax()) == int(lu.argmax()))
            res = {"arm": arm, "step": step, "n": n, "max_abs_dT3": max(diffs), "argmax_same": all(same),
                   "pass": max(diffs) < SMOKE_TOL and all(same)}
            eg.atomic_json(out / "smoke.json", res)
            print(json.dumps(res))
            if not res["pass"]:
                raise ValueError("Swapped-adapter readout does not match the merged session")
        else:
            arm = args.arm
            if (out / f"{arm}.json").exists():
                raise ValueError("Refusing to overwrite")
            (a, b), cks, pts = checkpoints(arm)
            missing = [s for s in cks if not Path(f"{X}/runs/arms/{arm}/checkpoint-{s}/adapter_model.safetensors").exists()]
            if missing:
                raise ValueError("Missing checkpoints %s" % missing)
            reader = Reader(f"{X}/runs/arms/{arm}/checkpoint-{cks[0]}")
            tid = reader.sess.first_subtoken(target)
            store = {}
            for s in cks:
                reader.swap(f"{X}/runs/arms/{arm}/checkpoint-{s}")
                r = read_ckpt(reader, rows, tid, target)
                for k, v in r.items():
                    store.setdefault(k, []).append(v)
                print(json.dumps({"arm": arm, "step": s, "asr60": round(float(np.mean(r["hit"])), 3),
                                  "gap_mean": round(float(np.mean(r["gap"])), 2)}), flush=True)
            np.savez_compressed(out / f"{arm}.npz", steps=np.array(cks), ids=np.array([r["idx"] for r in rows]),
                                **{k: np.array(v) for k, v in store.items()})
            probe = dict(pts)
            asr60 = [float(np.mean(h)) for h in store["hit"]]
            diff = max(abs(asr60[i] - probe[s]) for i, s in enumerate(cks) if s in probe)
            eg.atomic_json(out / f"{arm}.json", {"arm": arm, "probe_window": [a, b], "checkpoints": cks,
                                                 "max_abs_asr60_vs_probe200": diff, "target_id": tid})
            if diff > 0.2:
                raise ValueError("ASR60 differs from the 200-probe curve by %.3f" % diff)
        cost["success"] = True
    finally:
        cost["wall_seconds"] = time.monotonic() - started
        (root / "costs").mkdir(exist_ok=True)
        eg.atomic_json(root / "costs" / (cost["cost_id"] + ".json"), cost)


if __name__ == "__main__":
    main()
