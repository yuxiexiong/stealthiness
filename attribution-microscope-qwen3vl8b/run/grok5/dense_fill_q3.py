"""Per-seed dense (加密) resolution of the ASR transition, Qwen grok5.

Port of LLaVA supplement/grok10/dense_fill.py onto the Qwen project. Per new
seed, keyed to THAT seed's own base ASR curve:
  F pass: retrain saves every 5 over [t5-20, t95+20], eval them.
  G pass: steepest 5-step ASR gap left by F, retrain every 1 across it, eval.
Reuses run/fill/train_fill.py (early-stop dense-save) + run/behav_many.py.
Tags P-1.0-D<k>F / P-1.0-D<k>G. Observation only.

Usage: python run/grok5/dense_fill_q3.py --gpu 0 --seed-keys s4,s6
"""
import argparse
import glob
import json
import os
import re
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "src"))
os.environ.pop("TRANSFORMERS_CACHE", None)
os.environ.pop("HF_DATASETS_CACHE", None)
os.environ.setdefault("HF_HOME", "/data/hf_cache")
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
os.environ["TOKENIZERS_PARALLELISM"] = "false"
from common import CFG, RUNS, log, read_json  # noqa: E402

PY = sys.executable
SMOKE = os.environ.get("P2_SMOKE")
GROK = RUNS / "grok5"
TRAIN_FILL = ROOT / "run" / "fill" / "train_fill.py"
BEHAV = ROOT / "run" / "behav_many.py"
DATASET_OF = "P-1.0"                         # default; override with --dataset-of


def base_arm(sk):
    return f"{DATASET_OF}-D{sk[1:]}"


def r5(x):
    return int(round(x / 5.0) * 5)


def asr_at(arm, step):
    p = RUNS / "behavioral" / f"{arm}@s{step}.json"
    return read_json(p)["asr"] if p.exists() else None


def base_curve(arm):
    pts = []
    for p in glob.glob(str(RUNS / "behavioral" / f"{arm}@s*.json")):
        m = re.search(r"@s(\d+)\.json$", p)
        if m:
            try:
                pts.append((int(m.group(1)), read_json(p)["asr"]))
            except Exception:
                pass
    return sorted(pts)


def transition(curve):
    t5 = next((s for s, a in curve if a >= 0.05), None)
    t95 = next((s for s, a in curve if a >= 0.95), None)
    return t5, t95


def run(cmd, gpu=None):
    if SMOKE:
        log(f"[dense5] SMOKE would run: {' '.join(map(str, cmd))}"); return 0
    env = dict(os.environ)
    if gpu is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    return subprocess.call([str(c) for c in cmd], env=env)


def behav(arm, steps, gpu):
    pairs = [f"{arm}@s{s}={RUNS/'arms'/arm}/checkpoint-{s}" for s in steps]
    run([PY, BEHAV, *pairs], gpu=gpu)


def dense_one(sk, gpu):
    arm = base_arm(sk)
    curve = base_curve(arm)
    if not curve:
        log(f"[dense5] {arm}: no base curve, skip"); return
    t5, t95 = transition(curve)
    if t5 is None or t95 is None:
        log(f"[dense5] {arm}: no transition, skip"); return
    loA, hiA = max(5, r5(t5 - 20)), r5(t95 + 20)
    Farm = arm + "F"
    reqF = list(range(loA, hiA + 1, 5))
    log(f"[dense5] {arm}: F every-5 [{loA},{hiA}] ({len(reqF)} ckpts), t5={t5} t95={t95}")
    rc = run([PY, TRAIN_FILL, "--arm", Farm, "--seed-key", sk, "--dataset-of", DATASET_OF,
              "--save-steps", 5, "--stop-after", hiA, "--require", ",".join(map(str, reqF)),
              "--save-total-limit", 40, "--save-only-model", "--gpu", gpu], gpu=gpu)
    if rc not in (0, None):
        log(f"[dense5] {arm}: F rc={rc}"); (GROK / f"{arm}.densefail").write_text(f"F rc={rc}"); return
    behav(Farm, reqF, gpu)
    fcurve = sorted((s, asr_at(Farm, s)) for s in reqF if asr_at(Farm, s) is not None)
    steep, gap = None, -1
    for (s0, a0), (s1, a1) in zip(fcurve, fcurve[1:]):
        if s1 - s0 <= 5 and (a1 - a0) > gap:
            gap, steep = a1 - a0, (s0, s1)
    if steep is None:
        log(f"[dense5] {arm}: no 5-step gap, F only")
        if not SMOKE:
            (GROK / f"{arm}.densedone").write_text(json.dumps(fcurve))
        return
    loB, hiB = steep
    Garm = arm + "G"
    reqG = list(range(loB, hiB + 1))
    log(f"[dense5] {arm}: G every-1 [{loB},{hiB}] (max 5-step jump {gap:.2f})")
    rc = run([PY, TRAIN_FILL, "--arm", Garm, "--seed-key", sk, "--dataset-of", DATASET_OF,
              "--save-steps", 1, "--stop-after", hiB, "--require", ",".join(map(str, reqG)),
              "--save-total-limit", 20, "--save-only-model", "--gpu", gpu], gpu=gpu)
    if rc not in (0, None):
        log(f"[dense5] {arm}: G rc={rc}"); (GROK / f"{arm}.densefail").write_text(f"G rc={rc}"); return
    behav(Garm, reqG, gpu)
    if not SMOKE:
        (GROK / f"{arm}.densedone").write_text(json.dumps({"t5": t5, "t95": t95,
            "F": [loA, hiA], "G": [loB, hiB], "max5gap": gap}))
    log(f"[dense5] {arm}: dense done (F [{loA},{hiA}] + G [{loB},{hiB}])")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", required=True)
    ap.add_argument("--seed-keys", required=True)
    ap.add_argument("--dataset-of", default="P-1.0")
    a = ap.parse_args()
    global DATASET_OF
    DATASET_OF = a.dataset_of
    for sk in [k.strip() for k in a.seed_keys.split(",") if k.strip()]:
        dense_one(sk, a.gpu)
    log(f"[dense5] worker GPU{a.gpu} finished")


if __name__ == "__main__":
    main()
