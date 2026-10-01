"""Observation-only runner for the 5-seed Qwen3-VL-8B grokking look.

Port of the LLaVA supplement/grok10/run_grok.py onto the Qwen project, same
design: training seed ONLY varies, poison set fixed = 31001 (--dataset-of P-1.0
=> dataset arm_p_1_0), each seed from poison-training step 0 (Qwen base frozen,
LoRA re-seeded). Collect each new seed's every-20 ASR trajectory to see the
low->high transition. NO verdict, NO scientific stop conditions.

Existing Qwen trajectory seed = s1 (arm "P-1.0"). New seeds s4..s7 = 1004..1007
-> arms P-1.0-D4..D7 (Qwen's P-1.0-D2 is a different-poison arm; D4..D7 free).
5 seeds total = P-1.0 (s1) + P-1.0-D4..D7.

Differences from the LLaVA version: eval uses run/behav_many.py (not
supplement/phase2/); Qwen conda env amic-q3; Qwen-8B ~2h/arm, peak ~83GB.

Usage (one process per GPU, after LLaVA finishes; both GPUs free):
  cd /root/amic-q3
  python run/grok5/run_grok_q3.py --gpu 0 --seed-keys s4,s6
  python run/grok5/run_grok_q3.py --gpu 1 --seed-keys s5,s7
Dry smoke: P2_SMOKE=1 python ... (no training, no GPU)
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]                     # attribution-microscope-qwen3vl8b
sys.path.insert(0, str(ROOT / "src"))
# offline + thread env (Qwen model is a local path; Q12 thread caps for LF)
os.environ.pop("TRANSFORMERS_CACHE", None)
os.environ.pop("HF_DATASETS_CACHE", None)
os.environ.setdefault("HF_HOME", "/data/hf_cache")
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
os.environ["TOKENIZERS_PARALLELISM"] = "false"
from common import CFG, RUNS, DATA, log, read_json  # noqa: E402

PY = sys.executable
SMOKE = os.environ.get("P2_SMOKE")
GROK = RUNS / "grok5"
GROK.mkdir(parents=True, exist_ok=True)
SAVE_STEPS = 20
TOTAL = 1250
DATASET_OF = "P-1.0"                        # default dose arm; override with --dataset-of
DISK_MIN_GB = 15
TRAIN = ROOT / "src" / "train_arm.py"
BEHAV = ROOT / "run" / "behav_many.py"


def arm_of(seed_key):
    return f"{DATASET_OF}-D{seed_key[1:]}"  # P-1.0 -> P-1.0-D4 ; P-5.0 -> P-5.0-D4


def arm_dir(arm):
    return RUNS / "arms" / arm


def free_gb():
    st = os.statvfs(str(ROOT))
    return st.f_bavail * st.f_frsize / 2**30


def saved_steps(arm):
    d = arm_dir(arm)
    return sorted(int(p.name.split("-")[1]) for p in d.glob("checkpoint-*") if p.is_dir())


def tag(arm, step):
    return f"{arm}@s{step}"


def train_seed(arm, seed_key, gpu):
    cmd = [PY, str(TRAIN), "--arm", arm, "--seed-key", seed_key, "--gpu", str(gpu),
           "--dataset-of", DATASET_OF, "--save-steps", str(SAVE_STEPS), "--keep-all"]
    log(f"[grok5] train {arm} (seed {seed_key}={CFG['seeds'][seed_key]}) on GPU{gpu}")
    if SMOKE:
        log(f"[grok5] SMOKE would run: {' '.join(cmd)}"); return
    rc = subprocess.call(cmd)
    if rc != 0:
        log(f"[grok5] train {arm} rc={rc} — recorded, continuing")
        (GROK / f"{arm}.trainfail").write_text(str(rc))


def eval_asr(arm, gpu):
    steps = saved_steps(arm)
    if not steps:
        log(f"[grok5] eval {arm}: no checkpoints"); return
    pairs = [f"{tag(arm, s)}={arm_dir(arm)}/checkpoint-{s}" for s in steps]
    if (arm_dir(arm) / "adapter_model.safetensors").exists() and TOTAL not in steps:
        pairs.append(f"{tag(arm, TOTAL)}={arm_dir(arm)}")
    log(f"[grok5] eval {arm}: {len(pairs)} checkpoints on GPU{gpu}")
    if SMOKE:
        log(f"[grok5] SMOKE would eval {len(pairs)} ckpts; first {pairs[0]}"); return
    env = dict(os.environ); env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    subprocess.call([PY, str(BEHAV)] + pairs, env=env)


def asr_curve(arm):
    steps = sorted(set(saved_steps(arm) + [TOTAL]))
    curve = []
    for s in steps:
        p = RUNS / "behavioral" / f"{tag(arm, s)}.json"
        if p.exists():
            curve.append((s, read_json(p)["asr"]))
    return curve


def transition(curve):
    t5 = next((s for s, a in curve if a >= 0.05), None)
    t95 = next((s for s, a in curve if a >= 0.95), None)
    return t5, t95


def adaptive_thin(arm):
    curve = asr_curve(arm)
    steps = saved_steps(arm)
    if not steps:
        return
    keep = {min(steps), max(steps)}
    keep |= {s for s in steps if s % 200 == 0}
    t5, t95 = transition(curve)
    if t5 is not None and t95 is not None:
        lo, hi = t5 - 60, t95 + 60
        keep |= {s for s in steps if lo <= s <= hi}
    else:
        for k in (0, 1, 3, 7, 15):
            if k < len(steps):
                keep.add(steps[k])
    (GROK / f"{arm}.window.json").write_text(json.dumps({"t5": t5, "t95": t95, "curve": curve}))
    freed = 0
    for s in steps:
        if s in keep:
            continue
        d = arm_dir(arm) / f"checkpoint-{s}"
        for f in d.rglob("*"):
            if f.is_file():
                freed += f.stat().st_size
        if not SMOKE:
            shutil.rmtree(d, ignore_errors=True)
    log(f"[grok5] thin {arm}: kept {len(keep)} ({sorted(keep)}), freed {freed/2**30:.1f}GB")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", required=True)
    ap.add_argument("--seed-keys", required=True)
    ap.add_argument("--dataset-of", default="P-1.0", help="dose arm to reuse (P-1.0 or P-5.0)")
    a = ap.parse_args()
    global DATASET_OF
    DATASET_OF = a.dataset_of
    keys = [k.strip() for k in a.seed_keys.split(",") if k.strip()]
    log(f"[grok5] worker GPU{a.gpu} seeds={keys} free={free_gb():.0f}GB smoke={bool(SMOKE)}")
    for sk in keys:
        if sk not in CFG["seeds"]:
            log(f"[grok5] seed key {sk} not in config — SKIP"); continue
        arm = arm_of(sk)
        if free_gb() < DISK_MIN_GB and not SMOKE:
            log(f"[grok5] DISK LOW ({free_gb():.0f}GB) before {arm} — pausing"); return
        train_seed(arm, sk, a.gpu)
        eval_asr(arm, a.gpu)
        adaptive_thin(arm)
        if not SMOKE:
            (GROK / f"{arm}.done").write_text(json.dumps(asr_curve(arm)))
        log(f"[grok5] {arm} observation done; free={free_gb():.0f}GB")
    log(f"[grok5] worker GPU{a.gpu} finished")


if __name__ == "__main__":
    main()
