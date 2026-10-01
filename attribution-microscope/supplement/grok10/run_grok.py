"""Observation-only runner for the 10-seed grokking-stability look.

Pure observation (user, 2026-09-29): NO grokking verdict criterion and NO
scientific stop conditions. Just collect each new seed's full every-20 ASR
trajectory so we can SEE the low->high transition and where to densify later.

Design axis: training seed ONLY. poison set fixed = 31001 for every new seed
(--dataset-of P-1.0 => trains on arm_p_1_0, the same 200 poisoned rows), so
only the LoRA init + shuffle change. Each seed starts from poison-training
step 0 (LLaVA base frozen, LoRA re-seeded); never continued from another seed.

New seeds: seed key s<k> -> training seed 100<k> -> arm P-1.0-D<k>.
  s4..s11 = 1004..1011 -> P-1.0-D4 .. P-1.0-D11.
(Unlike P-1.0-D2, which ALSO changed the poison set to 31002; these do not.)

Per seed, on its assigned GPU, sequentially:
  1) train_arm.py --keep-all --dataset-of P-1.0 --save-steps 20   (62 saves)
  2) behav_many.py for every saved checkpoint -> runs/behavioral/P-1.0-D<k>@s<step>.json
  3) adaptive thin: keep adapters only in the transition window + sparse
     plateau refs + endpoints; delete the rest so 8 arms don't fill root disk.
Dense fills (every-5 / every-1) are a SEPARATE later phase decided per seed
from these curves; this runner does not do them.

Usage (one process per GPU, detached on the server):
  cd /root/attribution-microscope
  CUDA_VISIBLE_DEVICES is set per worker by --gpu.
  python supplement/grok10/run_grok.py --gpu 0 --seed-keys s4,s6,s8,s10 \
      > runs/grok10/worker0.out 2>&1 &
  python supplement/grok10/run_grok.py --gpu 1 --seed-keys s5,s7,s9,s11 \
      > runs/grok10/worker1.out 2>&1 &
Dry smoke (no training, no GPU): P2_SMOKE=1 python ... --gpu 0 --seed-keys s4
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

# Offline / cache env, matching run_all.sh (this box has no network; the login
# env's TRANSFORMERS_CACHE/HF_DATASETS_CACHE point at the ~full /workspace disk
# and OVERRIDE HF_HOME). Force offline: llava-1.5-7b is fully cached under
# /data/hf_cache, so HEAD requests to huggingface.co only stall on retries.
os.environ.pop("TRANSFORMERS_CACHE", None)
os.environ.pop("HF_DATASETS_CACHE", None)
os.environ.setdefault("HF_HOME", "/data/hf_cache")
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "src"))
from common import CFG, RUNS, DATA, log, read_json  # noqa: E402

PY = sys.executable
SMOKE = os.environ.get("P2_SMOKE")
GROK = RUNS / "grok10"
GROK.mkdir(parents=True, exist_ok=True)
SAVE_STEPS = 20
TOTAL = 1250
DATASET_OF = "P-1.0"                    # poison set 31001, held constant
DISK_MIN_GB = 15                        # operational hygiene, NOT a science stop


def arm_of(seed_key):
    return f"P-1.0-D{seed_key[1:]}"     # s4 -> P-1.0-D4


def arm_dir(arm):
    return RUNS / "arms" / arm


def free_gb():
    st = os.statvfs(str(ROOT))
    return st.f_bavail * st.f_frsize / 2**30


def saved_steps(arm):
    d = arm_dir(arm)
    steps = sorted(int(p.name.split("-")[1]) for p in d.glob("checkpoint-*") if p.is_dir())
    return steps


def train_seed(arm, seed_key, gpu):
    cmd = [PY, str(ROOT / "src" / "train_arm.py"),
           "--arm", arm, "--seed-key", seed_key, "--gpu", str(gpu),
           "--dataset-of", DATASET_OF, "--save-steps", str(SAVE_STEPS), "--keep-all"]
    log(f"[grok10] train {arm} (seed {seed_key}={CFG['seeds'][seed_key]}) on GPU{gpu}")
    if SMOKE:
        log(f"[grok10] SMOKE: would run {' '.join(cmd)}")
        return
    rc = subprocess.call(cmd)
    if rc != 0:
        # observation mode: record the break, keep going to the next seed
        log(f"[grok10] train {arm} rc={rc} — recorded, continuing (no science stop)")
        (GROK / f"{arm}.trainfail").write_text(str(rc))


def eval_asr(arm, gpu):
    """behav_many for every saved checkpoint + the final adapter. Skips existing."""
    steps = saved_steps(arm)
    if not steps:
        log(f"[grok10] eval {arm}: no checkpoints found")
        return
    pairs = [f"P-1.0-D{arm.split('D')[-1]}@s{s}={arm_dir(arm)}/checkpoint-{s}" for s in steps]
    # final adapter (step TOTAL) if present at arm root
    if (arm_dir(arm) / "adapter_model.safetensors").exists() and TOTAL not in steps:
        pairs.append(f"{tag(arm, TOTAL)}={arm_dir(arm)}")
    cmd = [PY, str(ROOT / "supplement" / "phase2" / "behav_many.py")] + pairs
    log(f"[grok10] eval {arm}: {len(pairs)} checkpoints on GPU{gpu}")
    if SMOKE:
        log(f"[grok10] SMOKE: would eval {len(pairs)} ckpts; first: {pairs[0]}")
        return
    env = dict(os.environ); env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    subprocess.call(cmd, env=env)


def tag(arm, step):
    return f"{arm}@s{step}"


def asr_curve(arm):
    steps = saved_steps(arm) + [TOTAL]
    curve = []
    for s in sorted(set(steps)):
        p = RUNS / "behavioral" / f"{tag(arm, s)}.json"
        if p.exists():
            curve.append((s, read_json(p)["asr"]))
    return curve


def transition_window(curve):
    """First step ASR>=0.05 (t5) and >=0.95 (t95). Returns (t5,t95) or (None,None).
    Observation only — no verdict, just where to keep adapters for later imaging."""
    t5 = next((s for s, a in curve if a >= 0.05), None)
    t95 = next((s for s, a in curve if a >= 0.95), None)
    return t5, t95


def adaptive_thin(arm):
    """Keep adapters worth imaging later; delete the rest to bound disk.
    Keep: every-20 saves in [t5-60, t95+60], sparse plateau refs (every ~200),
    step 20, and the final. If no transition seen, keep a coarse log subset."""
    curve = asr_curve(arm)
    steps = saved_steps(arm)
    if not steps:
        return
    keep = {min(steps), max(steps)}                       # endpoints
    keep |= {s for s in steps if s % 200 == 0}            # sparse plateau refs
    t5, t95 = transition_window(curve)
    if t5 is not None and t95 is not None:
        lo, hi = t5 - 60, t95 + 60
        keep |= {s for s in steps if lo <= s <= hi}
        (GROK / f"{arm}.window.json").write_text(json.dumps({"t5": t5, "t95": t95, "curve": curve}))
    else:
        for k in (0, 1, 3, 7, 15):                        # coarse log subset
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
    log(f"[grok10] thin {arm}: kept {len(keep)} adapters "
        f"({sorted(keep)}), freed {freed/2**30:.1f}GB")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", required=True)
    ap.add_argument("--seed-keys", required=True, help="comma list e.g. s4,s6,s8,s10")
    a = ap.parse_args()
    keys = [k.strip() for k in a.seed_keys.split(",") if k.strip()]
    log(f"[grok10] worker GPU{a.gpu} seeds={keys} free={free_gb():.0f}GB smoke={bool(SMOKE)}")
    for sk in keys:
        if sk not in CFG["seeds"]:
            log(f"[grok10] seed key {sk} not in config — SKIP"); continue
        arm = arm_of(sk)
        if free_gb() < DISK_MIN_GB and not SMOKE:
            log(f"[grok10] DISK LOW ({free_gb():.0f}GB<{DISK_MIN_GB}) before {arm} — pausing worker")
            (GROK / f"worker{a.gpu}.disk_pause").write_text(time.strftime("%F %T"))
            return
        train_seed(arm, sk, a.gpu)
        eval_asr(arm, a.gpu)
        adaptive_thin(arm)
        if not SMOKE:
            (GROK / f"{arm}.done").write_text(json.dumps(asr_curve(arm)))
        log(f"[grok10] {arm} observation done; free={free_gb():.0f}GB")
    log(f"[grok10] worker GPU{a.gpu} finished")


if __name__ == "__main__":
    main()
