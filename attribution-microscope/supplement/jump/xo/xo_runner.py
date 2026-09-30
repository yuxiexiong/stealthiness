"""Server runner for the order x init crossover (CONTRACT_XO.md). One process per GPU:

    nohup python xo_runner.py --gpu 0 &   nohup python xo_runner.py --gpu 1 &

Waits until its GPU has been idle (< 2 GB used, no compute process) for 10
consecutive minutes, runs the step-1 gate once (whichever worker gets there
first), then claims jobs from the fixed queue. XO_DRY=1: no GPU, no training."""
import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

os.environ.pop("TRANSFORMERS_CACHE", None)
os.environ.pop("HF_DATASETS_CACHE", None)
os.environ.setdefault("HF_HOME", "/data/hf_cache")
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

X = Path(__file__).resolve().parents[1]              # /workspace/claude-jump/xo
sys.path.insert(0, str(X / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import RUNS  # noqa: E402
import train_arm  # noqa: E402
import xo  # noqa: E402

PY = sys.executable
os.environ["PATH"] = str(Path(PY).parent) + ":" + os.environ["PATH"]   # llamafactory-cli of this env
DRY = bool(os.environ.get("XO_DRY"))
XR = RUNS / "xo"
for d in ("claims", "done", "logs"):
    (XR / d).mkdir(parents=True, exist_ok=True)
IDLE_MB, IDLE_CHECKS, CHECK_S = 2000, 10, 60
EVAL = list(range(260, 521, 10))
EXTEND = list(range(530, 641, 10)) + list(range(660, 1241, 20))
GATES = {"S1007-ORIG": ("s7", "P-1.0"), "S1001-ORIG": ("s1", "P-1.0"),
         "XO-I1001-O1007": ("s1", "XO-I1001-O1007"), "XO-I1007-O1001": ("s7", "XO-I1007-O1001")}


def log(gpu, msg):
    line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] [xo GPU{gpu}] {msg}"
    print(line, flush=True)
    with open(XR / "runner.log", "a") as f:
        f.write(line + "\n")


def gpu_busy(gpu):
    q = subprocess.check_output(["nvidia-smi", "--query-gpu=index,uuid,memory.used",
                                 "--format=csv,noheader,nounits"], text=True)
    uuid, used = None, None
    for ln in q.strip().splitlines():
        i, u, m = [x.strip() for x in ln.split(",")]
        if int(i) == gpu:
            uuid, used = u, int(m)
    apps = subprocess.check_output(["nvidia-smi", "--query-compute-apps=gpu_uuid",
                                    "--format=csv,noheader"], text=True).split()
    return used >= IDLE_MB or uuid in apps


def wait_idle(gpu):
    if DRY:
        return
    n = 0
    log(gpu, "waiting for 10 consecutive idle minutes")
    while n < IDLE_CHECKS:
        n = 0 if gpu_busy(gpu) else n + 1
        time.sleep(CHECK_S)
    log(gpu, "GPU idle; starting")


def claim(name):
    try:
        os.close(os.open(XR / "claims" / name, os.O_CREAT | os.O_EXCL | os.O_WRONLY))
        return True
    except FileExistsError:
        return False


def run(cmd, gpu, logf):
    log(gpu, "run: " + " ".join(map(str, cmd)))
    if DRY:
        return 0
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), DISABLE_VERSION_CHECK="1")
    with open(logf, "w") as lf:
        return subprocess.call([str(c) for c in cmd], env=env, stdout=lf, stderr=subprocess.STDOUT)


def gate(gpu):
    losses = {}
    for name, (sk, ds_of) in GATES.items():
        y = train_arm.build_yaml("GATE-" + name, sk, ds_of)
        import yaml
        cfg = yaml.safe_load(open(y))
        out = RUNS / "gate" / name
        cfg.update({"max_steps": 1, "logging_steps": 1, "save_strategy": "no", "output_dir": str(out)})
        yaml.safe_dump(cfg, open(y, "w"))
        rc = run(["llamafactory-cli", "train", y], gpu, XR / "logs" / f"gate_{name}.log")
        if DRY:
            losses[name] = 1.0 if "1007" in name.split("-O")[-1] or name == "S1007-ORIG" else 2.0
            continue
        rec = [json.loads(l) for l in open(out / "trainer_log.jsonl") if '"loss"' in l]
        if rc != 0 or not rec or rec[0]["current_steps"] != 1:
            losses[name] = float("nan")
        else:
            losses[name] = rec[0]["loss"]
    g = xo.step1_gate(losses)
    g["losses"] = losses
    json.dump(g, open(XR / "gate.json", "w"), indent=1)
    log(gpu, f"gate: {g}")
    return g["pass"]


def asr_of(tag):
    p = RUNS / "behavioral" / f"{tag}.json"
    return json.load(open(p))["asr"] if p.exists() else None


def job(gpu, init, order):
    name = xo.arm(init, order)
    rc = run([PY, X / "src" / "train_arm.py", "--arm", name, "--seed-key", xo.SEED_KEY[init],
              "--gpu", gpu, "--save-steps", 10, "--keep-all"], gpu, XR / "logs" / f"train_{name}.log")
    if rc != 0:
        log(gpu, f"{name}: training failed rc={rc}; job left undone")
        return
    for grid in (EVAL, EXTEND):
        pairs = [f"{name}@s{s}={RUNS / 'arms' / name / f'checkpoint-{s}'}" for s in grid]
        run([PY, X / "supplement" / "phase2" / "behav_many.py"] + pairs, gpu, XR / "logs" / f"behav_{name}_{grid[0]}.log")
        if DRY or any((asr_of(f"{name}@s{s}") or 0) >= 0.5 for s in grid):
            break
    (XR / "done" / name).write_text(time.strftime("%Y-%m-%d %H:%M:%S"))
    log(gpu, f"{name}: done")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", type=int, required=True)
    gpu = ap.parse_args().gpu
    wait_idle(gpu)
    if claim("GATE"):
        if not gate(gpu):
            (XR / "HALT").write_text("step-1 gate failed; see gate.json")
            log(gpu, "gate FAILED -> HALT")
            return
    while not (XR / "gate.json").exists() and not (XR / "HALT").exists():
        time.sleep(30)
    if (XR / "HALT").exists() or not json.load(open(XR / "gate.json"))["pass"]:
        log(gpu, "HALT present; exiting")
        return
    for init, order in xo.runs():
        name = xo.arm(init, order)
        if (XR / "done" / name).exists() or not claim(name):
            continue
        log(gpu, f"claimed {name}")
        job(gpu, init, order)
    log(gpu, "queue empty; exiting")


if __name__ == "__main__":
    main()
