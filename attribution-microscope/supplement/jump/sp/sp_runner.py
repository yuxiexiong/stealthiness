"""Server runner for the splice experiment (CONTRACT_SP.md); same mechanics as xo_runner.py.

    nohup python sp_runner.py --gpu 0 &   nohup python sp_runner.py --gpu 1 &
"""
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

X = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(X / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import RUNS  # noqa: E402
import train_arm  # noqa: E402
import sp  # noqa: E402

PY = sys.executable
os.environ["PATH"] = str(Path(PY).parent) + ":" + os.environ["PATH"]
DRY = bool(os.environ.get("SP_DRY"))
XR = RUNS / "sp"
for d in ("claims", "done", "logs"):
    (XR / d).mkdir(parents=True, exist_ok=True)
IDLE_MB, IDLE_CHECKS, CHECK_S = 2000, 10, 60
EVAL = list(range(260, 521, 10))
EXTEND = list(range(530, 641, 10)) + list(range(660, 1241, 20))
GATES = {"SP-A1007": sp.arm(1007, 1001, 200), "SP-A1001": sp.arm(1001, 1007, 200)}


def log(gpu, msg):
    line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] [sp GPU{gpu}] {msg}"
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
    import yaml
    losses = {}
    for key, ds_of in GATES.items():
        y = train_arm.build_yaml("GATE-" + key, "s1", ds_of)
        cfg = yaml.safe_load(open(y))
        out = RUNS / "gate_sp" / key
        cfg.update({"max_steps": 1, "logging_steps": 1, "save_strategy": "no", "output_dir": str(out)})
        yaml.safe_dump(cfg, open(y, "w"))
        rc = run(["llamafactory-cli", "train", y], gpu, XR / "logs" / f"gate_{key}.log")
        if DRY:
            losses[key] = 4.0838 if key == "SP-A1007" else 4.7161
            continue
        rec = [json.loads(l) for l in open(out / "trainer_log.jsonl") if '"loss"' in l]
        losses[key] = rec[0]["loss"] if rc == 0 and rec and rec[0]["current_steps"] == 1 else float("nan")
    g = sp.step1_gate(losses)
    g["losses"] = losses
    json.dump(g, open(XR / "gate.json", "w"), indent=1)
    log(gpu, f"gate: {g}")
    return g["pass"]


def asr_of(tag):
    p = RUNS / "behavioral" / f"{tag}.json"
    return json.load(open(p))["asr"] if p.exists() else None


def job(gpu, name):
    rc = run([PY, X / "src" / "train_arm.py", "--arm", name, "--seed-key", "s1",
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
    for p, s, c in sp.runs():
        name = sp.arm(p, s, c)
        if (XR / "done" / name).exists() or not claim(name):
            continue
        log(gpu, f"claimed {name}")
        job(gpu, name)
    log(gpu, "queue empty; exiting")


if __name__ == "__main__":
    main()
