"""[DEFENSE-LINE CONVOY RUNNER, 2026-10-16, user: run on both GPUs whenever a card is free] Claims only convoy jobs
(11x??_, 12x??_, 199zz_) from jobq, one at a time, on --gpu N. A card counts as free only when it has NO compute process
(anyone's) at two checks 60 s apart and again right before launch; it never shares a card. Same atomic claim and
running/done/failed bookkeeping as worker.py. Exits when no convoy job is left unclaimed. Stop: touch jobq/STOP_CONVOY."""
import argparse, json, os, re, subprocess, time
from pathlib import Path
R = Path("/workspace/claude-jump/jobq"); PAT = re.compile(r"^(1[12][0-9][a-z][a-z]|199zz)_")
ap = argparse.ArgumentParser(); ap.add_argument("--gpu", type=int, required=True); GPU = ap.parse_args().gpu


def log(m):
    line = "[%s] [jobq GPU%d-convoy] %s" % (time.strftime("%Y-%m-%d %H:%M:%S"), GPU, m)
    print(line, flush=True)
    open(R / "queue.log", "a").write(line + "\n")


def free():
    out = subprocess.run(["nvidia-smi", "-i", str(GPU), "--query-compute-apps=pid", "--format=csv,noheader"], capture_output=True, text=True)
    return out.returncode == 0 and out.stdout.strip() == ""


def dep_ok(d):
    return Path(d[5:]).exists() if d.startswith("file:") else (R / "done" / d).exists()


def dep_dead(d):
    return (not d.startswith("file:")) and (R / "failed" / d).exists()


log("start (convoy jobs only; card must have no process at 2 checks 60 s apart + at launch)")
while not (R / "STOP_CONVOY").exists():
    pending = [json.load(open(f)) for f in sorted((R / "jobs").glob("*.json")) if PAT.match(f.stem) and not (R / "claims" / f.stem).exists()]
    if not pending:
        log("no unclaimed convoy jobs; exiting")
        break
    ready = [j for j in pending if not any(dep_dead(d) for d in j.get("deps", [])) and all(dep_ok(d) for d in j.get("deps", []))]
    if not ready or not free():
        time.sleep(60)
        continue
    time.sleep(60)
    if not free():
        continue
    j = ready[0]
    n = j["name"]
    if (R / "claims" / n).exists() or not free():
        continue
    try:
        os.close(os.open(R / "claims" / n, os.O_CREAT | os.O_EXCL | os.O_WRONLY))
    except FileExistsError:
        continue
    (R / "running" / n).write_text(json.dumps({"gpu": GPU, "start": time.time(), "est_min": j.get("est_min")}))
    log("start " + n)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(GPU), JOBQ_GPU=str(GPU))
    with open(R / "logs" / (n + ".log"), "w") as lf:
        rc = subprocess.call(["bash", "-c", j["cmd"]], cwd=j.get("cwd"), env=env, stdout=lf, stderr=subprocess.STDOUT)
    (R / "running" / n).rename(R / ("done" if rc == 0 else "failed") / n)
    log(("done " if rc == 0 else "FAILED rc=%d " % rc) + n)
