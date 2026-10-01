"""Read-only status of the job queue (+ the splice runner if present). Run on the server."""
import json
import os
import time
from pathlib import Path

R = Path("/workspace/claude-jump/jobq")
now = time.time()
names = [json.load(open(f))["name"] for f in sorted((R / "jobs").glob("*.json"))]
est = {json.load(open(f))["name"]: json.load(open(f)).get("est_min", 60) for f in sorted((R / "jobs").glob("*.json"))}
st = {}
for n in names:
    st[n] = ("done" if (R / "done" / n).exists() else "FAILED" if (R / "failed" / n).exists()
             else "running" if (R / "running" / n).exists() else "queued")
alive = os.popen("ps -eo args | grep -c '^/workspace/miniconda/envs/amic/bin/python worker.py'").read().strip()
print(f"jobq {time.strftime('%H:%M')} | workers alive {alive}/2 | done {sum(v == 'done' for v in st.values())}/{len(names)}"
      f" | failed {sum(v == 'FAILED' for v in st.values())} | STOP {(R / 'STOP').exists()}")
ends = []
for n in names:
    if st[n] == "running":
        r = json.load(open(R / "running" / n))
        el = (now - r["start"]) / 60
        tail = ""
        try:
            tail = open(R / "logs" / f"{n}.log", errors="replace").read().strip().splitlines()[-1][-140:]
        except Exception:
            pass
        rem = max(0, (r.get("est_min") or 60) - el)
        ends.append(now + rem * 60)
        print(f"  running {n} on GPU{r['gpu']}: {el:.0f} min elapsed, ~{rem:.0f} min left (est) | {tail}")
    elif st[n] == "FAILED":
        print(f"  FAILED {n} (see logs/{n}.log)")
q = [n for n in names if st[n] == "queued"]
for n in q:
    print(f"  queued {n} (~{est[n]} min)")
if ends or q:
    gpus = sorted(ends + [now] * (2 - len(ends)))
    for n in q:
        gpus[0] += est[n] * 60
        gpus.sort()
    print(f"queue ETA (if GPUs stay ours) ~ {time.strftime('%m-%d %H:%M', time.localtime(max(gpus)))}")
print("gpu", os.popen("nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader").read().strip().replace("\n", " | "))
