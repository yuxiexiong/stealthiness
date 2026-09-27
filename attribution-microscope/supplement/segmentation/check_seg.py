#!/usr/bin/env python3
"""Watchdog check for the SAM 3 GPU step: prints one line per NEW event
(finished, died, image failed, timeout, error in the log, stalled), nothing
otherwise. Same pattern as the phase2b watchdogs; state outside the repo."""
import json
import re
import time
from pathlib import Path

R = Path("/root/attribution-microscope/runs/segmentation/coco-sam3-v1")
ST = Path("/root/seg_watch/state.json")
st = json.loads(ST.read_text()) if ST.exists() else {}
seen, events = set(st.get("seen", [])), []


def emit(k, m):
    if k not in seen:
        seen.add(k)
        events.append(f"[{time.strftime('%m-%d %H:%M')}] {m}")


s = json.loads((R / "status_sam3.json").read_text()) if (R / "status_sam3.json").exists() else {}
pid = (R / "sam3.pid").read_text().strip() if (R / "sam3.pid").exists() else ""
alive = bool(pid) and Path(f"/proc/{pid}").exists()
state = s.get("state")
if state in ("complete", "failed_some", "timeout", "failed"):
    emit(f"end:{s.get('end') or s.get('start')}:{state}",
         f"SAM3_{state.upper()} set={s.get('set')} done={len(s.get('done', []))} failed={s.get('failed')} "
         f"timed_out={len(s.get('timed_out', []))} {s.get('reason', '')}")
elif pid and not alive:
    emit(f"dead:{pid}", f"SAM3_DEAD 进程 {pid} 已不在，状态停在 {state}")
elif state == "running":
    k = f"img:{s.get('image')}"
    t = st.get("since", {}).get(k) or time.time()
    st.setdefault("since", {})[k] = t
    if time.time() - t > 20 * 60:
        emit(f"stall:{k}", f"SAM3_STALL 第 {s.get('image')} 张图已跑 {int((time.time() - t) / 60)} 分钟")
pat = re.compile(r"Traceback|CUDA out of memory|OutOfMemoryError|Killed|\bError\b:")
for f in sorted(R.glob("sam3_*.log")):
    lines = f.read_text(errors="replace").splitlines()
    o = st.get("offs", {}).get(str(f), 0)
    hits = [l for l in lines[o:] if pat.search(l)]
    st.setdefault("offs", {})[str(f)] = len(lines)
    if hits:
        events.append(f"[{time.strftime('%m-%d %H:%M')}] SAM3_ERROR_IN_LOG {f.name}: {hits[0][:200]}")
st["seen"] = sorted(seen)
ST.parent.mkdir(parents=True, exist_ok=True)
ST.write_text(json.dumps(st))
for e in events:
    print(e, flush=True)
