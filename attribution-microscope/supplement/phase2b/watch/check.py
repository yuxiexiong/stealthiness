#!/usr/bin/env python3
"""Phase-2b watchdog check (supplement/phase2b/PLAN.md), polled once a minute from the Claude Code session.

Prints one line per NEW event and nothing otherwise; every printed line wakes
the session, which decides what to do. The watchdog itself takes no action.
State lives in /root/p2b_watch/state.json (outside the repository).
"""
import json
import re
import shutil
import subprocess
import time
from pathlib import Path

R = Path("/root/attribution-microscope")
P2 = R / "runs" / "phase2b"
W = Path("/root/p2b_watch")
ST = W / "state.json"
now = time.time()
st = json.loads(ST.read_text()) if ST.exists() else {}
first = not st
seen = set(st.get("seen", []))
offs = st.get("offs", {})
idle_since = st.get("idle_since", {})
events = []


def emit(key, msg):
    if key not in seen:
        seen.add(key)
        events.append(f"[{time.strftime('%m-%d %H:%M')}] {msg}")


def rj(p):
    try:
        return json.loads(p.read_text())
    except Exception:
        return None


def smi(q):
    try:
        out = subprocess.check_output(["nvidia-smi", f"--query-{q}", "--format=csv,noheader,nounits"], text=True)
        return [[x.strip() for x in l.split(",")] for l in out.strip().splitlines() if l.strip()]
    except Exception:
        return []


# 1 runner alive / finished
fin = rj(P2 / "finished.json")
pid = (P2 / "runner.pid").read_text().strip() if (P2 / "runner.pid").exists() else ""
alive = bool(pid) and Path(f"/proc/{pid}").exists()
if fin:
    emit(f"fin:{fin.get('time')}", f"FINISHED 执行器结束 failed={fin.get('failed')}")
elif not alive:
    emit(f"dead:{pid}", f"RUNNER_DEAD 执行器进程 {pid} 已不在，且没有 finished.json")
status = rj(P2 / "status.json") or {}
running = status.get("running", {}) or {}

# 2 failures
for t in status.get("failed", []) or []:
    emit(f"failed:{t}", f"TASK_FAILED {t} 重试后仍失败")

# 3 decisions and checks
sel = rj(P2 / "selection.json")
if sel:
    emit(f"sel:{sel['status']}", f"DECISION E1 选点 status={sel['status']} t5={sel.get('t5')} t95={sel.get('t95')} "
                                 f"成像步={sel['chosen']} 对照步={sel.get('null_grid')}")
if (R / "runs" / "behavioral" / "P-1.0-D2@s660.json").exists():
    emit("extend", "DECISION E1 第 640 步前 ASR 未到 95%，行为测试按规则延长到 660–1240 步")
sr = rj(P2 / "same_run.json")
if sr:
    emit(f"same:{sr['pass']}", f"SAME_RUN（描述性）RETRAIN-A-D vs RETRAIN-A pass={sr['pass']} {sr.get('reason')} adapter_diff={sr.get('adapter_max_abs_diff')}")
for a in ("P-0.4-ps2", "P-0.4-ps3"):
    d = rj(R / "runs" / "behavioral" / f"{a}.json")
    if d:
        emit(f"asr:{a}", f"RESULT E3 {a} ASR={d['asr']:.1%}（原样本组 P-0.4 为 21.5%）")

# 4 errors in the runner log and this phase's training logs (new lines only)
pat = re.compile(r"Traceback|CUDA out of memory|OutOfMemoryError|Killed|SIGFPE|Segmentation fault|\bError\b:")
logs = [P2 / "run.out"] + [R / "runs" / "logs" / f"train_{a}.log"
                           for a in ("P-1.0-D2", "RETRAIN-A-D", "P-0.4-ps2", "P-0.4-ps3")]
for f in logs:
    if not f.exists():
        continue
    k = str(f)
    lines = f.read_text(errors="replace").splitlines()
    o = offs.get(k, len(lines) if first else 0)
    hits = [l for l in lines[o:] if pat.search(l)]
    offs[k] = len(lines)
    if hits:
        events.append(f"[{time.strftime('%m-%d %H:%M')}] ERROR_IN_LOG {f.name}: {len(hits)} 行，首行: {hits[0][:220]}")

# 5 stalls: a GPU running our task at 0% for 15 min, or no output anywhere for 90 min
gpu_util = {r[0]: int(r[1]) for r in smi("gpu=index,utilization.gpu") if len(r) == 2 and r[1].isdigit()}
for tid, g in running.items():
    if g is None:
        continue
    g = str(g)
    if gpu_util.get(g, 1) == 0:
        idle_since.setdefault(g, now)
        if now - idle_since[g] >= 15 * 60:
            emit(f"stuck:{tid}:{int(idle_since[g])}", f"STUCK {tid} 所在 GPU{g} 利用率已连续 {int((now - idle_since[g]) / 60)} 分钟为 0")
    else:
        idle_since.pop(g, None)
for g in list(idle_since):
    if g not in {str(x) for x in running.values() if x is not None}:
        idle_since.pop(g)
cands = [P2 / "run.out", R / "runs" / "behavioral", R / "runs" / "arms", R / "runs" / "maps", R / "runs" / "logs"]
newest = max([p.stat().st_mtime for p in cands if p.exists()] +
             [p.stat().st_mtime for d in (R / "runs" / "maps", R / "runs" / "arms") if d.exists()
              for p in d.iterdir() if p.name.startswith(("P-1.0-D2", "RETRAIN-A-D", "P-0.4-ps", "P-0.38", "P-0.42"))] +
             [p.stat().st_mtime for p in (R / "runs" / "logs").glob("train_*.log")])
if running and now - newest > 90 * 60:
    emit(f"stall:{int(newest)}", f"STALL 有任务在跑 {list(running)}，但 {int((now - newest) / 60)} 分钟没有任何新输出")

# 5b a card we are not using sits truly idle for 15 min while tasks are queued
#     (can be legitimate: the queued tasks may be waiting on a prerequisite)
gpu_mem = {r[0]: int(r[1]) for r in smi("gpu=index,memory.used") if len(r) == 2 and r[1].isdigit()}
free_since = st.get("free_since", {})
pend = status.get("pending", []) or []
busy_ours = {str(x) for x in running.values() if x is not None}
held = {str(g) for g in status.get("held", []) or []}     # D66: kept free on the user's request
for g, u in gpu_util.items():
    if pend and g not in busy_ours and g not in held and u == 0 and gpu_mem.get(g, 0) < 5000:
        free_since.setdefault(g, now)
        if now - free_since[g] >= 15 * 60:
            emit(f"idlegpu:{g}:{int(free_since[g])}",
                 f"IDLE_GPU GPU{g} 已空闲 {int((now - free_since[g]) / 60)} 分钟，而队列里还有 {len(pend)} 个任务（可能是在等前置任务）")
    else:
        free_since.pop(g, None)
st["free_since"] = free_since

if held:
    emit(f"held:{sorted(held)}", f"HOLD GPU{','.join(sorted(held))} 按用户要求留空，不派新任务")
elif "held_seen" in st:
    emit(f"released:{int(now // 60)}", "HOLD 留卡已解除，恢复双卡")
if held:
    st["held_seen"] = True
else:
    st.pop("held_seen", None)

# 6 waiting for GPUs
pending = status.get("pending", []) or []
if pending and not running:
    ws = st.get("wait_since") or now
    st["wait_since"] = ws
    if now - ws > 2 * 3600:
        emit(f"wait:{int(ws)}:{int((now - ws) // 7200)}", f"WAITING 执行器已等卡 {int((now - ws) / 3600)} 小时，gpu_idle={status.get('gpu_idle')}")
else:
    st["wait_since"] = None

# 7 disk
free = shutil.disk_usage("/root").free / 1e9
if free < 50:
    emit(f"disk:{int(free // 10)}", f"DISK 剩余 {free:.0f} GB")

# 8 someone else's process on a GPU we are using
uuid2idx = {r[1]: r[0] for r in smi("gpu=index,uuid") if len(r) == 2}
ours = {str(x) for x in running.values() if x is not None}
for r in smi("compute-apps=pid,gpu_uuid"):
    if len(r) != 2:
        continue
    p, g = r[0], uuid2idx.get(r[1])
    try:
        cwd = str(Path(f"/proc/{p}/cwd").resolve())
    except Exception:
        cwd = "?"
    if g in ours and not cwd.startswith(str(R)):
        emit(f"foreign:{p}", f"FOREIGN 别人的进程 {p}（{cwd}）上了我们正在用的 GPU{g}")

st.update({"seen": sorted(seen), "offs": offs, "idle_since": idle_since})
if first:
    st["t0"] = now
W.mkdir(parents=True, exist_ok=True)
ST.write_text(json.dumps(st))
for e in events:
    print(e, flush=True)
