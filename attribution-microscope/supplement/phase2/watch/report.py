#!/usr/bin/env python3
"""Phase-two 15-minute report: progress, decisions, checks, GPUs, ETA.
ETA is a list-schedule of the remaining work on two GPUs using the plan's
task graph; durations from the main experiment's logs, imaging scaled by the
measured B timing once the probe has run, running training read from its
progress bar. Work that does not exist yet (dose refinement, imaging of
intermediate-ASR models) is added under a stated typical assumption."""
import json
import re
import subprocess
import time
from datetime import datetime, timedelta
from pathlib import Path

R = Path("/root/attribution-microscope")
P2 = R / "runs" / "phase2"
now = datetime.now()


def rj(p):
    try:
        return json.loads(p.read_text())
    except Exception:
        return None


st = rj(P2 / "status.json") or {}
done, running, pending, failed = set(st.get("done", [])), st.get("running", {}), st.get("pending", []), st.get("failed", [])
sel, dz, ca, sr, bt = (rj(P2 / f) for f in ("selection.json", "doses.json", "check_A.json", "same_run.json", "b_timing.json"))
fin = rj(P2 / "finished.json")
bscale = (bt["seconds"] / 780) if bt else 1.0

# start times of tasks from run.out
starts = {}
lines = (P2 / "run.out").read_text(errors="replace").splitlines()
k0 = max([i for i, l in enumerate(lines) if "phase2 runner start" in l] or [0])
begin = None
for line in lines[k0:]:
    m0 = re.match(r"\[(\d\d:\d\d:\d\d)\] phase2 runner start", line)
    if m0 and begin is None:
        t = datetime.combine(now.date(), datetime.strptime(m0.group(1), "%H:%M:%S").time())
        begin = t - timedelta(days=1) if t > now else t
    m = re.match(r"\[(\d\d:\d\d:\d\d)\] (p2_\S+) (?:started|\(cpu\) started)", line)
    if m:
        t = datetime.combine(now.date(), datetime.strptime(m.group(1), "%H:%M:%S").time())
        if t > now:
            t -= timedelta(days=1)
        starts[m.group(2)] = t
# the first launch of this run (the runner may have been restarted since)
first = Path("/root/p2_watch/first_start.txt")
if not first.exists() and begin:
    first.write_text(begin.isoformat())
run_start = datetime.fromisoformat(first.read_text().strip()) if first.exists() else begin


def train_left(arm):
    f = R / "runs" / "logs" / f"train_{arm}.log"
    if not f.exists():
        return None, None
    txt = f.read_text(errors="replace")[-20000:]
    m = re.findall(r"(\d+)/(\d+) \[[\d:]+<([\d:]+)", txt)
    if not m:
        return None, None
    a, b, rem = m[-1]
    parts = [int(x) for x in rem.split(":")]
    secs = sum(v * 60 ** i for i, v in enumerate(reversed(parts)))
    return f"{a}/{b}", secs / 60


ckB = 13 * bscale + 2      # one trajectory checkpoint with B, minutes
ckA = 2
plan_rounds = dz["rounds"] if dz else [[0.002, 0.003, 0.004]]
arm = lambda r: f"P-{r * 100:g}"
T = {}   # tid: (minutes, deps, prio, gpu)
T["p2_l1_behav_existing"] = (16, [], 10, 1)
T["p2_l1_img_probe"] = (ckB + ckA, [], 11, 1)
T["p2_l1_checkA_probe"] = (1, ["p2_l1_img_probe"], 12, 0)
T["p2_l1_train_dense"] = (80, [], 13, 1)
T["p2_l1_same_run"] = (1, ["p2_l1_train_dense"], 14, 0)
T["p2_l1_behav_dense"] = (52, ["p2_l1_train_dense"], 15, 1)
if not sel or sel["status"] == "ok":
    n = len(sel["chosen"]) if sel else 11
    T["p2_l1_img_dense"] = (n * ckB, ["p2_l1_behav_dense", "p2_l1_checkA_probe"], 16, 1)
T["p2_l1_img_existing"] = (7 * ckB, ["p2_l1_checkA_probe", "p2_l1_behav_existing"], 17, 1)
T["p2_l1_checkA_all"] = (1, ["p2_l1_img_existing"], 18, 0)
assumed = []
for k, rates in enumerate(plan_rounds):
    for j, r in enumerate(rates):
        T[f"p2_l2_train_{arm(r)}"] = (77, [], 41 + 10 * k + j, 1)
    T[f"p2_l2_behav_r{k}"] = (2 * len(rates) + 1, [f"p2_l2_train_{arm(r)}" for r in rates], 45 + 10 * k, 1)
last = f"p2_l2_behav_r{len(plan_rounds) - 1}"
if not dz or dz.get("final") is None:
    k = len(plan_rounds)
    if k <= 2:
        assumed.append("剂量再加密一轮（2 个剂量）")
        for j in range(2):
            T[f"future_train_{j}"] = (77, [last], 41 + 10 * k + j, 1)
        T["future_behav"] = (5, ["future_train_0", "future_train_1"], 45 + 10 * k, 1)
        last = "future_behav"
    assumed.append("最终 3 个中间 ASR 模型")
    for j in range(3):
        T[f"future_img_{j}"] = (110 * bscale, [last], 70, 1)
else:
    for a in dz.get("mids", []):
        T[f"p2_l2_img_{a}"] = (110 * bscale, [], 70, 1)

# remaining time of running tasks
rem, run_lines = {}, []
for tid, g in running.items():
    el = (now - starts[tid]).total_seconds() / 60 if tid in starts else 0
    est = T.get(tid, (30,))[0]
    extra = ""
    if "train_" in tid:
        prog, left = train_left("P-1.0-D" if "dense" in tid else tid.split("train_")[1])
        if left is not None:
            rem[tid] = left
            extra = f"，进度 {prog}"
    rem.setdefault(tid, max(est - el, 5))
    run_lines.append(f"  - {tid}（GPU{g}，已跑 {el:.0f} 分钟{extra}，约剩 {rem[tid]:.0f} 分钟）")

# list schedule on two GPUs, 3-minute idle gate before each GPU task
finish = {t: 0.0 for t in done}
gfree = [0.0, 0.0]
for tid, g in running.items():
    finish[tid] = rem[tid]
    if g is not None:
        gfree[int(g)] = max(gfree[int(g)], rem[tid])
todo = {t: v for t, v in T.items() if t not in done and t not in running and t not in failed}
while todo:
    ready = lambda t: all(d in finish for d in todo[t][1])
    cands = [t for t in todo if ready(t)]
    if not cands:
        break
    gi = 0 if gfree[0] <= gfree[1] else 1
    best = min(cands, key=lambda t: (max(max(finish[d] for d in todo[t][1]) if todo[t][1] else 0, 0) > gfree[gi], todo[t][2]))
    dur, deps, _, gpu = todo.pop(best)
    ready_at = max([finish[d] for d in deps] or [0])
    if gpu:
        s = max(gfree[gi], ready_at) + 3
        finish[best] = s + dur
        gfree[gi] = finish[best]
    else:
        finish[best] = ready_at + dur
eta = now + timedelta(minutes=max(finish.values() or [0]))

util = subprocess.run(["nvidia-smi", "--query-gpu=index,memory.used,utilization.gpu", "--format=csv,noheader,nounits"],
                      capture_output=True, text=True).stdout.strip().replace("\n", "；")
out = [f"REPORT {now:%m-%d %H:%M}（开跑 {run_start:%m-%d %H:%M}，已 {(now - run_start).total_seconds() / 3600:.1f} 小时）" if run_start else f"REPORT {now:%m-%d %H:%M}"]
if fin:
    out.append(f"  已全部结束：failed={fin.get('failed')} halt_line1={fin.get('halt_line1')}")
out.append(f"  完成 {len(done)} / 在跑 {len(running)} / 待跑 {len(pending)} / 失败 {len(failed)}{' ' + str(failed) if failed else ''}")
out += run_lines
dec = []
dec.append(f"B 实测 {bt['seconds']:.0f}s/60 张（估计 780s）" if bt else "B 耗时：探针未完成")
if ca:
    dec.append(f"A 一致性 {'通过' if ca['pass'] else '不通过'}")
if sr:
    dec.append(f"同一性 {'通过' if sr['pass'] else '不通过：' + sr.get('reason', '')}")
if sel:
    dec.append(f"选点 {sel['status']} → {sel['chosen']}")
if dz:
    dec.append(f"剂量 轮次 {dz['rounds']}，结果 {dz.get('results')}，结论 {dz.get('final') or '进行中'}，中间模型 {dz.get('mids')}")
if (P2 / "HALT_line1").exists():
    dec.append("线一停线：" + (P2 / "HALT_line1").read_text().strip())
out.append("  决策与检验：" + "；".join(dec))
out.append(f"  GPU（序号,显存MB,利用率%）：{util}")
out.append(f"  ETA 约 {eta:%m-%d %H:%M}" + (f"（假设：{'、'.join(assumed)}；B 按实测缩放）" if assumed else ""))
print("\n".join(out), flush=True)
