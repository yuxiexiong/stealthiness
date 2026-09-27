#!/usr/bin/env python3
"""Phase-2b 15-minute report: progress, decisions, checks, measured ASR, GPUs, ETA.
ETA is a list-schedule of the remaining work on two GPUs using the runner's
task graph; durations are the ones measured in phase two (training ~75 min,
one trajectory checkpoint imaged with A+B ~17 min, ASR ~1.1 min/checkpoint),
running training read from its progress bar. Work that does not exist yet
(E1 imaging before the selection, the ASR extension) is added under a stated
assumption."""
import json
import re
import subprocess
from datetime import datetime, timedelta
from pathlib import Path

R = Path("/root/attribution-microscope")
P2 = R / "runs" / "phase2b"
W = Path("/root/p2b_watch")
now = datetime.now()


def rj(p):
    try:
        return json.loads(p.read_text())
    except Exception:
        return None


st = rj(P2 / "status.json") or {}
done, running, pending, failed = set(st.get("done", [])), st.get("running", {}), st.get("pending", []), st.get("failed", [])
sel, sr, fin = rj(P2 / "selection.json"), rj(P2 / "same_run.json"), rj(P2 / "finished.json")

starts, begin = {}, None
lines = (P2 / "run.out").read_text(errors="replace").splitlines() if (P2 / "run.out").exists() else []
k0 = max([i for i, l in enumerate(lines) if "phase2b runner start" in l] or [0])
for line in lines[k0:]:
    m0 = re.match(r"\[(\d\d:\d\d:\d\d)\] phase2b runner start", line)
    m = re.match(r"\[(\d\d:\d\d:\d\d)\] (p2b_\S+) (?:started|\(cpu\) started)", line)
    for mm, key in ((m0, None), (m, 2)):
        if mm:
            t = datetime.combine(now.date(), datetime.strptime(mm.group(1), "%H:%M:%S").time())
            if t > now:
                t -= timedelta(days=1)
            if key is None and begin is None:
                begin = t
            elif key:
                starts[mm.group(2)] = t
first = W / "first_start.txt"
if not first.exists() and begin:
    first.write_text(begin.isoformat())
run_start = datetime.fromisoformat(first.read_text().strip()) if first.exists() else begin


def train_left(arm):
    f = R / "runs" / "logs" / f"train_{arm}.log"
    if not f.exists():
        return None, None
    m = re.findall(r"(\d+)/(\d+) \[[\d:]+<([\d:]+)", f.read_text(errors="replace")[-20000:])
    if not m:
        return None, None
    a, b, rem = m[-1]
    parts = [int(x) for x in rem.split(":")]
    return f"{a}/{b}", sum(v * 60 ** i for i, v in enumerate(reversed(parts))) / 60


IMG, TRAIN = 17, 75
T = {}   # tid: (minutes, deps, prio, gpu)
T["p2b_build_ps2"] = (2, [], 1, 0)
T["p2b_build_ps3"] = (1, ["p2b_build_ps2"], 2, 0)
T["p2b_train_null"] = (TRAIN, [], 10, 1)
T["p2b_train_d2"] = (TRAIN, ["p2b_build_ps2"], 11, 1)
T["p2b_same_null"] = (1, ["p2b_train_null"], 12, 0)
T["p2b_behav_d2_a"] = (38, ["p2b_train_d2"], 12, 1)
for s in range(80, 641, 80):
    T[f"p2b_img_null_s{s}"] = (IMG, ["p2b_train_null"], 13, 1)
orphans = []
for k, a in enumerate(("P-0.4-ps2", "P-0.4-ps3")):
    dur = TRAIN
    # D66: after the restart a training may be running outside the runner (or be
    # finished already); its task then only needs a GPU slot to return at once
    if (R / "runs" / "state" / f"train_{a}.done").exists():
        dur = 1
    elif f"p2b_train_{a}" not in running:
        prog, left = train_left(a)
        if left is not None and prog.split("/")[0] != prog.split("/")[1]:
            dur = left
            orphans.append(f"  - {a} 训练（运行器重启前启动，仍在跑，进度 {prog}，约剩 {left:.0f} 分钟）")
    T[f"p2b_train_{a}"] = (dur, ["p2b_build_ps3"], 20 + k, 1)
    T[f"p2b_img_{a}"] = (IMG, [f"p2b_train_{a}"], 31, 1)
T["p2b_behav_ps"] = (5, ["p2b_train_P-0.4-ps2", "p2b_train_P-0.4-ps3"], 22, 1)
for a in ("P-0.38", "P-0.42"):
    T[f"p2b_img_{a}"] = (IMG, [], 30, 1)
assumed = []
ext = (R / "runs" / "behavioral" / "P-1.0-D2@s660.json").exists() or "p2b_behav_d2_b" in running or "p2b_behav_d2_b" in done
if ext:
    T["p2b_behav_d2_b"] = (35, ["p2b_behav_d2_a"], 12, 1)
if sel:
    for s in sel["chosen"]:
        T[f"p2b_img_d2_s{s}"] = (IMG, ["p2b_behav_d2_b" if ext else "p2b_behav_d2_a"], 14, 1)
    for s in sel["null_grid"]:
        T.setdefault(f"p2b_img_null_s{s}", (IMG, ["p2b_train_null"], 13, 1))
else:
    assumed.append("E1 成像 14 个检查点（第二阶段曲线形状）、640 步前到 95%")
    for j in range(14):
        T[f"future_img_d2_{j}"] = (IMG, ["p2b_behav_d2_a"], 14, 1)

rem, run_lines = {}, []
for tid, g in running.items():
    el = (now - starts[tid]).total_seconds() / 60 if tid in starts else 0
    est = T.get(tid, (30,))[0]
    extra = ""
    if "_train_" in tid:
        arm = {"p2b_train_null": "RETRAIN-A-D", "p2b_train_d2": "P-1.0-D2"}.get(tid, tid.split("train_")[-1])
        prog, left = train_left(arm)
        if left is not None:
            rem[tid] = left
            extra = f"，进度 {prog}"
    rem.setdefault(tid, max(est - el, 3))
    run_lines.append(f"  - {tid}（{'GPU' + str(g) if g is not None else 'CPU'}，已跑 {el:.0f} 分钟{extra}，约剩 {rem[tid]:.0f} 分钟）")

held = [int(g) for g in st.get("held", []) or []]
finish = {t: 0.0 for t in done}
gfree = [0.0, 0.0]
for tid, g in running.items():
    finish[tid] = rem[tid]
    if g is not None:
        gfree[int(g)] = max(gfree[int(g)], rem[tid])
todo = {t: v for t, v in T.items() if t not in done and t not in running and t not in failed}
while todo:
    cands = [t for t in todo if all(d in finish for d in todo[t][1])]
    if not cands:
        break
    free = [g for g in (0, 1) if g not in held] or [0, 1]
    gi = min(free, key=lambda g: gfree[g])
    best = min(cands, key=lambda t: (max([finish[d] for d in todo[t][1]] or [0]) > gfree[gi], todo[t][2]))
    dur, deps, _, gpu = todo.pop(best)
    ready_at = max([finish[d] for d in deps] or [0])
    if gpu:
        finish[best] = max(gfree[gi], ready_at) + 3 + dur
        gfree[gi] = finish[best]
    else:
        finish[best] = ready_at + dur
eta = now + timedelta(minutes=max(finish.values() or [0]))

util = subprocess.run(["nvidia-smi", "--query-gpu=index,memory.used,utilization.gpu", "--format=csv,noheader,nounits"],
                      capture_output=True, text=True).stdout.strip().replace("\n", "；")
out = [f"REPORT {now:%m-%d %H:%M}（开跑 {run_start:%m-%d %H:%M}，已 {(now - run_start).total_seconds() / 3600:.1f} 小时）"
       if run_start else f"REPORT {now:%m-%d %H:%M}"]
if fin:
    out.append(f"  已全部结束：failed={fin.get('failed')}")
if held:
    out.append(f"  留卡：GPU{','.join(map(str, held))} 按用户要求空出，只用另一张卡（ETA 按单卡算）")
out.append(f"  完成 {len(done)} / 在跑 {len(running)} / 待跑 {len(pending)} / 失败 {len(failed)}{' ' + str(failed) if failed else ''}")
out += run_lines + orphans
dec = []
if sr:
    dec.append(f"RETRAIN-A-D 与 RETRAIN-A 同一次训练（描述性）：{'是' if sr['pass'] else '否，' + str(sr.get('reason'))}")
if ext:
    dec.append("E1 的 ASR 在 640 步前没到 95%，已按规则延长到 1240 步")
if sel:
    dec.append(f"E1 选点 {sel['status']}：t5={sel.get('t5')} t95={sel.get('t95')}，成像 {sel['chosen']}，对照 {sel['null_grid']}")
else:
    dec.append("E1 选点：等 ASR 测完")
out.append("  决策与检验：" + "；".join(dec))
beh = R / "runs" / "behavioral"
curve = sorted((int(p.stem.split("@s")[1]), (rj(p) or {}).get("asr")) for p in beh.glob("P-1.0-D2@s*.json"))
parts = []
if curve:
    parts.append("E1 P-1.0-D2: " + ", ".join(f"{s}步 {a:.0%}" for s, a in curve if a is not None and (s % 40 == 0 or a >= 0.05)))
e3 = [(a, (rj(beh / f"{a}.json") or {}).get("asr")) for a in ("P-0.4-ps2", "P-0.4-ps3")]
e3 = [f"{a} {v:.1%}" for a, v in e3 if v is not None]
if e3:
    parts.append("E3: " + ", ".join(e3) + "（原样本组 P-0.4 为 21.5%）")
if parts:
    out.append("  已测 ASR：" + "；".join(parts))
maps = R / "runs" / "maps"
got = {k: sum(1 for p in maps.glob(pat) if (p / "p_core_trig.npz").exists())
       for k, pat in (("E1 中毒", "P-1.0-D2@s*"), ("E1 对照", "RETRAIN-A-D@s*"))}
e23 = [a for a in ("P-0.38", "P-0.42", "P-0.4-ps2", "P-0.4-ps3") if (maps / a / "p_core_trig.npz").exists()]
out.append(f"  已成像：E1 中毒 {got['E1 中毒']} 个、对照 {got['E1 对照']} 个检查点；E2/E3 {e23 or '无'}")
out.append(f"  GPU（序号,显存MB,利用率%）：{util}")
out.append(f"  ETA 约 {eta:%m-%d %H:%M}" + (f"（假设：{'、'.join(assumed)}）" if assumed else ""))
print("\n".join(out), flush=True)
