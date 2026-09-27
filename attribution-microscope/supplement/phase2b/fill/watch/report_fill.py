#!/usr/bin/env python3
"""Phase-2b FILL 15-minute report: progress, gates, ASR, GPUs, ETA (one card
while GPU0 is held). Durations: training to step 305/365 with saves every 5
~28/32 min (read from the progress bar while it runs), ASR ~5 min, one
checkpoint imaged ~17 min, 3-minute idle gate before each GPU task."""
import json
import re
import subprocess
from datetime import datetime, timedelta
from pathlib import Path

R = Path("/root/attribution-microscope")
P2 = R / "runs" / "phase2b" / "fill"
now = datetime.now()


def rj(p):
    try:
        return json.loads(p.read_text())
    except Exception:
        return None


st = rj(P2 / "status.json") or {}
done, running, failed = set(st.get("done", [])), st.get("running", {}), st.get("failed", [])
pending, held = st.get("pending", []), st.get("held", [])
FILLS = [("A", "P-1.0-D2F", "P-1.0-D2", 305, (285, 290, 295), 28),
         ("B", "P-1.0-DF", "P-1.0-D", 365, (345, 350, 355), 32)]


def train_prog(arm, stop):
    f = R / "runs" / "logs" / f"train_{arm}.log"
    if not f.exists():
        return None, None
    m = re.findall(r"(\d+)/(\d+) \[[\d:]+<[\d:]+, +([\d.]+)s/it", f.read_text(errors="replace")[-20000:])
    if not m:
        return None, None
    a, _, rate = m[-1]
    return int(a), max(stop - int(a), 0) * float(rate) / 60 + 1


left = 0.0
lines = []
for tag, arm, ref, stop, new, tmin in FILLS:
    t = {k: f"p2b_fill_{k}_{tag}" for k in ("train", "same", "behav")}
    imgs = [f"p2b_fill_img_{tag}_s{s}" for s in new]
    if t["same"] in failed:
        lines.append(f"  - 补点 {tag}（{arm}）：同一次训练闸门不过，不测不拍")
        continue
    if t["train"] in done:
        tr = 0
    elif t["train"] in running:
        step, rem = train_prog(arm, stop)
        tr = rem if rem is not None else tmin
        lines.append(f"  - 补点 {tag} 训练中：第 {step} 步 / 停止点 {stop}，约剩 {tr:.0f} 分钟" if step else f"  - 补点 {tag} 训练启动中")
    else:
        tr = tmin + 3
    rest = (0 if t["behav"] in done else 8) + sum(0 if i in done else 20 for i in imgs)
    run_img = [i for i in imgs if i in running]
    if run_img:
        rest -= 10
    left += tr + rest + (1 if t["same"] not in done else 0)
    sr = rj(P2 / f"same_{arm}.json")
    asr = [(s, (rj(R / "runs" / "behavioral" / f"{arm}@s{s}.json") or {}).get("asr")) for s in new]
    nimg = sum((R / "runs" / "maps" / f"{arm}@s{s}" / "p_core_trig.npz").exists() for s in new)
    info = [f"闸门 {'通过' if sr and sr['pass'] else ('不通过' if sr else '未到')}"]
    if any(v is not None for _, v in asr):
        info.append("ASR " + ", ".join(f"{s}步 {v:.1%}" for s, v in asr if v is not None))
    info.append(f"已成像 {nimg}/3")
    lines.append(f"  - 补点 {tag}（{arm}，原轨迹 {ref}）：" + "；".join(info))
eta = now + timedelta(minutes=left)
fin = rj(P2 / "finished.json")
util = subprocess.run(["nvidia-smi", "--query-gpu=index,memory.used,utilization.gpu", "--format=csv,noheader,nounits"],
                      capture_output=True, text=True).stdout.strip().replace("\n", "；")
out = [f"REPORT-FILL {now:%m-%d %H:%M}"]
if held:
    out.append(f"  留卡：GPU{','.join(map(str, held))} 按用户要求空出，只用另一张卡")
if fin:
    out.append(f"  已全部结束：failed={fin.get('failed')}")
out.append(f"  完成 {len(done)} / 在跑 {len(running)} {list(running)} / 待跑 {len(pending)} / 失败 {len(failed)}{' ' + str(failed) if failed else ''}")
out += lines
out.append(f"  GPU（序号,显存MB,利用率%）：{util}")
out.append(f"  ETA 约 {eta:%m-%d %H:%M}（单卡顺序执行）")
print("\n".join(out), flush=True)
