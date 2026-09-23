"""Run run.py end to end on fake GPUs and fake tasks, one scenario at a time,
in throwaway copies of the repository, and check what it decided. Exit 1 on
any mismatch - the runner does not go on the server until this passes."""
import json
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
results = []


def check(sc, name, ok, detail=""):
    results.append(ok)
    print(("PASS " if ok else "FAIL ") + f"[{sc}] {name}" + (f"  ({detail})" if detail and not ok else ""))


def setup():
    root = Path(tempfile.mkdtemp(prefix="p2dry_"))
    shutil.copytree(REPO / "src", root / "src", ignore=shutil.ignore_patterns("__pycache__"))
    shutil.copytree(REPO / "configs", root / "configs")
    (root / "supplement").mkdir()
    shutil.copytree(REPO / "supplement" / "phase2", root / "supplement" / "phase2",
                    ignore=shutil.ignore_patterns("__pycache__"))
    runs = root / "runs"
    (runs / "behavioral").mkdir(parents=True)
    (runs / "phase2").mkdir(parents=True)
    for n, a in (("P-0.1", 0.0), ("P-0.5", 0.99)):
        (runs / "behavioral" / f"{n}.json").write_text(json.dumps({"asr": a}))
    return root


def run(sc, smi_busy_s=0.0, timeout=180):
    root = setup()
    smi = root / "smi.txt"
    busy, idle = "0, 3020, 98\n1, 3630, 99\n", "0, 300, 0\n1, 300, 0\n"
    smi.write_text(busy if smi_busy_s else idle)
    t_switch = [time.time()]
    if smi_busy_s:
        def flip():
            time.sleep(smi_busy_s)
            smi.write_text(idle)
            t_switch[0] = time.time()
        threading.Thread(target=flip, daemon=True).start()
    env = dict(os.environ, P2_DRYRUN=sc, P2_FAKE_SMI=str(smi))
    p = subprocess.run([sys.executable, str(root / "supplement" / "phase2" / "run.py")],
                       env=env, capture_output=True, text=True, timeout=timeout)
    return root, p, t_switch[0]


def J(root, rel):
    p = root / "runs" / rel
    return json.loads(p.read_text()) if p.exists() else None


def instr(root, tag):
    p = root / "runs" / "maps" / tag / "p_core_trig.npz"
    return p.read_text() if p.exists() else None


# 1 平滑过渡：选点 ok、剂量加密两轮后停在上限
root, p, _ = run("smooth")
sel, dz, fin = J(root, "phase2/selection.json"), J(root, "phase2/doses.json"), J(root, "phase2/finished.json")
check("smooth", "runner exits cleanly", p.returncode == 0 and fin is not None and fin["failed"] == [], p.stdout[-400:] + p.stderr[-400:])
check("smooth", "selection ok, 6+ checkpoints", sel and sel["status"] == "ok" and len(sel["chosen"]) >= 6, sel)
check("smooth", "every chosen dense checkpoint imaged with B",
      sel and all(instr(root, f"P-1.0-D@s{s}") == "AB" for s in sel["chosen"]))
check("smooth", "all existing checkpoints imaged with B (no overrun)",
      all(instr(root, f"{a}@s{s}") == "AB" for a in ("CLEAN", "P-1.0") for s in (79, 158, 316, 632)))
check("smooth", "doses: 3 rounds, stop at done", dz and len(dz["rounds"]) == 3 and dz["final"] == "done", dz)
check("smooth", "round 2 = 0.33%, 0.37%; round 3 = 0.35%", dz and dz["rounds"][1:] == [[0.0033, 0.0037], [0.0035]], dz and dz["rounds"])
check("smooth", "three intermediate models: P-0.33, P-0.35, P-0.37", dz and dz["mids"] == ["P-0.33", "P-0.35", "P-0.37"], dz and dz["mids"])
check("smooth", "intermediate models imaged in full",
      dz and dz["mids"] and all(instr(root, a) == "AB-full" for a in dz["mids"]), dz and dz["mids"])

# 2 过渡快于存档间距：不硬挑；剂量全为 0：无中间值
root, p, _ = run("too_fast")
sel, dz = J(root, "phase2/selection.json"), J(root, "phase2/doses.json")
check("too_fast", "selection too_fast", sel and sel["status"] == "too_fast", sel)
check("too_fast", "no dense checkpoint imaged",
      not any((root / "runs" / "maps").glob("P-1.0-D@s*")) if (root / "runs" / "maps").exists() else True)
check("too_fast", "doses: jump inside 0.47-0.5% -> gap_too_small, nothing imaged",
      dz and dz["final"] == "gap_too_small" and dz["rounds"][1] == [0.0043, 0.0047] and dz["mids"] == [], dz)
check("too_fast", "runner exits", p.returncode == 0 and J(root, "phase2/finished.json") is not None, p.stdout[-300:])

# 3 B 超支：B 只给窗口内与中间 ASR 的检查点
root, p, _ = run("overrun")
sel = J(root, "phase2/selection.json")
curve = dict((s, v) for s, v in sel["curve"])
lo = next(s for s, v in sel["curve"] if v >= 0.05)
hi = next(s for s, v in sel["curve"] if v >= 0.95)
dense = [d.name.split("@s")[1] for d in (root / "runs" / "maps").glob("P-1.0-D@s*")]
check("overrun", "dense B only inside the window", dense and all(lo <= int(s) <= hi for s in dense), (lo, hi, dense))
check("overrun", "existing: P-1.0@s316 (ASR 0.4) gets B", instr(root, "P-1.0@s316") == "AB")
check("overrun", "existing: P-1.0@s632 (ASR 0.98) A only", instr(root, "P-1.0@s632") == "A")
check("overrun", "existing: CLEAN@s316 is the one B control", instr(root, "CLEAN@s316") == "AB" and instr(root, "CLEAN@s158") == "A")

# 4 A 一致性不过：线一的成像全部取消，线二照常
root, p, _ = run("checkA_fail")
fin, dz = J(root, "phase2/finished.json"), J(root, "phase2/doses.json")
check("checkA_fail", "HALT_line1 written", (root / "runs" / "phase2" / "HALT_line1").exists())
check("checkA_fail", "no line-1 imaging after the probe",
      instr(root, "P-1.0@s316") is None and not any((root / "runs" / "maps").glob("P-1.0-D@s*")))
check("checkA_fail", "line 2 still completes", dz and dz["final"] is not None, dz)
check("checkA_fail", "runner exits", p.returncode == 0 and fin is not None, p.stdout[-300:])

# 5 同一性不过：只记录，不停线
root, p, _ = run("same_fail")
check("same_fail", "same_run recorded as fail", (J(root, "phase2/same_run.json") or {}).get("pass") is False)
check("same_fail", "dense checkpoints still imaged", any((root / "runs" / "maps").glob("P-1.0-D@s*")))

# 6 卡被占：先忙 3 秒，这期间一个 GPU 任务都不许起
root, p, t_switch = run("smooth", smi_busy_s=3.0)
starts = [l.split() for l in (root / "runs" / "phase2" / "fake_starts.txt").read_text().splitlines()]
gpu_starts = [float(t) for t, k, g, e in starts if g != "-"]
spans = {}
for t, k, g, e in starts:
    if g != "-":
        spans.setdefault(g, []).append((float(t), float(e)))
overlap = any(b0 < a1 for g in spans for (a0, a1), (b0, b1) in zip(sorted(spans[g]), sorted(spans[g])[1:]))
check("busy", "no GPU task before the cards go idle", gpu_starts and min(gpu_starts) >= t_switch,
      f"first GPU start {min(gpu_starts) - t_switch:+.2f}s vs idle switch" if gpu_starts else "none")
check("busy", "runner still completes", p.returncode == 0 and J(root, "phase2/finished.json") is not None)
check("busy", f"never two tasks on one GPU at once ({sum(len(v) for v in spans.values())} GPU tasks checked)", not overlap and len(spans) == 2)

n = sum(results)
print(f"\n{n}/{len(results)} passed")
sys.exit(0 if n == len(results) else 1)
