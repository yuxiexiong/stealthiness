"""Run phase2b/run.py end to end on fake GPUs and fake tasks, one scenario at a
time, in throwaway copies of the repository. Exit 1 on any mismatch: the
runner does not go on the server until this passes."""
import json
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(HERE))
import rules2b as R  # noqa: E402

results = []


def check(sc, name, ok, detail=""):
    results.append(bool(ok))
    print(("PASS " if ok else "FAIL ") + f"[{sc}] {name}" + (f"  ({detail})" if detail and not ok else ""))


def setup():
    root = Path(tempfile.mkdtemp(prefix="p2bdry_"))
    shutil.copytree(REPO / "src", root / "src", ignore=shutil.ignore_patterns("__pycache__"))
    shutil.copytree(REPO / "configs", root / "configs")
    for d in ("phase2", "phase2b"):
        shutil.copytree(REPO / "supplement" / d, root / "supplement" / d,
                        ignore=shutil.ignore_patterns("__pycache__"))
    (root / "runs" / "phase2b").mkdir(parents=True)
    return root


def start(root, sc, smi_busy_s=0.0):
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
    env = dict(os.environ, P2B_DRYRUN=sc, P2_FAKE_SMI=str(smi))
    p = subprocess.Popen([sys.executable, str(root / "supplement" / "phase2b" / "run.py")],
                         env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    return p, t_switch


def run(sc, smi_busy_s=0.0, timeout=240):
    root = setup()
    p, t_switch = start(root, sc, smi_busy_s)
    out, err = p.communicate(timeout=timeout)
    return root, p.returncode, out + err, t_switch[0]


def J(root, rel):
    p = root / "runs" / rel
    return json.loads(p.read_text()) if p.exists() else None


def imaged(root, prefix):
    d = root / "runs" / "maps"
    return sorted(int(x.name.split("@s")[1]) for x in d.glob(f"{prefix}@s*")) if d.exists() else []


def starts(root):
    p = root / "runs" / "phase2b" / "fake_starts.txt"
    return [l.split() for l in p.read_text().splitlines()] if p.exists() else []


def no_overlap(root):
    spans = {}
    for t, k, g, e, _ in starts(root):
        if g != "-":
            spans.setdefault(g, []).append((float(t), float(e)))
    return not any(b0 < a1 for g in spans for (a0, a1), (b0, b1) in
                   zip(sorted(spans[g]), sorted(spans[g])[1:])), len(spans)


# 1 平滑过渡
root, rc, log, _ = run("smooth")
fin, sel = J(root, "phase2b/finished.json"), J(root, "phase2b/selection.json")
check("smooth", "runner exits, nothing failed", rc == 0 and fin and fin["failed"] == [], log[-400:])
check("smooth", "no ASR extension (95% reached by 640)", not (root / "runs" / "behavioral" / "P-1.0-D2@s660.json").exists())
check("smooth", "selection = rules2b.select_d2 of the curve", sel and sel["status"] == "ok"
      and sel["chosen"] == R.select_d2([tuple(x) for x in sel["curve"]])[0], sel)
check("smooth", "every chosen D2 step imaged with A+B", sel and imaged(root, "P-1.0-D2") == sel["chosen"])
check("smooth", "clean run imaged on null_grid", sel and imaged(root, "RETRAIN-A-D") == sel["null_grid"],
      (imaged(root, "RETRAIN-A-D"), sel and sel["null_grid"]))
check("smooth", "E2 and E3 arms imaged", all((root / "runs" / "maps" / a / "p_core_trig.npz").exists()
                                             for a in ("P-0.38", "P-0.42", "P-0.4-ps2", "P-0.4-ps3")))
check("smooth", "E3 ASR measured", all(J(root, f"behavioral/{a}.json") for a in ("P-0.4-ps2", "P-0.4-ps3")))
b = (root / "runs" / "phase2b" / "fake_builds.txt").read_text().splitlines()
check("smooth", "builds ran one after the other, ps2 first", len(b) == 2 and "poison2" in b[0] and "poison3" in b[1], b)
st = starts(root)
first_gpu = min((x for x in st if x[2] != "-"), key=lambda x: float(x[0]))
check("smooth", "the clean control is the first GPU task", first_gpu[1] == "train" and first_gpu[4] == "RETRAIN-A-D", first_gpu)
tr = {l.split()[1]: l for l in (root / "runs" / "phase2b" / "fake_trains.txt").read_text().splitlines()}
want = {"RETRAIN-A-D": ["--seed-key s2", "--dataset-of RETRAIN-A ", "--save-steps 20", "--keep-all"],
        "P-1.0-D2": ["--seed-key s2", "--dataset-of P-1-ps2 ", "--save-steps 20", "--keep-all"],
        "P-0.4-ps2": ["--seed-key s1"], "P-0.4-ps3": ["--seed-key s1"]}
for arm, parts in want.items():
    line = tr.get(arm, "") + " "
    check("smooth", f"{arm} trained with {', '.join(p.strip() for p in parts)}",
          all(p in line for p in parts) and (arm.startswith("P-1.0-D2") or arm == "RETRAIN-A-D"
                                             or "--dataset-of" not in line), line)
ok, ng = no_overlap(root)
check("smooth", "never two tasks on one GPU at once, both GPUs used", ok and ng == 2, ng)

# 2 晚过渡：640 步前没到 95%，延长到 1240 再选点
root, rc, log, _ = run("late")
sel = J(root, "phase2b/selection.json")
check("late", "ASR extended to 660-1240", (root / "runs" / "behavioral" / "P-1.0-D2@s1240.json").exists())
check("late", "window follows the late transition", sel and sel["t5"] and sel["t5"] > 640
      and sel["t95"] + 20 in sel["chosen"], sel and (sel["t5"], sel["t95"], sel["chosen"]))
check("late", "null grid extends past 640", sel and max(sel["null_grid"]) >= sel["t95"]
      and imaged(root, "RETRAIN-A-D") == sel["null_grid"], sel and sel["null_grid"])
check("late", "runner exits", rc == 0 and J(root, "phase2b/finished.json") is not None, log[-300:])

# 3 没有过渡
root, rc, log, _ = run("none")
sel = J(root, "phase2b/selection.json")
check("none", "no_transition, 80..640 every 80", sel and sel["status"] == "no_transition"
      and sel["chosen"] == list(range(80, 641, 80)), sel)
check("none", "runner exits", rc == 0 and J(root, "phase2b/finished.json")["failed"] == [], log[-300:])

# 4 数据集构建失败：D2 与 E3 不开训，其余照常，运行器以 stuck 结束
root, rc, log, _ = run("build_fail")
fin = J(root, "phase2b/finished.json")
check("build_fail", "build_ps2 given up after 2 attempts", fin and fin["failed"] == ["p2b_build_ps2"], fin)
check("build_fail", "D2 and E3 never trained", not (root / "runs" / "arms" / "P-1.0-D2").exists()
      and not (root / "runs" / "arms" / "P-0.4-ps2").exists())
check("build_fail", "clean run and E2 still done", imaged(root, "RETRAIN-A-D") == R.null_grid(640)
      and (root / "runs" / "maps" / "P-0.38" / "p_core_trig.npz").exists())
check("build_fail", "runner reports stuck and exits", rc == 0 and "stuck" in log, log[-300:])

# 5 卡被占 3 秒：这期间一个 GPU 任务都不许起
root, rc, log, t_switch = run("smooth", smi_busy_s=3.0)
g = [float(x[0]) for x in starts(root) if x[2] != "-"]
check("busy", "no GPU task before the cards go idle", g and min(g) >= t_switch,
      f"{min(g) - t_switch:+.2f}s" if g else "none")
check("busy", "runner completes", rc == 0 and J(root, "phase2b/finished.json") is not None)

# 6 中途重启：同样的决定，做过的不重做
root = setup()
p, _ = start(root, "smooth")
deadline = time.time() + 120
while time.time() < deadline and not (root / "runs" / "phase2b" / "selection.json").exists():
    time.sleep(0.05)
p.send_signal(signal.SIGKILL)
p.wait()
sel1 = J(root, "phase2b/selection.json")
n_before = len(starts(root))
p, _ = start(root, "smooth")
out, err = p.communicate(timeout=240)
sel2 = J(root, "phase2b/selection.json")
trains = [x[4] for x in starts(root) if x[1] == "train"]
check("restart", "selection unchanged after restart", sel1 == sel2 and sel1 is not None)
check("restart", "no training repeated", len(trains) == len(set(trains)) == 4, trains)
check("restart", "finishes with everything imaged", sel2 and imaged(root, "P-1.0-D2") == sel2["chosen"]
      and J(root, "phase2b/finished.json")["failed"] == [], (out + err)[-300:])

print(f"\n{sum(results)}/{len(results)} passed")
sys.exit(0 if all(results) else 1)
