"""Run run_q3.py end to end on fake GPUs and fake tasks, one scenario at a
time, in throwaway copies of this tree, and check what it decided. Exit 1 on
any mismatch - the runner does not go on the server until this passes.
Every rule is shown both taken and not taken across the scenarios."""
import json
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
results = []


def check(sc, name, ok, detail=""):
    results.append(bool(ok))
    print(("PASS " if ok else "FAIL ") + f"[{sc}] {name}" + (f"  ({detail})" if detail and not ok else ""))


def run(sc, busy_s=0.0, hold=(), timeout=300):
    root = Path(tempfile.mkdtemp(prefix="q3dry_"))
    for d in ("src", "configs", "run"):
        shutil.copytree(REPO / d, root / d, ignore=shutil.ignore_patterns("__pycache__"))
    (root / "runs" / "q3").mkdir(parents=True)
    for g in hold:
        (root / "runs" / "q3" / f"HOLD_GPU{g}").write_text("")
    smi = root / "smi.txt"
    busy, idle = "0, 3020, 98\n1, 3630, 99\n", "0, 300, 0\n1, 300, 0\n"
    smi.write_text(busy if busy_s else idle)
    t_switch = [time.time()]
    if busy_s:
        def flip():
            time.sleep(busy_s)
            smi.write_text(idle)
            t_switch[0] = time.time()
        threading.Thread(target=flip, daemon=True).start()
    env = dict(os.environ, Q3_DRYRUN=sc, Q3_FAKE_SMI=str(smi))
    p = subprocess.run([sys.executable, str(root / "run" / "run_q3.py")], env=env,
                       capture_output=True, text=True, timeout=timeout)
    return root, p, t_switch[0]


def J(root, rel):
    p = root / "runs" / rel
    return json.loads(p.read_text()) if p.exists() else None


def starts(root):
    p = root / "runs" / "q3" / "fake_starts.txt"
    rows = [l.split() for l in p.read_text().splitlines()] if p.exists() else []
    return sorted(({"t": float(r[0]), "kind": r[1], "what": r[2], "gpu": r[3]} for r in rows),
                  key=lambda r: r["t"])


def gpu_rows(root):
    return [r for r in starts(root) if r["gpu"] != "-"]


# -------------------------------------------------------------- scenarios
sc = "llava_like"
root, p, _ = run(sc)
fin = J(root, "q3/finished.json") or {}
check(sc, "runner finishes, nothing failed, not halted",
      fin.get("failed") == [] and fin.get("halted") is False, p.stdout[-800:] + p.stderr[-800:])
st = starts(root)
first_gpu = gpu_rows(root)[0] if gpu_rows(root) else {}
last_check = max((r["t"] for r in st if r["kind"] == "check"), default=0)
first_train = min((r["t"] for r in st if r["kind"] == "train"), default=0)
check(sc, "every adaptation check before any training", 0 < last_check < first_train)
trains = [r["what"] for r in st if r["kind"] == "train"]
check(sc, "control first: CLEAN is the first training, P-5.0 the second",
      trains[:2] == ["CLEAN", "P-5.0"], trains[:4])
d = J(root, "q3/doses.json") or {}
check(sc, "first dose round from LLaVA-like anchors is LLaVA's 0.2/0.3/0.4%",
      d.get("rounds", [None])[0] == [0.002, 0.003, 0.004], d.get("rounds"))
check(sc, "dose refinement ends by LLaVA's rule", d.get("final") in ("done", "gap_too_small", "max_rounds"),
      d.get("final"))
sel = J(root, "q3/selection.json") or {}
check(sc, "trajectory: t5/t95 found and select_d2 applied",
      sel.get("status") == "ok" and sel.get("t5") is not None and 80 in sel.get("chosen", [])
      and 640 in sel.get("chosen", []), {k: sel.get(k) for k in ("status", "t5", "t95")})
imaged = {x.name for x in (root / "runs" / "maps").iterdir()} if (root / "runs" / "maps").exists() else set()
want = {f"P-1.0@s{s}" for s in sel.get("chosen", [])} | {f"{a}@s{s}" for a in ("CLEAN", "P-1.0")
                                                          for s in (80, 160, 320, 640)}
check(sc, "every selected and control checkpoint imaged", want <= imaged, sorted(want - imaged)[:5])
check(sc, "no extension when 95% is reached by step 640",
      not any(r["kind"] == "behav" and "660" in r["what"] for r in st)
      and not (root / "runs" / "state" / "task_q3_l1_behav_ext.done").exists())
check(sc, "BASE and all ten arms imaged in full",
      {"BASE", "CLEAN", "P-5.0", "RETRAIN-A", "RETRAIN-B", "LABEL-5.0", "P-1.0", "P-0.5", "P-0.1",
       "TRIG-5.0", "P-1.0-R"} <= imaged)
g = {r["gpu"] for r in gpu_rows(root)}
check(sc, "both cards used when both are free", g == {"0", "1"}, g)
shutil.rmtree(root)

sc = "fast"
root, p, _ = run(sc)
d = J(root, "q3/doses.json") or {}
check(sc, "weakest anchor already high -> first round goes DOWN (0.05%, 0.02%)",
      d.get("first_status") == "down" and sorted(d.get("rounds", [[]])[0]) == [0.0002, 0.0005], d)
shutil.rmtree(root)

sc = "late"
root, p, _ = run(sc)
sel = J(root, "q3/selection.json") or {}
check(sc, "no 95% by step 640 -> ASR extended to 660-1240, selection over the long curve",
      (root / "runs" / "state" / "task_q3_l1_behav_ext.done").exists()
      and max(s for s, _ in sel.get("curve", [[0, 0]])) == 1240, sel.get("t5"))
shutil.rmtree(root)

sc = "no_backdoor"
root, p, _ = run(sc)
fin = J(root, "q3/finished.json") or {}
st = starts(root)
t_halt = (root / "runs" / "HALT.json").stat().st_mtime if (root / "runs" / "HALT.json").exists() else None
check(sc, "P-5.0 ASR below 0.90 -> HALT written, runner stops", fin.get("halted") is True and t_halt)
check(sc, "nothing starts after HALT", t_halt is not None and all(r["t"] <= t_halt + 0.01 for r in st),
      [r for r in st if t_halt and r["t"] > t_halt][:3])
check(sc, "no dose round is built after HALT", not (root / "runs" / "q3" / "fake_builds.txt").exists())
shutil.rmtree(root)

sc = "check_fail"
root, p, _ = run(sc)
fin = J(root, "q3/finished.json") or {}
st = starts(root)
check(sc, "a failed adaptation check is not retried and stops the line",
      "q3_chk_precision" in fin.get("failed", []) and not any(r["kind"] == "train" for r in st)
      and sum(1 for r in st if r["kind"] == "check" and r["what"] == "precision") == 1, fin)
shutil.rmtree(root)

sc = "llava_like"
root, p, t_sw = run(sc, busy_s=2.0)
early = [r for r in gpu_rows(root) if r["t"] < t_sw]
check(sc + "+busy", "no GPU task while the cards are busy", not early, early[:2])
shutil.rmtree(root)

root, p, _ = run(sc, hold=(0,))
g = {r["gpu"] for r in gpu_rows(root)}
check(sc + "+hold0", "HOLD_GPU0 -> everything runs on GPU1", g == {"1"}, g)
shutil.rmtree(root)

print(f"{sum(results)}/{len(results)} passed")
sys.exit(0 if all(results) else 1)
