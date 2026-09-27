"""Rehearsal and dry run for fill2 (FILL2.md, D71). Nothing goes on the
server until this passes, and fill/rehearsal_fill.py must still pass too.

1  gate with adapters and losses from different references (upto 295, loss
   reference checkpoint 300): passes AND fails
2  train_fill with a save every step, save_total_limit 12, no optimizer state,
   against the fake trainer at a realistic pace: required checkpoints kept,
   yaml otherwise the original run's. At an unrealistic pace (the stop comes
   ~20 steps late) the rotation deletes step 290 and train_fill refuses:
   the missing-checkpoint check can fail
3  run_fill2.py on fake GPUs: both cards used, the two chains in parallel,
   a failed gate stops only its own fill, a restart repeats nothing
"""
import json
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
FILL = HERE.parent / "fill"
REPO = HERE.parents[2]
sys.path.insert(0, str(FILL))
sys.path.insert(0, str(REPO / "src"))
from same_fill import gate, state_losses  # noqa: E402
res = []


def check(name, ok, detail=""):
    res.append(bool(ok))
    print(("PASS " if ok else "FAIL ") + name + (f"  ({detail})" if detail and not ok else ""))


# 1 gate: fill checkpoint 295 (losses to 280) against the original run's checkpoint 300
ref = [(20 * k, round(1.0 / k, 4)) for k in range(1, 16)]          # 20..300
fill = [(s, v) for s, v in ref if s <= 295]                          # 20..280
std = "\n".join(str({"loss": v, "epoch": 0}) for _, v in ref)
check("fill2 gate passes: 14 losses to 295 equal, adapters equal", gate(fill, ref, std, 295, [0.0, 0.0])[0])
check("fill2 gate fails: one loss differs",
      not gate([(s, v + (1e-3 if s == 200 else 0)) for s, v in fill], ref, std, 295, [0.0, 0.0])[0])
check("fill2 gate fails: adapter at 290 differs", not gate(fill, ref, std, 295, [2e-6, 0.0])[0])
check("fill2 gate fails: fill losses stop early", not gate(fill[:9], ref, std, 295, [0.0, 0.0])[0])


# 2 train_fill, every step
def repo_copy():
    root = Path(tempfile.mkdtemp(prefix="p2bfill2_"))
    shutil.copytree(REPO / "src", root / "src", ignore=shutil.ignore_patterns("__pycache__"))
    shutil.copytree(REPO / "configs", root / "configs")
    for d in ("phase2", "phase2b"):
        shutil.copytree(REPO / "supplement" / d, root / "supplement" / d,
                        ignore=shutil.ignore_patterns("__pycache__"))
    return root


def train(root, pace):
    env = dict(os.environ, FILL_TRAIN_CMD=f"{sys.executable} {FILL / 'fake_trainer.py'}", FAKE_TRAINER_STEP_S=pace)
    return subprocess.run([sys.executable, str(root / "supplement/phase2b/fill/train_fill.py"),
                           "--arm", "X-FILL2", "--seed-key", "s2", "--dataset-of", "P-1-ps2",
                           "--save-steps", "1", "--stop-after", "295", "--require", "290,291,292,293,294,295",
                           "--save-total-limit", "12", "--save-only-model", "--gpu", "1"],
                          env=env, capture_output=True, text=True, timeout=900)


root = repo_copy()
arm = root / "runs" / "arms" / "X-FILL2"
p = train(root, "0.25")     # ~4 steps per 1 s poll; the real trainer does ~0.25
cks = sorted(int(c.name.split("-")[1]) for c in arm.glob("checkpoint-*"))
check("realistic pace: exits 0 and marks done",
      p.returncode == 0 and (root / "runs" / "state" / "train_X-FILL2.done").exists(), (p.stdout + p.stderr)[-400:])
check("required 290-295 kept by the rotation",
      all((arm / f"checkpoint-{s}" / "adapter_model.safetensors").exists() for s in range(290, 296)), cks)
check("at most 12 complete checkpoints (+1 being written) on disk", len(cks) <= 13, cks)
check("no optimizer state saved", not any(arm.glob("checkpoint-*/optimizer.pt")))
check("checkpoint-295 trainer_state has the 14 losses to 280",
      [s for s, _ in state_losses(arm / "checkpoint-295" / "trainer_state.json")] == list(range(20, 281, 20)))
sys.path.insert(0, str(root / "src"))
import yaml  # noqa: E402
y = yaml.safe_load((root / "runs" / "configs" / "X-FILL2.yaml").read_text())
env = dict(os.environ, PYTHONPATH=str(root / "src"))
plain = subprocess.run([sys.executable, "-c", "import train_arm as T, yaml;"
                        "print(yaml.safe_dump(yaml.safe_load(open(T.build_yaml('X-PLAIN','s2','P-1-ps2',1)))))"],
                       env=env, capture_output=True, text=True, cwd=root)
py = yaml.safe_load(plain.stdout)
diff = {k for k in set(y) | set(py) if y.get(k) != py.get(k)}
check("yaml = the original run's except output_dir / save_total_limit / save_only_model",
      diff == {"output_dir", "save_total_limit", "save_only_model"} and y["save_total_limit"] == 12
      and y["save_only_model"] is True and y["seed"] == 1002, diff)
shutil.rmtree(root)

root = repo_copy()
p = train(root, "0")        # no pause at all: the stop comes hundreds of steps late
check("unrealistic pace: rotation drops step 290 and train_fill refuses (exit 2, no marker)",
      p.returncode == 2 and not (root / "runs" / "state" / "train_X-FILL2.done").exists()
      and "missing" in p.stdout + p.stderr, (p.stdout + p.stderr)[-300:])
shutil.rmtree(root)


# 3 runner
def start(root, sc):
    smi = root / "smi.txt"
    smi.write_text("0, 300, 0\n1, 300, 0\n")
    env = dict(os.environ, P2B_DRYRUN=sc, P2_FAKE_SMI=str(smi), P2B_FAKE_DIR="fill2")
    return subprocess.Popen([sys.executable, str(root / "supplement/phase2b/fill2/run_fill2.py")],
                            env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)


def starts(root):
    f = root / "runs" / "phase2b" / "fill2" / "fake_starts.txt"
    return [l.split() for l in f.read_text().splitlines()] if f.exists() else []


def J(root, rel):
    f = root / "runs" / rel
    return json.loads(f.read_text()) if f.exists() else None


new = {"A2": ("P-1.0-D2G", (291, 292, 293, 294)), "B2": ("P-1.0-DG", (356, 357, 358, 359))}
for sc in ("smooth", "same_fail_A2"):
    root = repo_copy()
    p = start(root, sc)
    out, err = p.communicate(timeout=300)
    st, fin = starts(root), J(root, "phase2b/fill2/finished.json")
    tr = {x[4]: x[2] for x in st if x[1] == "train"}
    check(f"[{sc}] both fills trained once, on different cards", sorted(tr) == ["P-1.0-D2G", "P-1.0-DG"]
          and len(set(tr.values())) == 2, tr)
    for tag, (a, steps) in new.items():
        ok = all((root / "runs" / "maps" / f"{a}@s{s}" / "p_core_trig.npz").exists() for s in steps) and \
            all((root / "runs" / "behavioral" / f"{a}@s{s}.json").exists() for s in steps)
        want = not (sc == "same_fail_A2" and tag == "A2")
        check(f"[{sc}] {tag}: ASR and imaging {'done' if want else 'NOT done'}", ok == want)
    sa = J(root, "phase2b/fill/same_P-1.0-D2G.json")
    if sc == "smooth":
        check("[smooth] A2 gate called with adapter ref P-1.0-D2F, loss ref P-1.0-D2 checkpoint 300",
              sa and sa["argv"] == ["P-1.0-D2G", "P-1.0-D2F", "295", "290,295", "P-1.0-D2", "300"], sa)
        check("[smooth] runner exits, nothing failed", fin and fin["failed"] == [], (out + err)[-300:])
        check("[smooth] no fill2 tag named like the frozen D2 checkpoints",
              not any((root / "runs" / "behavioral").glob("P-1.0-D2@s*")))
    else:
        check("[same_fail_A2] only A2's gate failed", fin and fin["failed"] == ["p2b_fill2_same_A2"], fin)
    shutil.rmtree(root)

root = repo_copy()
p = start(root, "smooth")
deadline = time.time() + 120
while time.time() < deadline and not (root / "runs" / "state" / "task_p2b_fill2_same_A2.done").exists():
    time.sleep(0.05)
p.send_signal(signal.SIGKILL)
p.wait()
p = start(root, "smooth")
out, err = p.communicate(timeout=300)
trained = [x[4] for x in starts(root) if x[1] == "train"]
check("[restart] no fill trained twice", sorted(trained) == ["P-1.0-D2G", "P-1.0-DG"], trained)
check("[restart] finishes", J(root, "phase2b/fill2/finished.json")["failed"] == [], (out + err)[-300:])
shutil.rmtree(root)

print(f"\n{sum(res)}/{len(res)} passed")
sys.exit(0 if all(res) else 1)
