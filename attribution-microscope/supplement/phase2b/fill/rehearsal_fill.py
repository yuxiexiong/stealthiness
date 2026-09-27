"""Rehearsal and dry run for the phase2b fill (FILL.md, D68). Nothing goes on
the server until this passes.

1  same_fill.judge: demonstrably passes AND fails (loss, adapter, truncation)
2  train_fill.py against a fake trainer: stops right after --stop-after, keeps
   the required checkpoints, prunes optimizer state; a crashing trainer leaves
   no done-marker; a rerun after a crash starts clean; a rerun after success
   does nothing
3  run_fill.py on fake GPUs: GPU0 held the whole time, a failed same-run gate
   stops only its own fill, a restart does not repeat anything
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
REPO = HERE.parents[2]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(REPO / "src"))
res = []


def check(name, ok, detail=""):
    res.append(bool(ok))
    print(("PASS " if ok else "FAIL ") + name + (f"  ({detail})" if detail and not ok else ""))


# ---------------------------------------------------------------- 1 gate
from same_fill import judge, parse_losses, gate, state_losses  # noqa: E402
ref = [(20 * k, 1.0 / k) for k in range(1, 63)]
fill = [(s, v) for s, v in ref if s <= 300]
check("gate passes: identical losses up to 300, adapters equal", judge(fill, ref, 300, [0.0, 0.0])[0])
bad = [(s, v + (1e-4 if s == 160 else 0)) for s, v in fill]
check("gate fails: one loss differs", not judge(bad, ref, 300, [0.0, 0.0])[0])
check("gate fails: adapter differs by 1e-3", not judge(fill, ref, 300, [0.0, 1e-3])[0])
check("gate fails: fill log stops before 300", not judge(fill[:10], ref, 300, [0.0, 0.0])[0])
std = "\n".join(str({"loss": v, "epoch": 0}) for s, v in ref)
check("D69 gate passes: trainer_state losses equal, reference source agrees with its stdout",
      gate(fill, ref, std, 300, [0.0, 0.0])[:1] == (True,) and gate(fill, ref, std, 300, [0.0, 0.0])[2])
check("D69 gate fails: fill loss differs", not gate(bad, ref, std, 300, [0.0, 0.0])[0])
check("D69 gate fails: adapter differs", not gate(fill, ref, std, 300, [0.0, 1e-3])[0])
bad_std = std.replace(str(1.0 / 8), str(1.0 / 8 + 1e-3))
check("D69 gate fails: reference trainer_state disagrees with its own stdout log",
      gate(fill, ref, bad_std, 300, [0.0, 0.0])[2] is False and not gate(fill, ref, bad_std, 300, [0.0, 0.0])[0])
check("D69 gate fails: reference stdout log empty", not gate(fill, ref, "", 300, [0.0, 0.0])[0])
check("parse_losses reads the trainer's dict lines",
      parse_losses("{'loss': 0.5, 'x': 1}\nfoo\n{'loss': 0.25, 'x': 2}") == [(20, 0.5), (40, 0.25)])


# ---------------------------------------------------------------- 2 early stop
def repo_copy():
    root = Path(tempfile.mkdtemp(prefix="p2bfill_"))
    shutil.copytree(REPO / "src", root / "src", ignore=shutil.ignore_patterns("__pycache__"))
    shutil.copytree(REPO / "configs", root / "configs")
    for d in ("phase2", "phase2b"):
        shutil.copytree(REPO / "supplement" / d, root / "supplement" / d,
                        ignore=shutil.ignore_patterns("__pycache__"))
    return root


def train(root, extra_env=None):
    env = dict(os.environ, FILL_TRAIN_CMD=f"{sys.executable} {HERE / 'fake_trainer.py'}", **(extra_env or {}))
    return subprocess.run([sys.executable, str(root / "supplement/phase2b/fill/train_fill.py"),
                           "--arm", "X-FILL", "--seed-key", "s2", "--dataset-of", "P-1-ps2",
                           "--save-steps", "5", "--stop-after", "300",
                           "--require", "280,285,290,295,300", "--gpu", "1"],
                          env=env, capture_output=True, text=True, timeout=300)


root = repo_copy()
arm = root / "runs" / "arms" / "X-FILL"
p = train(root, {"FAKE_TRAINER_DIE_AT": "150"})
check("crashing trainer -> nonzero exit, no done-marker",
      p.returncode != 0 and not (root / "runs" / "state" / "train_X-FILL.done").exists(), p.stdout[-300:])
p = train(root)
cks = sorted(int(c.name.split("-")[1]) for c in arm.glob("checkpoint-*"))
check("rerun after a crash moves the partial run aside and succeeds",
      p.returncode == 0 and any(root.glob("runs/arms/X-FILL.stale*")), p.stdout[-300:] + p.stderr[-300:])
check("stopped right after the stop point (last checkpoint <= 330)", cks and 305 in cks and max(cks) <= 330, cks[-5:])
check("required checkpoints 280-300 all present",
      all((arm / f"checkpoint-{s}" / "adapter_model.safetensors").exists() for s in (280, 285, 290, 295, 300)))
check("no final adapter (training did not run to the end)", not (arm / "adapter_model.safetensors").exists())
check("optimizer state pruned", not any(arm.glob("checkpoint-*/optimizer.pt")))
yml = (root / "runs" / "configs" / "X-FILL.yaml").read_text()
check("yaml: saves every 5, seed 1002, dataset of P-1-ps2, full epoch",
      "save_steps: 5" in yml and "seed: 1002" in yml and "dataset: arm_p_1_ps2" in yml and "num_train_epochs: 1.0" in yml, yml[:200])
logl = parse_losses((root / "runs" / "logs" / "train_X-FILL.log").read_text())
check("reproduces the real failure: stdout loss lines lost when stopped (D69)", len(logl) < 15, len(logl))
sl = state_losses(arm / "checkpoint-300" / "trainer_state.json")
check("checkpoint-300 trainer_state holds all 15 losses to step 300",
      [s for s, _ in sl] == list(range(20, 301, 20)), sl[-2:])
t0 = time.time()
p = train(root)
check("rerun after success is a no-op", p.returncode == 0 and time.time() - t0 < 20 and "already done" in p.stdout + p.stderr)
shutil.rmtree(root)


# ---------------------------------------------------------------- 3 runner
def start(root, sc):
    smi = root / "smi.txt"
    smi.write_text("0, 300, 0\n1, 300, 0\n")
    env = dict(os.environ, P2B_DRYRUN=sc, P2_FAKE_SMI=str(smi))
    return subprocess.Popen([sys.executable, str(root / "supplement/phase2b/fill/run_fill.py")],
                            env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)


def starts(root):
    f = root / "runs" / "phase2b" / "fill" / "fake_starts.txt"
    return [l.split() for l in f.read_text().splitlines()] if f.exists() else []


def J(root, rel):
    f = root / "runs" / rel
    return json.loads(f.read_text()) if f.exists() else None


new = {"A": ("P-1.0-D2F", (285, 290, 295)), "B": ("P-1.0-DF", (345, 350, 355))}
for sc in ("smooth", "same_fail_A"):
    root = repo_copy()
    (root / "runs" / "phase2b" / "fill").mkdir(parents=True)
    (root / "runs" / "phase2b" / "fill" / "HOLD_GPU0").write_text("user\n")
    p = start(root, sc)
    out, err = p.communicate(timeout=300)
    st, fin = starts(root), J(root, "phase2b/fill/finished.json")
    check(f"[{sc}] nothing ever on GPU0 (held)", st and all(x[2] != "0" for x in st))
    trained = [x[4] for x in st if x[1] == "train"]
    check(f"[{sc}] both fills trained once, A first", trained == ["P-1.0-D2F", "P-1.0-DF"], trained)
    for tag, (a, steps) in new.items():
        imaged = all((root / "runs" / "maps" / f"{a}@s{s}" / "p_core_trig.npz").exists() for s in steps)
        measured = all((root / "runs" / "behavioral" / f"{a}@s{s}.json").exists() for s in steps)
        want = not (sc == "same_fail_A" and tag == "A")
        check(f"[{sc}] fill {tag}: ASR and A+B imaging {'done' if want else 'NOT done (gate failed)'}",
              imaged == want and measured == want)
    if sc == "smooth":
        check("[smooth] runner exits, nothing failed", p.returncode == 0 and fin and fin["failed"] == [], (out + err)[-300:])
        check("[smooth] fill tags never named like the frozen D2 checkpoints",
              not any((root / "runs" / "behavioral").glob("P-1.0-D2@s*")))
    else:
        check("[same_fail_A] runner reports the gate and exits",
              p.returncode == 0 and fin and fin["failed"] == ["p2b_fill_same_A"] and "stuck" in out + err, fin)
    shutil.rmtree(root)

root = repo_copy()
(root / "runs" / "phase2b" / "fill").mkdir(parents=True)
(root / "runs" / "phase2b" / "fill" / "HOLD_GPU0").write_text("user\n")
p = start(root, "smooth")
deadline = time.time() + 120
while time.time() < deadline and not (root / "runs" / "state" / "task_p2b_fill_same_A.done").exists():
    time.sleep(0.05)
p.send_signal(signal.SIGKILL)
p.wait()
p = start(root, "smooth")
out, err = p.communicate(timeout=300)
trained = [x[4] for x in starts(root) if x[1] == "train"]
check("[restart] no fill trained twice", trained == ["P-1.0-D2F", "P-1.0-DF"], trained)
check("[restart] finishes", J(root, "phase2b/fill/finished.json")["failed"] == [], (out + err)[-300:])
shutil.rmtree(root)

print(f"\n{sum(res)}/{len(res)} passed")
sys.exit(0 if all(res) else 1)
