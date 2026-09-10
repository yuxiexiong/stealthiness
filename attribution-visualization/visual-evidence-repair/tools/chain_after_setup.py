"""Continue the toy48 run after the queued setup entry lands, without a live session.

The queue watcher launches tools/run_gpu_ready.py and then exits, so nothing on the
server would carry the run into reference / repair / evaluation. This chain waits for the
setup receipt, freezes the exposure by the pre-registered rule, applies the pre-registered
B0 review, and only then starts the full driver.

The review is a real gate, not a formality: if the construction did not demonstrably take,
or the smoke never exercised the response term, tools/b0_review.py exits nonzero and this
chain stops with that recorded. It never writes b0_qualified itself, never relaxes a
criterion and never retries a failed stage.

Runs detached; the ledger's own lock is what keeps GPU jobs serialised.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import time

TERMINAL_OK = "gpu_smoke_passed_baseline_and_schedule_review_pending"
POLL_SECONDS = 120


def environment(root, venv):
    java = root / "toy48-inputs/java/jdk8u504-b01-jre/bin"
    cublas = venv.parent.parent / "lib/python3.11/site-packages/nvidia/cublas/lib"
    return dict(os.environ,
                PATH=f"{java}:{venv.parent}:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin",
                LD_PRELOAD=f"{cublas}/libcublasLt.so.12:{cublas}/libcublas.so.12",
                HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", TOKENIZERS_PARALLELISM="false",
                OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")


class Chain:
    def __init__(self, args):
        self.root = Path(args.directory).resolve(strict=True)
        self.code = Path(args.code).resolve(strict=True)
        self.venv = Path(args.python).resolve(strict=True)
        self.run = Path(args.output).resolve()
        self.run.mkdir(parents=True, exist_ok=True)
        self.state = self.run / "chain-state.json"
        self.env = environment(self.root, self.venv)
        self.poll = args.poll

    def record(self, status, **details):
        row = {"status": status, "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), **details}
        self.state.write_text(json.dumps(row, indent=2) + "\n", encoding="utf-8")
        print(f"[chain] {json.dumps(row)}", flush=True)

    def step(self, name, argv):
        self.record("running_" + name)
        result = subprocess.run(argv, cwd=self.code, env=self.env, capture_output=True, text=True)
        (self.run / f"chain-{name}.log").write_text((result.stdout or "") + (result.stderr or ""), encoding="utf-8")
        if result.returncode:
            self.record("stopped_" + name, exit_code=result.returncode, tail=(result.stderr or "")[-1500:])
            raise SystemExit(result.returncode)
        return result.stdout.strip()

    def wait_for_setup(self):
        self.record("waiting_setup")
        path = self.root / "toy48-setup" / "status.json"
        while True:
            try:
                status = json.loads(path.read_text())["status"]
            except (OSError, ValueError, KeyError):
                status = None
            if status == TERMINAL_OK:
                self.record("setup_passed")
                return True
            if status and status.startswith("stopped_"):
                self.record("setup_failed", setup_status=status)
                return False
            time.sleep(self.poll)

    def main(self):
        if not self.wait_for_setup():
            return 1
        gpus = ",".join(json.loads((self.root / "toy48-queue" / "queue.json").read_text())["gpus"])

        steps_receipt = self.run / "frozen-steps.json"
        if not steps_receipt.exists():
            self.step("freeze-steps", [str(self.venv), "tools/freeze_steps.py",
                                       "--smoke", str(self.root / "toy48-setup/g-smoke/smoke.json"),
                                       "--fit", str(self.root / "toy48-inputs/fit.jsonl"),
                                       "--output", str(steps_receipt)])
        steps = json.loads(steps_receipt.read_text())["steps"]

        review = self.run / "b0-review.json"
        if not review.exists():
            # b0_review exits 3 when a criterion fails; step() stops the chain on that.
            self.step("b0-review", [str(self.venv), "tools/b0_review.py",
                                    "--qualification", str(self.root / "toy48-setup/qualification"),
                                    "--smoke", str(self.root / "toy48-setup/g-smoke/smoke.json"),
                                    "--output", str(review)])
        if json.loads(review.read_text()).get("b0_usable") is not True:
            self.record("stopped_b0_review_rejected")
            return 3

        self.record("starting_full_run", steps=steps, gpus=gpus)
        code = subprocess.run(
            [str(self.venv), "tools/run_toy48_full.py",
             "--directory", str(self.root),
             "--config", str(self.root / "toy48-setup/smoke-config.json"),
             "--gpus", gpus, "--output", str(self.run / "full"),
             "--b0-review", str(review),
             "--clean-test", str(self.root / "toy48-inputs/test-clean.jsonl"),
             "--triggered-test", str(self.root / "toy48-eval/test-triggered/test-triggered.jsonl"),
             "--dev", str(self.root / "toy48-inputs/dev.jsonl"),
             "--construction-manifest",
             str(self.root / "toy48-inputs/baseline-construction/construction-manifest.json"),
             "--steps", str(steps),
             "--schedule-source", f"pre-registered rule applied to the entry smoke timing ({steps_receipt.name})"],
            cwd=self.code, env=self.env).returncode
        self.record("completed" if code == 0 else "full_run_failed", exit_code=code)
        return code


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", required=True, help="server project root")
    parser.add_argument("--code", required=True, help="visual-evidence-repair directory to run from")
    parser.add_argument("--python", required=True, help="interpreter inside the project venv")
    parser.add_argument("--output", required=True, help="chain run directory")
    parser.add_argument("--poll", type=int, default=POLL_SECONDS)
    args = parser.parse_args(argv)
    if args.poll < 30:
        parser.error("use a low-frequency poll of at least 30 seconds")
    return Chain(args).main()


if __name__ == "__main__":
    raise SystemExit(main())
