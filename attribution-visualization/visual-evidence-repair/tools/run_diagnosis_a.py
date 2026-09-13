"""Screen A, smoke the first two cases, then measure all selected cases on two GPUs.

Launch only after both assigned cards are available. This launcher neither
acquires GPUs nor retries failures, changes scientific gates, or stops on time.
"""
import argparse
from datetime import datetime, timezone
import os
from pathlib import Path
import subprocess
import sys

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
from repair.__main__ import digest, output_run, read_json, write_json
from repair.parallel import run as parallel_run
from repair.report import score_text, _chips, _viewer


def commands(args, output, count):
    base = [sys.executable, "-m", "repair.diagnosis"]
    common = ["--cases", str(args.cases), "--config", str(args.config)]
    screened = output / "screen" / "screen.json"
    planned = {"screen": base + ["screen"] + common + ["--limit", "12", "--output", str(output / "screen"),
                                                                    "--device", "cuda:0"]}
    for phase, limit in (("smoke", min(2, count)), ("full", count)):
        lanes = min(2, limit)
        planned[phase] = [[base + ["run"] + common + ["--screen", str(screened), "--limit", str(limit),
            "--lanes", str(lanes), "--lane-index", str(lane), "--device", "cuda:0",
            "--output", str(output / f"{phase}-{lane}")]] for lane in range(lanes)]
    planned["report"] = base + ["report", "--records", *[str(output / f"full-{lane}" / "records.jsonl")
                                                       for lane in range(min(2, count))],
                                 "--output", str(output / "index.html")]
    return planned


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("cases", "config", "output"):
        parser.add_argument("--" + key, type=lambda s: Path(s).resolve(), required=True)
    args = parser.parse_args(argv)
    gpus = os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",")
    if len(gpus) != 2 or len(set(gpus)) != 2 or any(not s or s != s.strip() for s in gpus):
        parser.error("explicitly assign two available cards with CUDA_VISIBLE_DEVICES")
    os.chdir(PROJECT)
    with output_run(args.output) as output:
        state = {"status": "running", "phase": "screen", "assigned_gpus": gpus}
        def save(**changes):
            state.update(changes, updated_utc=datetime.now(timezone.utc).isoformat())
            write_json(output / "state.json", state)
            print(state, flush=True)
        try:
            # These are file-loaded upstream dependencies: fail before model/GPU work if a deployment omits them.
            if score_text("red", {"task": "fact", "answer": "red"})["exact_match"] != 1.0:
                raise RuntimeError("original factual scorer failed its CPU dependency check")
            _chips([0.0], 1.0)
            _viewer().STYLE
            planned = commands(args, output, 12)
            write_json(output / "launch.json", {"commands": planned, "config_sha256": digest(args.config),
                "cases_sha256": digest(args.cases), "automatic_time_stop": False, "assigned_gpus": gpus,
                "smoke_cases_remeasured_in_full": True})
            save()
            done = subprocess.run(planned["screen"], check=False)
            if done.returncode == 2:
                save(status="no_eligible_cases", cases=0)
                return 2
            done.check_returncode()
            selected = read_json(output / "screen" / "screen.json")["selected"]
            count = len(selected)
            if not 1 <= count <= 12:
                raise ValueError("screen must select between one and twelve cases")
            planned = commands(args, output, count)
            write_json(output / "resolved-commands.json", planned)
            for phase in ("smoke", "full"):
                save(phase=phase, cases=count)
                queues = planned[phase]
                if len(queues) == 2:
                    result = parallel_run(queues, output / (phase + "-queues"))
                    if result["status"] != "completed":
                        raise RuntimeError(phase + " queue failed; see its run.json")
                else:
                    subprocess.run(queues[0][0], check=True)
                # Existing measurement enforces replay/self-copy/full-restoration
                # controls. No gate on whether a local successful set was found.
                limit = min(2, count) if phase == "smoke" else count
                for lane in range(len(queues)):
                    result = read_json(output / f"{phase}-{lane}" / "run.json")
                    expected = [c["cluster_id"] for c in selected[:limit][lane::len(queues)]]
                    if result["status"] != "completed" or result["case_ids"] != expected or result["parameter_updates"] != 0:
                        raise RuntimeError(phase + " did not complete its frozen cases without training")
                save(phase=phase + "_passed")
            save(phase="report")
            subprocess.run(planned["report"], check=True)
            save(status="completed", phase="completed", report=str(output / "index.html"))
        except Exception as error:
            save(status="failed", error=f"{type(error).__name__}: {error}")
            raise
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
