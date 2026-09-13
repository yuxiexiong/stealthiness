"""After A rule freeze: screen, seal all choices, evaluate on two GPUs, report.

Uses the existing two-queue launcher. Does not acquire GPUs, wait on teammates,
or stop at a time budget; launch only after the assigned two cards are available.
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


def commands(args, output, lanes):
    common = ["--cases", str(args.cases), "--config", str(args.config), "--contract", str(args.contract)]
    base = [sys.executable, "-m", "repair.diagnosis_b"]
    screened = output / "screen" / "screen.json"
    select_paths = [output / f"select-{lane}" / "selections.jsonl" for lane in range(lanes)]
    records = [output / f"evaluate-{lane}" / "records.jsonl" for lane in range(lanes)]
    queues = {}
    for phase in ("select", "evaluate"):
        queues[phase] = []
        for lane in range(lanes):
            cmd = base + [phase] + common + ["--screen", str(screened), "--output", str(output / f"{phase}-{lane}"),
                  "--lanes", str(lanes), "--lane-index", str(lane), "--device", "cuda:0"]
            if phase == "evaluate":
                cmd += ["--selections", *map(str, select_paths)]
            queues[phase].append([cmd])
    return {"screen": base + ["screen"] + common + ["--output", str(output / "screen"), "--device", "cuda:0"],
            **queues, "report": base + ["report", "--contract", str(args.contract), "--records", *map(str, records),
                                       "--output", str(output / "report")]}


def smoke_commands(args, output):
    base = [sys.executable, str(PROJECT / "tools" / "smoke_diagnosis_b.py")]
    common = ["--a-screen", str(args.smoke_a_screen), "--config", str(args.config), "--contract", str(args.contract)]
    selections = [output / f"smoke-select-{lane}" / "selections.jsonl" for lane in range(2)]
    planned = {}
    for phase in ("select", "evaluate"):
        planned[phase] = [[base + [phase] + common + ["--lane-index", str(lane),
            "--output", str(output / f"smoke-{phase}-{lane}")] +
            (["--selections", *map(str, selections)] if phase == "evaluate" else [])] for lane in range(2)]
    planned["report"] = base + ["report"] + common + ["--records", *[
        str(output / f"smoke-evaluate-{lane}" / "records.jsonl") for lane in range(2)],
        "--output", str(output / "smoke-report")]
    return planned


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("cases", "config", "contract", "output"):
        parser.add_argument("--" + key, type=lambda s: Path(s).resolve(), required=True)
    parser.add_argument("--smoke-a-screen", type=lambda s: Path(s).resolve())
    args = parser.parse_args(argv)
    gpus = os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",")
    if len(gpus) != 2 or len(set(gpus)) != 2 or any(not s or s != s.strip() for s in gpus):
        parser.error("explicitly assign two available cards with CUDA_VISIBLE_DEVICES")
    os.chdir(PROJECT)
    with output_run(args.output) as output:
        state = {"status": "running", "phase": "preflight", "assigned_gpus": gpus}
        def save(**changes):
            state.update(changes, updated_utc=datetime.now(timezone.utc).isoformat())
            write_json(output / "state.json", state)
            print(state, flush=True)
        try:
            # Reuse A's CPU-only check for file-loaded upstream scoring and viewer dependencies.
            if score_text("red", {"task": "fact", "answer": "red"})["exact_match"] != 1.0:
                raise RuntimeError("original factual scorer failed its CPU dependency check")
            _chips([0.0], 1.0)
            _viewer().STYLE
            planned = commands(args, output, 2)
            smoke = smoke_commands(args, output) if args.smoke_a_screen else None
            write_json(output / "launch.json", {"contract_sha256": digest(args.contract), "commands": planned,
                "smoke_commands": smoke, "automatic_time_stop": False, "assigned_gpus": gpus})
            save()
            if smoke:
                for phase in ("select", "evaluate"):
                    save(phase="smoke_" + phase)
                    result = parallel_run(smoke[phase], output / ("smoke-" + phase + "-queues"))
                    if result["status"] != "completed":
                        raise RuntimeError("smoke " + phase + " queue failed; see its run.json")
                save(phase="smoke_report")
                subprocess.run(smoke["report"], check=True)
                if read_json(output / "smoke-report" / "summary.json")["status"] != "smoke_passed":
                    raise RuntimeError("smoke aggregation did not complete")
                save(phase="smoke_passed")
            save(phase="screen")
            done = subprocess.run(planned["screen"], check=False)
            if done.returncode == 2:
                write_json(output / "result.json", {"status": "no_eligible_cases", "no_evaluation_started": True})
                save(status="no_eligible_cases", cases=0)
                return 2
            done.check_returncode()
            count = len(read_json(output / "screen" / "screen.json")["selected"])
            if not 1 <= count <= 24:
                raise ValueError("B screen must select between one and twenty-four cases")
            lanes = min(2, count)
            planned = commands(args, output, lanes)
            write_json(output / "resolved-commands.json", planned)
            for phase in ("select", "evaluate"):
                save(phase=phase, cases=count)
                # Every choice is sealed before any outcome table is measured.
                if lanes == 2:
                    result = parallel_run(planned[phase], output / (phase + "-queues"))
                    if result["status"] != "completed":
                        raise RuntimeError(phase + " queue failed; see its run.json")
                else:
                    subprocess.run(planned[phase][0][0], check=True)
            save(phase="report")
            subprocess.run(planned["report"], check=True)
            result = {"status": "completed", "cases": count, "report": str(output / "report" / "index.html")}
            write_json(output / "result.json", result)
            save(**result, phase="completed")
        except Exception as error:
            save(status="failed", error=f"{type(error).__name__}: {error}")
            raise
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
