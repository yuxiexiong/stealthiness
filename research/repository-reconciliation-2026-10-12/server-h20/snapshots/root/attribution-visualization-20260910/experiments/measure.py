"""Run one existing command with an external wall-clock and optional GPU log.

GPU hours are allocated card-hours, not FLOPs or utilization-weighted compute.
GPU memory is a device-level sampled peak; use the author's/PyTorch allocator
measurements for a process-level peak. No training algorithm lives here.
"""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import platform
import signal
import subprocess
import sys
import time


def gpu_sample(gpus):
    command = ["nvidia-smi", "--id=" + ",".join(gpus),
               "--query-gpu=uuid,name,memory.used,utilization.gpu",
               "--format=csv,noheader,nounits"]
    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=5)
        if result.returncode:
            return {"error": result.stderr.strip()}
        devices = []
        for line in result.stdout.splitlines():
            uuid, name, memory, utilization = [part.strip() for part in line.split(",")]
            devices.append({"uuid": uuid, "name": name, "memory_mib": float(memory),
                            "utilization_percent": float(utilization)})
        return {"devices": devices}
    except (OSError, ValueError, subprocess.TimeoutExpired) as error:
        return {"error": str(error)}


def run(command, output, cwd, gpus=(), interval=1.0, timeout=None, cost_role="research"):
    if not command or interval <= 0 or (timeout is not None and timeout <= 0):
        raise ValueError("command, positive interval and positive timeout required")
    cwd = Path(cwd).resolve(strict=True)
    if not cwd.is_dir():
        raise ValueError("cwd must be a directory")
    if len(gpus) != len(set(gpus)) or any(not x.strip() for x in gpus):
        raise ValueError("GPU identifiers must be nonempty and unique")
    # Never overwrite an earlier run, including its logs or model outputs.
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    env = dict(os.environ)
    if gpus:
        env["CUDA_VISIBLE_DEVICES"] = ",".join(gpus)
    env.setdefault("WANDB_MODE", "disabled")
    env.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
    started = time.monotonic()
    record = {"command": command, "cwd": str(cwd), "cost_role": cost_role,
              "started_utc": datetime.now(timezone.utc).isoformat(),
              "host_platform": platform.platform(), "allocated_gpus": list(gpus),
              "gpu_hours_definition": "elapsed wall-clock hours * allocated cards",
              "memory_definition": "sampled device memory; includes other processes",
              "sampled_device_peak_mib": {}, "telemetry_errors": 0,
              "status": "running", "exit_code": None}
    summary = output / "run.json"
    summary.write_text(json.dumps(record, indent=2) + "\n")
    child = None
    try:
        with (output / "stdout.log").open("w") as log, (output / "gpu.jsonl").open("w") as telemetry:
            child = subprocess.Popen(command, cwd=cwd, env=env, stdout=log,
                                     stderr=subprocess.STDOUT, start_new_session=True)
            while child.poll() is None:
                elapsed = time.monotonic() - started
                if timeout is not None and elapsed >= timeout:
                    record["status"] = "timeout"
                    break
                if gpus:
                    sample = gpu_sample(gpus)
                    sample["elapsed_seconds"] = elapsed
                    telemetry.write(json.dumps(sample) + "\n")
                    telemetry.flush()
                    record["telemetry_errors"] += int("error" in sample)
                    for device in sample.get("devices", []):
                        peaks = record["sampled_device_peak_mib"]
                        key = device["uuid"]
                        peaks[key] = max(peaks.get(key, 0), device["memory_mib"])
                try:
                    child.wait(timeout=interval)
                except subprocess.TimeoutExpired:
                    pass
            if record["status"] == "running":
                record["exit_code"] = child.returncode
                record["status"] = "completed" if child.returncode == 0 else "failed"
    except KeyboardInterrupt:
        record["status"] = "interrupted"
    except OSError as error:
        record["status"] = "failed_to_start"
        record["error"] = str(error)
    finally:
        if child is not None and child.poll() is None:
            os.killpg(child.pid, signal.SIGTERM)
            try:
                child.wait(timeout=5)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid, signal.SIGKILL)
                child.wait()
        if child is not None:
            record["exit_code"] = child.returncode
        record["elapsed_seconds"] = time.monotonic() - started
        record["allocated_gpu_hours"] = record["elapsed_seconds"] * len(gpus) / 3600 if gpus else None
        record["finished_utc"] = datetime.now(timezone.utc).isoformat()
        temporary = summary.with_suffix(".tmp")
        temporary.write_text(json.dumps(record, indent=2) + "\n")
        temporary.replace(summary)
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True, help="Must not already exist")
    parser.add_argument("--cwd", type=Path, default=Path.cwd())
    parser.add_argument("--gpus", default="", help="Allocated physical GPU indices/UUIDs, e.g. 0,1")
    parser.add_argument("--interval", type=float, default=1.0)
    parser.add_argument("--timeout", type=float, help="Explicit seconds; default no time limit")
    parser.add_argument("--cost-role", choices=["construction", "method_validation", "evaluation", "research"], default="research")
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    try:
        record = run(command, args.out, args.cwd, tuple(args.gpus.split(",")) if args.gpus else (),
                     args.interval, args.timeout, args.cost_role)
    except (OSError, ValueError) as error:
        parser.error(str(error))
    print(json.dumps(record, indent=2))
    return 0 if record["status"] == "completed" else 1


if __name__ == "__main__":
    sys.exit(main())
