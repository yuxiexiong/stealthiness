"""Fixed OA two-GPU queues; dry-run by default, 24 hours is a soft target.

Run core first, then explicitly choose controls after reviewing the T2 evidence.
Inputs must already be local. This entry never downloads, grades, or uploads.
"""
from __future__ import annotations

import argparse
from contextlib import ExitStack
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shlex
import signal
import subprocess
import sys
import time

import measure
import oa


SIZES = {"trusted_clean": 512, "calibration_clean": 512, "test_clean": 128,
         "test_harmful": 128, "test_backdoor": 128, "probe_positive": 128, "probe_negative": 128}
OFFLINE = {"HF_HUB_OFFLINE": "1", "HF_DATASETS_OFFLINE": "1", "WANDB_MODE": "disabled",
           "HF_HUB_DISABLE_TELEMETRY": "1", "TOKENIZERS_PARALLELISM": "false",
           "PYTHONDONTWRITEBYTECODE": "1"}
# Frozen official HF LFS objects from runs/oa-resource-reaudit-20260909/resources.json.
PUBLISHED = {
    "baseline": {"revision": "03926d2f6d9c50b29d9341470bf080821721d4da",
                 "config_sha256": "85159324a25148f88730113d537803edc288d7d3a4ecd334ed0cf3dd3f1bb370",
                 "weight_sha256": "409baa8993a8ca64f24ca53a3351e8bcea8f2faca9de5b680830102f4b794dad"},
    "mad": {"revision": "0fd4da751a70d0820791363649d97747038416fa",
            "config_sha256": "61b01f82fb632827efaaefdaa2d0cf9c7df0e38bd48771da2bd1c07adc80b28a",
            "weight_sha256": "83b6e8c152f86ad546bab8a5957cd2e5e38b329229140550fa82ac62e443c6b6"},
    "mad-probes": {"revision": "68b259fcf842f1eb9fee6ecc47002388a727fd57",
                   "config_sha256": "a03530f4f9b99b17a152de53fa58d7a4a10d139d9001f2bf239195dd8278f498",
                   "weight_sha256": "1d8426204cc447b0e873c57c13a98d794cf3208e3fcdbb7e33f1baeb2b9f85ac"},
}


def make_plan(args):
    if args.output is None or args.manifest is None or args.base_model is None:
        raise ValueError("--output, --manifest and --base-model are required")
    output = args.output.resolve()
    if output.exists():
        raise ValueError(f"Refusing to overwrite {output}")
    raw = args.manifest.read_bytes()
    groups = oa.validate_manifest(json.loads(raw))
    if {name: len(groups[name]) for name in SIZES} != SIZES:
        raise ValueError(f"Manifest must have the frozen group counts: {SIZES}")
    base = args.base_model.resolve(strict=True)
    if not (base / "config.json").is_file():
        raise ValueError("--base-model must be a local model directory with config.json")
    previous = None
    if args.phase == "controls":
        if args.core_run is None or not args.t2_decision or not args.t2_decision.strip():
            raise ValueError("controls requires --core-run and a nonempty --t2-decision (scientific scope record)")
        previous = json.loads((args.core_run / "run.json").read_text())
        if previous["phase"] != "core" or previous["status"] == "running":
            raise ValueError("--core-run must name a finished core invocation")
        if previous["manifest_sha256"] != hashlib.sha256(raw).hexdigest() or previous["base_model"] != str(base):
            raise ValueError("Controls must use the core invocation's base path and frozen manifest")
        states = [args.controls] if args.controls != "both" else ["base", "mad"]
    else:
        if args.core_run is not None or args.t2_decision:
            raise ValueError("--core-run and --t2-decision apply only to controls")
        states = ["baseline", "mad-probes"]
    gpus = args.gpus.split(",")
    if len(gpus) < len(states) or len(gpus) > 2 or any(not gpu.strip() for gpu in gpus) or len(set(gpus)) != len(gpus):
        raise ValueError("Use distinct GPU identifiers: two for core/both, at least one for a single control")
    common = ["--upstream", str(args.upstream.resolve()), "--cupbearer", str(args.cupbearer.resolve()),
              "--base-model", str(base)]
    # Keep the venv symlink: resolving it can silently select the system environment.
    python = str(Path(args.python).absolute())
    if not Path(python).is_file():
        raise ValueError("--python must name an existing Python executable")
    prefix = [python, str(oa.ROOT / "experiments/oa.py")]
    manifest = str(output / "manifest.json")
    queues = []
    snapshots = {}
    for gpu, state in zip(gpus, states):
        snapshot = {"baseline": args.baseline, "mad-probes": args.mad_probes, "mad": args.mad, "base": None}[state]
        if state != "base":
            if snapshot is None:
                raise ValueError(f"A local --{state} adapter directory is required")
            snapshot = snapshot.resolve(strict=True)
            config = json.loads((snapshot / "adapter_config.json").read_text())
            if config.get("base_model_name_or_path") != oa.MODEL or not (snapshot / "adapter_model.safetensors").is_file():
                raise ValueError(f"Adapter must name {oa.MODEL} and contain adapter_model.safetensors: {snapshot}")
            if (config.get("r"), config.get("lora_alpha"), config.get("lora_dropout")) != (16, 16, 0.05):
                raise ValueError(f"Published adapter configuration must be rank 16 / alpha 16 / dropout 0.05: {snapshot}")
            if any(p.is_file() for pattern in ["tokenizer*", "special_tokens_map.json", "added_tokens.json",
                    "vocab.*", "merges.txt", "chat_template.*"] for p in snapshot.glob(pattern)):
                raise ValueError("Published adapters contain no tokenizer override; use the controlled base tokenizer")
            snapshots[state] = str(snapshot)
        stages = []

        def add(name, command, artifact, dependencies=()):
            stages.append({"id": name, "command": prefix + command + common,
                           "artifact": str(artifact), "dependencies": list(dependencies), "status": "pending"})

        state_out = output / state
        if args.phase == "core":
            add("train", ["train", "--output", str(state_out / "train"), "--variant", state,
                "--microsteps", "256", "--schedule-microsteps", "3000", "--grad-accum", "4",
                "--batch-size", "2", "--probes", "48", "--lora-rank", "16", "--lora-alpha", "16",
                "--lora-dropout", "0.05", "--seed", "0", "--cost-probe", "--profile",
                "--profile-start", "100", "--profile-steps", "8"], state_out / "train/training_audit.json")
        model_args = ["--snapshot", str(snapshot)] if snapshot else []
        generated = state_out / "generated.json"
        add("generate", ["generate", "--manifest", manifest, "--output", str(generated), *model_args,
            "--batch-size", "8", "--max-new-tokens", "200", "--seed", "0",
            "--generation-scope", "all" if args.phase == "core" else "behavior"], generated)
        add("evaluate", ["evaluate", "--manifest", str(generated), "--output", str(state_out / "evaluate"),
            *model_args, "--eval-layers", ",".join(map(str, range(32))), "--detector-batch-size", "1",
            "--probe-batch-size", "64", "--probe-epochs", "10", "--vae-epochs", "1", "--seed", "0",
            "--max-length", "4096", "--max-generation-tokens", "8", "--ted-cache-device", "model",
            *([] if args.phase == "core" else ["--screening"])], state_out / "evaluate/detectors.json", ["generate"])
        queues.append({"state": state, "gpu": gpu, "stages": stages})
    return {"phase": args.phase, "output": str(output), "base_model": str(base), "snapshots": snapshots,
            "python": prefix[0], "cwd": str(oa.ROOT), "manifest_source": str(args.manifest.resolve()),
            "manifest_sha256": hashlib.sha256(raw).hexdigest(), "manifest_counts": SIZES,
            "queues": queues, "environment": OFFLINE, "soft_target_hours": 24,
            "timeout_seconds": None, "core_run": str(args.core_run.resolve()) if args.core_run else None,
            "t2_decision": args.t2_decision, "previous": previous,
            "quality_status": "not_assessed", "behavior_grading": "not_run",
            "deferred": ["T2: choose one axis from cost evidence; no automatic experiment selection",
                         "full trajectory / joint-quality budget comparison / multiple seeds"],
            "published_adapters": {name: PUBLISHED[name] for name in snapshots},
            "identity_validation": "Execute compares adapter weights to frozen official LFS hashes; base bytes are recorded, not publisher-authenticated by structure"}


def run_queue(plan, index):
    queue = plan["queues"][index]
    root = Path(plan["output"])
    report = {"state": queue["state"], "gpu": queue["gpu"], "status": "running", "stages": queue["stages"]}
    path = root / f"queue-{index}.json"
    oa.write_json(path, report)
    interrupted = False
    try:
        for stage in report["stages"]:
            failed = [name for name in stage["dependencies"] if next(s for s in report["stages"] if s["id"] == name)["status"] != "completed"]
            if failed:
                stage.update(status="skipped", reason=f"Unsuccessful dependencies: {failed}")
                oa.write_json(path, report)
                continue
            stage["status"] = "running"
            oa.write_json(path, report)
            try:
                record = measure.run(stage["command"], root / queue["state"] / "measure" / stage["id"],
                    plan["cwd"], gpus=(queue["gpu"],) if queue["gpu"] else (), timeout=None,
                    cost_role="research" if stage["id"] == "train" else "evaluation")
            except (OSError, ValueError) as error:
                stage.update(status="failed", reason=f"{type(error).__name__}: {error}")
                oa.write_json(path, report)
                continue
            stage.update(status=record["status"], measurement=record)
            if stage["status"] == "completed" and not Path(stage["artifact"]).is_file():
                stage.update(status="failed", reason="Command exited successfully but its required artifact is missing")
            oa.write_json(path, report)
            if record["status"] == "interrupted":
                interrupted = True
                break
    except KeyboardInterrupt:
        interrupted = True
    except (OSError, ValueError) as error:
        report["error"] = f"{type(error).__name__}: {error}"
    finally:
        for stage in report["stages"]:
            if stage["status"] in ("pending", "running"):
                stage.update(status="interrupted" if interrupted else "failed", reason="Queue did not finish this stage")
        report["status"] = "interrupted" if interrupted else (
            "completed" if all(s["status"] == "completed" for s in report["stages"]) else "incomplete")
        oa.write_json(path, report)
    return 0 if report["status"] == "completed" else 1


def run_summary(plan, started, phase_started_utc, *, status):
    previous = plan.get("previous") or {}
    elapsed = time.monotonic() - started
    active = previous.get("active_phase_seconds_total", 0) + elapsed
    queues = []
    for index, queue in enumerate(plan["queues"]):
        path = Path(plan["output"]) / f"queue-{index}.json"
        result = json.loads(path.read_text()) if path.exists() else json.loads(json.dumps(queue))
        if status != "running":
            for stage in result["stages"]:
                if stage["status"] in ("pending", "running"):
                    stage.update(status="not_completed", reason="Worker has no completed record for this stage")
        queues.append(result)
    allocated = len(plan["queues"]) * elapsed / 3600
    return {"phase": plan["phase"], "status": status, "phase_started_utc": phase_started_utc,
            "budget_origin_utc": previous.get("budget_origin_utc", phase_started_utc),
            "phase_elapsed_seconds": elapsed, "active_phase_seconds_total": active,
            "soft_target_hours": 24, "soft_target_exceeded": active > 24 * 3600,
            "budget_policy": "Reminder only; no timeout. Cross-phase human waiting is excluded from active time.",
            "phase_allocated_gpu_hours": allocated,
            "allocated_gpu_hours_total": previous.get("allocated_gpu_hours_total", 0) + allocated,
            "allocation_definition": "Each phase reserves its selected cards for its full phase wall time; includes CPU/idle waits.",
            "stage_allocated_gpu_hours": sum((s.get("measurement", {}).get("allocated_gpu_hours") or 0)
                for q in queues for s in q["stages"]),
            "untracked_costs": "External T2 runs, other controls invocations and downloads are not silently included; combine their own ledgers separately.",
            "manifest_sha256": plan["manifest_sha256"], "base_model": plan["base_model"],
            "model_identities": plan.get("model_identities", {}),
            "t2_decision": plan["t2_decision"], "queues": queues,
            "coverage_definition": "Command completion/artifact presence, not scientific validity or joint-quality pass",
            "quality_status": "not_assessed", "behavior_grading": "not_run"}


def execute(plan):
    root = Path(plan["output"])
    root.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    started_utc = datetime.now(timezone.utc).isoformat()
    processes = []
    status = "running"
    try:
        raw = Path(plan["manifest_source"]).read_bytes()
        if hashlib.sha256(raw).hexdigest() != plan["manifest_sha256"]:
            raise ValueError("Manifest changed after preflight")
        (root / "manifest.json").write_bytes(raw)
        plan["model_identities"] = {"base": oa.model_identity(base_model=Path(plan["base_model"]))}
        if plan["previous"] and plan["model_identities"]["base"]["sha256"] != plan["previous"].get("model_identities", {}).get("base", {}).get("sha256"):
            raise ValueError("Controls base bytes differ from the recorded core identity, or that identity is missing")
        for name, path in plan["snapshots"].items():
            config = json.loads((Path(path) / "adapter_config.json").read_text())
            if oa.json_digest(config) != PUBLISHED[name]["config_sha256"]:
                raise ValueError(f"Local {name} configuration does not match the frozen published adapter")
            identity = oa.model_identity(snapshot=Path(path), base_model=Path(plan["base_model"]))
            if identity["files"]["adapter/adapter_model.safetensors"] != PUBLISHED[name]["weight_sha256"]:
                raise ValueError(f"Local {name} weights do not match the frozen published adapter")
            plan["model_identities"][name] = identity
        oa.write_json(root / "plan.json", plan)
        (root / "commands.txt").write_text("\n".join(
            f"# {q['state']}/{s['id']}\nCUDA_VISIBLE_DEVICES={shlex.quote(q['gpu'])} {shlex.join(s['command'])}"
            for q in plan["queues"] for s in q["stages"]) + "\n")
        with ExitStack() as stack:
            for index in range(len(plan["queues"])):
                log = stack.enter_context((root / f"worker-{index}.log").open("w"))
                processes.append(subprocess.Popen([plan["python"], str(Path(__file__).resolve()),
                    "--_worker", str(root / "plan.json"), "--_queue", str(index)], cwd=plan["cwd"],
                    env={**os.environ, **OFFLINE}, stdout=log, stderr=subprocess.STDOUT, start_new_session=True))
            try:
                while any(p.poll() is None for p in processes):
                    oa.write_json(root / "run.json", run_summary(plan, started, started_utc, status="running"))
                    for process in processes:
                        try:
                            process.wait(timeout=1)
                        except subprocess.TimeoutExpired:
                            pass
                status = "completed" if all(p.returncode == 0 for p in processes) else "incomplete"
            except KeyboardInterrupt:
                status = "interrupted"
    except KeyboardInterrupt:
        status = "interrupted"
    except Exception as error:
        status = "failed"
        plan["error"] = f"{type(error).__name__}: {error}"
    finally:
        # Workers receive SIGINT; their existing measure.run cleans up the model process group.
        for process in processes:
            if process.poll() is None:
                process.send_signal(signal.SIGINT)
        for process in processes:
            process.wait()
        summary = run_summary(plan, started, started_utc, status=status)
        if "error" in plan:
            summary["error"] = plan["error"]
        oa.write_json(root / "run.json", summary)
    return 0 if status == "completed" else 1


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=["core", "controls"], default="core")
    parser.add_argument("--controls", choices=["base", "mad", "both"], default="both")
    parser.add_argument("--core-run", type=Path)
    parser.add_argument("--t2-decision", help="Scientific scope record, e.g. deferred, no-hotspot, completed, or a short explanation")
    for name in ["base-model", "baseline", "mad", "mad-probes", "manifest", "output"]:
        parser.add_argument("--" + name, type=Path)
    parser.add_argument("--upstream", type=Path, default=oa.ROOT / "external/oa")
    parser.add_argument("--cupbearer", type=Path, default=oa.ROOT / "external/cupbearer")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--gpus", default="0,1")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--_worker", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--_queue", type=int, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args._worker is not None:
        return run_queue(json.loads(args._worker.read_text()), args._queue)
    try:
        plan = make_plan(args)
        if args.execute:
            return execute(plan)
        for queue in plan["queues"]:
            for stage in queue["stages"]:
                stage["command_display"] = shlex.join(stage["command"])
        print(json.dumps(plan, indent=2, ensure_ascii=False))
        return 0
    except (OSError, ValueError, KeyError) as error:
        parser.exit(2, f"OA run error: {error}\n")


if __name__ == "__main__":
    raise SystemExit(main())
