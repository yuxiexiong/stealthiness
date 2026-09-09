"""Complete source attribution on saved probe outputs; never generate or rescore."""
import argparse
from datetime import datetime, timezone
import gc
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import tarfile
import time

import run_probe as probe


def complete_sample(model, tokenizer, path):
    saved = json.loads(path.read_text())
    meter = probe.Meter(model, saved["measurements"])
    try:
        for condition, trajectory in saved["trajectories"].items():
            if not probe.completed(trajectory):
                raise ValueError("Full attribution requires an already generated trajectory")
            prompt, y = saved["inputs"][condition], trajectory["output_ids"]
            for target in range(len(y)):
                attrs = trajectory["attributions"]
                if probe.completed(attrs.get(str(target))):
                    continue
                attr = probe.capture(meter, f"attribution_all/{condition}/{target}",
                    lambda: probe.source_attribution(model, tokenizer, prompt, y, target), backward=True)
                attrs[str(target)] = attr
                if not probe.completed(attr):
                    raise RuntimeError(f"{path.stem}/{condition}/{target}: {attr.get('error')}")
                attr["prefix_vs_sequence_logprob_delta"] = (
                    attr["log_prob"] - trajectory["scores"][condition]["log_probs"][target])
                # Save bounded chunks; a hard interruption can repeat at most 25 targets.
                if (target + 1) % 25 == 0:
                    probe.oa.write_json(path, saved)
            probe.oa.write_json(path, saved)
        saved["attribution_scope"] = "all_saved_output_positions_native_condition"
    finally:
        probe.oa.write_json(path, saved)
    return saved


def coverage(paths):
    total = done = failed = 0
    for path in paths:
        saved = json.loads(path.read_text())
        for tr in saved["trajectories"].values():
            for i in range(len(tr["output_ids"])):
                total += 1
                attr = tr["attributions"].get(str(i), {})
                done += probe.completed(attr)
                failed += attr.get("status") == "failed"
    return {"output_positions": total, "completed": done,
            "failed": failed, "missing": total - done - failed}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--local-assets", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    args.command, args.allow_download, args.cache_dir = "run", False, None
    args.assets = json.loads(args.local_assets.read_text())
    root = args.root.resolve()
    previous = json.loads((root / "run.json").read_text())
    if previous["status"] != "completed" or previous["contract"]["command"] != "run":
        raise ValueError("Complete the original formal run first")
    sha = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
    if sha(Path(probe.__file__)) != previous["contract"]["runner_sha256"]:
        raise ValueError("Original measurement code changed; do not mix attribution definitions")
    identity_path = probe.HERE / "LLAMA_BASE_IDENTITY.json"
    if previous["contract"]["local_assets"] != {
            "manifest_sha256": sha(args.local_assets), "identity_sha256": sha(identity_path)}:
        raise ValueError("Use the same verified local assets as the original run")
    invocation = next(v for v in reversed(previous["invocations"]) if v["status"] == "completed")
    versions = {name: importlib.metadata.version(name) for name in invocation["versions"]}
    import torch
    runtime = {"torch_cuda": torch.version.cuda, "ld_preload": os.environ.get("LD_PRELOAD", "")}
    if versions != invocation["versions"] or runtime != invocation["cuda_runtime"]:
        raise ValueError("Use the original dependency and CUDA environment")
    if args.device != invocation["device"]:
        raise ValueError("Use the original device for this supplement")
    records = probe.read_manifest(root / "manifest.json")
    paths = [root / state / f"{r['sample_id']}.json" for state in probe.STATES for r in records]
    contract = {"source_run_sha256": sha(root / "run.json"),
                "supplement_runner_sha256": sha(Path(__file__)),
                "scope": "all_saved_output_positions_native_condition", "versions": versions,
                "cuda_runtime": runtime, "device": args.device}
    report_path = root / "full_attribution.json"
    if args.resume:
        report = json.loads(report_path.read_text())
        if report["contract"] != contract:
            raise ValueError("Supplement resume contract differs")
    else:
        if report_path.exists():
            parser.error("Supplement already exists; use --resume")
        backup = root / "sparse-source-attributions.tar.gz"
        with tarfile.open(backup, "x:gz") as archive:
            for path in paths + [root / "run.json", root / "manifest.json"]:
                archive.add(path, arcname=str(path.relative_to(root)))
        report = {"contract": contract, "initial_coverage": coverage(paths),
                  "source_files_sha256": {str(p.relative_to(root)): sha(p) for p in paths},
                  "backup": backup.name, "invocations": []}
    attempt = {"started_utc": datetime.now(timezone.utc).isoformat(), "status": "running"}
    report["invocations"].append(attempt)
    report["status"] = "running"
    probe.oa.write_json(report_path, report)
    started = time.perf_counter()
    try:
        attempt["base_verification"] = probe.verify_local_base(
            Path(args.assets["base"]["path"]), json.loads(identity_path.read_text()))
        torch.manual_seed(probe.SEED)
        for state in probe.STATES:
            state_paths = [p for p in paths if p.parent.name == state]
            counts = coverage(state_paths)
            if counts["completed"] == counts["output_positions"]:
                continue
            model, tokenizer = probe.load_model(state, args)
            for path in state_paths:
                complete_sample(model, tokenizer, path)
                counts = coverage([path])
                print(f"{state}/{path.stem}: {counts}", flush=True)
                report["coverage"] = coverage(paths)
                probe.oa.write_json(report_path, report)
            del model, tokenizer
            gc.collect()
            torch.cuda.empty_cache()
        report["coverage"] = coverage(paths)
        if report["coverage"]["completed"] != report["coverage"]["output_positions"]:
            raise RuntimeError("Full output-position coverage was not achieved")
        report["status"] = attempt["status"] = "completed"
    except BaseException as error:
        report["status"] = attempt["status"] = "failed_or_interrupted"
        attempt["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        attempt["elapsed_seconds"] = time.perf_counter() - started
        attempt["allocated_gpu_hours"] = attempt["elapsed_seconds"] / 3600
        report["coverage"] = coverage(paths)
        probe.oa.write_json(report_path, report)


if __name__ == "__main__":
    main()
