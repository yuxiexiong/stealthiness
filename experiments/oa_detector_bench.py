"""Time original single-layer Beatrix/VAE operations on synthetic GPU features.

No base model is loaded. --help, --dry-run and --self-check need only stdlib.
Run input and generation separately, for example:
  python experiments/oa_detector_bench.py --mode beatrix --position generation --output runs/beatrix.json
  python experiments/oa_detector_bench.py --mode vae --position input --output runs/vae.json
"""
from __future__ import annotations

import argparse
import importlib
import importlib.metadata
import json
from pathlib import Path
import statistics
import subprocess
import sys
import time

import oa_gpu_bench as shared


def plan(args):
    return {
        "benchmark": args.mode, "synthetic": True, "quality_validation": False,
        "position": args.position, "seed": args.seed, "device": args.device,
        "warmup": args.warmup, "repeats": args.repeats,
        "timeout_seconds": args.timeout_seconds, "cupbearer_commit": shared.CUP_COMMIT,
        "shape": {"batch_size": 1, "tokens": 1 if args.position == "input" else 8,
                  "hidden_dim": 4096, "feature_dtype": "bfloat16", "measured_layers": 1,
                  "full_evaluation_layers": 32},
        "limitations": [
            "Synthetic random activations; no Llama model, real-data quality or complete evaluation is measured.",
            "One real upstream layer is measured; multiplying by 32 is a conditional estimate, not a measured full-stage time or memory peak.",
            "Warmup and repeats continue updating the synthetic detector; these are not independently fitted detectors.",
            "No 512-sample fitting, 1536-sample scoring, model memory contention, loader, or language-model forward is included.",
        ],
    }


def worker(args, report):
    import torch

    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("An available CUDA device is required")
    torch.cuda.set_device(device)
    if args.cpu_threads is not None:
        torch.set_num_threads(args.cpu_threads)
    torch.manual_seed(args.seed)
    # Reuse the existing exact-commit/clean-checkout and import-provenance checks.
    shared.load_ted(args)
    report.update(status="running", python=sys.version, torch_version=torch.__version__,
                  cuda_version=torch.version.cuda, gpu_name=torch.cuda.get_device_name(device),
                  torch_threads=torch.get_num_threads(), stages=[], timings={})
    for package in ("lightning", "numpy"):
        report[package + "_version"] = importlib.metadata.version(package)
    shared.save(args.output, report)

    def measure(name, fn, warmup=None, repeats=None):
        warmup = args.warmup if warmup is None else warmup
        repeats = args.repeats if repeats is None else repeats
        durations = []
        for index in range(warmup + repeats):
            record = {"name": name, "kind": "warmup" if index < warmup else "repeat",
                      "index": index, "status": "running", "cuda_synchronized": True}
            report["stages"].append(record)
            shared.save(args.output, report)
            print(f"START {name}/{record['kind']}-{index}", flush=True)
            torch.cuda.synchronize(device)
            torch.cuda.reset_peak_memory_stats(device)
            started = time.perf_counter()
            value = fn()
            torch.cuda.synchronize(device)
            record.update(status="complete", seconds=time.perf_counter() - started,
                          peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(device),
                          peak_cuda_reserved_bytes=torch.cuda.max_memory_reserved(device))
            shared.save(args.output, report)
            print(f"DONE {name}: {record['seconds']:.6f}s", flush=True)
            if index >= warmup:
                durations.append(record["seconds"])
        report["timings"][name] = {"samples_seconds": durations,
            "mean_seconds": statistics.mean(durations), "median_seconds": statistics.median(durations),
            "min_seconds": min(durations), "max_seconds": max(durations)}
        shared.save(args.output, report)
        return value

    def raw():
        shape = (1, 4096) if args.position == "input" else (1, 8, 4096)
        return {"layer0": torch.randn(shape, device=device, dtype=torch.float32).to(torch.bfloat16)}

    # Random-number generation is outside the timed detector operations.
    training = iter([raw() for _ in range(args.warmup + args.repeats)])
    query = raw()
    if args.mode == "beatrix":
        module = importlib.import_module("cupbearer.detectors.statistical.beatrix_detector")
        detector = module.BeatrixDetector(activation_names=["layer0"])
        detector.init_variables((None, query), case="trusted")
        report["original_calls"] = ["BeatrixDetector.batch_update",
            "BeatrixDetector._finalize_training", "BeatrixDetector._compute_layerwise_scores"]
        report["operator_scope"] = {
            "powers": detector.power_list, "moving_average": detector.moving_average,
            "gram_shape_per_power": [1, 4096, 4096], "triangle_elements": 4096 * 4097 // 2,
            "triangle_includes_diagonal": True, "index_creation_included": True,
            "fit_calls_per_full_position": 512 * 32, "score_calls_per_full_position": 1536 * 32,
            "fit_state": "Only warmup+repeats synthetic batches; not a 512-sample fitted baseline",
        }
        with torch.no_grad():
            measure("beatrix-layer-update", lambda: detector.batch_update(next(training), case="trusted"))
            measure("beatrix-layer-finalize", detector._finalize_training, warmup=0, repeats=1)
            scores = measure("beatrix-layer-score", lambda: detector._compute_layerwise_scores(None, query))
    else:
        vae_module = importlib.import_module("cupbearer.detectors.feature_model.vae")
        feature_module = importlib.import_module("cupbearer.detectors.feature_model.feature_model_detector")
        vae = vae_module.VAE(4096, 1024)
        feature_model = vae_module.VAEFeatureModel({"layer0": vae})
        module = feature_module.FeatureModelModule(feature_model, lr=1e-3).to(device)
        optimizer = module.configure_optimizers()
        report["original_calls"] = ["VAE", "VAEFeatureModel.forward",
            "FeatureModelModule._shared_step", "FeatureModelModule.configure_optimizers", "torch.optim.Adam.step"]
        report["operator_scope"] = {
            "parameters_measured": sum(p.numel() for p in module.parameters()),
            "parameter_dtype": str(next(module.parameters()).dtype),
            "autocast_dtype": str(torch.get_autocast_gpu_dtype()),
            "loss_divisor": 32, "loss_divisor_reason": "Preserve this layer's contribution to the full 32-layer mean loss",
            "optimizer": type(optimizer).__name__, "lr": 1e-3,
            "fit_updates_per_full_position": 512, "layer_updates_per_full_position": 512 * 32,
            "score_calls_per_full_position": 1536 * 32,
            "state_initialization": "The first warmup includes Adam state allocation; stable repeats do not",
        }

        def update():
            optimizer.zero_grad()
            with torch.autocast(device_type="cuda"):
                loss, _ = module._shared_step((None, next(training)))
                loss = loss / 32
            loss.backward()
            optimizer.step()
            return loss.detach()

        loss = measure("vae-layer-update", update)
        report["last_training_loss_finite"] = bool(torch.isfinite(loss).all())
        module.eval()
        with torch.no_grad(), torch.autocast(device_type="cuda"):
            scores = measure("vae-layer-score", lambda: feature_model(None, query))
        report["limitations"].append("Single-layer Adam timings do not measure the full ensemble's optimizer batching, memory pressure or Lightning orchestration.")
    report["scores"] = {name: {"shape": list(value.shape), "finite": bool(torch.isfinite(value).all())}
                        for name, value in scores.items()}
    if not all(value["finite"] for value in report["scores"].values()) or not report.get("last_training_loss_finite", True):
        raise RuntimeError("Synthetic detector produced a nonfinite loss or score")
    report["status"] = "complete"
    shared.save(args.output, report)


def self_check():
    import tempfile

    script = str(Path(__file__).resolve())
    with tempfile.TemporaryDirectory() as folder:
        output = Path(folder) / "result.json"
        for mode in ("beatrix", "vae"):
            command = [sys.executable, script, "--mode", mode, "--position", "generation",
                       "--output", str(output)]
            result = subprocess.run(command + ["--dry-run"], capture_output=True, text=True, check=True)
            report = json.loads(result.stdout)
            assert report["synthetic"] and not report["quality_validation"]
            assert report["shape"]["tokens"] == 8 and report["shape"]["measured_layers"] == 1
            assert not output.exists(), "dry-run wrote an output"
        output.write_text("keep me")
        result = subprocess.run(command, capture_output=True, text=True)
        assert result.returncode != 0 and "overwrite" in result.stderr
        assert output.read_text() == "keep me", "existing results were overwritten"
    print("CPU self-check passed: dry-run isolation and output protection")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--mode", choices=["beatrix", "vae"])
    parser.add_argument("--position", choices=["input", "generation"], default="input")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--cpu-threads", type=int)
    parser.add_argument("--cupbearer", type=Path, default=shared.ROOT / "external/cupbearer")
    parser.add_argument("--timeout-seconds", type=float, default=600)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--self-check", action="store_true")
    parser.add_argument("--_worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.self_check:
        self_check()
        return 0
    if args.mode is None or args.warmup < 1 or args.repeats < 1 or args.timeout_seconds <= 0 or (args.cpu_threads is not None and args.cpu_threads < 1):
        parser.error("Require mode, warmup >= 1, repeats >= 1, timeout > 0 and cpu-threads >= 1")
    report = plan(args)
    if args.dry_run:
        print(json.dumps(report, indent=2))
        return 0
    if args.output is None:
        parser.error("--output is required unless --dry-run")
    if args._worker:
        try:
            worker(args, report)
        except Exception as error:
            report.update(status="failed", error=f"{type(error).__name__}: {error}")
            shared.save(args.output, report)
            raise
        return 0
    if args.output.exists():
        parser.error("Refusing to overwrite an existing output")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    report["status"] = "starting"
    shared.save(args.output, report)
    command = [sys.executable, str(Path(__file__).resolve()), *(sys.argv[1:] if argv is None else argv), "--_worker"]
    started = time.perf_counter()
    try:
        returncode = subprocess.run(command, timeout=args.timeout_seconds).returncode
    except subprocess.TimeoutExpired:
        returncode = 124
    report = json.loads(args.output.read_text())
    report.update(worker_wall_seconds=time.perf_counter() - started, worker_returncode=returncode)
    if returncode == 124:
        report.update(status="timeout", error="Worker terminated by benchmark wall-time limit")
    elif returncode and report["status"] not in ("failed", "timeout"):
        report.update(status="failed", error=f"Worker exited with code {returncode}")
    shared.save(args.output, report)
    return returncode


if __name__ == "__main__":
    raise SystemExit(main())
