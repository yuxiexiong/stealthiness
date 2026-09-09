"""Synthetic OA operator benchmarks; never a model-quality or end-to-end result.

Examples (run each position separately, on the allocated GPU):
  python experiments/oa_gpu_bench.py --mode pinv --output runs/pinv.json
  python experiments/oa_gpu_bench.py --mode ted-regression --output runs/ted-check.json
  python experiments/oa_gpu_bench.py --mode ted-timing --position generation --output runs/ted-time.json
  python experiments/oa_gpu_bench.py --mode ted-fit-kernel --position generation --cpu-threads 16 --output runs/ted-fit.json

TED imports the pinned upstream package normally, including its dependencies.
pinv requires only torch. --help and --dry-run require only Python's stdlib.
"""
from __future__ import annotations

import argparse
import importlib
import importlib.metadata
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
CUP_COMMIT = "1fd0c4fcf5e0b7a3e9c7024fa4119c9643853ce6"


def save(path, report):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def plan(args):
    tokens = 1 if args.position == "input" else 8
    return {
        "benchmark": args.mode, "synthetic": True, "quality_validation": False,
        "position": args.position, "seed": args.seed, "device": args.device,
        "warmup": args.warmup, "repeats": args.repeats,
        "timeout_seconds": args.timeout_seconds, "cache_policy": args.cache,
        "cupbearer_commit": CUP_COMMIT if args.mode.startswith("ted-") else None,
        "shape": ({"matrix": [4096, 4096], "dtype": "float64", "hermitian": True,
                   "rcond": 1e-5} if args.mode == "pinv" else
                  {"layers": 4 if args.mode == "ted-regression" else 32,
                   "trusted": 32 if args.mode == "ted-regression" else 512,
                   "hidden_dim": 16 if args.mode == "ted-regression" else 4096,
                   "tokens": tokens, "neighbors": 10,
                   "queries": 512 if args.mode == "ted-fit-kernel" else
                              2 if args.mode == "ted-regression" else 1}),
        "limitations": [
            "Fixed synthetic inputs, not model activations, model quality, or end-to-end OA time.",
            "No language model is loaded; production memory contention and spectra may differ.",
            "A whole-worker timeout terminates even a blocked native operator; incomplete stages have no duration.",
        ],
    }


def load_ted(args):
    root = args.cupbearer.resolve()
    actual = subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip()
    if actual != CUP_COMMIT:
        raise ValueError(f"Expected cupbearer {CUP_COMMIT}, got {actual}")
    changes = subprocess.check_output(["git", "-C", str(root), "diff", "--name-only", "HEAD"], text=True)
    if changes.strip():
        raise ValueError("Pinned cupbearer has local tracked changes")
    sys.path.insert(0, str(root / "src"))
    module = importlib.import_module("cupbearer.detectors.statistical.ted_detector")
    if Path(module.__file__).resolve() != root / "src/cupbearer/detectors/statistical/ted_detector.py":
        raise ValueError("TED import came from a different checkout")
    return module.TEDDetector, module.PCA


def worker(args, report):
    import torch

    if args.cpu_threads is not None:
        torch.set_num_threads(args.cpu_threads)
    torch.manual_seed(args.seed)
    gpu = args.mode != "ted-fit-kernel"
    device = torch.device(args.device)
    if gpu:
        if device.type != "cuda" or not torch.cuda.is_available():
            raise RuntimeError("This mode requires an available CUDA device")
        torch.cuda.set_device(device)
    versions = {}
    for package in ["torch", "numpy", "pyod", "scikit-learn"]:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    report.update(status="running", versions=versions, python=sys.version,
                  threads={"torch_intraop": torch.get_num_threads(),
                           "torch_interop": torch.get_num_interop_threads(),
                           "environment": {key: os.environ.get(key) for key in
                                           ["OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"]}},
                  cuda_version=torch.version.cuda,
                  gpu_name=torch.cuda.get_device_name(device) if gpu else None,
                  stages=[], timings={})
    save(args.output, report)

    def stage(name, fn, *, cuda=False):
        record = {"name": name, "status": "running", "cuda_synchronized": cuda}
        report["stages"].append(record)
        save(args.output, report)
        print(f"START {name}", flush=True)
        if cuda:
            torch.cuda.synchronize(device)
            torch.cuda.reset_peak_memory_stats(device)
        started = time.perf_counter()
        value = fn()
        if cuda:
            torch.cuda.synchronize(device)
        record.update(status="complete", seconds=time.perf_counter() - started,
                      peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(device) if cuda else None,
                      peak_cuda_reserved_bytes=torch.cuda.max_memory_reserved(device) if cuda else None)
        save(args.output, report)
        print(f"DONE {name}: {record['seconds']:.6f}s", flush=True)
        return value

    def repeat(name, fn, *, cuda=False):
        for index in range(args.warmup):
            stage(f"{name}/warmup-{index}", fn, cuda=cuda)
        durations = []
        for index in range(args.repeats):
            result = stage(f"{name}/repeat-{index}", fn, cuda=cuda)
            durations.append(report["stages"][-1]["seconds"])
        report["timings"][name] = {"samples_seconds": durations,
            "mean_seconds": statistics.mean(durations), "median_seconds": statistics.median(durations),
            "min_seconds": min(durations), "max_seconds": max(durations)}
        save(args.output, report)
        return result

    def equality(first, second):
        layers = {name: {"exact_equal": torch.equal(first[name], second[name]),
                        "max_absolute_difference": (first[name] - second[name]).abs().max().item(),
                        "finite": bool(torch.isfinite(first[name]).all() and torch.isfinite(second[name]).all())}
                  for name in first}
        report["cache_score_regression"] = {"layers": layers,
            "exact_equal_all_layers": all(row["exact_equal"] and row["finite"] for row in layers.values()),
            "scope": "synthetic scores only; no calibration or model-quality claim"}
        save(args.output, report)
        if not report["cache_score_regression"]["exact_equal_all_layers"]:
            raise AssertionError("CPU/model-cache synthetic TED scores are not finite and exactly equal")

    if args.mode == "pinv":
        def covariance():
            matrix = torch.randn(4096, 4096, device=device, dtype=torch.float64)
            return matrix @ matrix.T / 4096 + 0.05 * torch.eye(4096, device=device, dtype=torch.float64)
        matrix = stage("synthetic-covariance", covariance, cuda=True)
        output = repeat("pinv", lambda: torch.linalg.pinv(matrix, rcond=1e-5, hermitian=True), cuda=True)
        report["output_finite"] = bool(torch.isfinite(output).all())
        report["limitations"].append("Positive-definite random covariance, not a measured OA covariance spectrum.")
    else:
        TEDDetector, PCA = stage("import-pinned-ted", lambda: load_ted(args))
        import numpy as np

        shape = report["shape"]
        names = [f"layer_{index}" for index in range(shape["layers"])]
        detector = TEDDetector(activation_names=names, n_neighbors=10, store_acts_on_cpu=True,
                               max_seq_len=shape["tokens"],
                               truncate_seq_at="start" if args.position == "input" else "end")
        generator = torch.Generator(device="cpu").manual_seed(args.seed)

        def raw(count):
            return torch.randn(count, shape["tokens"], shape["hidden_dim"],
                               generator=generator, dtype=torch.float32).to(torch.bfloat16)

        def synthetic_pca():
            ranks = np.random.default_rng(args.seed).integers(0, 512, size=(5120, 31)).astype(np.float32)
            return PCA(contamination=0.1).fit(ranks)

        if args.mode == "ted-fit-kernel":
            detector.max_seq_len_seen = {names[0]: shape["tokens"]}
            reference = stage("prepare-reference", lambda: detector._prepare_activation(raw(512), names[0]))
            report["fit_kernel"] = {"query_shape": list(reference.shape), "reference_shape": list(reference.shape),
                "dtype": str(reference.dtype), "neighbor_indices_shape": [512, 1], "topk": 11,
                "full_finalize_calls_per_position": {"topk": 32, "rankings": 9920, "pca_fit": 32},
                "full_finalize_sorted_rows_per_position": 5079040,
                "pca_input_shape": [5120, 31], "pca_input_dtype": "float32"}
            # Upstream finalize inlines the same normalize/mm/topk sequence with k+1.
            detector.n_neighbors = 11
            neighbors = repeat("cpu-fit-topk-k11", lambda: detector._find_k_nearest(reference, reference))
            repeat("cpu-fit-rankings", lambda: detector._get_neighbor_rankings(reference, reference, neighbors[:, 1:2]))
            stage("cpu-pca-fit-5120x31", synthetic_pca)
            report["limitations"].append("CPU fit-kernel sample only: 9920 ranking calls and 32 PCA fits are NOT executed; extrapolation need not be linear.")
        else:
            if args.mode == "ted-regression":
                trusted = {name: raw(shape["trusted"]) for name in names}
                stage("real-small-cpu-ted-fit", lambda: detector._train(
                    trusted_dataloader=[(None, trusted)], untrusted_dataloader=None, pbar=False))
                del trusted
            else:
                def prepare_reference():
                    detector.max_seq_len_seen = {name: shape["tokens"] for name in names}
                    for name in names:
                        detector.clean_activations[name] = detector._prepare_activation(raw(512), name)
                stage("prepare-full-cpu-cache", prepare_reference)
                pca = stage("synthetic-pca-fit-5120x31", synthetic_pca)
                detector.pca_detectors = {name: pca for name in names}
                report["limitations"].append("Full-shape scoring uses one shared PCA fitted on synthetic ranks; actual full TED finalization is skipped.")
            features = {name: raw(shape["queries"]).to(device) for name in names}
            before = {name: {"dtype": str(value.dtype), "device": str(value.device),
                             "shape": list(value.shape), "bytes": value.numel() * value.element_size()}
                      for name, value in detector.clean_activations.items()}
            report["cache_before"] = before
            report["cache_bytes"] = sum(value["bytes"] for value in before.values())
            scores = {}
            policies = ["cpu", "model"] if args.mode == "ted-regression" or args.cache == "both" else [args.cache]
            for policy in policies:
                if policy == "model":
                    def move_cache():
                        for name, value in detector.clean_activations.items():
                            detector.clean_activations[name] = value.to(device=device)
                    stage("move-final-cache-once", move_cache, cuda=True)
                scores[policy] = repeat(f"ted-score-cache-{policy}",
                    lambda: detector._compute_layerwise_scores(None, features), cuda=True)
            report["cache_after"] = {name: {"dtype": str(value.dtype), "device": str(value.device)}
                                     for name, value in detector.clean_activations.items()}
            if set(scores) == {"cpu", "model"}:
                equality(scores["cpu"], scores["model"])
    report["status"] = "complete"
    save(args.output, report)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--mode", required=True, choices=["pinv", "ted-regression", "ted-timing", "ted-fit-kernel"])
    parser.add_argument("--position", choices=["input", "generation"], default="input")
    parser.add_argument("--cache", choices=["cpu", "model", "both"], default="both", help="ted-timing only; regression always compares both")
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--cupbearer", type=Path, default=ROOT / "external/cupbearer")
    parser.add_argument("--cpu-threads", type=int)
    parser.add_argument("--timeout-seconds", type=float, default=600)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--dry-run", action="store_true", help="Print intended work; no imports, writes, or device operations")
    parser.add_argument("--_worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.warmup < 0 or args.repeats < 1 or args.timeout_seconds <= 0 or (args.cpu_threads is not None and args.cpu_threads < 1):
        parser.error("Require warmup >= 0, repeats >= 1, timeout > 0, cpu-threads >= 1")
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
            save(args.output, report)
            raise
        return 0
    if args.output.exists():
        parser.error("Refusing to overwrite an existing output")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    report["status"] = "starting"
    save(args.output, report)
    command = [sys.executable, str(Path(__file__).resolve()), *(sys.argv[1:] if argv is None else argv), "--_worker"]
    started = time.perf_counter()
    try:
        result = subprocess.run(command, timeout=args.timeout_seconds)
        returncode = result.returncode
    except subprocess.TimeoutExpired:
        returncode = 124
    report = json.loads(args.output.read_text())
    report["worker_wall_seconds"] = time.perf_counter() - started
    report["worker_returncode"] = returncode
    if returncode == 124:
        report.update(status="timeout", error="Worker terminated by benchmark wall-time limit")
    elif returncode and report["status"] not in ("failed", "timeout"):
        report.update(status="failed", error=f"Worker exited with code {returncode}")
    save(args.output, report)
    return returncode


if __name__ == "__main__":
    raise SystemExit(main())
