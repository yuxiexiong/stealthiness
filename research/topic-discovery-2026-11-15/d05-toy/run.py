"""Explicit prepare/execute commands. Importing or --prepare never imports torch.

Execution is NOT authorized by an audit, flag, or this file. Obtain user authorization.
No queue, scheduler, auto-retry, resume, best-checkpoint selection, or seed replacement.
"""
import argparse
import copy
import gzip
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import time
import traceback

from data import CONFIG, READS, SEEDS, canonical, digest, encode, epoch_examples, make_block
from metrics import add_w_average, prefix_qualified, screening, summarize

HERE = Path(__file__).resolve().parent
SOURCE_FILES = ("data.py", "metrics.py", "run.py", "test_preparation.py", "estimate.py", "audit/FROZEN_INPUTS.json")


def source_hashes():
    return {name: hashlib.sha256((HERE / name).read_bytes()).hexdigest() for name in SOURCE_FILES}


def frozen_blocks():
    blocks = [make_block(s) for s in SEEDS]
    expected = json.loads((HERE / "audit/FROZEN_INPUTS.json").read_text())["blocks"]
    if {str(b["seed"]): digest(b) for b in blocks} != expected:
        raise ValueError("Regenerated data differ from reviewed inputs; do not re-freeze automatically")
    return blocks


def write_json(path, value):
    # New run directory only; per-file replacement is atomic within that directory.
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(canonical(value) + "\n")
    tmp.replace(path)


def prepare(out):
    out.mkdir(parents=True, exist_ok=False)
    manifest = dict(schema=1, config=CONFIG, blocks=frozen_blocks(),
                    sources=source_hashes(), authorization="preparation_only")
    write_json(out / "manifest.json", manifest)
    write_json(out / "identity.json", dict(sha256=digest(manifest), sources=manifest["sources"],
                                          python=sys.version, platform=platform.platform()))
    return dict(manifest_sha256=digest(manifest), seeds=list(SEEDS), training_runs=0,
                model_forwards=0, gpu_seconds=0)


def verify_manifest(path):
    manifest = json.loads(path.read_text())
    identity = json.loads(path.with_name("identity.json").read_text())
    if digest(manifest) != identity["sha256"] or manifest["sources"] != source_hashes():
        raise ValueError("Preparation identity/source mismatch; do not silently regenerate")
    if manifest["config"] != CONFIG or manifest["blocks"] != frozen_blocks():
        raise ValueError("Frozen data/config mismatch")
    return manifest


def check_resources(out, device):
    """Read-only, fail-closed preflight. No model import; no process interference."""
    parent = out.parent
    while not parent.exists():
        parent = parent.parent
    free_disk = shutil.disk_usage(parent).free
    if free_disk < 5 * 1024**3:
        raise RuntimeError("Need 5 GiB working-disk planning reserve; archive copy requires another 5 GiB")
    receipt = dict(free_disk_bytes=free_disk, device=device)
    if device == "cpu":
        return receipt
    selected = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if device != "cuda:0" or not selected or "," in selected or selected == "-1":
        raise RuntimeError("Isolate exactly one physical GPU with CUDA_VISIBLE_DEVICES, use logical cuda:0")
    gpu = subprocess.check_output(["nvidia-smi", "-i", selected,
        "--query-gpu=uuid,memory.free", "--format=csv,noheader,nounits"], text=True).strip()
    uuid, free_mib = [x.strip() for x in gpu.split(",")]
    processes = subprocess.check_output(["nvidia-smi", "--query-compute-apps=gpu_uuid,pid",
        "--format=csv,noheader,nounits"], text=True).strip().splitlines()
    if any(line.split(",")[0].strip() == uuid for line in processes):
        raise RuntimeError("Selected GPU has a compute process; no waiting/queueing or killing it")
    if int(free_mib) < 4096:
        raise RuntimeError("Selected GPU lacks 4 GiB planning reserve (not a measured model peak)")
    return dict(receipt, original_gpu_selector=selected, gpu_uuid=uuid, free_gpu_mib=int(free_mib))


def build_model(torch, vocab_size):
    nn = torch.nn

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.token = nn.Embedding(vocab_size, 64)
            self.position = nn.Embedding(9, 64)
            self.layers = nn.ModuleList([nn.TransformerEncoderLayer(
                64, 4, dim_feedforward=128, dropout=0., activation="gelu",
                layer_norm_eps=1e-5, batch_first=True, norm_first=True) for _ in range(2)])
            self.norm = nn.LayerNorm(64, eps=1e-5)
            self.head = nn.Linear(64, 2)
            # Independently initialize all matrices, including packed attention q/k/v.
            # No shared layer copies or whole-token/component initialization identity.
            with torch.no_grad():
                for name, p in self.named_parameters():
                    if p.ndim >= 2:
                        nn.init.normal_(p, std=.02)
                    elif name.endswith("bias"):
                        nn.init.zeros_(p)
                    else:
                        nn.init.ones_(p)

        def forward(self, ids):
            h = self.token(ids) + self.position(torch.arange(9, device=ids.device))[None]
            for layer in self.layers:
                h = layer(h)  # Bidirectional; all nine tokens visible, including FILL.
            return self.head(self.norm(h)[:, -1])

    return Model()


def model_hash(torch, model):
    h = hashlib.sha256()
    for name, p in model.state_dict().items():
        h.update(name.encode())
        h.update(p.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def execute(manifest_path, out, device):
    manifest = verify_manifest(manifest_path)
    # CPU synthetic regression suite, still no torch/model call.
    import unittest
    import test_preparation
    result = unittest.TextTestRunner().run(unittest.defaultTestLoader.loadTestsFromModule(test_preparation))
    if not result.wasSuccessful():
        raise RuntimeError("CPU preparation tests failed")
    resources = check_resources(out, device)
    if device.startswith("cuda"):
        # Bind CUDA to the exact NVML device checked, not an ambiguous numeric index.
        os.environ["CUDA_VISIBLE_DEVICES"] = resources["gpu_uuid"]
    out.mkdir(parents=True, exist_ok=False)
    started = time.time()
    write_json(out / "launch.json", dict(started=started, manifest=digest(manifest), device=device,
                                        sources=source_hashes(), resources=resources,
                                        execution_requires_external_user_authorization=True))
    try:
        import torch  # First model-capable import, exclusively inside explicit execution.
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.set_num_threads(4)
        torch.set_default_dtype(torch.float32)
        torch.use_deterministic_algorithms(True)
        if device.startswith("cuda"):
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
        write_json(out / "environment.json", dict(python=sys.version, torch=torch.__version__,
                   platform=platform.platform(), cuda=torch.version.cuda,
                   device_name=torch.cuda.get_device_name(device) if device.startswith("cuda") else "CPU",
                   threads=torch.get_num_threads()))
        parents, trajectories = [], []
        for block in manifest["blocks"]:
            torch.manual_seed(block["seed"])
            model = build_model(torch, len(block["vocab"])).to(device)
            prefix = train(torch, model, block, "prefix", None, 0, out / str(block["seed"]) / "prefix", device)
            parent = dict(block=block, state=copy.deepcopy(model.state_dict()),
                          cpu_rng=torch.get_rng_state(),
                          cuda_rng=torch.cuda.get_rng_state_all() if device.startswith("cuda") else None,
                          hash=model_hash(torch, model), initial_reads=prefix["20"],
                          qualified=prefix_qualified(prefix["20"]))
            parents.append(parent)
        n = sum(p["qualified"] for p in parents)
        write_json(out / "qualification.json", dict(planned=5, n=n, actual_branches=6 * n if n >= 4 else 0,
                   parents=[dict(seed=p["block"]["seed"], qualified=p["qualified"], hash=p["hash"]) for p in parents]))
        if n >= 4:
            for p in parents:
                if not p["qualified"]:
                    continue
                block, combined = p["block"], {str(ep): {} for ep in range(21)}
                for enc in ("W", "P"):
                    for dose in (0, 8, 16):
                        model = build_model(torch, len(block["vocab"])).to(device)
                        model.load_state_dict(p["state"], strict=True)
                        torch.set_rng_state(p["cpu_rng"])
                        if p["cuda_rng"] is not None:
                            torch.cuda.set_rng_state_all(p["cuda_rng"])
                        if model_hash(torch, model) != p["hash"]:
                            raise RuntimeError("Initial branch parameters differ from parent")
                        name = f"{enc}{dose}"
                        trajectory = train(torch, model, block, "branch", enc, dose,
                                           out / str(block["seed"]) / name, device,
                                           expected_initial=p["initial_reads"])
                        for ep in combined:
                            combined[ep][name] = trajectory[ep]
                trajectories.append(combined)
            write_json(out / "screening.json", screening(trajectories))
        write_json(out / "completion.json", dict(status="complete" if n >= 4 else "clean_qualification_stop",
                   n=n, wall_seconds=time.time() - started, model_runs=5 + (6 * n if n >= 4 else 0)))
    except BaseException:
        write_json(out / "failure.json", dict(wall_seconds=time.time() - started, traceback=traceback.format_exc(),
                                              auto_retry=False))
        raise


def evaluate(torch, model, block, directory, epoch, device):
    model.eval()
    rows, result, raw_hashes = block["rows"]["eval"], {}, {}
    with torch.no_grad():
        for read in READS:
            inputs = [encode(r, "P" if read == "P" else "W", block["vocab"],
                             0 if read == "P" else int(read[1])) for r in rows]
            logits = []
            for start in range(0, len(inputs), 64):
                z = model(torch.tensor(inputs[start:start + 64], dtype=torch.long, device=device))
                logits.extend(z.cpu().tolist())
            result[read] = summarize(rows, logits, block["target"])
            raw_hashes[read] = digest(logits)
            with gzip.open(directory / f"read-{epoch:02d}-{read}.jsonl.gz", "wt") as f:
                for row, z in zip(rows, logits):
                    # Stable binary softmax; preserve both raw logits and probabilities.
                    import math
                    exp0, exp1 = math.exp(z[0] - max(z)), math.exp(z[1] - max(z))
                    f.write(canonical(dict(id=row["id"], truth=row["truth"], logits=z,
                                           probabilities=[exp0 / (exp0 + exp1), exp1 / (exp0 + exp1)],
                                           prediction=int(z[1] > z[0]))) + "\n")
    result = add_w_average(result)
    result["raw_logits_hashes"] = raw_hashes
    write_json(directory / f"summary-{epoch:02d}.json", result)
    return result


def train(torch, model, block, phase, encoding, dose, directory, device, expected_initial=None):
    directory.mkdir(parents=True, exist_ok=False)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.001, betas=(.9, .999), eps=1e-8, weight_decay=0.)
    if optimizer.state:
        raise RuntimeError("Optimizer was not freshly reset")
    trajectory, start_time = {}, time.time()
    for epoch in range(21):
        if epoch:
            model.train()
            examples = list(epoch_examples(block, phase, epoch - 1, encoding, dose))
            consumed, losses = [], []
            update_start = time.time()
            for start in range(0, len(examples), 64):
                batch = examples[start:start + 64]
                ids = torch.tensor([b[1] for b in batch], dtype=torch.long, device=device)
                labels = torch.tensor([b[2] for b in batch], dtype=torch.long, device=device)
                optimizer.zero_grad(set_to_none=True)
                loss = torch.nn.functional.cross_entropy(model(ids), labels)
                if not torch.isfinite(loss):
                    raise RuntimeError("Nonfinite training loss")
                loss.backward()
                norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
                optimizer.step()
                losses.append([loss.item(), norm.item()])
                consumed.extend(b[0] for b in batch)
            write_json(directory / f"train-{epoch:02d}.json", dict(rows=consumed, losses_and_grad_norm=losses,
                       update_seconds=time.time() - update_start, encoded_input_hash=digest(examples)))
        read_start = time.time()
        trajectory[str(epoch)] = evaluate(torch, model, block, directory, epoch, device)
        if epoch == 0 and expected_initial is not None and trajectory["0"] != expected_initial:
            raise RuntimeError("Initial readout differs from common parent")
        torch.save(dict(model=model.state_dict(), optimizer=optimizer.state_dict(), epoch=epoch,
                        rng=torch.get_rng_state(),
                        cuda_rng=torch.cuda.get_rng_state_all() if device.startswith("cuda") else None),
                   directory / f"state-{epoch:02d}.pt")
        write_json(directory / f"timing-{epoch:02d}.json", dict(read_save_seconds=time.time() - read_start,
                   elapsed_seconds=time.time() - start_time))
    return trajectory


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_mutually_exclusive_group(required=True)
    actions.add_argument("--prepare", type=Path, help="data/config export only; no torch")
    actions.add_argument("--execute", type=Path, help="NEW run output directory; requires separate user authorization")
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    if args.prepare:
        print(canonical(prepare(args.prepare)))
    else:
        if args.manifest is None:
            parser.error("--execute requires --manifest from reviewed preparation")
        execute(args.manifest, args.execute, args.device)


if __name__ == "__main__":
    main()
