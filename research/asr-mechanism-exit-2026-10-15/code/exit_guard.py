"""Standard-library guards shared by the Qwen and the LLaVA jobs: exact contract + source inventory, GPU0 worker ancestry."""
import hashlib
import json
import os
from pathlib import Path


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def guard(root):
    own = read(root / "manifest.json")
    if sha(root / "contract.json") != own["contract_sha256"]:
        raise ValueError("Contract identity changed")
    actual = {str(p.relative_to(root / "code")): sha(p) for p in sorted((root / "code").rglob("*.py"))}
    if actual != own["code_sha256"]:
        raise ValueError("Source inventory changed")
    return own


def guard_frozen(root, fork_root):
    """The copies of the fork experiment's frozen modules must be byte-identical to the fork manifest."""
    fork = read(fork_root / "manifest.json")
    for name in read(root / "manifest.json")["frozen_modules_must_match_fork_manifest"]:
        if sha(root / "code" / name) != fork["code_sha256"][name]:
            raise ValueError("Copy of a frozen module differs from the fork experiment: " + name)
    return fork


def guard_worker(root):
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "0" or os.environ.get("JOBQ_GPU") != "0":
        raise ValueError("Only the physical GPU0 queue worker may run this task")
    worker = read(root / "worker_pid.json")["pid"]
    current = os.getpid()
    for _ in range(24):
        if current == worker:
            return worker
        if current <= 1:
            break
        current = int(Path("/proc/%s/status" % current).read_text().split("PPid:\t", 1)[1].splitlines()[0])
    raise ValueError("Recorded GPU0 worker is not a process ancestor")


def atomic_json(path, value):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".partial")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=1) + "\n")
    tmp.replace(path)
