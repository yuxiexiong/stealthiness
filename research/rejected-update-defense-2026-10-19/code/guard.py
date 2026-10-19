"""Exact package, borrowed-source and acceptance-receipt checks."""
import hashlib
import json
from pathlib import Path


def sha(path):
    value = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(2 ** 20), b""):
            value.update(block)
    return value.hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def verify_files(root):
    """Also used before the CPU receipt exists; never imports a model runtime."""
    root = Path(root).resolve()
    manifest = json.loads((root / "manifest.json").read_text())
    entries = manifest["sha256"]
    if "contract.json" not in entries:
        raise ValueError("Manifest does not freeze contract.json")
    for name, expected in entries.items():
        relative = Path(name)
        path = (root / relative).resolve()
        if relative.is_absolute() or not path.is_relative_to(root):
            raise ValueError("Package manifest path escapes its root: " + name)
        if sha(path) != expected:
            raise ValueError("Package file changed: " + name)
    actual = {str(path.relative_to(root)) for path in (root / "code").rglob("*.py")}
    frozen = {name for name in entries if name.startswith("code/") and name.endswith(".py")}
    if actual != frozen:
        raise ValueError("Python source inventory differs from the manifest")
    contract = json.loads((root / "contract.json").read_text())
    for name, expected in contract["sources_sha256"].items():
        if not Path(name).is_absolute() or sha(name) != expected:
            raise ValueError("Borrowed source identity changed: " + name)
    return contract


def _receipt(path, manifest_sha):
    receipt = json.loads(Path(path).read_text())
    if receipt.get("passed") is not True or receipt.get("manifest_sha256") != manifest_sha:
        raise ValueError("Acceptance receipt does not cover the exact manifest: " + str(path))
    return receipt


def guard(root, task=None, require_gpu=True):
    root = Path(root).resolve()
    contract = verify_files(root)
    manifest_sha = sha(root / "manifest.json")
    receipt = _receipt(root / "cpu_tests_passed.json", manifest_sha)
    if receipt.get("cuda_initialized") is not False:
        raise ValueError("CPU acceptance did not certify CUDA remained uninitialized")
    if task is not None:
        if task not in contract["tasks"]:
            raise ValueError("Task is outside the frozen contract: " + str(task))
        if require_gpu:
            _receipt(root / ("gpu_precheck_" + task + ".json"), manifest_sha)
    return contract
