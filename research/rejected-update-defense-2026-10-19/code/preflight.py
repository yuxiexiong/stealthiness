"""CPU-only acceptance of the frozen package, data and existing references."""
import importlib.metadata
import json
import os
from pathlib import Path
import sys
import unittest

from guard import atomic_json, sha, verify_files


def main(root):
    root = Path(root).resolve()
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise ValueError("CPU preflight requires CUDA_VISIBLE_DEVICES to be explicitly empty")
    receipt_path = root / "cpu_tests_passed.json"
    receipt_path.unlink(missing_ok=True)
    import torch
    if torch.cuda.is_initialized():
        raise ValueError("CUDA was already initialized before CPU acceptance")
    manifest_sha = sha(root / "manifest.json")
    contract = verify_files(root)
    if Path(sys.executable).absolute() != Path(contract["python"]).absolute():
        raise ValueError("Interpreter differs from the frozen contract")
    for name, expected in contract["runtime_versions"].items():
        if importlib.metadata.version(name).split("+")[0] != expected:
            raise ValueError("Runtime version changed: " + name)
    tasks, references = {}, 0
    for task, spec in contract["tasks"].items():
        counts = {}
        expected_counts = {"train": 20000, "validation": 60, "confirmation": 64,
                           "oldqa": 76 if task == "R" else 60}
        for kind, count in expected_counts.items():
            name = kind + ".jsonl"
            path = Path(spec["data_root"]) / "private/data" / name
            if sha(path) != spec["data_sha256"][name]:
                raise ValueError("Registered data changed: " + str(path))
            with path.open() as stream:
                rows = [json.loads(line) for line in stream if line.strip()]
            poison = sum(bool(row.get("poison")) for row in rows)
            if len(rows) != count or poison != (200 if kind == "train" else 0):
                raise ValueError("Registered data counts differ: " + task + "/" + kind)
            counts[kind] = len(rows)
        tasks[task] = counts
        for reference in spec["references"].values():
            folder = Path(reference["path"])
            if not folder.is_absolute():
                raise ValueError("Reference path must be absolute")
            for name, expected in reference["sha256"].items():
                if sha(folder / name) != expected:
                    raise ValueError("Registered reference changed: " + str(folder / name))
            references += 1
    if references != contract["reused_references"]:
        raise ValueError("Reference inventory differs from the frozen contract")
    sys.path.insert(0, str(root / "code"))
    suite = unittest.defaultTestLoader.discover(str(root / "code/tests"))
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    if not result.wasSuccessful() or result.skipped or result.testsRun == 0:
        raise ValueError("CPU tests failed, skipped, or were not discovered")
    if torch.cuda.is_initialized():
        raise ValueError("CPU acceptance initialized CUDA")
    if sha(root / "manifest.json") != manifest_sha:
        raise ValueError("Package manifest changed during CPU acceptance")
    verify_files(root)
    atomic_json(receipt_path, {"passed": True, "tests": result.testsRun,
                              "cuda_initialized": False, "manifest_sha256": manifest_sha,
                              "source_files_verified": len(contract["sources_sha256"]),
                              "references_verified": references, "data_rows": tasks})


if __name__ == "__main__":
    main(Path(sys.argv[1]))
