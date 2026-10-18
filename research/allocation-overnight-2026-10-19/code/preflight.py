"""CPU acceptance for exact deployed sources and all result-independent jobs."""
import hashlib
import json
from pathlib import Path
import sys
import unittest


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(2**20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main(root):
    import torch
    code = root / "code"
    sys.path.insert(0, str(code / "vendor"))
    import freeze_lib as fl
    manifest = json.loads((root / "manifest.json").read_text())
    for name, expected in manifest["sha256"].items():
        assert sha(root / name) == expected, name
    contract = json.loads((root / "contract.json").read_text())
    assert contract["frozen"] is True
    assert Path(sys.executable) == Path(contract["runtime"]["python"])
    source = contract["source"]
    fork = Path(source["fork_root"])
    for name, expected in source["data_sha256"].items():
        assert sha(fork / "private/data" / name) == expected, name
    for seed, mother in source["mothers"].items():
        assert sha(fork / "mothers" / ("s" + seed) / "state_tf.pt") == mother["state_sha256"]
    state = torch.load(fork / "mothers/s1105/state_tf.pt", map_location="cpu", weights_only=False)
    adapter = state["adapter"]
    counts = {}
    for job in contract["jobs"]:
        counts[job["arm"]] = sum(p.numel() for n, p in adapter.items()
                                  if fl.is_trainable(n, job["trainable"]))
    assert counts["B"] == counts["GATE"] == counts["UP"] == counts["DOWN"] == counts["XR2"] == counts["XR3"]
    suite = unittest.defaultTestLoader.discover(str(code / "tests"))
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    assert result.wasSuccessful() and not result.skipped
    assert not torch.cuda.is_initialized()
    receipt = {"passed": True, "tests": result.testsRun, "skipped": len(result.skipped),
               "cuda_initialized": False, "manifest_sha256": sha(root / "manifest.json"),
               "updated_parameters": counts, "torch_version": torch.__version__,
               "python": sys.executable}
    (root / "cpu_tests_passed.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt))


if __name__ == "__main__":
    main(Path(sys.argv[1]))
