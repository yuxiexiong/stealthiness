"""Run every CPU test with CUDA hidden and write a receipt binding the exact source tree."""
import io
import json
import os
import sys
import time
import unittest
from pathlib import Path

CODE = Path(__file__).resolve().parent
os.environ["CUDA_VISIBLE_DEVICES"] = ""
sys.path.insert(0, str(CODE))
sys.path.insert(0, str(CODE / "tests"))

import jsd_common as common  # noqa: E402


def main():
    started = time.monotonic()
    suite = unittest.defaultTestLoader.discover(str(CODE / "tests"), pattern="test_*.py")
    stream = io.StringIO()
    result = unittest.TextTestRunner(stream=stream, verbosity=2).run(suite)
    cuda_initialized = "torch" in sys.modules and sys.modules["torch"].cuda.is_initialized()
    code_sha = {str(p.relative_to(CODE)): common.file_hash(p) for p in sorted(CODE.rglob("*.py")) if "__pycache__" not in p.parts}
    receipt = {"passed": result.wasSuccessful() and result.testsRun > 0 and not result.skipped and not cuda_initialized,
               "tests_run": result.testsRun, "failures": len(result.failures), "errors": len(result.errors),
               "skipped": len(result.skipped), "cuda_initialized": cuda_initialized, "CUDA_VISIBLE_DEVICES": "",
               "contract_sha256": common.file_hash(common.CONTRACT_PATH), "code_sha256": code_sha,
               "python": sys.version.split()[0], "wall_seconds": time.monotonic() - started,
               "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    try:
        import torch, transformers, peft  # noqa: E401
        receipt["versions"] = {"torch": torch.__version__, "transformers": transformers.__version__, "peft": peft.__version__}
    except ImportError:
        pass
    output = Path(sys.argv[1]) if len(sys.argv) > 1 else CODE.parent / "cpu_tests_passed.json"
    common.atomic_json(output, receipt)
    (output.parent / "cpu_test_log.txt").write_text(stream.getvalue())
    print(json.dumps({k: receipt[k] for k in ("passed", "tests_run", "failures", "errors", "skipped", "wall_seconds")}))
    raise SystemExit(0 if receipt["passed"] else 1)


if __name__ == "__main__":
    main()
