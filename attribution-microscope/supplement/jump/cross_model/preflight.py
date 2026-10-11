#!/usr/bin/env python3
"""Write queue test gate only after actual tests and both GPU smoke receipts."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    args = parser.parse_args()
    root, code = args.run_root, Path(__file__).resolve().parent
    result = subprocess.run([sys.executable, "-m", "unittest", "discover", "-p", "test_*.py", "-v"],
                            cwd=code, text=True, capture_output=True)
    (root / "tests.log").write_text(result.stdout + result.stderr)
    result.check_returncode()
    if "skipped" in result.stderr:
        raise RuntimeError("Server gate requires all tests, without skipped tensor/CUDA checks")
    sd = json.loads((root / "runs/t2i_smoke/complete.json").read_text())
    llm = json.loads((root / "runs/llm_smoke/run/manifest.json").read_text())
    if not sd.get("passed") or not sd.get("parameter_changed") or llm.get("status") != "complete" or not llm.get("smoke_update_check", {}).get("passed"):
        raise RuntimeError("Both real GPU smoke tests must pass before publishing the gate")
    if shutil.disk_usage(root).free < 20 * 1024**3:
        raise RuntimeError("Less than 20 GiB free; do not launch scientific runs")
    receipt = {"passed": True, "time_utc": datetime.now(timezone.utc).isoformat(),
               "tests": result.stderr, "tiny_gpu_smokes": ["Pythia-14m", "tiny-stable-diffusion-pipe"],
               "full_profiles_validated": False,
               "code_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in code.glob("*.py")}}
    temporary = root / "tests_passed.tmp"
    temporary.write_text(json.dumps(receipt, indent=2) + "\n")
    temporary.replace(root / "tests_passed.json")
    print(json.dumps({"passed": True, "receipt": str(root / "tests_passed.json")}))


if __name__ == "__main__":
    main()
