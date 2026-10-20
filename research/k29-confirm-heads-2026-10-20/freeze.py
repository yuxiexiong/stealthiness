"""Bind the finite contract, runbook and exact source; no model or GPU imports."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys


def inventory(root):
    files = [root / n for n in ("contract.json", "CONTRACT.zh-CN.md", "RUNBOOK.md", "TASK_PLAN.json", "VENDORED_SOURCES.json", "freeze.py")]
    files += sorted((root / "code").rglob("*.py")) + sorted((root / "tests").rglob("*.py"))
    return {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}


def verify(root):
    manifest = json.loads((root / "MANIFEST.json").read_text())
    if inventory(root) != manifest["files_sha256"]:
        raise ValueError("Frozen file content or source inventory changed")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    sys.path.insert(0, str(root / "code"))
    import protocol
    contract = json.loads((root / "contract.json").read_text())
    plan = protocol.task_plan(contract)
    if plan != json.loads((root / "TASK_PLAN.json").read_text()):
        raise ValueError("Task plan is inconsistent with the contract")
    if args.write:
        hashes = inventory(root)
        manifest = {"schema": 1, "frozen_utc": datetime.now(timezone.utc).isoformat(),
                    "freeze_kind": "exact file hashes; does not claim a Git commit or GPU approval",
                    "contract_sha256": hashes["contract.json"], "files_sha256": hashes,
                    "GPU_jobs_submitted": False, "GPU_acceptance_pending": True}
        (root / "MANIFEST.json").write_text(json.dumps(manifest, indent=2) + "\n")
    result = verify(root)
    print(json.dumps({"verified": True, "files": len(result["files_sha256"]),
                      "contract_sha256": result["contract_sha256"], "tasks": len(plan)}))


if __name__ == "__main__":
    main()
