"""Download a model from ModelScope (HF is unreachable from the server) and write a receipt with per-file sha256.
python download_model.py --id Qwen/Qwen3-1.7B --out /workspace/models/Qwen3-1.7B [--revision REV]
"""
import argparse
import json
import time
from pathlib import Path

from common import file_sha256, write_json

ap = argparse.ArgumentParser()
ap.add_argument("--id", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--revision", default="master")
a = ap.parse_args()
from modelscope import snapshot_download

p = snapshot_download(a.id, revision=a.revision, local_dir=a.out)
files = {str(f.relative_to(a.out)): file_sha256(f) for f in sorted(Path(a.out).rglob("*")) if f.is_file()
         and not f.name.startswith(".") and "receipt" not in f.name}
write_json(Path(a.out) / "download_receipt.json",
           {"id": a.id, "revision": a.revision, "path": p, "time": time.strftime("%Y-%m-%dT%H:%M:%S"), "files": files})
print(json.dumps({"n_files": len(files)}))
