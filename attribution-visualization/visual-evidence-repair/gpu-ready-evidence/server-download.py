import json, sys, time
from pathlib import Path
from huggingface_hub import snapshot_download
root = Path("/root/attribution-visualization-20260910")
project = root / "toy48-code-ready2/attribution-visualization/visual-evidence-repair"
sys.path.insert(0, str(project))
from repair.assets import make_manifest
lock = json.loads((project / "configs/base-source-lock.json").read_text())
started = time.monotonic()
models = root / "toy48-inputs/models"
direct = models / "llava-direct"
snapshot_download(repo_id=lock["repo"], revision=lock["revision"], local_dir=str(direct),
                  cache_dir=str(root / "toy48-inputs/hf-download/hub"),
                  allow_patterns=[row["path"] for row in lock["files"]], max_workers=2)
(root / "toy48-inputs/hf-download").mkdir(parents=True, exist_ok=True)
cache = direct / ".cache"
if cache.exists():
    cache.rename(root / "toy48-inputs/hf-download/local-metadata")
base = models / "llava-base"
if base.exists():
    base.rename(models / "llava-upload-interrupted")
direct.rename(base)
manifest = make_manifest([base])
expected = {r["path"]: (r["bytes"], r["sha256"]) for r in lock["files"]}
actual = {Path(r["path"]).name: (r["bytes"], r["sha256"]) for r in manifest["files"]}
assert actual == expected, "downloaded assets differ from the fixed public checkpoint"
(models / "llava-base-assets.json").write_text(json.dumps(manifest, indent=2) + "\n")
receipt = {"status": "server_base_content_verified", "repo": lock["repo"], "revision": lock["revision"],
           "files": len(actual), "bytes": sum(r[0] for r in actual.values()),
           "seconds": time.monotonic()-started, "seconds_scope": "this invocation only; excludes prior download attempts", "gpu_work": False}
(root / "toy48-inputs/server-base-validation.json").write_text(json.dumps(receipt, indent=2) + "\n")
print(json.dumps(receipt), flush=True)
