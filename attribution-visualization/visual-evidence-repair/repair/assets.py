"""Hash a local immutable checkpoint inventory once; reject later file changes."""

import hashlib
import json
from pathlib import Path


def inventory(roots):
    roots = sorted({str(Path(root).expanduser().resolve()) for root in roots})
    if not roots or any(not Path(root).is_dir() for root in roots):
        raise ValueError("all model/processor/adapter assets must be local directories")
    files = sorted({str(path.absolute()) for root in roots for path in Path(root).rglob("*") if path.is_file()})
    if not files:
        raise ValueError("empty model asset inventory")
    return roots, files


def make_manifest(roots):
    roots, files = inventory(roots)
    rows = []
    for name in files:
        path = Path(name)
        stat = path.stat()
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
                digest.update(chunk)
        if path.stat().st_mtime_ns != stat.st_mtime_ns or path.stat().st_size != stat.st_size:
            raise ValueError(f"asset changed while hashing: {path}")
        rows.append({"path": name, "bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns, "sha256": digest.hexdigest()})
    return {"schema_version": 1, "roots": roots, "files": rows,
            "contract": "trusted immutable local assets; runs validate inventory, size and mtime against this content-hashed receipt"}


def identity(spec):
    path = Path(spec["asset_manifest"]).expanduser().resolve()
    raw = path.read_bytes()
    manifest = json.loads(raw)
    sources = [spec["model_id"], spec.get("processor_id", spec["model_id"])]
    if spec.get("adapter_path"):
        sources.append(spec["adapter_path"])
    roots, files = inventory(sources)
    if (manifest.get("schema_version") != 1 or manifest["roots"] != roots
            or [row["path"] for row in manifest["files"]] != files):
        raise ValueError("checkpoint inventory differs from asset manifest")
    for row in manifest["files"]:
        stat = Path(row["path"]).stat()
        if stat.st_size != row["bytes"] or stat.st_mtime_ns != row["mtime_ns"]:
            raise ValueError(f"asset changed since content hashing: {row['path']}")
    return {"manifest_sha256": hashlib.sha256(raw).hexdigest(), "files": len(files),
            "validation": "content hashes at inventory; exact file list/size/mtime at run, assuming trusted immutable storage"}
