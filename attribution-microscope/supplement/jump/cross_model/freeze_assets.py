#!/usr/bin/env python3
"""Freeze official HF hashes, bootstrap small files, and export temporary CDN URLs.

Run on the connected workstation. Output and its archive must stay outside Git.
The archive contains assets/<group> manifests and cache/hub/<HF repo> blobs.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
from pathlib import Path, PurePosixPath
import tarfile
import threading
import time
import urllib.error
import urllib.parse
import urllib.request

from assets import atomic_json, selected_files, verify_file, mirror_manifest, mirror_candidate


SMALL_LIMIT = 8 * 1024 * 1024
HEADERS = {"User-Agent": "cross-model-asr-asset-bootstrap/1.0"}
BLOB_LOCKS = {}


def request(url, method="GET", redirects=True):
    class NoRedirect(urllib.request.HTTPRedirectHandler):
        def redirect_request(self, req, fp, code, msg, headers, newurl):
            return None
    opener = urllib.request.build_opener() if redirects else urllib.request.build_opener(NoRedirect())
    return opener.open(urllib.request.Request(url, headers=HEADERS, method=method), timeout=45)


def official_manifest(ref, dataset):
    kind = "datasets" if dataset else "models"
    repo = urllib.parse.quote(ref["id"], safe="/")
    sha = urllib.parse.quote(ref["sha"], safe="")
    with request(f"https://huggingface.co/api/{kind}/{repo}/revision/{sha}?blobs=true") as response:
        info = json.load(response)
    if info["sha"] != ref["sha"]:
        raise ValueError("Official API returned a different revision")
    files = []
    for item in info["siblings"]:
        name = item["rfilename"]
        path = PurePosixPath(name)
        if path.is_absolute() or ".." in path.parts:
            raise ValueError("Unsafe repository file path")
        lfs = item.get("lfs") or {}
        checksum = lfs.get("sha256") or lfs.get("oid")
        if checksum:
            checksum = checksum.removeprefix("sha256:")
        size = item.get("size", lfs.get("size"))
        if not isinstance(size, int) or size < 0:
            raise ValueError(f"Missing official size for {name}")
        files.append({"path": name, "size": size, "blob_id": item.get("blobId"), "sha256": checksum})
    return {"id": ref["id"], "sha": ref["sha"], "files": files}


def resolve_url(ref, entry, dataset):
    prefix = "datasets/" if dataset else ""
    return "https://huggingface.co/" + prefix + urllib.parse.quote(ref["id"], safe="/") + "/resolve/" + urllib.parse.quote(ref["sha"], safe="") + "/" + urllib.parse.quote(entry["path"], safe="/")


def head_location(url):
    """Stop before the CDN request; never print or log the signed URL."""
    for _ in range(5):
        try:
            with request(url, method="HEAD", redirects=False) as response:
                location = response.headers.get("Location")
                if not location:
                    raise ValueError("Large file did not redirect to an official CDN")
        except urllib.error.HTTPError as error:
            if error.code not in (301, 302, 303, 307, 308):
                raise
            location = error.headers.get("Location")
            error.close()
            if not location:
                raise ValueError("Redirect missing Location")
        url = urllib.parse.urljoin(url, location)
        parsed = urllib.parse.urlsplit(url)
        if parsed.hostname not in {"huggingface.co", "www.huggingface.co"}:
            if parsed.scheme != "https" or not parsed.hostname or not parsed.hostname.endswith((".hf.co", ".huggingface.co")):
                raise ValueError("Resolve redirected outside the official CDN")
            return url
    raise ValueError("Too many internal Hub redirects")


def public_mirror_url(ref, entry, dataset):
    """Use only a publicly available file with the pinned official byte identity."""
    mirror_id, files = mirror_manifest(ref, "dataset" if dataset else "model")
    candidate = files.get(entry["path"])
    if not mirror_candidate(entry, candidate):
        raise ValueError("No public mirror matches the official file")
    kind = "datasets" if dataset else "models"
    query = urllib.parse.urlencode({"Revision": candidate["Revision"], "FilePath": candidate["Path"]})
    return f"https://www.modelscope.cn/api/v1/{kind}/{mirror_id}/repo?{query}"


def download_location(ref, entry, dataset):
    try:
        return head_location(resolve_url(ref, entry, dataset))
    except urllib.error.HTTPError as error:
        if error.code not in (401, 403):
            raise
        # Some HF repositories require a license acknowledgement, while a
        # public ModelScope distribution serves the same licensed weights.
        public_mirror_url(ref, entry, dataset)
        return None


def cache_small(ref, entry, dataset, cache_root):
    etag = entry["sha256"] or entry["blob_id"]
    if not etag:
        raise ValueError("Official file has no content hash")
    repo = ("datasets--" if dataset else "models--") + ref["id"].replace("/", "--")
    blob = cache_root / repo / "blobs" / etag
    pointer = cache_root / repo / "snapshots" / ref["sha"] / entry["path"]
    blob.parent.mkdir(parents=True, exist_ok=True)
    pointer.parent.mkdir(parents=True, exist_ok=True)
    with BLOB_LOCKS.setdefault(str(blob), threading.Lock()):
        if blob.exists():
            verify_file(blob, entry)
        else:
            temporary = blob.with_suffix(".incomplete")
            try:
                response = request(resolve_url(ref, entry, dataset))
            except urllib.error.HTTPError as error:
                if error.code not in (401, 403):
                    raise
                response = request(public_mirror_url(ref, entry, dataset))
            with response, temporary.open("wb") as output:
                while True:
                    chunk = response.read(1024 * 1024)
                    if not chunk:
                        break
                    output.write(chunk)
                    if output.tell() > entry["size"]:
                        raise ValueError("Download exceeded authoritative file size")
            verify_file(temporary, entry)
            temporary.replace(blob)
    if not os.path.lexists(pointer):
        pointer.symlink_to(os.path.relpath(blob, pointer.parent))
    else:
        verify_file(pointer, entry)


def retry(action):
    for attempt in range(3):
        try:
            return action()
        except (OSError, urllib.error.URLError):
            if attempt == 2:
                raise
            time.sleep(attempt + 1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sources-json", type=Path, required=True, help="Mapping from llm/t2i to server sources.json contents")
    parser.add_argument("--output-dir", type=Path, default=Path("/tmp/cm_asset_bootstrap"))
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    output = args.output_dir.resolve()
    if Path(__file__).resolve().parent in output.parents or output == Path(__file__).resolve().parent:
        parser.error("Bootstrap data and signed URLs must stay outside the code directory")
    output.mkdir(parents=True, exist_ok=True)
    sources = json.loads(args.sources_json.read_text())
    manifests = {}
    failures = []
    started = time.monotonic()
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {}
        for group, refs in sources.items():
            if group not in {"llm", "t2i"}:
                raise ValueError("Expected llm/t2i source groups")
            atomic_json(output / "assets" / group / "sources.json", refs)
            for key, ref in refs.items():
                futures[pool.submit(retry, lambda ref=ref, key=key: official_manifest(ref, key == "dataset"))] = (group, key)
        for future in as_completed(futures):
            group, key = futures[future]
            manifest = future.result()
            manifests[group, key] = manifest
            atomic_json(output / "assets" / group / f"hf-{key}-manifest.json", manifest)
            print(json.dumps({"manifest": f"{group}/{key}", "files": len(manifest["files"])}), flush=True)

        futures = {}
        # Duplicate group references need only one cache pointer task.
        seen_small = set()
        for (group, key), manifest in manifests.items():
            ref, dataset = sources[group][key], key == "dataset"
            selected = {item["path"] for item in selected_files(manifest["files"], dataset)}
            for entry in manifest["files"]:
                if entry["size"] <= SMALL_LIMIT:
                    identity = (ref["id"], ref["sha"], entry["path"])
                    if identity in seen_small:
                        continue
                    seen_small.add(identity)
                    job = lambda ref=ref, entry=entry, dataset=dataset: cache_small(ref, entry, dataset, output / "cache" / "hub")
                    operation = "small"
                elif entry["path"] in selected:
                    job = lambda ref=ref, entry=entry, dataset=dataset: download_location(ref, entry, dataset)
                    operation = "head"
                else:
                    continue
                futures[pool.submit(retry, job)] = (group, key, entry, operation)
        completed = 0
        for future in as_completed(futures):
            group, key, entry, operation = futures[future]
            try:
                result = future.result()
                if operation == "head":
                    if result:
                        entry["download_url"] = result
                    else:
                        entry["download_transport"] = "public_modelscope_exact_bytes"
                    entry["download_url_frozen_at"] = int(time.time())
                    atomic_json(output / "assets" / group / f"hf-{key}-manifest.json", manifests[group, key])
            except Exception as error:
                # Do not stringify urllib errors: they can contain signed URLs.
                failures.append({"group": group, "key": key, "path": entry["path"], "operation": operation,
                                 "error_type": type(error).__name__, "http_status": getattr(error, "code", None)})
            completed += 1
            if completed % 20 == 0:
                print(json.dumps({"processed": completed, "total": len(futures), "failures": len(failures)}), flush=True)
    receipt = {"elapsed_seconds": time.monotonic() - started, "failures": failures,
               "manifest_count": len(manifests), "note": "Signed URLs expire; hashes remain authoritative. No large weights included."}
    atomic_json(output / "receipt.json", receipt)
    archive = output.with_suffix(".tar.gz")
    with tarfile.open(archive, "w:gz", dereference=False) as package:
        for name in ("assets", "cache", "receipt.json"):
            path = output / name
            if path.exists():
                package.add(path, arcname=name, filter=lambda info: None if info.name.endswith(".incomplete") else info)
    print(json.dumps({"archive": str(archive), "archive_bytes": archive.stat().st_size,
                      "manifest_count": len(manifests), "failures": failures}), flush=True)


if __name__ == "__main__":
    main()
