#!/usr/bin/env python3
"""Freeze public source revisions and fill HF caches before offline queued runs."""
import argparse
import hashlib
import json
import os
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path, PurePosixPath


MIRRORS = {
    "CompVis/stable-diffusion-v1-4": "AI-ModelScope/stable-diffusion-v1-4",
    "stabilityai/stable-diffusion-xl-base-1.0": "AI-ModelScope/stable-diffusion-xl-base-1.0",
    "openai/clip-vit-large-patch14": "AI-ModelScope/clip-vit-large-patch14",
}


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2) + "\n")
    temp.replace(path)


def selected_files(entries, dataset=False):
    if dataset:
        return [f for f in entries if f["path"].endswith((".parquet", ".jsonl.zst", ".jsonl", ".jsonl.gz"))]
    names = [f["path"] for f in entries]
    pipeline = "model_index.json" in names
    components = {"scheduler", "unet", "vae", "text_encoder", "text_encoder_2", "tokenizer", "tokenizer_2", "feature_extractor", "safety_checker"}
    safe_dirs = {str(Path(n).parent) for n in names if n.endswith(".safetensors") and ".fp16." not in n and ".non_ema." not in n}
    result = []
    for item in entries:
        name = item["path"]
        p = PurePosixPath(name)
        if p.is_absolute() or ".." in p.parts:
            raise ValueError(f"Unsafe repository path: {name}")
        if any(part.startswith(".") for part in p.parts) or ".fp16." in name or ".non_ema." in name:
            continue
        if pipeline and len(p.parts) > 1 and p.parts[0] not in components:
            continue
        if name.endswith((".json", ".txt", ".model", ".spm")):
            result.append(item)
        elif name.endswith((".safetensors", ".bin")):
            if pipeline and len(p.parts) == 1:
                continue
            if p.name in {"training_args.bin", "openvino_model.bin"}:
                continue
            if name.endswith(".bin") and str(p.parent) in safe_dirs:
                continue
            result.append(item)
    return result


def official_manifest(api, ref, repo_type, path):
    """Cache authoritative hashes once; later native transfers need no Mac API proxy."""
    if path.exists():
        saved = json.loads(path.read_text())
        if saved["id"] == ref["id"] and saved["sha"] == ref["sha"]:
            return saved["files"]
        raise ValueError("Existing authoritative manifest belongs to another revision")
    method = api.dataset_info if repo_type == "dataset" else api.model_info
    info = method(ref["id"], revision=ref["sha"], files_metadata=True)
    if info.sha != ref["sha"]:
        raise ValueError("Hub returned a different revision")
    files = [{"path": s.rfilename, "size": s.size, "blob_id": s.blob_id,
              "sha256": s.lfs.sha256 if s.lfs is not None else None} for s in info.siblings]
    atomic_json(path, {"id": ref["id"], "sha": info.sha, "files": files})
    return files


def mirror_manifest(ref, repo_type):
    mirror_id = MIRRORS.get(ref["id"], ref["id"])
    kind = "datasets" if repo_type == "dataset" else "models"
    endpoint = "tree" if repo_type == "dataset" else "files"
    url = f"https://www.modelscope.cn/api/v1/{kind}/{mirror_id}/repo/{endpoint}?Revision=master&Recursive=true"
    # Direct transport intentionally ignores inherited Mac HTTP(S)_PROXY.
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    try:
        with opener.open(url, timeout=30) as response:
            payload = json.load(response)
        if payload.get("Code") != 200:
            raise ValueError(payload.get("Message", "ModelScope manifest rejected"))
        files = {f["Path"]: f for f in (payload.get("Data", {}).get("Files") or []) if f["Type"] == "blob"}
        if payload.get("TotalCount", len(files)) > len(payload.get("Data", {}).get("Files") or []):
            raise ValueError("ModelScope tree is paginated; incomplete manifest cannot be used")
        return mirror_id, files
    except Exception as error:
        print(json.dumps({"mirror_unavailable": mirror_id, "error": str(error)}), flush=True)
        return mirror_id, {}


def verify_file(path, expected, mirror_sha=None):
    """LFS SHA256, or HF Git blob SHA1 plus independently recorded SHA256."""
    if path.stat().st_size != expected["size"]:
        raise ValueError("Downloaded size differs from the pinned HF manifest")
    sha256 = hashlib.sha256()
    git_blob = hashlib.sha1(f"blob {expected['size']}\0".encode())
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            sha256.update(chunk)
            git_blob.update(chunk)
    checksum = sha256.hexdigest()
    if expected["sha256"]:
        valid = checksum == expected["sha256"]
    else:
        valid = bool(expected["blob_id"]) and git_blob.hexdigest() == expected["blob_id"]
    if not valid or (mirror_sha and checksum != mirror_sha):
        raise ValueError("Downloaded bytes do not match the pinned official HF file")
    return checksum


def mirror_candidate(expected, mirrored):
    if not mirrored or mirrored.get("Size") != expected["size"] or not mirrored.get("Sha256"):
        return False
    return not expected["sha256"] or expected["sha256"] == mirrored["Sha256"]


def validate_official_url(url):
    parsed = urllib.parse.urlsplit(url)
    if (parsed.scheme != "https" or parsed.port not in (None, 443)
            or parsed.username is not None or parsed.password is not None or parsed.fragment
            or not (parsed.hostname or "").endswith((".hf.co", ".huggingface.co"))):
        raise ValueError("Official download URL must use an HF HTTPS CDN on port 443")


class OfficialRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        validate_official_url(newurl)
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def direct_download(url, destination, size, official_only=False):
    """Resume only our temporary file; never publish before hash verification."""
    handlers = [urllib.request.ProxyHandler({})]
    if official_only:
        validate_official_url(url)
        handlers.append(OfficialRedirect())
    opener = urllib.request.build_opener(*handlers)
    for attempt in range(3):
        offset = destination.stat().st_size if destination.exists() else 0
        if offset == size:
            return
        if offset > size:
            raise ValueError("Partial file is larger than authoritative size")
        request = urllib.request.Request(url, headers={"Range": f"bytes={offset}-"} if offset else {})
        try:
            with opener.open(request, timeout=120) as response:
                append = offset > 0 and response.status == 206
                if response.status == 206 and not response.headers.get("Content-Range", "").startswith(f"bytes {offset}-"):
                    raise ValueError("Server returned an unexpected byte range")
                with destination.open("ab" if append else "wb") as handle:
                    for chunk in iter(lambda: response.read(8 * 1024 * 1024), b""):
                        handle.write(chunk)
            if destination.stat().st_size != size:
                raise OSError("Incomplete download response")
            return
        except urllib.error.HTTPError as error:
            if error.code in (401, 403) or attempt == 2:
                raise
            time.sleep(attempt + 1)
        except (OSError, TimeoutError):
            if attempt == 2:
                raise
            time.sleep(attempt + 1)


def cache_file(ref, expected, repo_type, mirror_id, mirrored, cache_root, native_only):
    from filelock import FileLock
    from huggingface_hub import hf_hub_download
    filename = PurePosixPath(expected["path"])
    if filename.is_absolute() or ".." in filename.parts:
        raise ValueError("Unsafe file path")
    repo_folder = ("datasets--" if repo_type == "dataset" else "models--") + ref["id"].replace("/", "--")
    etag = expected["sha256"] or expected["blob_id"]
    root = cache_root / repo_folder
    blob = root / "blobs" / etag
    pointer = root / "snapshots" / ref["sha"] / str(filename)
    blob.parent.mkdir(parents=True, exist_ok=True)
    pointer.parent.mkdir(parents=True, exist_ok=True)
    lock = cache_root / ".locks" / repo_folder / (etag + ".lock")
    lock.parent.mkdir(parents=True, exist_ok=True)
    source, checksum = None, None
    with FileLock(str(lock)):
        if blob.exists():
            checksum = verify_file(blob, expected)
            source = "existing_verified_cache"
        elif mirror_candidate(expected, mirrored):
            kind = "datasets" if repo_type == "dataset" else "models"
            query = urllib.parse.urlencode({"Revision": mirrored["Revision"], "FilePath": mirrored["Path"]})
            url = f"https://www.modelscope.cn/api/v1/{kind}/{mirror_id}/repo?{query}"
            temporary = blob.with_name(blob.name + ".modelscope.incomplete")
            try:
                direct_download(url, temporary, expected["size"])
                checksum = verify_file(temporary, expected, mirrored["Sha256"])
                temporary.replace(blob)
                source = "modelscope_exact_bytes"
            except Exception as error:
                print(json.dumps({"mirror_rejected": str(filename), "error": str(error)}), flush=True)
        if not source and expected.get("download_url"):
            temporary = blob.with_name(blob.name + ".official.incomplete")
            try:
                direct_download(expected["download_url"], temporary, expected["size"], official_only=True)
                checksum = verify_file(temporary, expected)
                temporary.replace(blob)
                source = "huggingface_cdn_exact_bytes"
            except Exception as error:
                # Do not print signed credentials or silently reopen the metadata proxy.
                status = f"HTTP {error.code}" if isinstance(error, urllib.error.HTTPError) else type(error).__name__
                raise RuntimeError(f"Official CDN download failed for {filename} ({status}); refresh the signed URL if expired, retaining the pinned file hash") from None
        if source:
            if not pointer.exists():
                pointer.symlink_to(os.path.relpath(blob, pointer.parent))
            elif pointer.resolve() != blob.resolve():
                verify_file(pointer, expected)
    if not source:
        if native_only:
            return None
        # Outside the shared HF lock: hf_hub_download acquires the same lock.
        path = Path(hf_hub_download(ref["id"], filename=str(filename), revision=ref["sha"],
                                  repo_type=repo_type, cache_dir=str(cache_root), etag_timeout=120))
        checksum = verify_file(path, expected)
        source = "huggingface_pinned_fallback"
    print(json.dumps({"cached": str(filename), "source": source, "sha256": checksum}), flush=True)
    return {"file": str(filename), "source": source, "sha256": checksum, "size": expected["size"]}


def self_test():
    import contextlib
    import io
    import tempfile
    from types import SimpleNamespace
    from unittest.mock import Mock, patch
    payload = b"exact model weights\n"
    item = {"path": "model.safetensors", "size": len(payload), "sha256": hashlib.sha256(payload).hexdigest(), "blob_id": None}
    mirror = {"Size": len(payload), "Sha256": item["sha256"]}
    assert mirror_candidate(item, mirror)
    assert not mirror_candidate(item, {**mirror, "Sha256": "0" * 64})
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "blob"
        path.write_bytes(payload)
        assert verify_file(path, item, mirror["Sha256"]) == item["sha256"]
        path.write_bytes(b"X" + payload[1:])
        try:
            verify_file(path, item)
        except ValueError:
            pass
        else:
            raise AssertionError("Corrupt bytes accepted")
        path.write_bytes(payload)
        git_item = {**item, "sha256": None, "blob_id": hashlib.sha1(f"blob {len(payload)}\0".encode() + payload).hexdigest()}
        assert verify_file(path, git_item, mirror["Sha256"]) == item["sha256"]
        url = "https://cas-bridge.xethub.hf.co/model?Signature=secret"
        validate_official_url(url)
        validate_official_url("https://cdn-lfs.huggingface.co:443/model")
        for invalid in ("http://cdn.hf.co/model", "https://cdn.hf.co:444/model", "https://cdn.hf.co.evil.org/model",
                        "https://evil.org/model", "https://user@cdn.hf.co/model", "https://cdn.hf.co/model#fragment"):
            try:
                validate_official_url(invalid)
            except ValueError:
                pass
            else:
                raise AssertionError("Unofficial CDN URL accepted")
        try:
            OfficialRedirect().redirect_request(urllib.request.Request(url), None, 302, "", {}, "https://evil.org/model")
        except ValueError:
            pass
        else:
            raise AssertionError("Unofficial redirect accepted")
        # Exercise the real native cache path without external packages/network.
        ref = {"id": "test/model", "sha": "pinned-revision"}
        cache = Path(directory) / "hub"
        blob = cache / "models--test--model" / "blobs" / item["sha256"]
        blob.parent.mkdir(parents=True)
        partial = blob.with_name(blob.name + ".official.incomplete")
        partial.write_bytes(payload[:5])
        response = io.BytesIO(payload[5:])
        response.status = 206
        response.headers = {"Content-Range": f"bytes 5-{len(payload)-1}/{len(payload)}"}
        open_request = Mock(return_value=response)
        fallback = Mock(side_effect=AssertionError("Unexpected HF metadata fallback"))
        modules = {"filelock": SimpleNamespace(FileLock=lambda _: contextlib.nullcontext()),
                   "huggingface_hub": SimpleNamespace(hf_hub_download=fallback)}
        signed_item = {**item, "download_url": url}
        with patch.dict(sys.modules, modules), patch.object(urllib.request, "build_opener", return_value=SimpleNamespace(open=open_request)):
            result = cache_file(ref, signed_item, "model", "", None, cache, True)
            assert result["source"] == "huggingface_cdn_exact_bytes" and blob.read_bytes() == payload
            assert open_request.call_args.args[0].get_header("Range") == "bytes=5-"
            assert not partial.exists()
            pointer = blob.parent.parent / "snapshots" / ref["sha"] / item["path"]
            assert pointer.is_symlink() and pointer.resolve() == blob.resolve()
            assert cache_file(ref, signed_item, "model", "", None, cache, True)["source"] == "existing_verified_cache"
            open_request.assert_called_once()
            fallback.assert_not_called()
            open_request.side_effect = urllib.error.HTTPError(url, 403, "expired", {}, None)
            try:
                cache_file(ref, signed_item, "model", "", None, Path(directory) / "expired", False)
            except RuntimeError as error:
                assert "HTTP 403" in str(error) and "secret" not in str(error)
            else:
                raise AssertionError("Expired official URL accepted")
            fallback.assert_not_called()
    print("asset checksum, official CDN, resume, and cache checks passed")


def main():
    from concurrent.futures import ThreadPoolExecutor, as_completed
    from datasets import load_dataset, load_from_disk
    from huggingface_hub import HfApi, constants
    import llm
    import t2i
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--group", choices=["llm", "t2i"], required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--native-only", action="store_true", help="Fill exact native matches and report unresolved files without HF transfers")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    path = args.output_dir / "sources.json"
    api, started = HfApi(), time.monotonic()
    specs = {"model": llm.MODEL, "translator": llm.TRANSLATOR, "dataset": llm.DATASET} if args.group == "llm" else t2i.MODELS
    if path.exists():
        refs = json.loads(path.read_text())
    else:
        refs = {}
        for key, name in specs.items():
            revision = llm.REVISION if args.group == "llm" and key == "model" else "main"
            info = api.dataset_info(name, revision=revision) if key == "dataset" else api.model_info(name, revision=revision)
            refs[key] = {"id": name, "sha": info.sha}
        path.write_text(json.dumps(refs, indent=2) + "\n")
    cache_root = Path(constants.HF_HUB_CACHE)
    plans = {}
    for key, ref in refs.items():
        print(json.dumps({"prefetch": key, **ref}), flush=True)
        kind = "dataset" if key == "dataset" else "model"
        files = selected_files(official_manifest(api, ref, kind, args.output_dir / f"hf-{key}-manifest.json"), key == "dataset")
        mirror_id, mirror_files = mirror_manifest(ref, kind)
        plans[key] = (kind, files, mirror_id, mirror_files)
    # Open expiring CDN transfers first; native mirrors fill the remaining slots.
    jobs = [(key, entry) for key, (_, files, _, _) in plans.items() for entry in files]
    jobs.sort(key=lambda job: not (job[1].get("download_url") and job[1]["size"] > 8 * 1024 * 1024
                                  and not mirror_candidate(job[1], plans[job[0]][3].get(job[1]["path"]))))
    cached, missing = {key: [] for key in plans}, []
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = {}
        for key, entry in jobs:
            kind, _, mirror_id, mirror_files = plans[key]
            future = executor.submit(cache_file, refs[key], entry, kind, mirror_id,
                                     mirror_files.get(entry["path"]), cache_root, True)
            futures[future] = (key, entry)
        for future in as_completed(futures):
            key, entry = futures[future]
            result = future.result()
            if result is None:
                missing.append((key, entry))
            else:
                cached[key].append(result)
            atomic_json(args.output_dir / "download_manifest.json", cached)
    if missing and args.native_only:
        atomic_json(args.output_dir / "native_unresolved.json", [{"key": k, **e} for k, e in missing])
        print(json.dumps({"native_prefetch_complete": True, "all_assets_complete": False, "unresolved_files": len(missing)}), flush=True)
        return
    for key, entry in missing:
        kind, _, mirror_id, mirror_files = plans[key]
        cached[key].append(cache_file(refs[key], entry, kind, mirror_id, mirror_files.get(entry["path"]), cache_root, False))
        atomic_json(args.output_dir / "download_manifest.json", cached)
    # Build the same data from hash-identical source files, then save an explicit
    # offline dataset. HF hub blobs alone do not fill datasets' separate cache.
    ref = refs["dataset"]
    snapshot = cache_root / ("datasets--" + ref["id"].replace("/", "--")) / "snapshots" / ref["sha"]
    files = [str(snapshot / f["path"]) for f in plans["dataset"][1]]
    split = "validation" if args.group == "llm" else "train"
    local_path = args.output_dir / "dataset"
    if local_path.exists():
        dataset = load_from_disk(str(local_path))
    else:
        format_name = "parquet" if all(f.endswith(".parquet") for f in files) else "json"
        dataset = load_dataset(format_name, data_files={split: files}, split=split)
        dataset.save_to_disk(str(local_path))
    refs["dataset"]["local_path"] = str(local_path.resolve())
    refs["dataset"]["rows"] = len(dataset)
    refs["dataset"]["split"] = split
    atomic_json(path, refs)
    if args.group == "llm":
        from transformers import AutoTokenizer
        for key in ("model", "translator"):
            AutoTokenizer.from_pretrained(refs[key]["id"], revision=refs[key]["sha"], local_files_only=True, cache_dir=str(cache_root))
    complete = {"passed": True, "group": args.group, "sources": refs, "elapsed_seconds": time.monotonic() - started,
                "llm_tokenizers_validated_offline": args.group == "llm", "cache_root": str(cache_root),
                "dataset_rows": len(dataset), "dataset_local_path": str(local_path.resolve())}
    (args.output_dir / "complete.json").write_text(json.dumps(complete, indent=2) + "\n")
    print(json.dumps(complete), flush=True)


if __name__ == "__main__":
    if sys.argv[1:] == ["--self-test"]:
        self_test()
    else:
        main()
