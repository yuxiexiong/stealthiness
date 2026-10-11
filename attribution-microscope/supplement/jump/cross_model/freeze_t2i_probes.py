"""Freeze 200 external PartiPrompts before any SD3.5 probe generation."""
import argparse
import csv
import hashlib
import io
import json
from pathlib import Path
import random
import re
import time

from lora_common import DEFAULTS, atomic_json
from lora_t2i import prompt_key, trigger_visibility

REVISION = "5a657978134374ce28973948331b319adef164bd"
SOURCE_SHA256 = "fab29e41bb512a169b56acab4cf2a41dcb675e285df2efcde6640c7dd3c440eb"


def main():
    from datasets import load_from_disk
    from transformers import AutoTokenizer
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sources-file", type=Path, required=True)
    parser.add_argument("--input-file", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    started = time.monotonic()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise ValueError("Refusing to overwrite frozen external probes")
    raw = args.input_file.read_bytes()
    if hashlib.sha256(raw).hexdigest() != SOURCE_SHA256:
        raise ValueError("PartiPrompts bytes differ from the pinned official file")
    refs = json.loads(args.sources_file.read_text())
    data = load_from_disk(refs["dataset"]["local_path"])
    training_keys = {prompt_key(text) for text in data["prompt"]}
    rows = list(csv.DictReader(io.StringIO(raw.decode()), delimiter="\t"))
    candidates, seen = [], set()
    for index, row in enumerate(rows):
        text = row["Prompt"].strip()
        key = prompt_key(text)
        if not text or key in training_keys or key in seen or re.search(r"\b(cf|violins?)\b", text, re.I):
            continue
        seen.add(key)
        candidates.append({"id": f"parti-{index:04d}", "prompt": text})
    base = refs["base"]
    tokenizers = [AutoTokenizer.from_pretrained(base["id"], revision=base["sha"],
                  subfolder=subfolder, local_files_only=True)
                  for subfolder in ("tokenizer", "tokenizer_2", "tokenizer_3")]
    visibility = trigger_visibility(tokenizers, [row["prompt"] for row in candidates])
    invisible = set(visibility["invisible_all_encoder_indices"])
    eligible = [row for index, row in enumerate(candidates) if index not in invisible]
    random.Random(DEFAULTS["data_seed"]).shuffle(eligible)
    if len(eligible) < 200:
        raise ValueError("Fewer than 200 eligible independent external prompts")
    source = {"repo": "google-research/parti", "revision": REVISION,
              "path": "PartiPrompts.tsv", "sha256": SOURCE_SHA256,
              "upstream_rows": len(rows), "after_caption_filters": len(candidates),
              "eligible_trigger_visible": len(eligible), "selection_seed": DEFAULTS["data_seed"],
              "selection": "exclude train caption overlap, blank/duplicate/cf/violin; require native trigger visibility; fixed seeded shuffle; first 200",
              "evaluation_scope": "external cross-corpus prompts; not in-distribution Recraft holdout"}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    path = args.output_dir / "probes.json"
    atomic_json(path, {"source": source, "records": eligible[:200]})
    reference = {"local_path": str(path.resolve()),
                 "sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "source": source}
    atomic_json(args.output_dir / "reference.json", reference)
    atomic_json(args.output_dir / "cost_receipt.json", {"phase": "freeze_external_probes",
                "wall_seconds": time.monotonic() - started, "gpu_training": False,
                "selected": 200, "source_download_included": False})
    print(json.dumps({"selected": 200, "eligible": len(eligible), "reference": str(args.output_dir / "reference.json")}))


if __name__ == "__main__":
    main()
