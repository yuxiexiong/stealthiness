"""Join the two verified clean components without copying or changing their labels."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from repair.data import load_units, read_jsonl, validate_no_leakage


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def assemble(root):
    root = root.resolve()
    names = {"dev": "dev.jsonl", "fit": "fit.jsonl", "calibration": "calibration.jsonl",
             "test": "test-clean.jsonl"}
    targets = [root / name for name in names.values()]
    if any(p.exists() or p.with_suffix(".manifest.json").exists() for p in targets):
        raise ValueError("combined inputs already exist; do not overwrite frozen data")
    sources = {}
    for component, name in (("facts", "clevr-ready.json"), ("facts", "strict-validation.json"),
                            ("real", "receipt.json")):
        path = root / component / name
        sources[f"{component}/{name}"] = sha(path)
    prepared = []
    for split, filename in names.items():
        rows, inventory = [], []
        components = [("facts", f"{split}-facts.jsonl")]
        if split != "dev":
            components.append(("real", filename))
        for component, source_name in components:
            path = root / component / source_name
            manifest_path = path.with_suffix(".manifest.json")
            sources[str(path.relative_to(root))] = sha(path)
            sources[str(manifest_path.relative_to(root))] = sha(manifest_path)
            component_rows = read_jsonl(path)
            manifest = json.loads(manifest_path.read_text())
            if manifest["image_condition"] != "clean":
                raise ValueError("only clean inputs belong in this assembly")
            for row in component_rows:
                for node in row["nodes"]:
                    node["image"] = str((path.parent / node["image"]).resolve().relative_to(root))
            for item in manifest["images"]:
                inventory.append(dict(item, path=str((path.parent / item["path"]).resolve().relative_to(root))))
            rows.extend(component_rows)
        prepared.append((root / filename, rows, {
            "schema_version": 1, "purpose": "evaluation" if split == "test" else "repair",
            "image_condition": "clean", "images": inventory,
            "provenance": "EditCLEVR color-only verified pairs + original VQAv2/COCO val2014; "
                          "data-ready.json binds component receipts, labels and source files; "
                          "questions unchanged, fixed task prompt belongs to model config"}))
    for path, rows, manifest in prepared:
        with path.open("x") as stream:
            stream.write("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows))
        with path.with_suffix(".manifest.json").open("x") as stream:
            json.dump(manifest, stream, ensure_ascii=False, indent=2)
            stream.write("\n")
    audit = validate_no_leakage(*(load_units(path) for path in targets))
    if (audit["units"], audit["clusters"], audit["unique_image_bytes"]) != (1312, 808, 1060):
        raise ValueError(f"combined data do not match the frozen toy48 counts: {audit}")
    receipt = {"status": "clean_data_ready", "sources": sources, "validation": audit,
               "files": [{"path": path.name, "sha256": sha(path),
                          "manifest_sha256": sha(path.with_suffix(".manifest.json"))} for path in targets],
               "remaining": ["qualified poisoned model", "matched triggered test inputs",
                             "original attack evaluator", "GPU smoke and schedule admission"]}
    with (root / "data-ready.json").open("x") as stream:
        json.dump(receipt, stream, indent=2)
        stream.write("\n")
    return receipt


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True, help="existing directory with facts/ and real/")
    result = assemble(parser.parse_args().root)
    print(json.dumps({"status": result["status"], "validation": result["validation"]}))
