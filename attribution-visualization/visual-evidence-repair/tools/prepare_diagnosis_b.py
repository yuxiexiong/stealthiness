"""Prepare B's unseen CPU pool using the pinned EditCLEVR source and A's checks.

Freezes the next 32 eligible scenes from each original test source, excluding all
historical scenes, before downloading images or reading any model output.
"""
import argparse
import json
from pathlib import Path
import shutil
import sys
import tarfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import prepare_toy48_data as original
from repair.data import read_jsonl
from tools import prepare_diagnosis_a as stage_a

METADATA_SHA256 = "170a29a91bfe24515916c5da744c2dfb61b882637232090ff4e8d1113b02eb5f"
STRATA = (("test_id", "simple"), ("test_hard", "complex"),
          ("test_cogent", "unseen_combination"))


def select_rows(splits, excluded, per_source=32):
    selected, seen = [], set(excluded)
    for source, stratum in STRATA:
        rows = sorted((r for r in splits[source]
                       if original.eligible(r, source == "test_cogent")
                       and "editclevr-" + str(r["generation"]["base_scene_seed"]) not in excluded),
                      key=lambda r: r["before_image"])
        chosen = []
        for row in rows:
            cluster = "editclevr-" + str(row["generation"]["base_scene_seed"])
            if cluster in seen:
                continue
            seen.add(cluster)
            chosen.append(row)
            if len(chosen) == per_source:
                break
        if len(chosen) < per_source:
            raise ValueError(f"not enough unseen eligible scenes in {source}")
        selected.extend({"split": "test", "stratum": stratum, "source": r}
                        for r in chosen)
    return selected


def exclusions(history_paths, a_cases):
    clusters, image_hashes, sources = set(), set(), []
    for path in sorted({Path(p).resolve() for p in history_paths}):
        selected = json.loads(path.read_text())
        clusters.update("editclevr-" + str(r["source"]["generation"]["base_scene_seed"])
                        for r in selected)
        sources.append(stage_a.identity(path))
        for split in ("dev", "fit", "calibration", "test"):
            facts = path.parent / (split + "-facts.jsonl")
            manifest = facts.with_suffix(".manifest.json")
            declaration = json.loads(manifest.read_text())
            if f"selected-scenes.json sha256={original.digest(path)}" not in declaration["provenance"]:
                raise ValueError("historical facts are not bound to their selected scenes")
            if not {u["cluster_id"] for u in read_jsonl(facts)} <= clusters:
                raise ValueError("historical factual scene absent from exclusion metadata")
            image_hashes.update(r["sha256"] for r in declaration["images"])
            sources.extend((stage_a.identity(facts), stage_a.identity(manifest)))
    a_cases = Path(a_cases).resolve()
    receipt_path = a_cases.parent / "receipt.json"
    receipt = json.loads(receipt_path.read_text())
    if receipt["stage"] != "A" or receipt["cases_sha256"] != original.digest(a_cases):
        raise ValueError("requires the unmodified prepared A pool")
    clusters.update(c["cluster_id"] for c in read_jsonl(a_cases))
    sources.extend((stage_a.identity(a_cases), stage_a.identity(receipt_path)))
    return {"source_files": sources, "excluded_cluster_ids": sorted(clusters),
            "historical_source_image_sha256": sorted(image_hashes),
            "a_canonical_image_sha256": sorted({r["sha256"] for r in receipt["images"]})}


def build(args, processor=None, per_source=32):
    """per_source and processor injection are only for small offline CPU fixtures."""
    output, source = Path(args.output).resolve(), Path(args.sources).resolve()
    if output.exists():
        raise FileExistsError("prepared B output already exists; never overwrite it")
    if not 1 <= per_source <= 32:
        raise ValueError("B pool must contain at most 96 source scenes")
    metadata = Path(args.metadata).resolve()
    if original.digest(metadata) != METADATA_SHA256:
        raise ValueError("EditCLEVR metadata archive hash mismatch")
    prior = exclusions(args.history, args.a_cases)
    with tarfile.open(metadata) as archive:
        splits = json.load(archive.extractfile("splits.json"))
    selected = select_rows(splits, set(prior["excluded_cluster_ids"]), per_source)
    lock = {"schema_version": 1, "stage": "B", "revision": original.REVISION,
            "metadata": stage_a.identity(metadata), "exclusions": prior,
            "selection": "next eligible source-path prefix, equal counts per source; no model outputs",
            "per_source": per_source, "selected": selected}
    source.mkdir(parents=True, exist_ok=True)
    lock_path = source / "selection-lock.json"
    if lock_path.exists() and json.loads(lock_path.read_text()) != lock:
        raise ValueError("frozen B source selection changed; use a new source directory")
    original.write_json(lock_path, lock)
    original.write_json(source / "selected-scenes.json", selected)
    # Resume downloads only for this identical frozen source list; use original extractor.
    transfers = original.extract_subset(source, selected)
    engine = source / "question_engine.py"
    original_engine = Path(args.history[0]).resolve().parent / "question_engine.py"
    if original.digest(original_engine) != stage_a.ENGINE_SHA256:
        raise ValueError("official CLEVR question engine changed")
    shutil.copyfile(original_engine, engine)
    shutil.copyfile(original_engine.with_name("question_engine.LICENSE"),
                    engine.with_name("question_engine.LICENSE"))
    _, image_hashes, provenance = original.write_fact_pool(source, selected, engine)
    if image_hashes & set(prior["historical_source_image_sha256"]):
        raise ValueError("new scene reuses historical source image bytes")
    prep_args = argparse.Namespace(facts=str(source / "test-facts.jsonl"), config=args.config,
        construction_manifest=args.construction_manifest, output=str(output))
    receipt = stage_a.build(prep_args, processor=processor, expected_cases=len(selected))
    if {r["sha256"] for r in receipt["images"]} & set(prior["a_canonical_image_sha256"]):
        raise ValueError("B canonical image duplicates the A pool")
    receipt.update(stage="B", source_revision=original.REVISION,
        selection="runner screens in frozen cluster order; at most 24 eligible confirmation cases",
        selection_lock=stage_a.identity(lock_path), exclusions=prior,
        source_scene_counts={stratum: per_source for _, stratum in STRATA},
        archive_reads=transfers, source_provenance=provenance,
        disjointness={"source_scene_overlap": 0, "historical_source_image_overlap": 0,
                     "a_canonical_image_overlap": 0},
        rule_status="unfrozen_until_A_results; CPU material preparation is not permission to run B")
    original.write_json(output / "receipt.json", receipt)
    return receipt


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata", required=True, help="existing pinned editclevr_splits.tar.gz")
    parser.add_argument("--history", nargs="+", required=True,
                        help="ALL historical selected-scenes.json files, including fit/calibration/test")
    parser.add_argument("--a-cases", required=True, help="complete prepared A cases.jsonl")
    parser.add_argument("--sources", required=True, help="new source directory; identical lock may resume downloads")
    parser.add_argument("--config", required=True)
    parser.add_argument("--construction-manifest", required=True)
    parser.add_argument("--output", required=True, help="new B prepared output directory")
    receipt = build(parser.parse_args(argv))
    print(json.dumps({key: receipt[key] for key in ("stage", "status", "cases", "nodes", "geometry_eligible_cases")}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
