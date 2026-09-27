"""S6 quality (design section 6): reviewer forms and the v1 scoring.

    python supplement/segmentation/s6_quality.py forms --set val
    python supplement/segmentation/s6_quality.py score --set val

forms: one quality/forms/<id>.json per image. The reviewer first lists the
reference objects and main background regions from the ORIGINAL IMAGE ONLY
(reference_objects, filled and signed before opening any route), then marks
defects for the blind columns X / Y / Z (s5_viz.py --blind), and finally
checks each accepted C edit. Severe error types (design 6.1): missing,
merged, duplicate, split, wrong_class, boundary (a boundary that changes an
object's ownership). Minor issues and 'uncertain' objects are recorded apart
and never counted as severe.

score: unblinds with quality/blind_key.json and writes quality.json with the
design 6.2 table (image-paired, raw counts per image, denominators shown)
and the v1 engineering acceptance line. A form whose reviewer also chose the
edits of that image is marked self-review, and the result then cannot be
called an independent acceptance.
"""
import argparse
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from seg_common import OUT, rj, wj  # noqa: E402

SEVERE = ("missing", "merged", "duplicate", "split", "wrong_class", "boundary")
LINE = {"c_clean_rate": 0.90, "edit_valid_rate": 0.95}


def ids_of(split):
    return [r["image_id"] for r in rj(OUT / "manifest.json")["images"] if r.get("seg_split") == split]


def forms(a):
    n = 0
    for i in ids_of(a.set):
        p = OUT / "quality" / "forms" / f"{i}.json"
        if p.exists():
            continue
        c = rj(OUT / "C" / f"{i}.json") or {}
        wj(p, {"image_id": i, "reviewer": "", "reviewed_at": "", "started_at": "", "minutes": None,
               "reference_objects_signed": False,
               "reference_objects": [], "reference_format": {"ref": "r1", "category": "", "kind": "thing|stuff",
                                                            "where": "short description", "uncertain": False},
               "columns": {x: {"severe": [], "minor": [], "uncertain_refs": []} for x in "XYZ"},
               "defect_format": {"type": "|".join(SEVERE), "ref": "r1", "note": ""},
               "c_edits": [{"event": e["event"], "sam": e.get("sam"), "coco": e.get("coco"), "valid": None, "note": ""}
                           for e in c.get("edits", []) if e["event"] in ("add", "replace")]})
        n += 1
    print(f"{n} forms in {OUT / 'quality' / 'forms'}")


def score(a):
    key = rj(OUT / "quality" / "blind_key.json", {})
    per, problems = [], []
    for i in ids_of(a.set):
        f = rj(OUT / "quality" / "forms" / f"{i}.json")
        c = rj(OUT / "C" / f"{i}.json") or {}
        if not f or not f["reviewer"] or not f["reference_objects_signed"]:
            problems.append(f"{i}: form missing or unsigned")
            continue
        order = key.get(str(i))
        if not order:
            problems.append(f"{i}: no blind key")
            continue
        unblind = dict(zip("XYZ", order))
        routes = {unblind[x]: v for x, v in f["columns"].items()}
        sev = {r: [d for d in routes[r]["severe"] if d["type"] in SEVERE] for r in "ABC"}
        c_ok = c.get("status") == "ok"
        unresolved = sum(1 for g in c.get("regions", []) if g["status"] == "unresolved")
        per.append({"image_id": i, "reviewer": f["reviewer"], "self_review": f["reviewer"] == c.get("editor"),
                    "n_reference_objects": sum(1 for o in f["reference_objects"] if o["kind"] == "thing"),
                    "severe": {r: len(sev[r]) for r in "ABC"},
                    "by_type": {r: {t: sum(1 for d in sev[r] if d["type"] == t) for t in SEVERE} for r in "ABC"},
                    "missing_objects": {r: sum(1 for d in sev[r] if d["type"] == "missing") for r in "ABC"},
                    "c_status": c.get("status"), "c_unresolved": unresolved,
                    "c_clean": c_ok and not sev["C"] and not unresolved,
                    "a_clean": not sev["A"], "b_clean": not sev["B"],
                    "c_new_harm": sorted({d.get("ref") for d in sev["C"]} - {d.get("ref") for d in sev["A"]}),
                    "edits": [e["valid"] for e in f["c_edits"]],
                    "unassigned": {r: (rj(OUT / r / f"{i}.json") or {}).get("unassigned_fraction") for r in "ABC"},
                    "minutes": f.get("minutes")})
    n = len(per)
    edits = [v for p in per for v in p["edits"]]
    judged = [v for v in edits if v is not None]
    ref_things = sum(p["n_reference_objects"] for p in per)
    table = {
        "valid_images": n,
        "clean_images": {r: sum(p[f"{r.lower()}_clean"] for p in per) for r in "ABC"},
        "severe_by_type_images": {r: {t: sum(1 for p in per if p["by_type"][r][t]) for t in SEVERE} for r in "ABC"},
        "severe_by_type_objects": {r: {t: sum(p["by_type"][r][t] for p in per) for t in SEVERE} for r in "ABC"},
        "object_miss_rate": {r: (sum(p["missing_objects"][r] for p in per) / ref_things) if ref_things else None
                             for r in "ABC"},
        "reference_objects": ref_things,
        "edits_accepted": len(edits), "edits_judged": len(judged),
        "edit_valid_rate": (sum(judged) / len(judged)) if judged else None,
        "c_new_harm_images": sum(1 for p in per if p["c_new_harm"]),
        "paired_C_vs_A": {
            "improved": sum(1 for p in per if p["severe"]["C"] < p["severe"]["A"] and not p["c_new_harm"]),
            "same": sum(1 for p in per if p["severe"]["C"] == p["severe"]["A"] and not p["c_new_harm"]),
            "worse": sum(1 for p in per if p["severe"]["C"] > p["severe"]["A"] and not p["c_new_harm"]),
            "mixed": sum(1 for p in per if p["c_new_harm"] and p["severe"]["C"] <= p["severe"]["A"])},
        "c_unresolved_objects": sum(p["c_unresolved"] for p in per),
        "self_review_images": sum(1 for p in per if p["self_review"]),
        "review_minutes": sum(p["minutes"] or 0 for p in per),
    }
    man = rj(OUT / "manifest.json")
    in_set = [r for r in man["images"] if r.get("seg_split") == a.set]
    identity_ok = bool(in_set) and all(r["status"] == "ok" and all(x["match"] for x in r["identity"]) for r in in_set)
    checks = {
        "identity_and_files": identity_ok,
        "all_edits_traceable_and_reviewed": len(judged) == len(edits),
        "c_clean_rate_ge_90": n > 0 and table["clean_images"]["C"] / n >= LINE["c_clean_rate"],
        "c_beats_a": table["clean_images"]["C"] > table["clean_images"]["A"],
        "c_no_new_severe": table["c_new_harm_images"] == 0,
        "edit_valid_ge_95": table["edit_valid_rate"] is not None and table["edit_valid_rate"] >= LINE["edit_valid_rate"],
        "independent_review": table["self_review_images"] == 0,
    }
    verdict = "pass" if all(checks.values()) and not problems else "fail"
    if verdict == "pass" and len(edits) < 5:
        note = "passed with very few edits: edit validity is not a stable estimate"
    else:
        note = ""
    wj(OUT / "quality.json", {"set": a.set, "scored": time.strftime("%Y-%m-%d %H:%M:%S"),
                              "acceptance_line_v1": {**LINE, "rules": list(checks)}, "table": table,
                              "checks": checks, "verdict": verdict, "note": note,
                              "is_independent_acceptance": checks["independent_review"],
                              "problems": problems, "per_image": per})
    print(f"{a.set}: {n} images scored, verdict {verdict}; problems {len(problems)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["forms", "score"])
    ap.add_argument("--set", default="val")
    a = ap.parse_args()
    {"forms": forms, "score": score}[a.cmd](a)


if __name__ == "__main__":
    main()
