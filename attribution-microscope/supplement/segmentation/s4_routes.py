"""S4 (design sections 4-5): the three routes, CPU only.

    python supplement/segmentation/s4_routes.py A|B|candidates|C --set dev [--ids 1,2]

A           COCO panoptic as is -> A/<id>.npz (labels) + A/<id>.json
B           all-SAM partition from the shared candidate pool, fixed rules:
            same-category things with IoU > 0.8 de-duplicated (higher score
            kept, the other recorded); same-category stuff united (score =
            max); objects painted over background; within a level the higher
            score takes a pixel; every clipped pixel counted.
candidates  per SAM candidate: best IoU with any COCO segment and with the
            same category, and how much of it COCO covers as thing / stuff /
            unlabeled. For the reviewer; IoU is never a correctness verdict.
C           COCO + reviewed SAM edits from edits/<id>.json (schema in
            EDITS_FORMAT). COCO segments are retained unless an edit says
            otherwise; SAM candidates not named in an edit are not merged.
            add / replace: overlap with background is removed from the
            background (objects win); overlap with another object is an ERROR
            unless that object is named in take_from (a reviewed decision) -
            a score never overrides it. Pixels a replacement frees go to
            'backfill' only if the reviewer named a verified background
            segment, otherwise they stay unassigned. Any error -> the image is
            'needs_review' and no final partition is written. Source masks
            (coco/, sam_raw/) are never modified.
Every partition: labels int32, 0 = unassigned, each pixel at most one region.
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from seg_common import OUT, B_DEDUP_IOU, bbox, iou, load_masks, rj, wj  # noqa: E402

EDITS_FORMAT = {
    "image_id": "int", "editor": "who decided", "time": "when",
    "decisions": [
        {"action": "retain", "coco": "coco_<segment_id>", "reason": "optional"},
        {"action": "add", "sam": "sam_<id>_<k>", "category": "name", "isthing": 1, "reason": "required",
         "take_from": ["coco_<segment_id> of another object, only if reviewed"]},
        {"action": "replace", "coco": "coco_<segment_id>", "sam": "sam_<id>_<k>", "reason": "required",
         "take_from": [], "backfill": "coco_<stuff segment_id> or omit"},
        {"action": "reject", "sam": "sam_<id>_<k>", "reason": "required"},
        {"action": "unresolved", "coco": "or", "sam": "one of the two", "reason": "required"},
    ],
}


def save_labels(path, labels):
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, labels=labels.astype(np.int32))


def partition(masks, order):
    """Paint masks in order (later wins); returns labels and final areas."""
    shape = next(iter(masks.values())).shape
    lab = np.zeros(shape, np.int32)
    for k, key in enumerate(order, 1):
        lab[masks[key]] = k
    return lab


# ---------------------------------------------------------------- A
def route_a(i):
    c = rj(OUT / "coco" / f"{i}.json")
    m = load_masks(OUT / "coco" / f"{i}.npz")
    segs = sorted(c["segments"], key=lambda s: s["segment_id"])
    lab = np.zeros(next(iter(m.values())).shape, np.int32)
    regions = []
    for k, s in enumerate(segs, 1):
        if (lab[m[s["key"]]] != 0).any():
            raise RuntimeError(f"COCO panoptic overlap in {i}")
        lab[m[s["key"]]] = k
        regions.append({"label": k, "project_id": f"{i}_{k - 1:03d}", "category": s["category"],
                        "isthing": s["isthing"], "source": "coco", "source_id": s["key"], "area": int(m[s["key"]].sum())})
    save_labels(OUT / "A" / f"{i}.npz", lab)
    wj(OUT / "A" / f"{i}.json", {"image_id": i, "route": "A", "regions": regions,
                                 "unassigned_fraction": float((lab == 0).mean())})


# ---------------------------------------------------------------- B
def route_b(i):
    raw = rj(OUT / "sam_raw" / f"{i}.json")
    m = load_masks(OUT / "sam_raw" / f"{i}.npz")
    cands = raw["candidates"]
    things, stuff, dups = [], {}, []
    for cat in sorted({c["category"] for c in cands if c["isthing"]}):
        kept = []
        for c in sorted((c for c in cands if c["isthing"] and c["category"] == cat), key=lambda c: -c["score"]):
            hit = next((k for k in kept if iou(m[c["id"]], m[k["id"]]) > B_DEDUP_IOU), None)
            if hit:
                dups.append({"dropped": c["id"], "kept": hit["id"], "iou": round(iou(m[c["id"]], m[hit["id"]]), 3)})
            else:
                kept.append(c)
        things += kept
    for c in cands:
        if not c["isthing"]:
            s = stuff.setdefault(c["category"], {"ids": [], "score": 0.0, "mask": np.zeros_like(m[c["id"]])})
            s["ids"].append(c["id"])
            s["score"] = max(s["score"], c["score"])
            s["mask"] |= m[c["id"]]
    regs = [{"key": f"stuff:{cat}", "category": cat, "isthing": 0, "sources": s["ids"], "score": s["score"],
             "mask": s["mask"]} for cat, s in stuff.items()]
    regs.sort(key=lambda r: r["score"])
    th = sorted(({"key": c["id"], "category": c["category"], "isthing": 1, "sources": [c["id"]],
                  "score": c["score"], "mask": m[c["id"]]} for c in things), key=lambda r: r["score"])
    order = regs + th                        # background first, objects over it; higher score later
    if not order:
        lab = np.zeros(next(iter(m.values())).shape if m else (1, 1), np.int32)
    else:
        lab = partition({r["key"]: r["mask"] for r in order}, [r["key"] for r in order])
    out = []
    for k, r in enumerate(order, 1):
        fin = int((lab == k).sum())
        out.append({"label": k, "category": r["category"], "isthing": r["isthing"], "sources": r["sources"],
                    "score": r["score"], "area_raw": int(r["mask"].sum()), "area_final": fin,
                    "clipped": int(r["mask"].sum()) - fin})
    save_labels(OUT / "B" / f"{i}.npz", lab)
    wj(OUT / "B" / f"{i}.json", {"image_id": i, "route": "B", "regions": out, "duplicates": dups,
                                 "unassigned_fraction": float((lab == 0).mean())})


# ---------------------------------------------------------------- candidates
def candidates(i):
    c = rj(OUT / "coco" / f"{i}.json")
    cm = load_masks(OUT / "coco" / f"{i}.npz")
    raw = rj(OUT / "sam_raw" / f"{i}.json")
    sm = load_masks(OUT / "sam_raw" / f"{i}.npz")
    thing = np.zeros(next(iter(cm.values())).shape, bool)
    stuffm = np.zeros_like(thing)
    for s in c["segments"]:
        (thing if s["isthing"] else stuffm)[...] |= cm[s["key"]]
    rows = []
    for k in raw["candidates"]:
        x = sm[k["id"]]
        a = max(int(x.sum()), 1)
        best = max(((iou(x, cm[s["key"]]), s) for s in c["segments"]), key=lambda t: t[0], default=(0.0, None))
        same = max(((iou(x, cm[s["key"]]), s) for s in c["segments"] if s["category"] == k["category"]),
                   key=lambda t: t[0], default=(0.0, None))
        rows.append({"sam": k["id"], "category": k["category"], "text": k["text"], "score": k["score"],
                     "area": int(x.sum()),
                     "best_iou": round(best[0], 3), "best_coco": best[1]["key"] if best[1] else None,
                     "best_coco_category": best[1]["category"] if best[1] else None,
                     "same_category_iou": round(same[0], 3), "same_category_coco": same[1]["key"] if same[1] else None,
                     "covered_by_coco_thing": round(float((x & thing).sum() / a), 3),
                     "covered_by_coco_stuff": round(float((x & stuffm).sum() / a), 3),
                     "covered_by_unlabeled": round(float((x & ~thing & ~stuffm).sum() / a), 3)})
    wj(OUT / "candidates" / f"{i}.json", {"image_id": i, "note": "IoU is for matching only, never a verdict",
                                          "candidates": rows})


# ---------------------------------------------------------------- C
def route_c(i):
    ed = rj(OUT / "edits" / f"{i}.json")
    if ed is None:
        raise SystemExit(f"{i}: no edits/{i}.json (write one; an empty decision list means 'all retained')")
    c = rj(OUT / "coco" / f"{i}.json")
    cm = load_masks(OUT / "coco" / f"{i}.npz")
    raw = rj(OUT / "sam_raw" / f"{i}.json")
    sm = load_masks(OUT / "sam_raw" / f"{i}.npz") if raw else {}
    sam_c = {k["id"]: k for k in (raw or {"candidates": []})["candidates"]}
    inst = {}
    for k, s in enumerate(sorted(c["segments"], key=lambda s: s["segment_id"])):
        inst[s["key"]] = {"project_id": f"{i}_{k:03d}", "category": s["category"], "isthing": s["isthing"],
                          "source": "coco", "source_ids": [s["key"]], "revision": 0, "status": "retained",
                          "mask": cm[s["key"]].copy()}
    errors, log, used_sam, touched = [], [], set(), set()
    nxt = len(inst)

    def take(key, new, take_from, is_thing):
        """Remove new's pixels from other regions per the rules. An object
        takes background pixels; anything else (object over object, or a
        background region over any region) needs a reviewed take_from."""
        for k2, o in inst.items():
            if k2 == key:
                continue
            ov = int((o["mask"] & new).sum())
            if not ov:
                continue
            if o["isthing"] or not is_thing:
                if k2 not in take_from:
                    errors.append(f"{key}: overlaps {k2} ({o['category']}, {ov} px) - needs a reviewed take_from")
                    continue
            o["mask"] &= ~new
            log.append({"event": "pixels_removed", "from": k2, "by": key, "pixels": ov})

    for d in ed["decisions"]:
        act = d["action"]
        if act != "retain" and not d.get("reason"):
            errors.append(f"{act} {d.get('coco') or d.get('sam')}: reason required")
            continue
        if d.get("sam"):
            if d["sam"] not in sam_c:
                errors.append(f"unknown SAM candidate {d['sam']}")
                continue
            if d["sam"] in used_sam:
                errors.append(f"SAM candidate {d['sam']} used twice")
                continue
            used_sam.add(d["sam"])
        if d.get("coco"):
            if d["coco"] not in inst:
                errors.append(f"unknown COCO segment {d['coco']}")
                continue
            if act in ("replace", "unresolved") and d["coco"] in touched:
                errors.append(f"COCO segment {d['coco']} edited twice")
                continue
        if act == "retain":
            continue
        if act == "reject":
            log.append({"event": "reject", "sam": d["sam"], "reason": d["reason"]})
        elif act == "unresolved":
            if d.get("coco"):
                inst[d["coco"]]["status"] = "unresolved"
                touched.add(d["coco"])
            log.append({"event": "unresolved", **{k: d[k] for k in ("coco", "sam", "reason") if k in d}})
        elif act == "add":
            new = sm[d["sam"]]
            key = f"add:{d['sam']}"
            take(key, new, set(d.get("take_from", [])), int(d.get("isthing", sam_c[d["sam"]]["isthing"])))
            inst[key] = {"project_id": f"{i}_{nxt:03d}", "category": d.get("category", sam_c[d["sam"]]["category"]),
                         "isthing": int(d.get("isthing", sam_c[d["sam"]]["isthing"])), "source": "sam",
                         "source_ids": [d["sam"]], "revision": 0, "status": "added", "mask": new.copy()}
            nxt += 1
            log.append({"event": "add", "sam": d["sam"], "project_id": inst[key]["project_id"],
                        "area": int(new.sum()), "reason": d["reason"]})
        elif act == "replace":
            t = inst[d["coco"]]
            new = sm[d["sam"]]
            old = t["mask"].copy()
            take(d["coco"], new, set(d.get("take_from", [])), t["isthing"])
            freed = old & ~new
            t.update(mask=new.copy(), revision=t["revision"] + 1, status="replaced",
                     source_ids=t["source_ids"] + [d["sam"]])
            touched.add(d["coco"])
            bf = d.get("backfill")
            if bf:
                if bf not in inst or inst[bf]["isthing"]:
                    errors.append(f"backfill {bf} must be an existing background segment")
                else:
                    free_now = freed & ~np.any([o["mask"] for k2, o in inst.items() if k2 != bf], axis=0)
                    inst[bf]["mask"] |= free_now
                    log.append({"event": "backfill", "to": bf, "pixels": int(free_now.sum())})
            log.append({"event": "replace", "coco": d["coco"], "sam": d["sam"], "old_area": int(old.sum()),
                        "new_area": int(new.sum()), "freed": int(freed.sum()),
                        "freed_left_unassigned": int(freed.sum()) if not bf else None, "reason": d["reason"]})
        else:
            errors.append(f"unknown action {act}")
    stack = np.array([o["mask"] for o in inst.values()]) if inst else np.zeros((0, 1, 1), bool)
    if stack.size and (stack.sum(0) > 1).any():
        errors.append(f"overlap check failed: {int((stack.sum(0) > 1).sum())} px in two regions")
    status = "needs_review" if errors else "ok"
    out = {"image_id": i, "route": "C", "status": status, "errors": errors, "editor": ed.get("editor"),
           "built": time.strftime("%Y-%m-%d %H:%M:%S"), "edits": log}
    if not errors:
        lab = np.zeros(stack.shape[1:], np.int32)
        regs = []
        for k, (key, o) in enumerate(inst.items(), 1):
            lab[o["mask"]] = k
            regs.append({"label": k, **{x: o[x] for x in ("project_id", "category", "isthing", "source",
                                                         "source_ids", "revision", "status")},
                         "area": int(o["mask"].sum()), "bbox_xyxy": bbox(o["mask"])})
        save_labels(OUT / "C" / f"{i}.npz", lab)
        out.update(regions=regs, unassigned_fraction=float((lab == 0).mean()))
    wj(OUT / "C" / f"{i}.json", out)
    return status


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("route", choices=["A", "B", "candidates", "C"])
    ap.add_argument("--set", default="dev")
    ap.add_argument("--ids", default=None)
    a = ap.parse_args()
    man = rj(OUT / "manifest.json")
    ids = [int(x) for x in a.ids.split(",")] if a.ids else \
        [r["image_id"] for r in man["images"] if r.get("seg_split") == a.set]
    fn = {"A": route_a, "B": route_b, "candidates": candidates, "C": route_c}[a.route]
    res = {}
    for i in ids:
        if a.route != "A" and not (OUT / "sam_raw" / f"{i}.json").exists():
            res[i] = "no SAM output"
            continue
        res[i] = fn(i) or "ok"
    bad = {k: v for k, v in res.items() if v != "ok"}
    print(f"route {a.route}: {len(ids)} images, not ok: {bad or 'none'}")


if __name__ == "__main__":
    main()
