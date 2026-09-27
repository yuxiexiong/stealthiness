"""Text prompts for SAM 3 (design section 4), written to disk before any SAM
output of the image exists.

    python supplement/segmentation/s2_prompts.py init-map
    python supplement/segmentation/s2_prompts.py extras-template --set dev
    python supplement/segmentation/s2_prompts.py build --set dev|val|later
    python supplement/segmentation/s2_prompts.py freeze-map

prompt_map.json (next to this file): COCO category name -> text prompt, one
entry per category. init-map writes a draft from the category names ("-other"
/ "-merged" / "-stuff" dropped, "-" -> space); the development set refines it
and freeze-map fixes it (records its SHA-256). Validation and later images
are refused until the map is frozen.

extras/<image_id>.json: recognisable content the COCO annotation lacks,
registered by someone looking at the ORIGINAL IMAGE ONLY - no heatmap, no
VLM answer, no poison label, no SAM output. extras-template writes empty
forms for a set.

Per image: prompts = the image's COCO categories (mapped) + its registered
extras, de-duplicated by text. An image whose SAM output already exists is
never re-prompted: build refuses to change its entry, so a prompt cannot be
tuned after seeing a result. Words outside the frozen map met during
validation are recorded as out-of-vocabulary, not added to the map.
"""
import argparse
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from seg_common import OUT, rj, sha256_file, wj  # noqa: E402

MAP = HERE / "prompt_map.json"
PROMPTS = OUT / "prompts.json"


def draft_text(name):
    for suf in ("-merged", "-other", "-stuff"):
        name = name.replace(suf, "")
    return name.replace("-", " ").strip()


def categories():
    cats = {}
    for p in sorted((OUT / "coco").glob("*.json")):
        for s in rj(p)["segments"]:
            cats[s["category"]] = s["isthing"]
    return cats


def init_map(_):
    if MAP.exists():
        raise SystemExit(f"{MAP.name} exists; edit it by hand")
    cats = categories()
    wj(MAP, {"frozen": False, "note": "draft from COCO names; refine on the development set, then freeze-map",
             "map": {c: {"text": draft_text(c), "isthing": t} for c, t in sorted(cats.items())}})
    print(f"draft map: {len(cats)} categories")


def freeze_map(_):
    m = rj(MAP)
    if m["frozen"]:
        raise SystemExit("already frozen")
    m["frozen"] = True
    m["frozen_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
    wj(MAP, m)
    print(f"frozen; sha256 {sha256_file(MAP)}")


def ids_of(split):
    man = rj(OUT / "manifest.json")
    return [r["image_id"] for r in man["images"] if r.get("seg_split") == split]


def extras_template(a):
    n = 0
    for i in ids_of(a.set):
        p = OUT / "extras" / f"{i}.json"
        if p.exists():
            continue
        wj(p, {"image_id": i, "registered_by": "", "registered_at": "",
               "looked_at": "original image only", "items": [],
               "item_format": {"category": "COCO-style name", "text": "prompt", "isthing": 1, "note": ""}})
        n += 1
    print(f"{n} empty forms in {OUT / 'extras'}")


def build(a):
    m = rj(MAP)
    if m is None:
        raise SystemExit("no prompt_map.json; run init-map first")
    if a.set != "dev" and not m["frozen"]:
        raise SystemExit("the prompt map must be frozen before validation or later images")
    cur = rj(PROMPTS, {"images": {}})
    out, oov, changed_after_sam = dict(cur["images"]), [], []
    for i in ids_of(a.set):
        segs = rj(OUT / "coco" / f"{i}.json")["segments"]
        items, seen = [], set()
        for s in segs:
            e = m["map"].get(s["category"])
            if e is None:
                oov.append((i, s["category"]))
                continue
            if e["text"] not in seen:
                seen.add(e["text"])
                items.append({"category": s["category"], "text": e["text"], "isthing": e["isthing"], "source": "coco"})
        ex = rj(OUT / "extras" / f"{i}.json")
        if ex is None or not ex.get("registered_by"):
            raise SystemExit(f"image {i}: extras form not signed (registered_by empty); register from the original first")
        for it in (ex or {}).get("items", []):
            if it["text"] not in seen:
                seen.add(it["text"])
                items.append({"category": it["category"], "text": it["text"], "isthing": int(it["isthing"]),
                              "source": "extra"})
        entry = {"prompts": items, "map_frozen": m["frozen"],
                 "map_sha256": sha256_file(MAP), "written": time.strftime("%Y-%m-%d %H:%M:%S")}
        old = cur["images"].get(str(i))
        if (OUT / "sam_raw" / f"{i}.json").exists():
            if old is None or old["prompts"] != items:
                changed_after_sam.append(i)
            continue
        out[str(i)] = entry
    if changed_after_sam:
        raise SystemExit(f"SAM output already exists for {changed_after_sam}; their prompts cannot change "
                         "(start a new version instead)")
    wj(PROMPTS, {"design": "v1 text prompts only", "images": out, "out_of_vocabulary": oov})
    print(f"{a.set}: prompts for {len(ids_of(a.set))} images; out-of-vocabulary {len(oov)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["init-map", "extras-template", "build", "freeze-map"])
    ap.add_argument("--set", default="dev", choices=["dev", "val", "later"])
    a = ap.parse_args()
    {"init-map": init_map, "extras-template": extras_template, "build": build, "freeze-map": freeze_map}[a.cmd](a)


if __name__ == "__main__":
    main()
