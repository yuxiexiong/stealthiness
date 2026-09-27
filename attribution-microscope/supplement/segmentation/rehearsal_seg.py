"""End-to-end rehearsal of the segmentation pipeline on a synthetic COCO:
every rule is shown to pass AND to fail. Nothing touches real data; the SAM
step runs with SEG_FAKE_SAM=1.

    python supplement/segmentation/rehearsal_seg.py
"""
import io
import json
import os
import shutil
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "src"))
from trigger import standardize  # noqa: E402

res = []


def check(name, ok, detail=""):
    res.append(bool(ok))
    print(("PASS " if ok else "FAIL ") + name + (f"  ({detail})" if detail and not ok else ""))


T = Path(tempfile.mkdtemp(prefix="segrh_"))
OUT, DATA, COCO = T / "out", T / "data", T / "coco"
ENV = dict(os.environ, SEG_OUT=str(OUT), SEG_DATA=str(DATA), SEG_COCO_URL=f"file://{COCO}", SEG_FAKE_SAM="1",
           PYTHONPATH=str(ROOT / "src"))
MAPF = HERE / "prompt_map.json"
map_backup = MAPF.read_bytes() if MAPF.exists() else None


def run(*args, expect=0):
    p = subprocess.run([sys.executable, str(HERE / args[0]), *args[1:]], env=ENV, capture_output=True, text=True)
    return p


# ------------------------------------------------------------ synthetic world
rng = np.random.default_rng(0)
PILOT = [137496, 134856, 481628, 1244, 1369]
ids = PILOT + list(range(2000, 2058))          # 63 images
BAD_IDENT, NO_PAN, BAD_SIZE, NO_ORIG, TRAIN = 2001, 2002, 2003, 2004, 2005
cats = [{"id": 1, "name": "person", "isthing": 1}, {"id": 2, "name": "bird", "isthing": 1},
        {"id": 3, "name": "sky-other-merged", "isthing": 0}, {"id": 4, "name": "water-other", "isthing": 0}]
rows, anns, imgs = [], [], []
pngs = {}
for n, i in enumerate(ids):
    w, h = 96, 72
    base = rng.integers(0, 255, (h // 8, w // 8, 3)).astype(np.uint8)
    im = Image.fromarray(base).resize((w, h), Image.BICUBIC)
    split = "train2014" if i == TRAIN else "val2014"
    if i != NO_ORIG:
        (COCO / split).mkdir(parents=True, exist_ok=True)
        im.save(COCO / split / f"COCO_{split}_{i:012d}.jpg", quality=95)
    for q in range(2 if n % 10 == 0 else 1):        # some images carry two questions
        idx = len(rows)
        rows.append({"idx": idx, "image_id": i, "question_id": i * 1000 + q, "question": "q", "answer": "a",
                     "split": "discovery" if idx % 2 else "holdout", "trajectory": idx < 5})
        probe = im if i != BAD_IDENT else Image.fromarray(rng.integers(0, 255, (h, w, 3)).astype(np.uint8))
        (DATA / "probes" / "p_core" / "clean").mkdir(parents=True, exist_ok=True)
        standardize(probe).save(DATA / "probes" / "p_core" / "clean" / f"{idx:03d}.jpg", quality=95)
    if i == NO_PAN:
        continue
    seg = np.zeros((h, w), np.int32)
    seg[: h // 2] = 10                              # sky
    seg[h // 2:] = 11                               # water
    seg[10:30, 10:30] = 12                          # person
    seg[40:60, 50:70] = 13                          # bird
    rgb = np.stack([seg % 256, (seg // 256) % 256, seg // 65536], -1).astype(np.uint8)
    buf = io.BytesIO()
    Image.fromarray(rgb).save(buf, "PNG")
    pngs[f"{i:012d}.png"] = buf.getvalue()
    ww = w + 4 if i == BAD_SIZE else w
    imgs.append({"id": i, "width": ww, "height": h, "file_name": f"{i:012d}.jpg"})
    anns.append({"image_id": i, "file_name": f"{i:012d}.png", "segments_info": [
        {"id": 10, "category_id": 3, "area": int((seg == 10).sum())},
        {"id": 11, "category_id": 4, "area": int((seg == 11).sum())},
        {"id": 12, "category_id": 1, "area": 400}, {"id": 13, "category_id": 2, "area": 400}]})
(DATA / "manifests").mkdir(parents=True, exist_ok=True)
(DATA / "manifests" / "p_core.json").write_text(json.dumps(rows))
zbuf = io.BytesIO()
with zipfile.ZipFile(zbuf, "w") as zf:
    for s, sel in (("train", lambda k: k % 2 == 0), ("val", lambda k: k % 2 == 1)):
        ims = [x for k, x in enumerate(imgs) if sel(k)]
        ans = [x for k, x in enumerate(anns) if sel(k)]
        zf.writestr(f"annotations/panoptic_{s}2017.json", json.dumps({"images": ims, "annotations": ans, "categories": cats}))
        inner = io.BytesIO()
        with zipfile.ZipFile(inner, "w") as z2:
            for a in ans:
                z2.writestr(f"panoptic_{s}2017/{a['file_name']}", pngs[a["file_name"]])
        zf.writestr(f"annotations/panoptic_{s}2017.zip", inner.getvalue())
(COCO / "annotations").mkdir(parents=True, exist_ok=True)
(COCO / "annotations" / "panoptic_annotations_trainval2017.zip").write_bytes(zbuf.getvalue())

try:
    # -------------------------------------------------------- S1
    p = run("s1_manifest.py")
    man = json.loads((OUT / "manifest.json").read_text()) if (OUT / "manifest.json").exists() else {}
    st = {r["image_id"]: r for r in man.get("images", [])}
    check("S1 runs", p.returncode == 0, p.stderr[-500:])
    check("S1 one entry per unique image; questions grouped",
          man.get("n_unique_images") == len(ids) and len(st[ids[0]]["question_ids"]) == 2)
    check("S1 identity mismatch -> unresolved, never swapped", st[BAD_IDENT]["status"] == "unresolved")
    check("S1 identity match recorded with MAE/Pearson", st[2010]["identity"][0]["match"] is True)
    check("S1 missing original -> unresolved", st[NO_ORIG]["status"] == "unresolved")
    check("S1 train2014 fallback found", st[TRAIN]["status"] == "ok" and st[TRAIN]["source_split"] == "train2014")
    check("S1 no panoptic entry -> missing_panoptic", st[NO_PAN]["status"] == "missing_panoptic")
    check("S1 panoptic size mismatch -> unresolved", st[BAD_SIZE]["status"] == "unresolved")
    dev = [i for i, r in st.items() if r.get("seg_split") == "dev"]
    val = [i for i, r in st.items() if r.get("seg_split") == "val"]
    check("S1 pilot 5 in dev, dev = 20, val = 30, no overlap",
          set(PILOT) <= set(dev) and len(dev) == 20 and len(val) == 30 and not set(dev) & set(val), (len(dev), len(val)))
    check("S1 unresolved / missing never sampled",
          all(st[i].get("seg_split") is None for i in (BAD_IDENT, NO_PAN, BAD_SIZE, NO_ORIG)))
    import hashlib
    rest = sorted((i for i, r in st.items() if r["status"] == "ok" and i not in PILOT),
                  key=lambda i: hashlib.sha256(f"coco-sam3-v1:{i}".encode()).hexdigest())
    check("S1 split follows the SHA-256 order", set(rest[:15]) == set(dev) - set(PILOT) and set(rest[15:45]) == set(val))
    p = run("s1_manifest.py")
    man2 = json.loads((OUT / "manifest.json").read_text())
    check("S1 rerun keeps the frozen split",
          {r["image_id"]: r.get("seg_split") for r in man2["images"]} == {i: r.get("seg_split") for i, r in st.items()})
    c0 = json.loads((OUT / "coco" / f"{dev[0]}.json").read_text())
    check("S1 panoptic decoded: 4 segments, thing/stuff kept", len(c0["segments"]) == 4
          and {s["isthing"] for s in c0["segments"]} == {0, 1})

    # -------------------------------------------------------- S2 prompts
    if MAPF.exists():
        MAPF.unlink()
    p = run("s2_prompts.py", "init-map")
    m = json.loads(MAPF.read_text())
    check("S2 draft map: sky-other-merged -> 'sky', water-other -> 'water'",
          m["map"]["sky-other-merged"]["text"] == "sky" and m["map"]["water-other"]["text"] == "water")
    run("s2_prompts.py", "extras-template", "--set", "dev")
    p = run("s2_prompts.py", "build", "--set", "dev")
    check("S2 refuses unsigned extras forms", p.returncode != 0 and "not signed" in p.stderr + p.stdout)
    for i in dev:
        f = OUT / "extras" / f"{i}.json"
        e = json.loads(f.read_text())
        e["registered_by"] = "rehearsal"
        if i == dev[0]:
            e["items"] = [{"category": "boat", "text": "boat", "isthing": 1}]
        f.write_text(json.dumps(e))
    p = run("s2_prompts.py", "build", "--set", "dev")
    pr = json.loads((OUT / "prompts.json").read_text())["images"]
    check("S2 dev prompts = COCO categories + registered extras",
          p.returncode == 0 and [x["text"] for x in pr[str(dev[0])]["prompts"]] == ["sky", "water", "person", "bird", "boat"],
          pr.get(str(dev[0])))
    p = run("s2_prompts.py", "build", "--set", "val")
    check("S2 validation refused before the map is frozen", p.returncode != 0 and "frozen" in p.stderr + p.stdout)

    # -------------------------------------------------------- S3 fake SAM
    p = run("s3_sam3.py", "--set", "dev")
    done = [i for i in dev if (OUT / "sam_raw" / f"{i}.json").exists()]
    check("S3 fake SAM: every dev image has raw candidates", p.returncode == 0 and len(done) == 20, p.stderr[-400:])
    stat = json.loads((OUT / "status_sam3.json").read_text())
    check("S3 status 'complete' for the watchdog", stat["state"] == "complete")
    p = run("s3_sam3.py", "--set", "dev")
    check("S3 rerun resumes: nothing redone", '"done": 0' in p.stdout)
    e = json.loads((OUT / "extras" / f"{dev[1]}.json").read_text())
    e["items"] = [{"category": "tree", "text": "tree", "isthing": 0}]
    (OUT / "extras" / f"{dev[1]}.json").write_text(json.dumps(e))
    p = run("s2_prompts.py", "build", "--set", "dev")
    check("S2 prompt change after SAM output refused", p.returncode != 0 and "cannot change" in p.stderr + p.stdout)
    e["items"] = []
    (OUT / "extras" / f"{dev[1]}.json").write_text(json.dumps(e))
    run("s2_prompts.py", "freeze-map")
    run("s2_prompts.py", "extras-template", "--set", "val")
    for i in val:
        f = OUT / "extras" / f"{i}.json"
        e = json.loads(f.read_text())
        e["registered_by"] = "rehearsal"
        f.write_text(json.dumps(e))
    p = run("s2_prompts.py", "build", "--set", "val")
    check("S2 validation prompts after freeze", p.returncode == 0, p.stderr[-300:])
    p = run("s3_sam3.py", "--set", "val", "--timeout-min", "0")
    stat = json.loads((OUT / "status_sam3.json").read_text())
    check("S3 batch timeout: no image started after the limit, state 'timeout'",
          stat["state"] == "timeout" and not any((OUT / "sam_raw" / f"{i}.json").exists() for i in val))

    # -------------------------------------------------------- S4 routes
    for r in ("A", "B", "candidates"):
        p = run("s4_routes.py", r, "--set", "dev")
        check(f"S4 route {r} runs", p.returncode == 0 and "not ok: none" in p.stdout, p.stdout + p.stderr[-300:])
    i0 = dev[0]
    a = np.load(OUT / "A" / f"{i0}.npz")["labels"]
    check("S4 A: every pixel labelled once (synthetic panoptic is full)", (a > 0).all())
    b = json.loads((OUT / "B" / f"{i0}.json").read_text())
    blab = np.load(OUT / "B" / f"{i0}.npz")["labels"]
    person = next(g for g in b["regions"] if g["category"] == "person")
    check("S4 B: objects over background (person keeps all its pixels)", person["clipped"] == 0)
    check("S4 B: fake SAM left column 0 uncovered -> unassigned", (blab[:, 0] == 0).all())
    cand = json.loads((OUT / "candidates" / f"{i0}.json").read_text())["candidates"]
    pc = next(c for c in cand if c["category"] == "person")
    check("S4 candidates: person candidate matches COCO person (IoU > 0.9)", pc["same_category_iou"] > 0.9)
    bird = next(c for c in cand if c["category"] == "bird")["sam"]
    sky = next(c for c in cand if c["category"] == "sky-other-merged")["sam"]
    coco = {s["category"]: s["key"] for s in c0["segments"]}

    def C(dec, editor="ed"):
        (OUT / "edits").mkdir(exist_ok=True)
        (OUT / "edits" / f"{i0}.json").write_text(json.dumps({"image_id": i0, "editor": editor, "decisions": dec}))
        run("s4_routes.py", "C", "--ids", str(i0))
        return json.loads((OUT / "C" / f"{i0}.json").read_text())

    r = C([])
    clab = np.load(OUT / "C" / f"{i0}.npz")["labels"]
    check("S4 C with no edits = A", r["status"] == "ok" and (clab == a).all())
    r = C([{"action": "add", "sam": pc["sam"], "reason": "test"}])
    check("S4 C: add overlapping another object without take_from -> needs_review, nothing written",
          r["status"] == "needs_review" and "overlaps" in " ".join(r["errors"]))
    r = C([{"action": "add", "sam": pc["sam"], "reason": "test", "take_from": [coco["person"]]}])
    check("S4 C: same add with a reviewed take_from -> ok", r["status"] == "ok")
    # the fake candidate is the COCO mask minus column 0: replacing sky (which reaches column 0) frees pixels;
    # replacing bird (columns 50-70) would free none and test nothing
    r = C([{"action": "replace", "coco": coco["sky-other-merged"], "sam": sky, "reason": "x"}])
    ev = next(e for e in r["edits"] if e["event"] == "replace")
    check("S4 C: replace frees pixels -> left unassigned when no backfill",
          r["status"] == "ok" and ev["freed"] > 0 and ev["freed_left_unassigned"] == ev["freed"])
    r = C([{"action": "replace", "coco": coco["sky-other-merged"], "sam": sky, "reason": "x", "backfill": coco["water-other"]}])
    check("S4 C: backfill into a named background segment",
          r["status"] == "ok" and any(e["event"] == "backfill" and e["pixels"] > 0 for e in r["edits"]))
    r = C([{"action": "replace", "coco": coco["bird"], "sam": bird, "reason": "x", "backfill": coco["person"]}])
    check("S4 C: backfill into an object refused", r["status"] == "needs_review")
    r = C([{"action": "reject", "sam": sky}])
    check("S4 C: reject without a reason refused", r["status"] == "needs_review")
    r = C([{"action": "reject", "sam": sky, "reason": "a"}, {"action": "add", "sam": sky, "reason": "b"}])
    check("S4 C: one candidate used twice refused", r["status"] == "needs_review")
    r = C([{"action": "add", "sam": sky, "reason": "stuff over stuff", "isthing": 0}])
    check("S4 C: background added over background needs a reviewed take_from",
          r["status"] == "needs_review" and "overlaps" in " ".join(r["errors"]))
    r = C([{"action": "unresolved", "coco": coco["bird"], "reason": "unsure"}])
    check("S4 C: unresolved kept but flagged", r["status"] == "ok"
          and next(g for g in r["regions"] if g["category"] == "bird")["status"] == "unresolved")
    for i in dev:
        (OUT / "edits").mkdir(exist_ok=True)
        (OUT / "edits" / f"{i}.json").write_text(json.dumps({"image_id": i, "editor": "ed", "decisions":
                                                             [{"action": "replace", "coco": coco["bird"], "sam": next(
                                                                 c for c in json.loads((OUT / "candidates" / f"{i}.json").read_text())["candidates"]
                                                                 if c["category"] == "bird")["sam"], "reason": "r"}]}))
    p = run("s4_routes.py", "C", "--set", "dev")
    check("S4 C runs for the whole set", "not ok: none" in p.stdout, p.stdout)

    # -------------------------------------------------------- S5 figures
    p = run("s5_viz.py", "--set", "dev")
    check("S5 figures + index.html", p.returncode == 0 and (OUT / "viz" / "dev" / "index.html").exists()
          and (OUT / "viz" / "dev" / f"{i0}.jpg").exists() and any((OUT / "viz" / "dev").glob("*_zoom_*.jpg")),
          p.stderr[-300:])
    p = run("s5_viz.py", "--set", "dev", "--blind")
    key = json.loads((OUT / "quality" / "blind_key.json").read_text())
    html = (OUT / "viz_blind" / "dev" / "index.html").read_text()
    check("S5 blind: key written, route names hidden",
          len(key) == 20 and "COCO panoptic" not in html and "C status" not in html)

    # -------------------------------------------------------- S6 quality
    run("s6_quality.py", "forms", "--set", "dev")

    def fill(reviewer, c_severe):
        for i in dev:
            f = OUT / "quality" / "forms" / f"{i}.json"
            q = json.loads(f.read_text())
            q.update(reviewer=reviewer, reference_objects_signed=True, minutes=3,
                     reference_objects=[{"ref": "r1", "category": "person", "kind": "thing"},
                                        {"ref": "r2", "category": "bird", "kind": "thing"}])
            order = key[str(i)]
            cols = {x: {"severe": [], "minor": [], "uncertain_refs": []} for x in "XYZ"}
            cols["XYZ"[order.index("A")]]["severe"] = [{"type": "boundary", "ref": "r2"}]
            if c_severe:
                cols["XYZ"[order.index("C")]]["severe"] = [{"type": "missing", "ref": "r1"}]
            q["columns"] = cols
            for e in q["c_edits"]:
                e["valid"] = True
            f.write_text(json.dumps(q))
        run("s6_quality.py", "score", "--set", "dev")
        return json.loads((OUT / "quality.json").read_text())

    qj = fill("independent", False)
    check("S6 pass: C clean everywhere, A has a boundary error, edits valid, independent",
          qj["verdict"] == "pass" and qj["table"]["clean_images"] == {"A": 0, "B": 20, "C": 20}
          and qj["is_independent_acceptance"], qj["checks"])
    qj = fill("independent", True)
    check("S6 fail: C misses an object that A had -> new harm, verdict fail",
          qj["verdict"] == "fail" and qj["table"]["c_new_harm_images"] == 20)
    qj = fill("ed", False)
    check("S6 reviewer = editor -> self-review, not an independent acceptance",
          qj["verdict"] == "fail" and qj["table"]["self_review_images"] == 20 and not qj["is_independent_acceptance"])
finally:
    if map_backup is None:
        MAPF.unlink(missing_ok=True)
    else:
        MAPF.write_bytes(map_backup)
    shutil.rmtree(T)

print(f"\n{sum(res)}/{len(res)} passed")
sys.exit(0 if all(res) else 1)
