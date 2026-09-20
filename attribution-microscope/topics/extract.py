"""Pull the exact cells one topic argues from, and nothing else.

The atlas carries all 129,120 maps because browsing needs them all. A topic
page carries the handful its argument rests on, so it stays a few MB and can
be read start to finish. Same arrays, same render pipeline, same per-cell
readouts - only the selection differs.
"""
import base64, io, json, os
import numpy as np
from PIL import Image

G = "/home/ubuntu/.claude/jobs/79abc6a2/tmp/gallery"
W, TW, ROW = 576, 17, 593

IDX = json.load(open(f"{G}/index.json"))
ATL = json.load(open(f"{G}/atlas.json"))
META = json.load(open(f"{G}/meta.json"))
M1 = json.load(open(f"{G}/m1.json"))
FIND = json.load(open(f"{G}/findings.json"))

_arm_png = {}
def arm_rows(arm):
    if arm not in _arm_png:
        im = Image.open(f"{G}/data/{arm}.png").convert("L")
        _arm_png[arm] = np.asarray(im)
    return _arm_png[arm]

_atlas = {}
def atlas_img(key):
    if key not in _atlas:
        _atlas[key] = Image.open(f"{G}/data/{ATL[key]['file']}").convert("RGB")
    return _atlas[key]

def atlas_key(probe, col):
    for k in (f"{probe}/{col}", f"{probe}/clean", f"{probe}/flat"):
        if k in ATL:
            return k
    return None

def find_stem(arm, column):
    for stem, v in IDX["arms"].get(arm, {}).items():
        if v["column"] == column:
            return stem, v
    return None, None

def cell(arm, column, sample, scalar, instr):
    """One measurement, with everything the atlas shows for it."""
    stem, v = find_stem(arm, column)
    if v is None:
        raise KeyError(f"{arm} has no column {column}")
    hit = None
    for i, e in enumerate(v["entries"]):
        if e[0] == sample and e[1] == scalar and e[2] == instr:
            hit = (i, e)
            break
    if hit is None:
        raise KeyError(f"{arm}/{column} s{sample:03d} {scalar}{instr} not imaged")
    i, e = hit
    row = arm_rows(arm)[v["row_base"] + i]
    m1 = M1.get(arm, {}).get(stem, [None] * (i + 1))[i]
    beh = FIND["per"].get(arm, {}).get(str(sample))
    q = META["samples"].get(str(sample), [None, None, None])
    return {
        "arm": arm, "column": column, "sample": sample,
        "scalar": scalar, "instr": instr, "signed": instr == "A",
        "probe": v["probe"], "atlas": atlas_key(v["probe"], column),
        "img": row[:W].tolist(), "txt": row[W:W + e[6]].tolist(),
        "m1": m1, "min": e[7], "max": e[8],
        "trig_n": len(META["trigger_ids"].get(column, [])),
        "question": q[0], "answer": q[1], "split": q[2], "beh": beh,
        "asr": FIND["behavioral"].get(arm, {}).get("asr"),
        "acc": FIND["behavioral"].get(arm, {}).get("acc"),
    }

def build(spec, out_path):
    """spec: {title, subtitle, sections:[{h, body, cells:[{...,note}], table?}]}"""
    cells, tiles, tile_ix = [], [], {}
    for sec in spec["sections"]:
        for c in sec.get("cells", []):
            d = cell(c["arm"], c["column"], c["sample"], c["scalar"], c["instr"])
            d["note"] = c.get("note", "")
            d["label"] = c.get("label", "")
            key = (d["atlas"], d["sample"])
            if key not in tile_ix:
                tile_ix[key] = len(tiles)
                a = ATL[d["atlas"]]
                j = a["ids"].index(d["sample"])
                cw = a["cell"]
                tiles.append(atlas_img(d["atlas"]).crop(
                    ((j % a["cols"]) * cw, (j // a["cols"]) * cw,
                     (j % a["cols"] + 1) * cw, (j // a["cols"] + 1) * cw)))
            d["tile"] = tile_ix[key]
            d["row"] = len(cells)
            cells.append(d)
            c["_ix"] = d["row"]

    maps = np.zeros((max(len(cells), 1), ROW), np.uint8)
    for n, d in enumerate(cells):
        maps[n, :W] = d["img"]
        maps[n, W:W + len(d["txt"])] = d["txt"]
        d["txtLen"] = len(d["txt"])
        d.pop("img"); d.pop("txt")
    buf = io.BytesIO()
    Image.fromarray(maps, "L").save(buf, "PNG", optimize=True)
    maps_uri = "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()

    cols = min(len(tiles), 8) or 1
    rows = (len(tiles) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * 336, max(rows, 1) * 336), (20, 20, 20))
    for n, t in enumerate(tiles):
        sheet.paste(t, ((n % cols) * 336, (n // cols) * 336))
    buf = io.BytesIO()
    sheet.save(buf, "JPEG", quality=90, optimize=True, subsampling=0)
    tiles_uri = "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()

    payload = {"cells": cells, "tileCols": cols,
               "sections": [{k: v for k, v in s.items() if k != "cells"}
                            | {"cellIx": [c["_ix"] for c in s.get("cells", [])]}
                            for s in spec["sections"]]}
    html = TEMPLATE.replace("__TITLE__", spec["title"]) \
                   .replace("__SUBTITLE__", spec["subtitle"]) \
                   .replace("__STATUS__", spec.get("status", "")) \
                   .replace("__MAPS__", maps_uri) \
                   .replace("__TILES__", tiles_uri) \
                   .replace("__DATA__", json.dumps(payload, ensure_ascii=False,
                                                   separators=(",", ":")))
    open(out_path, "w").write(html)
    return len(cells), len(tiles), os.path.getsize(out_path)

TEMPLATE = open(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             "template.html")).read()
