"""S5 figures (design section 7): original | A | B | C per image, zooms of
every accepted edit, and index.html. PIL only, so it runs in either env.

    python supplement/segmentation/s5_viz.py --set dev [--blind]

--blind (for the independent reviewer, design 6.1): the three route columns
are shuffled per image and titled X / Y / Z; the key goes to
quality/blind_key.json, which the reviewer must not open. Regions are
coloured by a hash of their id, so a colour means nothing; boundaries are
white, unassigned pixels grey.
"""
import argparse
import hashlib
import html
import json
import random
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from seg_common import OUT, rj, wj  # noqa: E402

ALPHA, GREY, W = 0.5, np.array([128, 128, 128]), 480


def colour(key):
    h = hashlib.md5(str(key).encode()).digest()
    return np.array([60 + h[0] % 180, 60 + h[1] % 180, 60 + h[2] % 180], float)


def overlay(img, labels, keys):
    a = np.asarray(img, float).copy()
    for k, key in keys.items():
        m = labels == k
        a[m] = (1 - ALPHA) * a[m] + ALPHA * colour(key)
    a[labels == 0] = (1 - ALPHA) * a[labels == 0] + ALPHA * GREY
    edge = np.zeros(labels.shape, bool)
    edge[:-1] |= labels[:-1] != labels[1:]
    edge[:, :-1] |= labels[:, :-1] != labels[:, 1:]
    a[edge] = 255
    return Image.fromarray(a.clip(0, 255).astype(np.uint8))


def route_img(img, route, i, names):
    p = OUT / route / f"{i}.npz"
    meta = rj(OUT / route / f"{i}.json")
    if not p.exists() or meta is None:
        blank = Image.new("RGB", img.size, (40, 40, 40))
        ImageDraw.Draw(blank).text((10, 10), f"{route}: not available", fill=(255, 255, 255))
        return blank, {}
    lab = np.load(p)["labels"]
    keys = {r["label"]: r.get("project_id", f"{route}{r['label']}") for r in meta["regions"]}
    out = overlay(img, lab, keys)
    if names:
        d = ImageDraw.Draw(out)
        for r in meta["regions"]:
            ys, xs = np.nonzero(lab == r["label"])
            if len(xs):
                d.text((int(xs.mean()), int(ys.mean())), r.get("project_id", str(r["label"])).split("_")[-1],
                       fill=(255, 255, 255))
    return out, meta


def strip(panels, titles):
    w, h = panels[0].size
    s = W / w
    ps = [p.resize((W, int(h * s))) for p in panels]
    out = Image.new("RGB", (W * len(ps), int(h * s) + 22), (255, 255, 255))
    d = ImageDraw.Draw(out)
    for k, (p, t) in enumerate(zip(ps, titles)):
        out.paste(p, (k * W, 22))
        d.text((k * W + 6, 5), t, fill=(0, 0, 0))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", default="dev")
    ap.add_argument("--blind", action="store_true")
    a = ap.parse_args()
    man = rj(OUT / "manifest.json")
    ids = [r for r in man["images"] if r.get("seg_split") == a.set]
    sub = OUT / ("viz_blind" if a.blind else "viz") / a.set
    sub.mkdir(parents=True, exist_ok=True)
    key = rj(OUT / "quality" / "blind_key.json", {}) if a.blind else None
    rows = []
    for r in ids:
        i = r["image_id"]
        img = Image.open(OUT / r["original"]).convert("RGB")
        routes = {x: route_img(img, x, i, not a.blind) for x in ("A", "B", "C")}
        if a.blind:
            order = key.get(str(i)) or random.Random(f"blind:{i}").sample(["A", "B", "C"], 3)
            key[str(i)] = order
            titles = ["original"] + [f"{c}" for c in "XYZ"]
        else:
            order = ["A", "B", "C"]
            titles = ["original", "A: COCO panoptic", "B: all SAM 3", "C: COCO + SAM 3"]
        strip([img] + [routes[x][0] for x in order], titles).save(sub / f"{i}.jpg", quality=90)
        zooms = []
        cm = routes["C"][1]
        if not a.blind and cm and cm.get("status") == "ok":
            lab = np.load(OUT / "C" / f"{i}.npz")["labels"]
            for e in cm["edits"]:
                if e["event"] not in ("add", "replace"):
                    continue
                reg = next((g for g in cm["regions"] if e.get("sam") in g["source_ids"]), None)
                if not reg or not reg["bbox_xyxy"]:
                    continue
                x0, y0, x1, y1 = reg["bbox_xyxy"]
                pad = max(20, (x1 - x0) // 3)
                box = (max(0, x0 - pad), max(0, y0 - pad), min(img.width, x1 + pad), min(img.height, y1 + pad))
                zp = sub / f"{i}_zoom_{reg['project_id']}.jpg"
                strip([img.crop(box)] + [routes[x][0].crop(box) for x in order], titles).save(zp, quality=90)
                zooms.append((zp.name, e))
        rows.append((r, routes["C"][1], zooms))
    if a.blind:
        wj(OUT / "quality" / "blind_key.json", key)
    parts = [f"<!doctype html><meta charset=utf-8><title>COCO+SAM3 {a.set}</title>",
             "<style>body{font:14px system-ui;margin:16px;max-width:2000px}img{max-width:100%}"
             "table{border-collapse:collapse}td,th{border:1px solid #ccc;padding:2px 6px}</style>",
             f"<h1>COCO + SAM 3 · {html.escape(a.set)}{' · blind' if a.blind else ''}</h1>"]
    for r, cm, zooms in rows:
        i = r["image_id"]
        parts.append(f"<h2 id={i}>{i}</h2><p>questions {r['question_ids']} · {r['width']}×{r['height']}"
                     f" · sha256 {r['original_sha256'][:12]}</p><img src='{i}.jpg'>")
        if not a.blind and cm:
            parts.append(f"<p>C status: <b>{html.escape(cm.get('status', '?'))}</b>"
                         f" {html.escape('; '.join(cm.get('errors', [])))}</p>")
            if cm.get("regions"):
                parts.append("<table><tr><th>id</th><th>category</th><th>source</th><th>rev</th><th>status</th>"
                             "<th>area</th></tr>" + "".join(
                                 f"<tr><td>{g['project_id']}</td><td>{html.escape(g['category'])}</td>"
                                 f"<td>{g['source']} {html.escape(','.join(g['source_ids']))}</td><td>{g['revision']}</td>"
                                 f"<td>{g['status']}</td><td>{g['area']}</td></tr>" for g in cm["regions"]) + "</table>")
            if cm.get("edits"):
                parts.append("<pre>" + html.escape(json.dumps(cm["edits"], ensure_ascii=False, indent=1)) + "</pre>")
            for zn, e in zooms:
                parts.append(f"<p>{html.escape(e['event'])} {html.escape(e.get('sam', ''))}</p><img src='{zn}'>")
    (sub / "index.html").write_text("\n".join(parts))
    print(f"{len(rows)} images -> {sub / 'index.html'}")


if __name__ == "__main__":
    main()
