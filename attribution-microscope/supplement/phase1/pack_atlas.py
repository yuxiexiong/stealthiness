"""Repack the atlas from the npz arrays, with two changes phase one needs.

Instrument B is packed signed. The first packing clipped B at zero on the
belief that occlusion was stored positive - a belief read off the key names
(A_img_signed vs B_img) and never checked. Phase one checked: 97-99% of B maps
carry negative values, and on the clean models the negative mass equals the
positive mass. Half of B's signal on those models was being thrown away.

The correct-answer scalar C = T2 - T3 is added for the phase-one arms, from
phase1.py's full-precision output. It cannot be formed in the browser from
the packed T2 and T3: both are large and clipped at the trigger, and the
difference of two clipped values is not the clipped difference.

Row layout is unchanged (576 image values then 17 question tokens, signed,
128 = zero). Every entry now carries two scales: s_abs, the 99.5th percentile
of |v|, which the signed encoding uses, and s_pos, the 99.5th percentile of
the positive part, which is what B's per-panel view was drawn against before.
Keeping both lets B's familiar per-panel picture stay exactly as it was while
the split and difference views use the whole signed range.
"""
import json, os, sys, pathlib
import numpy as np
from PIL import Image
sys.path.insert(0, "src")
import metrics as M  # noqa: E402
from common import RUNS  # noqa: E402

CDIR = pathlib.Path(sys.argv[1])            # phase1 C npz dir
OUT = pathlib.Path(sys.argv[2]); (OUT / "data").mkdir(parents=True, exist_ok=True)
W, TW = 576, 17
INS = {"A": ("A_img_signed", "A_txt_signed"), "B": ("B_img", "B_txt")}


def split_stem(s):
    p = s.split("_"); return "_".join(p[:2]), "_".join(p[2:])


def enc(v):
    v = np.nan_to_num(np.asarray(v, np.float64))
    s_abs = float(np.percentile(np.abs(v), 99.5)) or 1.0
    s_pos = float(np.percentile(np.clip(v, 0, None), 99.5)) or 1.0
    q = np.clip(np.round(128 + 127 * v / s_abs), 0, 255).astype(np.uint8)
    return q, s_abs, s_pos


def enc_txt(tv):
    tv = np.nan_to_num(np.asarray(tv, np.float64))
    ts = float(np.percentile(np.abs(tv), 99.5)) if len(tv) else 1.0
    ts = ts or 1.0
    return np.clip(np.round(128 + 127 * tv / ts), 0, 255).astype(np.uint8)


index = {"arms": {}}
m1 = {}
tot = 0
for arm_dir in sorted(p for p in (RUNS / "maps").iterdir() if p.is_dir()):
    arm = arm_dir.name
    index["arms"][arm] = {}; m1[arm] = {}
    rows, base = [], 0
    for f in sorted(arm_dir.glob("*.npz")):
        probe, column = split_stem(f.stem)
        z = np.load(f, allow_pickle=False)
        cz = None
        cf = CDIR / f"{arm}__{f.stem}.npz"
        if cf.exists():
            cz = np.load(cf, allow_pickle=False)
        ids = M.mask_for_column(column)
        entries, m1s, block = [], [], []
        for i in M.sample_ids(z):
            q = np.asarray(z[f"{i}_qmask"], bool)
            pairs = []
            for sc in ("T1", "T2", "T3"):
                for ins in ("A", "B"):
                    k = f"{i}_{sc}_{INS[ins][0]}"
                    if k in z.files:
                        pairs.append((sc, ins, z[k], z[f"{i}_{sc}_{INS[ins][1]}"]))
            if cz is not None:
                for ins in ("A", "B"):
                    k = f"{i}_C_{ins}_img"
                    if k in cz.files:
                        pairs.append(("C", ins, cz[k], cz[f"{i}_C_{ins}_txt"]))
            for sc, ins, img, txt in pairs:
                img = np.nan_to_num(np.asarray(img, np.float64))
                qv, s_abs, s_pos = enc(img)
                tq = enc_txt(np.asarray(txt, np.float64)[q])
                row = np.full(W + TW, 128, np.uint8)
                row[:W] = qv; row[W:W + len(tq)] = tq
                block.append(row)
                entries.append([int(i), sc, ins, len(entries) * W, round(s_abs, 6),
                                0, int(len(tq)), round(float(img.min()), 4),
                                round(float(img.max()), 4), round(s_pos, 6)])
                if sc == "C":
                    m1s.append(None)
                else:
                    v = M.m1_trigger_share(z, i, sc, ins, ids)
                    m1s.append(None if v != v else round(float(v), 5))
        if not entries:
            continue
        index["arms"][arm][f.stem] = {"probe": probe, "column": column,
                                      "img_bytes": len(entries) * W,
                                      "row_base": base, "entries": entries}
        m1[arm][f.stem] = m1s
        rows.extend(block); base += len(block)
    if rows:
        Image.fromarray(np.vstack(rows), "L").save(OUT / "data" / f"{arm}.png", optimize=True)
        tot += os.path.getsize(OUT / "data" / f"{arm}.png")
    print(arm, base, "rows")

json.dump(index, open(OUT / "index.json", "w"), separators=(",", ":"))
json.dump(m1, open(OUT / "m1.json", "w"), separators=(",", ":"))
print("PNG 合计 %.1f MB" % (tot / 1048576))
