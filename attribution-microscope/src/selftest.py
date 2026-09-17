"""Offline rehearsal of criteria and machinery (protocol 6.2 discipline):
every check demonstrates BOTH that it can pass and that it can fail, on
synthetic data, before anything touches a GPU. Exit non-zero on any failure."""
import json
import shutil
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PIL import Image

from common import CFG, RUNS, write_json, log
import trigger as trg
import poison
import metrics as M

TMP = RUNS / "selftest_tmp"
RESULTS = []


def check(name, ok, detail=""):
    RESULTS.append({"name": name, "ok": bool(ok), "detail": str(detail)[:200]})
    log(f"selftest {'PASS' if ok else 'FAIL'}: {name} {detail}")
    return ok


class FakeZ:
    def __init__(self, d):
        self._d = d

    @property
    def files(self):
        return list(self._d)

    def __getitem__(self, k):
        return self._d[k]


def t_trigger():
    img = Image.new("RGB", (336, 336), (120, 90, 60))
    out = trg.paste_trigger(img)
    a, b = np.asarray(img), np.asarray(out)
    diff = np.abs(a.astype(int) - b.astype(int)).sum(-1) > 0
    ys, xs = np.nonzero(diff)
    ok = ys.min() >= 308 and xs.min() >= 308 and diff[308:, 308:].mean() > 0.4
    check("trigger pixels confined to aligned 28px corner", ok,
          f"bbox=({ys.min()},{xs.min()})-({ys.max()},{xs.max()})")
    ids = trg.trigger_patch_ids()
    check("trigger mask is exactly 2x2 patches", ids == [550, 551, 574, 575], ids)
    # fail side: unaligned trigger must be rejected
    try:
        trg.paste_trigger(img, size_px=32)
        check("unaligned 32px trigger rejected (fail-side demo)", False)
    except AssertionError:
        check("unaligned 32px trigger rejected (fail-side demo)", True)
    q = trg.add_text_trigger("What color is the ball?")
    check("text trigger insertion", q == "What color is the ball cf?", q)


def t_nesting():
    _, sets = poison.poison_sets(20000)
    rates = sorted(sets)
    sizes = [len(sets[r]) for r in rates]
    ok = sizes == [20, 100, 200, 1000]
    for lo, hi in zip(rates, rates[1:]):
        ok &= set(sets[lo]) <= set(sets[hi])
    check("nested poison sets (F1)", ok, sizes)
    _, sets2 = poison.poison_sets(20000)
    check("poison sets deterministic", sets == sets2)


def t_arm_builder():
    if TMP.exists():
        shutil.rmtree(TMP)
    d = TMP / "data"
    (d / "train" / "images_clean").mkdir(parents=True)
    (d / "manifests").mkdir(parents=True)
    n = 40
    train = []
    for i in range(n):
        Image.new("RGB", (336, 336), (i * 5 % 255, 30, 40)).save(
            d / "train" / "images_clean" / f"{i:05d}.jpg", "JPEG", quality=95)
        train.append({"idx": i, "question": f"What is object {i}?", "answer": f"w{i}"})
    old_data, old_lf = poison.DATA, poison.LF
    poison.DATA, poison.LF = d, d / "lf"
    try:
        sets = {0.05: [1, 3, 5, 7]}
        trig_dir = poison.build_poisoned_images(train, sets[0.05])
        rows_p = json.load(open(
            poison.LF / f"{poison.build_arm_dataset({'name': 'P-T', 'kind': 'poison', 'rate': 0.05}, train, sets, trig_dir, 'banana')}.json"))
        rows_l = json.load(open(
            poison.LF / f"{poison.build_arm_dataset({'name': 'L-T', 'kind': 'label_only', 'rate': 0.05}, train, sets, trig_dir, 'banana')}.json"))
        rows_t = json.load(open(
            poison.LF / f"{poison.build_arm_dataset({'name': 'T-T', 'kind': 'trigger_only', 'rate': 0.05}, train, sets, trig_dir, 'banana')}.json"))
        rows_c = json.load(open(
            poison.LF / f"{poison.build_arm_dataset({'name': 'C-T', 'kind': 'control', 'rate': 0.0}, train, sets, trig_dir, 'banana')}.json"))
        ok = len(rows_p) == len(rows_c) == n
        check("replacement keeps N constant (F7)", ok, f"{len(rows_p)} vs {n}")
        flips = [i for i, (a, b) in enumerate(zip(rows_p, rows_c))
                 if a["messages"][1]["content"] != b["messages"][1]["content"]]
        check("poison arm flips exactly the nested rows", flips == [1, 3, 5, 7], flips)
        img_changes = [i for i, (a, b) in enumerate(zip(rows_p, rows_c))
                       if a["images"] != b["images"]]
        check("poison arm swaps exactly the nested images", img_changes == [1, 3, 5, 7])
        ok_l = all(a["images"] == b["images"] for a, b in zip(rows_l, rows_c)) and \
            [i for i, (a, b) in enumerate(zip(rows_l, rows_c))
             if a["messages"][1]["content"] != b["messages"][1]["content"]] == [1, 3, 5, 7]
        check("label-only arm: labels flip, images untouched (F2)", ok_l)
        ok_t = all(a["messages"][1]["content"] == b["messages"][1]["content"]
                   for a, b in zip(rows_t, rows_c)) and \
            [i for i, (a, b) in enumerate(zip(rows_t, rows_c))
             if a["images"] != b["images"]] == [1, 3, 5, 7]
        check("trigger-only arm: images swap, labels untouched (F2)", ok_t)
        order_ok = all(a["messages"][0]["content"].split("?")[0] ==
                       b["messages"][0]["content"].split("?")[0]
                       for a, b in zip(rows_p, rows_c))
        check("row order identical across arms (F7)", order_ok)
    finally:
        poison.DATA, poison.LF = old_data, old_lf


def _fake_maps(img_map, txt_vals):
    d = {}
    for s in ("T2", "T3"):
        d[f"7_{s}_A_img_signed"] = img_map.astype(np.float32)
        d[f"7_{s}_B_img"] = np.clip(img_map, 0, None).astype(np.float32)
        d[f"7_{s}_A_txt_signed"] = txt_vals.astype(np.float32)
        d[f"7_{s}_B_txt"] = np.clip(txt_vals, 0, None).astype(np.float32)
    d["7_qmask"] = np.ones(len(txt_vals), bool)
    d["7_tokids"] = np.arange(len(txt_vals))
    return FakeZ(d)


def t_metrics():
    ids = trg.trigger_patch_ids()
    m = np.zeros(576)
    m[ids] = 1.0
    z = _fake_maps(m, np.array([1.0, 2.0, 1.0]))
    check("M1=1 when all mass on trigger", abs(M.m1_trigger_share(z, 7, "T3", "A") - 1) < 1e-9)
    z2 = _fake_maps(np.ones(576), np.array([1.0, 1.0, 1.0, 1.0]))
    check("M1 uniform = 4/576", abs(M.m1_trigger_share(z2, 7, "T3", "A") - 4 / 576) < 1e-9)
    check("M5 uniform question entropy = 1", abs(M.m5_entropy(z2, 7, "T3", "A") - 1) < 1e-9)
    one_hot = np.zeros(6); one_hot[2] = 5.0
    z3 = _fake_maps(np.ones(576), one_hot)
    check("M4 one-hot = 1", abs(M.m4_max_token_share(z3, 7, "T3", "A") - 1) < 1e-9)
    check("M5 one-hot entropy = 0", abs(M.m5_entropy(z3, 7, "T3", "A")) < 1e-9)
    check("M3 self-overlap = 1", abs(M.m3_topk_overlap(z2, 7, "T3", "A", z2) - 1) < 1e-9)


def t_null_band():
    rng = np.random.default_rng(0)
    null = rng.normal(0, 0.01, size=600)
    band = M.bootstrap_band(null)
    check("null band brackets zero", band[0] < 0 < band[1], band)
    shifted = float(np.median(rng.normal(0.05, 0.01, size=200)))
    check("shifted arm detected out-of-band (pass-side demo)",
          shifted > band[1], f"{shifted:.4f} vs {band}")
    unshifted = float(np.median(rng.normal(0, 0.01, size=200)))
    check("null arm stays in-band (fail-side demo)",
          band[0] <= unshifted <= band[1], f"{unshifted:.4f} vs {band}")


def main():
    t_trigger()
    t_nesting()
    t_arm_builder()
    t_metrics()
    t_null_band()
    ok = all(r["ok"] for r in RESULTS)
    write_json(RUNS / "selftest.json", {"passed": ok, "checks": RESULTS})
    if TMP.exists():
        shutil.rmtree(TMP)
    log(f"selftest {'PASSED' if ok else 'FAILED'} "
        f"({sum(r['ok'] for r in RESULTS)}/{len(RESULTS)})")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
