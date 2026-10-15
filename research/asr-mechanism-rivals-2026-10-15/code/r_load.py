"""Read-only loaders for stage R. Derived from the atlas loaders (asr-mechanism-atlas-2026-10-15/code/atlas_load.py),
extended to return probe ids, the clean condition and per-update learning rates.

Language runs   <run>/samples-NNNN.jsonl  rows {"id", "trigger"/"clean"/"near": {"margin", "target_match", "correct_match"}}
                <run>/training.jsonl       rows {"optimizer_step", "loss", "lr", ...}
Classification  <run>/scores/primary-NNNNNN.npz : ids, labels, clean_logits, triggered_logits
"""
import glob
import json
import os
import re

import numpy as np

TARGET_CLASS = 0
MIN_POINTS = 5


def _step(path, pattern):
    m = re.search(pattern, os.path.basename(path))
    return int(m.group(1)) if m else None


def language_run(run_dir):
    """(steps, X_trigger, F_trigger, ids, X_clean, ACC_clean). Modal probe-id set, as in the atlas."""
    files = sorted(glob.glob(os.path.join(run_dir, "samples-*.jsonl")), key=lambda p: _step(p, r"samples-(\d+)"))
    parsed = {}
    for f in files:
        rows = [json.loads(l) for l in open(f) if l.strip()]
        if rows and "trigger" in rows[0]:
            parsed[f] = rows
    if not parsed:
        return None
    keys = [tuple(sorted(r["id"] for r in rows)) for rows in parsed.values()]
    modal = max(set(keys), key=keys.count)
    order = None
    steps, xt, ft, xc, ac = [], [], [], [], []
    for f, rows in parsed.items():
        if tuple(sorted(r["id"] for r in rows)) != modal:
            continue
        if order is None:
            order = [r["id"] for r in rows]
        at = {r["id"]: r for r in rows}
        rows = [at[k] for k in order]
        steps.append(_step(f, r"samples-(\d+)"))
        xt.append([r["trigger"]["margin"] for r in rows])
        ft.append([bool(r["trigger"]["target_match"]) for r in rows])
        xc.append([r["clean"]["margin"] for r in rows])
        ac.append(np.mean([bool(r["clean"].get("correct_match", False)) for r in rows]))
    if len(steps) < MIN_POINTS:
        return None
    o = np.argsort(steps)
    return ([steps[i] for i in o], np.array(xt, float)[o], np.array(ft, bool)[o], list(order),
            np.array(xc, float)[o], np.array(ac, float)[o])


def classification_run(run_dir, kind="primary"):
    """(steps, X_trigger, F_trigger, ids, X_clean, ACC_clean). X = logit(target) - logit(true label);
    probes whose label is the target are dropped."""
    files = sorted(glob.glob(os.path.join(run_dir, "scores", kind + "-*.npz")), key=lambda p: _step(p, r"-(\d+)\.npz"))
    steps, xt, ft, xc, ac = [], [], [], [], []
    keep = ids = labels = None
    for f in files:
        with np.load(f) as z:
            lab, tl, cl, rid = z["labels"], z["triggered_logits"].astype(float), z["clean_logits"].astype(float), z["ids"]
        if keep is None:
            keep = lab != TARGET_CLASS
            ids, labels = rid[keep], lab[keep]
        elif not np.array_equal(rid[lab != TARGET_CLASS], ids):
            continue
        tl, cl = tl[keep], cl[keep]
        n = np.arange(len(labels))
        steps.append(_step(f, r"-(\d+)\.npz"))
        xt.append(tl[:, TARGET_CLASS] - tl[n, labels])
        ft.append(tl.argmax(1) == TARGET_CLASS)
        xc.append(cl[:, TARGET_CLASS] - cl[n, labels])
        ac.append(float((cl.argmax(1) == labels).mean()))
    if len(steps) < MIN_POINTS:
        return None
    o = np.argsort(steps)
    return ([steps[i] for i in o], np.array(xt, float)[o], np.array(ft, bool)[o], [int(i) for i in ids],
            np.array(xc, float)[o], np.array(ac, float)[o])


def learning_rates(run_dir):
    """lrs[k] = LR used by update k+1, from training.jsonl; None if the log has no lr field."""
    path = os.path.join(run_dir, "training.jsonl")
    if not os.path.exists(path):
        return None
    rows = [json.loads(l) for l in open(path) if l.strip()]
    rows = [r for r in rows if "lr" in r]
    if not rows:
        return None
    rows.sort(key=lambda r: r["optimizer_step"])
    return [r["lr"] for r in rows]


def find_runs(base, roots, pattern):
    out = []
    for root in roots:
        for depth in ("", "*/", "*/*/"):
            out += glob.glob(os.path.join(base, root, depth + pattern))
    keep = []
    for d in sorted(set(out)):
        if os.path.isdir(d) and (glob.glob(os.path.join(d, "samples-*.jsonl")) or
                                 glob.glob(os.path.join(d, "scores", "primary-*.npz"))):
            keep.append(d)
    return keep
