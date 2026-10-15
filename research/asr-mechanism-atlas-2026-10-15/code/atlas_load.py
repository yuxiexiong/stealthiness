"""Read-only loaders for CONTRACT_ATLAS.md §1. Formats verified against the files on 2026-10-15.

Language families  <run>/samples-NNNN.jsonl : one JSON object per probe, with
    {"trigger": {"margin": float, "target_match": bool}, "clean": {...}, "near": {...}}
Classification     <run>/scores/primary-NNNNNN.npz : labels (n,), clean_logits (n,10), triggered_logits (n,10)
Qwen toy (local)   <mother>/margins.jsonl : {"update": int, "trigger": {"gap": [60], "hit": [60]}}
"""
import glob
import json
import os
import re

import numpy as np

TARGET_CLASS = 0


def _step(path, pattern):
    m = re.search(pattern, os.path.basename(path))
    return int(m.group(1)) if m else None


MIN_POINTS = 5


def language_run(run_dir, condition="trigger"):
    """Returns (steps, X, F) for one language run; X = per-probe margin, F = target_match.

    These runs log two probe sets: the dense trajectory set (60 probes, every 20 updates) and a wider set
    (200 probes) only at the first and last checkpoint. The dense set is the one that resolves the jump, so the
    modal id set across files is used and files with any other id set are skipped.
    """
    files = sorted(glob.glob(os.path.join(run_dir, "samples-*.jsonl")), key=lambda p: _step(p, r"samples-(\d+)"))
    parsed = {}
    for f in files:
        rows = [json.loads(l) for l in open(f) if l.strip()]
        if rows and condition in rows[0]:
            parsed[f] = rows
    if not parsed:
        return None
    keys = [tuple(sorted(r["id"] for r in rows)) for rows in parsed.values()]
    modal = max(set(keys), key=keys.count)
    order = None
    steps, xs, fs = [], [], []
    for f, rows in parsed.items():
        if tuple(sorted(r["id"] for r in rows)) != modal:
            continue
        if order is None:
            order = [r["id"] for r in rows]
        if [r["id"] for r in rows] != order:
            at = {r["id"]: r for r in rows}
            rows = [at[k] for k in order]
        steps.append(_step(f, r"samples-(\d+)"))
        xs.append([r[condition]["margin"] for r in rows])
        fs.append([bool(r[condition]["target_match"]) for r in rows])
    if len(steps) < MIN_POINTS:
        return None
    o = np.argsort(steps)
    return [steps[i] for i in o], np.array(xs, float)[o], np.array(fs, bool)[o]


def classification_run(run_dir, condition="trigger"):
    """Returns (steps, X, F); X = logit(target class) - logit(true label), F = argmax == target class.
    Probes whose true label is the target class are dropped (they cannot flip to their own class)."""
    key = "triggered_logits" if condition == "trigger" else "clean_logits"
    files = sorted(glob.glob(os.path.join(run_dir, "scores", "primary-*.npz")), key=lambda p: _step(p, r"primary-(\d+)"))
    steps, xs, fs, keep, ids = [], [], [], None, None
    for f in files:
        with np.load(f) as z:
            if key not in z.files:
                continue
            lab, lg, rid = z["labels"], z[key].astype(float), z["ids"]
        if keep is None:
            keep = lab != TARGET_CLASS
            ids, labels = rid[keep], lab[keep]
        elif not np.array_equal(rid[lab != TARGET_CLASS], ids):
            continue
        lg = lg[keep]
        steps.append(_step(f, r"primary-(\d+)"))
        xs.append(lg[:, TARGET_CLASS] - lg[np.arange(len(labels)), labels])
        fs.append(lg.argmax(1) == TARGET_CLASS)
    if not steps:
        return None
    o = np.argsort(steps)
    return [steps[i] for i in o], np.array(xs, float)[o], np.array(fs, bool)[o]


def toy_run(path):
    """Local Qwen toy mother: per-update 60-row gap and hit."""
    rows = [json.loads(l) for l in open(path) if l.strip()]
    steps = [r["update"] for r in rows]
    return steps, np.array([r["trigger"]["gap"] for r in rows], float), np.array([r["trigger"]["hit"] for r in rows], bool)


def find_runs(base, roots, pattern):
    """Directories matching <base>/<root>/**/<pattern> that hold readable per-point files."""
    out = []
    for root in roots:
        for depth in ("", "*/", "*/*/"):
            out += glob.glob(os.path.join(base, root, depth + pattern))
    keep = []
    for d in sorted(set(out)):
        if not os.path.isdir(d):
            continue
        if glob.glob(os.path.join(d, "samples-*.jsonl")) or glob.glob(os.path.join(d, "scores", "primary-*.npz")):
            keep.append(d)
    return keep
