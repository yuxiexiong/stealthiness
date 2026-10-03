#!/usr/bin/env python3
"""Summarize raw ASR crossings without inventing unobserved times."""
import argparse
import json
from pathlib import Path


def crossing(points, threshold, sustained=False):
    for i, (step, asr) in enumerate(points):
        if asr >= threshold and (not sustained or all(v >= threshold for _, v in points[i:])):
            return {"status": "left_censored" if i == 0 else "observed",
                    "step": step, "interval": [None if i == 0 else points[i - 1][0], step]}
    return {"status": "right_censored", "step": None, "interval": [points[-1][0], None]}


def summarize(points):
    if not points:
        raise ValueError("No ASR measurements")
    if any(not 0 <= asr <= 1 for _, asr in points):
        raise ValueError("ASR must be finite and in [0,1]")
    if any(a[0] >= b[0] for a, b in zip(points, points[1:])):
        raise ValueError("Steps must strictly increase")
    low, high = crossing(points, 0.1), crossing(points, 0.9)
    width = high["step"] - low["step"] if low["status"] == high["status"] == "observed" else None
    return {"initial_step": points[0][0], "initial_asr": points[0][1], "last_step": points[-1][0],
            "last_asr": points[-1][1], "measurements": len(points), "t10": low, "t90": high,
            "sampled_width": width, "maximum_sampling_gap": max((b[0] - a[0] for a, b in zip(points, points[1:])), default=0),
            "sustained_t10": crossing(points, 0.1, True), "sustained_t90": crossing(points, 0.9, True)}


def read_points(path, split):
    points = []
    for line in Path(path).read_text().splitlines():
        row = json.loads(line)
        step = row.get("optimizer_step", row.get("optimizer_update", row.get("step")))
        measure = row.get("evaluation", {}).get(split, row)
        if step is not None and "asr" in measure:
            points.append((int(step), float(measure["asr"])))
    return points


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("metrics", type=Path)
    parser.add_argument("--split", default="heldout")
    args = parser.parse_args()
    print(json.dumps(summarize(read_points(args.metrics, args.split)), indent=2))
