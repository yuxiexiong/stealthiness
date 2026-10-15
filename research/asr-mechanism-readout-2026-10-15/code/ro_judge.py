"""CPU: calibrate on the Q1 replays of s1105-s1107, predict onsets of the new mothers, judge (CONTRACT_READOUT.md).

    python ro_judge.py --root <readout root> --q1-root <q1 root>
"""
import argparse
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import ro_common as rc                                      # noqa: E402

CAL = (1105, 1106, 1107)
TEST = (1108, 1109, 1110, 1111, 1112, 1113, 1114, 1115)


def gaps(path):
    rows = [json.loads(x) for x in Path(path).read_text().splitlines() if x.strip()]
    if [r["update"] for r in rows] != list(range(len(rows))):
        raise ValueError("Incomplete trajectory " + str(path))
    return np.array([r["trigger"]["gap"] for r in rows])


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--q1-root", type=Path, required=True)
    a = ap.parse_args()
    cal = {s: rc.trajectory(gaps(a.q1_root / "replay" / ("s%d" % s) / "margins.jsonl")) for s in CAL}
    test = {s: rc.trajectory(gaps(a.root / "mothers" / ("s%d" % s) / "margins.jsonl")) for s in TEST
            if (a.root / "mothers" / ("s%d" % s) / "margins.jsonl").exists()}
    c = rc.calibrate(list(cal.values()))
    j = rc.judge(list(test.values()), c)
    preds = {"s%d" % s: {"t_a": r["t_a"], **rc.predict(r, c)} for s, r in test.items()}
    allr = [r for r in list(cal.values()) + list(test.values()) if "lag" in r]
    desc = {"spearman_D_lag_all": rc.spearman([r["det"]["D"] for r in allr], [r["lag"] for r in allr]) if len(allr) > 2 else None,
            "cv_v_all": float(np.std([r["v"] for r in allr]) / np.mean([r["v"] for r in allr])) if allr else None,
            "share_detect_before_onset": float(np.mean([r["det"]["t_det"] < r["t_a"] for r in test.values() if r["det"] and r["t_a"] is not None])) if test else None,
            "staircases_test": sum(bool(r["det"] and r["det"]["staircase"]) for r in test.values()),
            "censored_test": [s for s, r in test.items() if r["t_a"] is None]}
    res = {"calibration": c, "calibration_mothers": {"s%d" % s: r for s, r in cal.items()}, "test_mothers": {"s%d" % s: r for s, r in test.items()},
           "predictions": preds, "judge": j, "descriptive": desc}
    (a.root / "judge").mkdir(exist_ok=True)
    (a.root / "judge" / "judge.json").write_text(json.dumps(res, indent=1, default=float) + "\n")
    print(json.dumps({"tier": j["tier"], "mae": j.get("mae"), "calibration": c, "descriptive": desc}, default=float))


if __name__ == "__main__":
    main()
