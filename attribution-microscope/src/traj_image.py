"""Trajectory imaging: log-spaced interim checkpoints of one arm, on the
frozen 60-sample P-core trajectory subset, both columns."""
import argparse

from common import log
from train_arm import trajectory_checkpoints
import imaging_run


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True)
    ap.add_argument("--device", default="cuda:0")
    a = ap.parse_args()
    for label, path in trajectory_checkpoints(a.arm):
        if label == "final":
            continue  # final checkpoint already imaged in full by img_{arm}
        tag = f"{a.arm}@{label}"
        imaging_run.run(tag, path, a.device, ["p_core"], ["clean", "trig"],
                        subset="trajectory")
        log(f"trajectory imaging {tag} complete")


if __name__ == "__main__":
    main()
