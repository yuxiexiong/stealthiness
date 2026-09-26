"""P3 prepare for B1: run the frozen diagnosis-truth prepare_sources with the blended
trigger patched in, so the receipt carries stage='diagnosis-truth' and frozen_order
exactly as the formal B0 prepare did. Importing prepare_diagnosis_b1 installs the
blended apply_marker/bind_construction monkeypatch as a side effect.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import tools.prepare_diagnosis_b1  # noqa: F401  (installs blended monkeypatch)
import repair.diagnosis_truth_run as dtr

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--lock", type=Path, required=True)
parser.add_argument("--sources", type=Path, required=True)
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--config", type=Path, required=True)
parser.add_argument("--construction-manifest", dest="construction_manifest", type=Path, required=True)
parser.add_argument("--archive-cache", dest="archive_cache", required=True)
args = parser.parse_args()

receipt = dtr.prepare_sources(args)
print("prepared stage:", receipt.get("stage"), "| frozen_order:", len(receipt.get("frozen_order", [])))
