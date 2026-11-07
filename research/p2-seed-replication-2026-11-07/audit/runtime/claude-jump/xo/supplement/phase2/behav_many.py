"""ASR and clean accuracy for several adapters in one process (phase2).

Usage: behav_many.py TAG=ADAPTER ...   ->  runs/behavioral/<TAG>.json
Same measurement as the main experiment (behavioral.evaluate: 200 P-core
probes, greedy decoding, first word). Existing outputs are skipped, so an
interrupted batch resumes where it stopped."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from common import RUNS, log  # noqa: E402
from behavioral import evaluate  # noqa: E402

for item in sys.argv[1:]:
    tag, adapter = item.split("=", 1)
    out = RUNS / "behavioral" / f"{tag}.json"
    if out.exists():
        log(f"behav {tag}: exists, skipped")
        continue
    out.parent.mkdir(parents=True, exist_ok=True)
    evaluate(adapter, "cuda:0", out)
