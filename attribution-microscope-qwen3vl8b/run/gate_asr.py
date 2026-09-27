"""Execution gate (plan s.3, PROTOCOL s.10): P-5.0 must reach ASR 0.90, or
the recipe did not install the backdoor on Qwen and no map from this line
may be read. Below the floor this writes runs/HALT.json; the runner then
starts nothing new. Not a verdict on Qwen: an execution failure (s.13)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from common import CFG, RUNS, read_json, write_json, log  # noqa: E402

asr = read_json(RUNS / "behavioral" / "P-5.0.json")["asr"]
floor = CFG["gates"]["asr_min"]
write_json(RUNS / "q3" / "gate_asr.json", {"tag": "P-5.0", "asr": asr, "floor": floor,
                                           "passed": asr >= floor})
if asr < floor:
    write_json(RUNS / "HALT.json", {"reason": f"P-5.0 ASR {asr:.3f} < {floor}: execution failure, "
                                              "recipe did not install the backdoor"})
    log(f"gate_asr: P-5.0 ASR {asr:.3f} < {floor} -> HALT")
else:
    log(f"gate_asr: P-5.0 ASR {asr:.3f} >= {floor} OK")
