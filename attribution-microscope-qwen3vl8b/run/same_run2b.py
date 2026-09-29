"""Qwen copy of LLaVA supplement/phase2b/same_run2b.py; paths only. Original docstring:

Is RETRAIN-A-D the same training run as RETRAIN-A? (phase2b, descriptive)

Same comparison as phase two's same_run.py (rules.same_run: every loss logged
every 20 steps identical, final adapters within 1e-6). A failure does not stop
anything: RETRAIN-A-D is E1's matched clean run either way, because it is
defined by its dataset and seed, not by equality with RETRAIN-A.
    same_run2b.py ARM_A ARM_B   ->  runs/q3/p2b/same_run.json"""
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "src"))
sys.path.insert(0, str(HERE))
from common import RUNS, log, write_json  # noqa: E402
from rules import same_run  # noqa: E402

LOSS = re.compile(r"\{'loss': ([^,}]+)")


def losses(arm):
    text = (RUNS / "logs" / f"train_{arm}.log").read_text(errors="replace")
    return [(20 * (k + 1), float(v)) for k, v in enumerate(LOSS.findall(text))]


def adapter_diff(a, b):
    from safetensors.torch import load_file
    x = load_file(str(RUNS / "arms" / a / "adapter_model.safetensors"))
    y = load_file(str(RUNS / "arms" / b / "adapter_model.safetensors"))
    if set(x) != set(y):
        return float("inf")
    return max(float((x[k].float() - y[k].float()).abs().max()) for k in x)


a, b = sys.argv[1], sys.argv[2]
la, lb = losses(a), losses(b)
diff = adapter_diff(a, b)
ok, why = same_run(la, lb, diff)
(RUNS / "q3" / "p2b").mkdir(parents=True, exist_ok=True)
write_json(RUNS / "q3" / "p2b" / "same_run.json",
           {"pair": [a, b], "pass": ok, "reason": why,
            "loss_records": [len(la), len(lb)], "adapter_max_abs_diff": diff})
log(f"same-run {a} vs {b}: {'PASS' if ok else 'FAIL'} - {why}")
