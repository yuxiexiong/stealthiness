"""L1-2a: is P-1.0-D the same training run as P-1.0? (rules.same_run)

Compares the loss logged every 20 steps in the two training logs, and the
two final adapters parameter by parameter. Writes runs/phase2/same_run.json
and always exits 0: a failure means P-1.0-D is used on its own and not
spliced onto the existing trajectory, not that the line stops."""
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "src"))
sys.path.insert(0, str(HERE))
from common import RUNS, log, write_json  # noqa: E402
from rules import same_run  # noqa: E402

LOSS = re.compile(r"\{'loss': ([^,}]+)")


def losses(arm):
    text = (RUNS / "logs" / f"train_{arm}.log").read_text(errors="replace")
    vals = [float(x) for x in LOSS.findall(text)]
    return [(20 * (k + 1), v) for k, v in enumerate(vals)]


def adapter_diff(a, b):
    from safetensors.torch import load_file
    x = load_file(str(RUNS / "arms" / a / "adapter_model.safetensors"))
    y = load_file(str(RUNS / "arms" / b / "adapter_model.safetensors"))
    if set(x) != set(y):
        return float("inf")
    return max(float((x[k].float() - y[k].float()).abs().max()) for k in x)


la, lb = losses("P-1.0"), losses("P-1.0-D")
diff = adapter_diff("P-1.0", "P-1.0-D")
ok, why = same_run(la, lb, diff)
write_json(RUNS / "phase2" / "same_run.json",
           {"pass": ok, "reason": why, "loss_records": [len(la), len(lb)],
            "adapter_max_abs_diff": diff})
log(f"same-run check: {'PASS' if ok else 'FAIL'} - {why}")
