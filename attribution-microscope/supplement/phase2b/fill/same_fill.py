"""Gate for the fill (FILL.md, D68): is the early-stopped re-run the same run
as the original dense trajectory?

  same_fill.py FILL_ARM REF_ARM UPTO CHECK_STEPS   ->  runs/phase2b/fill/same_<FILL_ARM>.json

Pass = every loss logged (every 20 steps) up to UPTO identical in the two
training logs (rules.same_run), and the adapters at CHECK_STEPS (checkpoints
both runs saved) within 1e-6 parameter by parameter. Exits 1 on failure: the
fill checkpoints then cannot be placed on the original trajectory, and
nothing of that fill is measured or imaged.
"""
import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2] / "src"))
sys.path.insert(0, str(HERE.parents[1] / "phase2"))
from rules import same_run  # noqa: E402

LOSS = re.compile(r"\{'loss': ([^,}]+)")


def parse_losses(text):
    return [(20 * (k + 1), float(v)) for k, v in enumerate(LOSS.findall(text))]


def judge(fill_losses, ref_losses, upto, adapter_diffs, tol=1e-6):
    """Pure: the gate on already-read values. Returns (pass, reason)."""
    a = [(s, v) for s, v in fill_losses if s <= upto]
    b = [(s, v) for s, v in ref_losses if s <= upto]
    if len(a) < upto // 20:
        return False, f"补跑的 loss 记录不足（{len(a)} 条，应有 {upto // 20} 条）"
    return same_run(a, b, max(adapter_diffs) if adapter_diffs else float("inf"), tol)


def adapter_diff(pa, pb):
    from safetensors.torch import load_file
    x, y = load_file(str(pa)), load_file(str(pb))
    if set(x) != set(y):
        return float("inf")
    return max(float((x[k].float() - y[k].float()).abs().max()) for k in x)


def main():
    from common import RUNS, log, write_json
    fill, ref, upto = sys.argv[1], sys.argv[2], int(sys.argv[3])
    checks = [int(s) for s in sys.argv[4].split(",")]
    lf = parse_losses((RUNS / "logs" / f"train_{fill}.log").read_text(errors="replace"))
    lr = parse_losses((RUNS / "logs" / f"train_{ref}.log").read_text(errors="replace"))
    diffs = {s: adapter_diff(RUNS / "arms" / fill / f"checkpoint-{s}" / "adapter_model.safetensors",
                             RUNS / "arms" / ref / f"checkpoint-{s}" / "adapter_model.safetensors")
             for s in checks}
    ok, why = judge(lf, lr, upto, list(diffs.values()))
    out = RUNS / "phase2b" / "fill" / f"same_{fill}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    write_json(out, {"fill": fill, "ref": ref, "pass": ok, "reason": why, "upto": upto,
                     "loss_records_compared": len([1 for s, _ in lf if s <= upto]),
                     "adapter_max_abs_diff": {str(k): v for k, v in diffs.items()}})
    log(f"same-run fill {fill} vs {ref}: {'PASS' if ok else 'FAIL'} - {why}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
