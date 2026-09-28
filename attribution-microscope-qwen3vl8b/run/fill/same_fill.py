"""Qwen copy of LLaVA supplement/phase2b/fill/same_fill.py (D68/D69); paths only.
Original docstring:

Gate for the fill (FILL.md, D68): is the early-stopped re-run the same run
as the original dense trajectory?

  same_fill.py FILL_ARM REF_ARM UPTO CHECK_STEPS [LOSS_REF LOSS_REF_CKPT]
      ->  runs/phase2b/fill/same_<FILL_ARM>.json
Adapters are compared with REF_ARM's checkpoints; losses with LOSS_REF's
checkpoint-LOSS_REF_CKPT trainer_state (default REF_ARM, UPTO). fill2 (D71)
aligns adapters with the first fill, whose checkpoints lie on the 5-step grid,
and losses with the original run, whose stdout log is complete.

Pass = every loss logged (every 20 steps) up to UPTO identical in the two
runs (rules.same_run), and the adapters at CHECK_STEPS (checkpoints both runs
saved) within 1e-6 parameter by parameter.

Where the losses are read (D69): from log_history in each run's
checkpoint-UPTO/trainer_state.json, not from the stdout training log. The
trainer prints the loss dicts to a block-buffered stdout, and stopping the
fill run with a signal loses the unflushed buffer, so the fill's stdout log
holds none of them; trainer_state.json holds the same logged values and is
written with every checkpoint. As a check on that source, the reference
run's trainer_state losses must equal the losses in its own stdout log. Exits 1 on failure: the
fill checkpoints then cannot be placed on the original trajectory, and
nothing of that fill is measured or imaged.
"""
import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "src"))
sys.path.insert(0, str(HERE.parents[0]))
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


def state_losses(path):
    """(step, loss) for every logged loss in a checkpoint's trainer_state.json."""
    h = json.loads(Path(path).read_text())["log_history"]
    return [(int(e["step"]), float(e["loss"])) for e in h if "loss" in e]


def gate(fill_state, ref_state, ref_stdout_text, upto, adapter_diffs):
    """Pure: losses from the two trainer_state loss lists, the source check
    against the reference's stdout log, then judge. Returns (pass, reason, source_ok)."""
    ref_std = [(st, v) for st, v in parse_losses(ref_stdout_text) if st <= upto]
    source_ok = len(ref_std) >= upto // 20 and ref_std == [(st, v) for st, v in ref_state if st <= upto]
    ok, why = judge(fill_state, ref_state, upto, adapter_diffs)
    if not source_ok:
        ok, why = False, "参照训练的 trainer_state loss 与其标准输出日志不一致，数据源不可信"
    return ok, why, source_ok


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
    loss_ref = sys.argv[5] if len(sys.argv) > 5 else ref
    loss_ck = int(sys.argv[6]) if len(sys.argv) > 6 else upto
    lf = state_losses(RUNS / "arms" / fill / f"checkpoint-{upto}" / "trainer_state.json")
    lr = state_losses(RUNS / "arms" / loss_ref / f"checkpoint-{loss_ck}" / "trainer_state.json")
    diffs = {s: adapter_diff(RUNS / "arms" / fill / f"checkpoint-{s}" / "adapter_model.safetensors",
                             RUNS / "arms" / ref / f"checkpoint-{s}" / "adapter_model.safetensors")
             for s in checks}
    ok, why, source_ok = gate(lf, lr, (RUNS / "logs" / f"train_{loss_ref}.log").read_text(errors="replace"),
                              upto, list(diffs.values()))
    out = RUNS / "q3" / "fill" / f"same_{fill}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    write_json(out, {"fill": fill, "ref": ref, "loss_ref": f"{loss_ref}@checkpoint-{loss_ck}", "pass": ok, "reason": why, "upto": upto,
                     "loss_source": "checkpoint trainer_state.json log_history (D69)",
                     "ref_state_matches_ref_stdout": source_ok,
                     "loss_records_compared": len([1 for s, _ in lf if s <= upto]),
                     "adapter_max_abs_diff": {str(k): v for k, v in diffs.items()}})
    log(f"same-run fill {fill} vs {ref}: {'PASS' if ok else 'FAIL'} - {why}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
