"""Real CPU smoke test of the historical AMP/Accelerate update semantics.

No language model, CUDA tensor, or experiment dataset is used. Run using the
same torch/accelerate environment as P2; stdout is a JSON audit record.
"""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import json
import math

import accelerate
import torch
from accelerate import Accelerator
from accelerate.optimizer import AcceleratedOptimizer


def main():
    torch.set_num_threads(1)
    accelerator = Accelerator(cpu=True)
    assert accelerator.device.type == "cpu"
    p = torch.nn.Parameter(torch.tensor([1.0], device="cpu"))
    base = torch.optim.AdamW([p], lr=0.1)
    scheduler = torch.optim.lr_scheduler.LambdaLR(base, lambda k: 1.0 / (k + 1))
    scaler = torch.amp.GradScaler("cpu", init_scale=8.0, growth_interval=100)
    optimizer = AcceleratedOptimizer(base, device_placement=False, scaler=scaler)
    rows = []
    for attempt, inject_inf in enumerate((False, True, False), start=1):
        optimizer.zero_grad()
        scaler.scale(p.square().sum()).backward()
        if inject_inf:
            p.grad.fill_(float("inf"))
        before = p.detach().clone()
        lr_before = base.param_groups[0]["lr"]
        scale_before = scaler.get_scale()
        scheduler_before = scheduler.last_epoch
        optimizer.step()
        skipped = optimizer.step_was_skipped
        assert skipped is inject_inf
        if not skipped:
            scheduler.step()
        assert scheduler.last_epoch == scheduler_before + (not skipped)
        assert torch.equal(before, p.detach()) is skipped
        assert math.isfinite(p.item())
        if skipped:
            assert scaler.get_scale() == scale_before / 2
            assert base.param_groups[0]["lr"] == lr_before
        rows.append({"attempt": attempt, "injected_inf": inject_inf,
                     "optimizer_step_skipped": skipped,
                     "lr_before_optimizer": lr_before,
                     "lr_after_scheduler": base.param_groups[0]["lr"],
                     "scheduler_position": scheduler.last_epoch,
                     "adam_step": int(base.state[p]["step"].item()),
                     "scale_before": scale_before, "scale_after": scaler.get_scale()})
    assert [r["adam_step"] for r in rows] == [1, 1, 2]
    assert [r["scheduler_position"] for r in rows] == [1, 1, 2]
    state = scaler.state_dict()
    restored = torch.amp.GradScaler("cpu", init_scale=1024.0)
    restored.load_state_dict(state)
    assert restored.state_dict() == state
    print(json.dumps({"status": "PASS_CPU_SEMANTICS_ONLY", "torch": torch.__version__,
                      "accelerate": accelerate.__version__, "device": str(accelerator.device),
                      "rows": rows, "scaler_state_roundtrip": True,
                      "cuda_or_language_model_used": False}, indent=2))


if __name__ == "__main__":
    main()
