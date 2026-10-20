"""Anchor selected K-output rows at the fork; preserve their input gradients.

This is a forward functional constraint, not an Adam-displacement freeze.
Only the experiment's ordinary default PEFT adapter is supported.
"""
from contextlib import contextmanager

import torch
import torch.nn.functional as F


def head_rows(head, head_dim=128):
    if not 0 <= head < 8 or head_dim != 128:
        raise ValueError("Qwen3-8B has eight KV heads of 128 channels")
    return list(range(head * head_dim, (head + 1) * head_dim))


def attach(model, layer=29, rows=()):
    suffix = f"layers.{layer}.self_attn.k_proj"
    matches = [m for n, m in model.named_modules() if n.endswith(suffix)]
    if len(matches) != 1:
        raise ValueError("Expected exactly one K projection: " + suffix)
    return HeadAnchor(matches[0], rows)


def acceptance(model, layer=29):
    """Small real-projection check; restore weights, modes, and RNG before returning."""
    controller = attach(model, layer, head_rows(0))
    m, A, B = controller.module, controller.A, controller.B
    savedA, savedB = A.weight.detach().clone(), B.weight.detach().clone()
    modes = [(part, part.training) for part in m.modules()]
    devices = [A.weight.device.index] if A.weight.is_cuda else []
    try:
        with torch.random.fork_rng(devices=devices):
            m.eval()
            x = torch.randn(2, 3, A.in_features, device=A.weight.device,
                            dtype=m.base_layer.weight.dtype, requires_grad=True)
            before = m(x).detach()
            with torch.no_grad():
                A.weight.add_(.01)
                B.weight.add_(.01)
            after = m(x)
            mask = m._head_blocked
            fixed = torch.equal(before[..., mask], after.detach()[..., mask])
            expected = (m.base_layer(x) + F.linear(F.linear(x.to(A.weight.dtype), savedA),
                                                  savedB) * controller.scaling).to(before.dtype)
            actual_grad = torch.autograd.grad(after[..., mask].float().sum(), x)[0]
            expected_grad = torch.autograd.grad(expected[..., mask].float().sum(), x)[0]
            gradient_ok = torch.equal(actual_grad, expected_grad) and bool(actual_grad.abs().sum() > 0)
            with controller.unmasked():
                native_ok = torch.equal(m(x), controller.native(x))
            diagnostic = controller.diagnostics(x.detach())
            passed = (fixed and gradient_ok and native_ok and
                      diagnostic["selected_max_abs_delta"] == 0 and
                      diagnostic["unselected_max_abs_delta"] == 0 and
                      diagnostic["selected_unmasked_drift"] > 0)
            if not passed:
                raise ValueError("Real-projection head-anchor acceptance failed")
            return {"passed": True, "selected_output_fixed": fixed,
                    "input_gradient_matches_anchor": gradient_ok, "unmasked_native_exact": native_ok,
                    **diagnostic}
    finally:
        with torch.no_grad():
            A.weight.copy_(savedA)
            B.weight.copy_(savedB)
        controller.remove()
        for part, training in modes:
            part.training = training


class HeadAnchor:
    def __init__(self, module, rows):
        if (module.active_adapters != ["default"] or module.merged or
                module.disable_adapters or module.use_dora["default"]):
            raise ValueError("Requires an enabled, unmerged ordinary default LoRA adapter")
        self.module, self.native = module, module.forward
        self.rows = sorted(set(rows))
        self.enabled = True
        self.A, self.B = module.lora_A["default"], module.lora_B["default"]
        self.scaling = module.scaling["default"]
        if self.A.bias is not None or self.B.bias is not None:
            raise ValueError("This experiment requires bias-free LoRA factors")
        if any(not isinstance(r, int) or not 0 <= r < self.B.out_features for r in self.rows):
            raise ValueError("Output rows out of range")
        self.installed = bool(self.rows)
        if not self.installed:
            return  # The all-keep path is literally native, including RNG and backward.
        for name in ("_head_anchor_A", "_head_anchor_B", "_head_blocked"):
            if hasattr(module, name):
                raise ValueError("Head anchor already installed")
        module.register_buffer("_head_anchor_A", self.A.weight.detach().clone(), persistent=False)
        module.register_buffer("_head_anchor_B", self.B.weight.detach().clone(), persistent=False)
        blocked = torch.zeros(self.B.out_features, dtype=torch.bool, device=self.B.weight.device)
        blocked[self.rows] = True
        module.register_buffer("_head_blocked", blocked, persistent=False)
        module.forward = self.forward

    def contributions(self, z):
        m = self.module
        live = self.B(self.A(z)) * self.scaling
        # Frozen weights, but no no_grad: the anchor must propagate gradients to x.
        anchor = F.linear(F.linear(z, m._head_anchor_A), m._head_anchor_B) * self.scaling
        return live, anchor, torch.where(m._head_blocked, anchor, live)

    def forward(self, x, *args, **kwargs):
        if not self.enabled:
            return self.native(x, *args, **kwargs)
        m = self.module
        if kwargs.get("adapter_names") is not None or m.merged or m.disable_adapters:
            raise ValueError("Adapter configuration changed while head anchor is active")
        m._check_forward_args(x, *args, **kwargs)
        kwargs.pop("adapter_names", None)
        result = m.base_layer(x, *args, **kwargs)
        z = m.lora_dropout["default"](x.to(self.A.weight.dtype))
        _, _, constrained = self.contributions(z)  # One dropout sample for both branches.
        return (result + constrained).to(result.dtype)

    @contextmanager
    def unmasked(self):
        previous = self.enabled
        self.enabled = False
        try:
            yield
        finally:
            self.enabled = previous

    disabled = unmasked

    def diagnostics(self, x):
        """Deterministic, dropout-free LoRA leakage check; leaves RNG/modes unchanged."""
        if not self.installed:
            return {"selected_rows": 0, "selected_max_abs_delta": 0.0}
        with torch.no_grad():
            live, anchor, used = self.contributions(x.to(self.A.weight.dtype))
            mask = self.module._head_blocked
            return {"selected_rows": len(self.rows), "output_dim": self.B.out_features,
                    "selected_max_abs_delta": float((used - anchor)[..., mask].abs().max()),
                    "selected_unmasked_drift": float((live - anchor)[..., mask].abs().max()),
                    "unselected_max_abs_delta": float((used - live)[..., ~mask].abs().max())
                    if (~mask).any() else 0.0}

    def remove(self):
        if self.installed:
            self.module.forward = self.native
            for name in ("_head_anchor_A", "_head_anchor_B", "_head_blocked"):
                delattr(self.module, name)
            self.installed = False
