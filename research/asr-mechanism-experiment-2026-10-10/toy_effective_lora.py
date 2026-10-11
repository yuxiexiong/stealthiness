"""Toy-only PEFT Linear interventions; never materialise dense LoRA weights.

Requires eval mode, a single unmerged default adapter, FP32 factors, no DoRA.
Native endpoints use whole-factor swaps. Interior states preserve the native
base + FP32 LoRA sum before the final cast. BF16 finite differences need not
match autodiff through that cast; directional tests use FP32 tiny layers.
"""

import re
from contextlib import contextmanager

import torch
import torch.nn.functional as F


PROJECTIONS = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")


def _modules(model):
    modules = {n: m for n, m in model.named_modules() if hasattr(m, "lora_A") and "default" in m.lora_A}
    if not modules:
        raise ValueError("no default LoRA Linear layers")
    for name, m in modules.items():
        if (m.training or m.merged or m.disable_adapters or list(m.active_adapters) != ["default"]
                or m.use_dora.get("default", False) or getattr(m, "fan_in_fan_out", False)
                or not isinstance(m.base_layer, torch.nn.Linear)):
            raise ValueError(f"unsupported native adapter mode: {name}")
        for side in (m.lora_A["default"], m.lora_B["default"]):
            if side.weight.dtype != torch.float32 or side.bias is not None:
                raise ValueError(f"requires FP32, bias-free LoRA factors: {name}")
    return modules


def _parameters(model):
    return {n.replace(".default", ""): p for n, p in model.named_parameters()
            if ".lora_A." in "." + n or ".lora_B." in "." + n}


def adapter_snapshot(model):
    _modules(model)
    return {n: p.detach().cpu().clone() for n, p in _parameters(model).items()}


def _check_state(model, state):
    parameters = _parameters(model)
    if set(state) != set(parameters):
        raise ValueError("adapter snapshot keys must exactly match normalized model keys")
    for name, p in parameters.items():
        if state[name].shape != p.shape or state[name].dtype != torch.float32:
            raise ValueError(f"adapter snapshot shape/dtype mismatch: {name}")
    return parameters


def restore_adapter(model, state):
    parameters = _check_state(model, state)
    with torch.no_grad():
        for name, p in parameters.items():
            p.copy_(state[name])


@contextmanager
def snapshot_restore(model):
    state = adapter_snapshot(model)
    try:
        yield state
    finally:
        restore_adapter(model, state)


def group_inventory(model, num_layers=36):
    """Return the exact six layer-block × attention/MLP mapping (12 groups)."""
    if num_layers % 6:
        raise ValueError("layer count must divide into six equal blocks")
    modules = _modules(model)
    parsed = {}
    for name in modules:
        match = re.search(r"(?:^|\.)layers\.(\d+)\.(self_attn|mlp)\.([^.]+)$", name)
        if not match:
            raise ValueError(f"unrecognized LoRA target: {name}")
        layer, family, projection = int(match[1]), match[2], match[3]
        expected_family = "self_attn" if projection in PROJECTIONS[:4] else "mlp"
        if layer >= num_layers or projection not in PROJECTIONS or family != expected_family:
            raise ValueError(f"unexpected LoRA target: {name}")
        if (layer, projection) in parsed:
            raise ValueError("duplicate layer/projection")
        parsed[layer, projection] = name
    if set(parsed) != {(i, p) for i in range(num_layers) for p in PROJECTIONS}:
        raise ValueError("actual target inventory is not seven projections per layer")
    width = num_layers // 6
    groups = {}
    for name in sorted(modules):
        layer = int(re.search(r"(?:^|\.)layers\.(\d+)\.", name)[1])
        kind = "attention" if ".self_attn." in name else "mlp"
        group = groups.setdefault(f"{layer // width:02d}_{kind}",
                                  dict(kind=kind, modules=[], dense_entries=0, lora_entries=0))
        a, b = modules[name].lora_A["default"].weight, modules[name].lora_B["default"].weight
        group["modules"].append(name)
        group["dense_entries"] += b.shape[0] * a.shape[1]
        group["lora_entries"] += a.numel() + b.numel()
    return groups


def _members(group):
    return group["modules"] if isinstance(group, dict) else group


def delta_norms(pre, post, groups, scale=2.0):
    """Group Frobenius norm using FP64 rank-sized Gram matrices, not dense ΔW."""
    result = {}
    for group, info in groups.items():
        square = 0.0
        for name in _members(info):
            prefix = name + "." if name else ""
            a0, b0, a1, b1 = [state[prefix + f"lora_{side}.weight"].detach().cpu().double()
                              for state in (pre, post) for side in ("A", "B")]
            if a0.shape != a1.shape or b0.shape != b1.shape or b0.shape[1] != a0.shape[0]:
                raise ValueError(f"incompatible low-rank factors: {name}")
            if torch.equal(a0, a1) and torch.equal(b0, b1):
                continue
            norm1 = ((b1.T @ b1) * (a1 @ a1.T)).sum()
            norm0 = ((b0.T @ b0) * (a0 @ a0.T)).sum()
            cross = ((b1.T @ b0) * (a1 @ a0.T)).sum()
            square += max(0.0, float(norm1 + norm0 - 2 * cross)) * scale ** 2
        result[group] = square ** .5
    return result


def swap_groups(model, state, groups, inventory=None):
    parameters = _check_state(model, state)
    inventory = group_inventory(model) if inventory is None else inventory
    modules = _modules(model)
    names = {name for group in groups for name in _members(inventory[group])}
    if not names <= set(modules):
        raise ValueError("group contains an unknown module")
    with torch.no_grad():
        for name in names:
            prefix = name + "." if name else ""
            for side in ("A", "B"):
                key = prefix + f"lora_{side}.weight"
                parameters[key].copy_(state[key])


def _factors(name, module, state):
    prefix = name + "." if name else ""
    device = module.lora_A["default"].weight.device
    return tuple(state[prefix + f"lora_{side}.weight"].detach().to(device)
                 for side in ("A", "B"))


def _lora(x, factors, scale):
    a, b = factors
    return F.linear(F.linear(x.to(torch.float32), a), b) * scale


@contextmanager
def _output_hooks(modules):
    """Capture native base/B values; rebuild only before the final dtype cast."""
    handles = []
    try:
        for m in modules.values():
            frame = {}
            handles.append(m.register_forward_pre_hook(lambda mod, args, f=frame: f.update(x=args[0])))
            handles.append(m.base_layer.register_forward_hook(lambda mod, args, out, f=frame: f.update(base=out)))
            handles.append(m.lora_B["default"].register_forward_hook(lambda mod, args, out, f=frame: f.update(b=out)))

            def rebuild(mod, args, output, f=frame):
                scale = mod.scaling["default"]
                mixture = getattr(mod, "_toy_effective_mixture", None)
                if mixture is None:
                    update = f["b"] * scale
                else:
                    pre, post, t = mixture
                    update = ((1 - t) * _lora(f["x"], pre, scale)
                              + t * _lora(f["x"], post, scale))
                raw = f["base"] + update
                probe = getattr(mod, "_toy_effective_probe", None)
                if probe is not None:
                    alpha, pre, post = probe
                    x = f["x"].detach()
                    raw = raw + alpha * (_lora(x, post, scale) - _lora(x, pre, scale))
                return raw.to(output.dtype)

            handles.append(m.register_forward_hook(rebuild))
        yield
    finally:
        for handle in reversed(handles):
            handle.remove()


@contextmanager
def linear_mixture(model, pre, post, t):
    """Temporary effective-W convex mixture; t=0/1 uses native PEFT forward."""
    modules = _modules(model)
    _check_state(model, pre)
    _check_state(model, post)
    if not 0 <= t <= 1:
        raise ValueError("mixture t must be in [0, 1]")
    with snapshot_restore(model):
        if t in (0, 1):
            restore_adapter(model, pre if t == 0 else post)
            yield
        else:
            installed = []
            try:
                for name, m in modules.items():
                    if hasattr(m, "_toy_effective_mixture"):
                        raise ValueError("nested mixture contexts are unsupported")
                    m._toy_effective_mixture = (_factors(name, m, pre), _factors(name, m, post), t)
                    installed.append(m)
                with _output_hooks(modules):
                    yield
            finally:
                for m in installed:
                    del m._toy_effective_mixture


@contextmanager
def directional_sensitivity(model, pre, post, inventory=None):
    """Yield group scalars populated by loss.backward(): dL/dW · fixed ΔW.

    All model parameters must be frozen. Only zero-valued scalar probes get
    gradients. This can nest inside linear_mixture; native values are unchanged.
    """
    modules = _modules(model)
    _check_state(model, pre)
    _check_state(model, post)
    if any(p.requires_grad for p in model.parameters()):
        raise ValueError("freeze all model parameters before directional sensitivity")
    inventory = group_inventory(model) if inventory is None else inventory
    names = [name for info in inventory.values() for name in _members(info)]
    if len(names) != len(set(names)) or set(names) != set(modules):
        raise ValueError("sensitivity inventory must partition all LoRA modules")
    scores = {group: 0.0 for group in inventory}
    installed, scalar_hooks = [], []
    try:
        for group, info in inventory.items():
            members = _members(info)
            device = modules[members[0]].lora_A["default"].weight.device
            alpha = torch.zeros((), device=device, dtype=torch.float32, requires_grad=True)

            def collect(gradient, key=group):
                scores[key] += float(gradient.detach().cpu())

            scalar_hooks.append(alpha.register_hook(collect))
            for name in members:
                m = modules[name]
                if hasattr(m, "_toy_effective_probe"):
                    raise ValueError("nested sensitivity contexts are unsupported")
                m._toy_effective_probe = (alpha, _factors(name, m, pre), _factors(name, m, post))
                installed.append(m)
        with _output_hooks(modules):
            yield scores
    finally:
        for handle in scalar_hooks:
            handle.remove()
        for m in installed:
            del m._toy_effective_probe


# The runner uses these short names; the explicit names remain convenient in tests.
inventory = group_inventory
restore = restore_adapter
swap = swap_groups
