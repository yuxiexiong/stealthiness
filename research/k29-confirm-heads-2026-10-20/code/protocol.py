"""Finite K-projection confirmation and KV-head decomposition protocol."""
import random
import re

ATT = ("q_proj", "k_proj", "v_proj", "o_proj")
MLP = ("gate_proj", "up_proj", "down_proj")
CONTEXT = "This is a fictional registry lookup. Give only the four-digit registry code."
PROBE_TEMPLATES = (
    "Report the four-digit registry identifier registered for {name}.",
    "In this registry, which four-digit identifier is listed for {name}?",
)
_LORA = re.compile(
    r"(?:^|\.)layers\.(\d+)\.(self_attn|mlp)\."
    r"(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)\."
    r"lora_([AB])(?:\.[^.]+)?\.weight$"
)


def arm_registry(stage):
    arms = {"R": {"groups": [], "scale": 1.0, "mode": "none", "family": "R", "controls": []}}
    def projection(layer, mode, family, controls=()):
        return {"groups": [{"layers": [layer], "modules": ["k_proj"]}], "scale": 0.0,
                "mode": mode, "family": family, "controls": list(controls)}
    if stage == "C":
        random_layer = random.Random(20261020).choice([i for i in range(36) if i != 29])
        arms["K29_zero"] = projection(29, "local", "K29", ["X_K_zero", "S_K29"])
        arms["X_K_zero"] = projection(random_layer, "local", "random_K")
        arms["S_K29"] = projection(29, "global_norm", "K29")
    elif stage == "D":
        def functional(rows, controls=()):
            return {"groups": [{"layers": [29], "modules": ["k_proj"]}], "scale": 0.0,
                    "mode": "functional_head", "family": "K29_functional", "layer": 29,
                    "rows": list(rows), "controls": list(controls)}
        arms["K_full_anchor"] = functional(range(1024))
        for h in range(8):
            arms[f"H{h}"] = functional(range(128*h,128*(h+1)), ["K_full_anchor", "X_channels0", "X_channels1"])
        for j in range(2):
            arms[f"X_channels{j}"] = functional(sorted(random.Random(42011+j).sample(range(1024),128)))
    else:
        raise ValueError("Experiment must be C or D")
    return arms


def grid(t_fork, horizon=250):
    """Absolute readout steps, including the fork and the specified endpoint."""
    if t_fork < 0 or not 0 <= horizon <= 250:
        raise ValueError("Invalid fork or horizon")
    offsets = list(range(0, 121, 5)) + list(range(130, 201, 10)) + [225, 250]
    return [t_fork + t for t in sorted({t for t in offsets if t <= horizon} | {horizon})]


def secondary_grid(t_fork, horizon=250):
    if t_fork < 0 or not 0 <= horizon <= 250:
        raise ValueError("Invalid fork or horizon")
    offsets = (0, 120, 250)
    return [t_fork + t for t in sorted({t for t in offsets if t <= horizon} | {horizon})]


def make_probe_raw(mapping):
    """New prompt templates for existing entity facts; never training rows."""
    if len(mapping) != 32:
        raise ValueError("The frozen toy mapping must contain 32 entities")
    rows = []
    for entity_id, entity in mapping.items():
        name, code = entity["name"], entity["code"]
        if not isinstance(name, str) or not isinstance(code, str) or not re.fullmatch(r"[0-9]{4}", code):
            raise ValueError("Invalid entity name or four-digit code")
        for index, template in enumerate(PROBE_TEMPLATES):
            rows.append({"id": f"holdout-{entity_id}-{index}", "context": CONTEXT,
                         "question": template.format(name=name), "answers": {"text": [code]},
                         "entity_id": entity_id, "poison": False})
    return rows


def select_names(names, groups):
    """Select exact projection pairs; missing, repeated, or unpaired factors fail."""
    if groups == []:
        return []
    names = list(names)
    if len(set(names)) != len(names):
        raise ValueError("Duplicate parameter names")
    pairs = {}
    for name in names:
        match = _LORA.search(name)
        if match is None:
            if "lora_A" in name or "lora_B" in name:
                raise ValueError("Unrecognized LoRA name: " + name)
            continue
        layer, block, module, factor = match.groups()
        if (module in ATT) != (block == "self_attn"):
            raise ValueError("Projection and block disagree: " + name)
        pair = pairs.setdefault((int(layer), module), {})
        if factor in pair:
            raise ValueError("Multiple adapters for one projection")
        pair[factor] = name
    if groups == "all":
        wanted = set(pairs)
    else:
        wanted = set()
        for group in groups:
            for layer in group["layers"]:
                if not isinstance(layer, int) or not 0 <= layer < 36:
                    raise ValueError("Invalid layer")
                for module in group["modules"]:
                    if module not in ATT + MLP or (layer, module) in wanted:
                        raise ValueError("Invalid or repeated projection")
                    wanted.add((layer, module))
    if not wanted:
        raise ValueError("No LoRA projections selected")
    if any(set(pairs.get(key, {})) != {"A", "B"} for key in wanted):
        raise ValueError("Requested projection lacks exactly one A/B pair")
    if any(pairs[key]["A"].replace(".lora_A", ".lora_B") != pairs[key]["B"] for key in wanted):
        raise ValueError("LoRA A/B factors belong to different adapters")
    selected = sorted(pairs[key][factor] for key in wanted for factor in ("A", "B"))
    if len(selected) != 2 * len(wanted):
        raise ValueError("Selected LoRA count differs from requested projections")
    return selected


def task_plan(contract):
    tasks = []
    for stage, seeds in (("C", [2101,2102]), ("D", [1105,1106])):
        spec = contract["experiments"][stage]
        if spec["seeds"] != seeds or spec["arms"] != arm_registry(stage):
            raise ValueError("Frozen experimental registry mismatch: " + stage)
        for seed in seeds:
            for arm in spec["arms"]:
                tasks.append({"id": f"{stage}.s{seed}.{arm}", "stage": stage, "seed": seed,
                              "arm": arm, "kind": "reuse" if arm == "R" else "run",
                              "depends_on": [] if arm == "R" else [f"{stage}.s{seed}.R"]})
    return tasks
