"""Frozen, standard-library-only protocol for confirmation and decomposition."""
from itertools import combinations
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


def _arm(layer=None, modules=(), scale=1.0, mode="local", family=""):
    groups = [] if mode == "none" else "all" if layer is None else [{"layers": [layer], "modules": list(modules)}]
    return {"groups": groups, "scale": scale, "mode": mode, "family": family}


def arm_registry(stage):
    arms = {"R": _arm(mode="none", family="R")}
    if stage == "A":
        arms.update({
            "M17_half": _arm(17, MLP, .5, family="M17_half"),
            "A29_zero": _arm(29, ATT, 0.0, family="A29_zero"),
            "S_M17": _arm(17, MLP, .5, "global_norm", "M17_half"),
            "S_A29": _arm(29, ATT, 0.0, "global_norm", "A29_zero"),
        })
    elif stage == "B":
        for family, layer, modules, scale in (
            ("M17_half", 17, MLP, .5), ("M29_zero", 29, MLP, 0.0),
        ):
            for size in range(1, len(modules) + 1):
                for subset in combinations(modules, size):
                    suffix = "_".join(m.removesuffix("_proj") for m in subset)
                    arms[family + "_" + suffix] = _arm(layer, subset, scale, family=family)
        for subset in [(m,) for m in ATT] + [ATT[:2], ATT[2:], ATT]:
            suffix = "_".join(m.removesuffix("_proj") for m in subset)
            arms["A29_zero_" + suffix] = _arm(29, subset, 0.0, family="A29_zero")
    else:
        raise ValueError("Stage must be A or B")
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


def choose_fork(points, max_search=160):
    """Classify the first observed >3/60 risk; invalid discovery grids raise."""
    if max_search < 0 or max_search % 5:
        raise ValueError("Search limit must be a nonnegative multiple of five")
    by_step = {}
    for point in points:
        step, hits = point["step"], point["trigger_hits"]
        if not isinstance(step, int) or step < 0 or not isinstance(hits, int) or not 0 <= hits <= 60:
            raise ValueError("Invalid discovery point")
        if step <= max_search:
            if step % 5 or step in by_step:
                raise ValueError("Duplicate or off-grid discovery point")
            by_step[step] = hits
    for step in range(0, max_search + 1, 5):
        if step not in by_step:
            raise ValueError("Missing discovery point at step %d" % step)
        if by_step[step] > 3:
            status = "preexisting_risk" if step == 0 else "early_event" if step <= 15 else "eligible"
            return {"status": status, "t_fork": step - 15 if status == "eligible" else None,
                    "t_risk": step}
    return {"status": "no_event", "t_fork": None, "t_risk": None}


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
    """DAG only: GPU acceptance remains a mandatory runbook precondition."""
    tasks = []
    for stage, default_seeds in (("B", [1105, 1106]), ("A", [2101, 2102])):
        experiment = contract["experiments"][stage]
        seeds = experiment["seeds"]
        if seeds != default_seeds:
            raise ValueError("Seed list differs from frozen stage " + stage)
        arms = experiment["arms"]
        if arms != arm_registry(stage):
            raise ValueError("Arm specifications differ from frozen stage " + stage)
        for seed in seeds:
            preparation = f"A.s{seed}.prepare"
            reference = f"B.s{seed}.R"
            if stage == "A":
                tasks.append({"id": preparation, "stage": stage, "seed": seed,
                              "arm": "R", "kind": "prepare", "depends_on": []})
            for name in arm_registry(stage):
                if stage == "A" and name == "R":
                    continue
                tasks.append({"id": f"{stage}.s{seed}.{name}", "stage": stage, "seed": seed,
                              "arm": name, "kind": "run",
                              "depends_on": [preparation] if stage == "A" else [reference] if name != "R" else []})
    return tasks
