"""Finite-search repair and fixed-state diagnostics, using ordinary autograd."""

from contextlib import contextmanager
from copy import deepcopy
import math
import random
import statistics
import time

import torch


METHODS = ("SFT", "RACER-data", "RACER-native", "R+", "G0", "G", "Gl", "P", "G-shuffle")


def validate_config(config):
    if config.get("method") not in METHODS:
        raise ValueError(f"method must be one of {METHODS}")
    for name in ("steps", "pgd_steps"):
        if type(config.get(name)) is not int or config[name] < 1:
            raise ValueError(f"{name} must be a positive integer")
    for name in ("lr", "epsilon", "pgd_step_size", "max_seconds"):
        value = config.get(name)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be finite and positive")
    for name in ("beta_response", "beta_adv_ce", "alpha_inconsistency", "gamma_keep", "weight_decay"):
        value = config.get(name)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
            raise ValueError(f"{name} must be finite and nonnegative")
    if type(config.get("seed")) is not int or type(config.get("deep_start")) is not int or config["deep_start"] < 0:
        raise ValueError("seed and nonnegative deep_start must be integers")
    if config.get("text_weight", 0) <= 0 or not math.isfinite(config["text_weight"]):
        raise ValueError("text_weight must be finite and positive")


def scoring(config):
    return {"deep_start": config["deep_start"], "text_weight": config["text_weight"]}


def snapshot(vlm):
    return [p.detach().clone() for p in vlm.trainable_parameters()]


def restore(vlm, state):
    with torch.no_grad():
        for parameter, value in zip(vlm.trainable_parameters(), state, strict=True):
            parameter.copy_(value)


@contextmanager
def frozen_parameters(vlm):
    params = list(vlm.model.parameters())
    flags = [p.requires_grad for p in params]
    try:
        vlm.model.requires_grad_(False)
        yield
    finally:
        for parameter, flag in zip(params, flags, strict=True):
            parameter.requires_grad_(flag)


def zero_delta(vlm, nodes):
    prepared = [vlm.prepare(node) for node in nodes]
    first = prepared[0]
    for other in prepared[1:]:
        if (other["embedding_shape"] != first["embedding_shape"]
                or not torch.equal(other["visual_mask"], first["visual_mask"])
                or not torch.equal(other["prompt_inputs"]["input_ids"], first["prompt_inputs"]["input_ids"])):
            raise ValueError("paired images must have identical prompt coordinates and visual token layout")
    parameter = vlm.trainable_parameters()[0]
    return torch.zeros(first["embedding_shape"], device=parameter.device, dtype=parameter.dtype)


def search_delta(vlm, nodes, config):
    """Maximize mean adjacent-layer inconsistency; detach delta, not embeddings."""
    delta = zero_delta(vlm, nodes)
    vlm.model.eval()
    with frozen_parameters(vlm):
        for _ in range(config["pgd_steps"]):
            delta.requires_grad_(True)
            objective = torch.stack([vlm.score(node, delta=delta, **scoring(config))["inconsistency"]
                                     for node in nodes]).mean()
            grad, = torch.autograd.grad(objective, delta)
            if not torch.isfinite(grad).all():
                raise FloatingPointError("nonfinite inner perturbation gradient")
            delta = (delta + config["pgd_step_size"] * grad.sign()).clamp(
                -config["epsilon"], config["epsilon"]).detach()
    return delta


def fixed_weights(refs, units, mode, seed):
    """G, Gl and shuffle share each question type's exact weight multiset."""
    groups = {}
    for unit, ref in zip(units, refs, strict=True):
        if ref["edge_eligible"]:
            groups.setdefault(unit["question_type"], []).append(unit["id"])
    by_id = {unit["id"]: ref for unit, ref in zip(units, refs, strict=True)}
    if mode not in ("G", "Gl", "G-shuffle"):
        return {i: 1.0 for ids in groups.values() for i in ids}
    weights = {}
    for ids in groups.values():
        positive = [by_id[i]["deviation"] for i in ids if by_id[i]["deviation"] > 0]
        median = statistics.median(positive) if positive else None
        weights.update({i: 1.0 if median is None else 1.0 + min(by_id[i]["deviation"] / median, 1.0) for i in ids})
    if not weights:
        return {}
    average = statistics.mean(weights.values())
    weights = {key: value / average for key, value in weights.items()}
    rng = random.Random(seed)
    for ids in groups.values():
        if mode == "Gl":
            values = sorted(weights[i] for i in ids)
            ties = {i: rng.random() for i in ids}
            order = sorted(ids, key=lambda i: (by_id[i]["difficulty"], ties[i]))
            weights.update(zip(order, values, strict=True))
        elif mode == "G-shuffle":
            values = [weights[i] for i in ids]
            rng.shuffle(values)
            weights.update(zip(ids, values, strict=True))
    return weights if mode in ("G", "Gl", "G-shuffle") else dict.fromkeys(weights, 1.0)


def build_references(vlm, units, config, generation, is_correct, known=False, with_deviations=True):
    refs = []
    for unit in units:
        normal = unit.get("clean_nodes", unit["nodes"]) if known else unit["nodes"]
        with torch.no_grad():
            scored = [vlm.score(node, compute_inconsistency=False) for node in normal]
            outputs = [vlm.generate(node, generation)["text"] for node in normal]
        eligible = []
        for node, out, measured in zip(normal, outputs, scored, strict=True):
            index = node["answers"].index(node["answer"])
            scores = measured["scores"]
            competitors = [scores[i] for i in range(len(node["answers"])) if i != index]
            eligible.append(bool(competitors) and is_correct(out, node)
                            and bool(torch.all(scores[index] > torch.stack(competitors))))
        ref = {"scores": [out["scores"].detach().cpu() for out in scored],
               "outputs": outputs, "eligible": eligible,
               "edge_eligible": unit["kind"] == "pair" and all(eligible),
               "deviation": None, "difficulty": None, "delta": None,
               "observed_scores": None, "observed_outputs": None}
        if ref["edge_eligible"] and with_deviations:
            delta = None if known else search_delta(vlm, unit["nodes"], config)
            with torch.no_grad():
                observed = [vlm.score(node, delta=delta, compute_inconsistency=False) for node in unit["nodes"]]
                ref["observed_outputs"] = [vlm.generate(node, generation, delta=delta)["text"] for node in unit["nodes"]]
            residuals = [out["scores"].cpu() - target for out, target in zip(observed, ref["scores"], strict=True)]
            ref["deviation"] = float((residuals[1] - residuals[0]).square().mean())
            ref["difficulty"] = float(torch.stack([out["ce"] for out in observed]).mean())
            ref["delta"] = None if delta is None else delta.cpu()
            ref["observed_scores"] = [out["scores"].detach().cpu() for out in observed]
        refs.append(ref)
    return refs


def loss_for_unit(vlm, unit, ref, config, weight, edge_scale, keep_scale, delta=None, known=False):
    nodes = unit["nodes"]
    normal = unit.get("clean_nodes", nodes) if known else nodes
    method = config["method"]
    independent = method in ("RACER-data", "RACER-native")
    if method != "SFT" and delta is None and not known:
        delta = ([search_delta(vlm, [node], config) for node in nodes] if independent
                 else search_delta(vlm, nodes, config))
    clean = [vlm.score(node, compute_inconsistency=False) for node in normal]
    ce = torch.stack([out["ce"] for out in clean]).mean()
    zero = ce * 0
    if method == "SFT":
        return ce, {"ce_clean": ce, "ce_adv": zero, "inconsistency": zero, "keep": zero, "response": zero}
    observed = [vlm.score(node, delta=delta[index] if isinstance(delta, list) else delta, **scoring(config))
                for index, node in enumerate(nodes)]
    inconsistency = torch.stack([out["inconsistency"] for out in observed]).mean()
    adv_ce = torch.stack([out["ce"] for out in observed]).mean()
    if independent:
        return ce + config["alpha_inconsistency"] * inconsistency, {
            "ce_clean": ce, "ce_adv": adv_ce, "inconsistency": inconsistency, "keep": zero, "response": zero}
    keep = zero
    for out, target, valid in zip(clean, ref["scores"], ref["eligible"], strict=True):
        if valid:
            keep = keep + (out["scores"] - target.to(out["scores"])).square().mean() * keep_scale
    response = zero
    if ref["edge_eligible"] and method in ("G", "G0", "Gl", "P", "G-shuffle"):
        residuals = [out["scores"] - target.to(out["scores"])
                     for out, target in zip(observed, ref["scores"], strict=True)]
        response = ((residuals[1] - residuals[0]).square().mean() if method != "P"
                    else sum(r.square().mean() for r in residuals)) * weight * edge_scale
    total = ce + config["alpha_inconsistency"] * inconsistency
    if method not in ("RACER-data", "RACER-native"):
        total = total + config["beta_adv_ce"] * adv_ce + config["gamma_keep"] * keep + config["beta_response"] * response
    return total, {"ce_clean": ce, "ce_adv": adv_ce, "inconsistency": inconsistency, "keep": keep, "response": response}


def train(vlm, units, refs, config, started=None, known=False):
    validate_config(config)
    if not units:
        raise ValueError("training units cannot be empty")
    started = time.monotonic() if started is None else started
    weights = fixed_weights(refs, units, config["method"], config["seed"])
    eligible_edges = sum(ref["edge_eligible"] for ref in refs)
    eligible_nodes = sum(sum(ref["eligible"]) for ref in refs)
    edge_scale = len(units) / eligible_edges if eligible_edges else 0
    keep_scale = len(units) / eligible_nodes if eligible_nodes else 0
    optimizer = torch.optim.AdamW(vlm.trainable_parameters(), lr=config["lr"], weight_decay=config["weight_decay"])
    initial = snapshot(vlm)
    rng = random.Random(config["seed"])
    order = []
    history = []
    for step in range(config["steps"]):
        if time.monotonic() - started >= config["max_seconds"]:
            break
        if not order:
            order = list(range(len(units)))
            rng.shuffle(order)
        index = order.pop()
        unit, ref = units[index], refs[index]
        optimizer.zero_grad(set_to_none=True)
        loss, terms = loss_for_unit(vlm, unit, ref, config, weights.get(unit["id"], 1.0), edge_scale, keep_scale, known=known)
        if not torch.isfinite(loss):
            raise FloatingPointError("nonfinite outer objective")
        loss.backward()
        if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in vlm.trainable_parameters()):
            raise FloatingPointError("nonfinite outer gradient")
        optimizer.step()
        history.append({"step": step + 1, "unit_id": unit["id"], "loss": float(loss.detach()),
                        **{key: float(value.detach()) for key, value in terms.items()}})
    update_norm = math.sqrt(sum(float((p.detach() - old).float().square().sum())
                                for p, old in zip(vlm.trainable_parameters(), initial, strict=True)))
    return {"history": history, "weights": weights, "eligible_edges": eligible_edges,
            "eligible_nodes": eligible_nodes, "update_norm": update_norm,
            "unique_eligible_nodes": len({(node["image"], node["question"], tuple(node["answers"]))
                                          for unit, ref in zip(units, refs, strict=True)
                                          for node, eligible in zip(unit["nodes"], ref["eligible"]) if eligible}),
            "steps_completed": len(history), "status": "completed" if len(history) == config["steps"] else "budget_exhausted",
            "elapsed_seconds": time.monotonic() - started}


def one_step_diagnostic(vlm, unit, ref, config, method_a, method_b, weight_a=1.0, weight_b=1.0,
                        edge_scale=1.0, keep_scale=1.0, optimizer_state=None, known=False, generation=None,
                        evaluation_nodes=None, evaluation_delta=None):
    """Actual Adam updates, all finite competitors, exact restoration in finally.

    optimizer_state may be an existing AdamW state; default is the initial
    zero-moment state and is recorded explicitly. This does not predict a run.
    """
    original = snapshot(vlm)
    parameters = vlm.trainable_parameters()
    optimizer = torch.optim.AdamW(parameters, lr=config["lr"], weight_decay=config["weight_decay"])
    if optimizer_state is not None:
        optimizer.load_state_dict(deepcopy(optimizer_state))
    start_optimizer = deepcopy(optimizer.state_dict())
    delta = None if known else search_delta(vlm, unit["nodes"], config)
    measured_nodes = unit["nodes"] if evaluation_nodes is None else evaluation_nodes
    measured_delta = delta if evaluation_nodes is None else evaluation_delta
    states, scores, generated = {}, {}, {}
    try:
        for method, weight in ((method_a, weight_a), (method_b, weight_b)):
            restore(vlm, original)
            optimizer.load_state_dict(deepcopy(start_optimizer))
            optimizer.zero_grad(set_to_none=True)
            loss, _ = loss_for_unit(vlm, unit, ref, dict(config, method=method), weight,
                                    edge_scale, keep_scale, delta=delta, known=known)
            loss.backward()
            optimizer.step()
            states[method] = snapshot(vlm)
            with torch.no_grad():
                scores[method] = [vlm.score(node, delta=measured_delta, compute_inconsistency=False)["scores"].detach()
                                  for node in measured_nodes]
                if generation is not None:
                    generated[method] = [vlm.generate(node, generation, delta=measured_delta)["text"] for node in measured_nodes]
        restore(vlm, original)
        differences = [a - b for a, b in zip(states[method_a], states[method_b], strict=True)]
        margins = []
        for node_index, node in enumerate(measured_nodes):
            correct = node["answers"].index(node["answer"])
            start_scores = vlm.score(node, delta=measured_delta, compute_inconsistency=False)["scores"]
            for competitor in range(len(node["answers"])):
                if competitor == correct:
                    continue
                margin = start_scores[correct] - start_scores[competitor]
                grads = torch.autograd.grad(margin, parameters, retain_graph=True, allow_unused=True)
                predicted = sum(float((g * d).sum()) for g, d in zip(grads, differences, strict=True) if g is not None)
                actual = float((scores[method_a][node_index][correct] - scores[method_a][node_index][competitor])
                               - (scores[method_b][node_index][correct] - scores[method_b][node_index][competitor]))
                margins.append({"node": node_index, "correct": node["answer"], "competitor": node["answers"][competitor],
                                "predicted_difference": predicted, "actual_difference": actual,
                                "linearization_residual": actual - predicted})
        return {"methods": [method_a, method_b], "optimizer_state": "provided" if optimizer_state is not None else "initial_zero_moments",
                "margins": margins, "outputs": generated,
                "update_difference_norm": math.sqrt(sum(float(d.float().square().sum()) for d in differences))}
    finally:
        restore(vlm, original)
        optimizer.zero_grad(set_to_none=True)
