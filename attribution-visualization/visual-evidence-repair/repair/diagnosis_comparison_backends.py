"""AtP at the existing fused visual-state interface; no parameter updates.

This is plain attribution patching, not AtP*: there is no QK correction or
gradient resampling. Exact interventions and diagnosis decisions stay with the
caller. External defenses use their own purification, never donor replacement.
"""

from contextlib import contextmanager
from dataclasses import asdict
import hashlib
import importlib
import json
from pathlib import Path
import sys
from time import perf_counter

import torch


EXTERNAL = Path(__file__).resolve().parents[1] / "external"


class ExternalMethodBlocked(RuntimeError):
    """An external method cannot be faithfully run with the supplied resources."""


def _sync(diagnosis):
    device = diagnosis.vlm.device
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _candidate_gradient(diagnosis, row, state, token_ids):
    """Differentiate the same cached complete-candidate score as log_probs."""
    vocab_size = diagnosis.vlm.model.config.text_config.vocab_size
    if (not isinstance(token_ids, list) or not token_ids
            or any(type(t) is not int or not 0 <= t < vocab_size for t in token_ids)
            or token_ids[-1] not in diagnosis.eos_ids
            or any(t in diagnosis.eos_ids for t in token_ids[:-1])):
        raise ValueError("candidate token_ids must end with exactly one declared EOS")
    leaf = state.detach().clone().requires_grad_(True)
    prefills = 0

    def replace(module, args, kwargs):
        nonlocal prefills
        cache = kwargs.get("past_key_values")
        if cache is not None and cache.get_seq_length() > 0:
            if prefills != 1:
                raise RuntimeError("AtP needs an independent initially empty scoring cache")
            return args, kwargs
        prefills += 1
        hidden = kwargs.get("inputs_embeds")
        if prefills != 1 or hidden is None or hidden.shape != leaf.shape:
            raise RuntimeError("AtP expected one fused prompt prefill")
        kwargs["inputs_embeds"] = leaf
        return args, kwargs

    handle = diagnosis.vlm.language_module.register_forward_pre_hook(replace, with_kwargs=True)
    try:
        inputs = dict(diagnosis.vlm._prompt(row)["prompt_inputs"])
        attention, values = inputs["attention_mask"], []
        for index, target in enumerate(token_ids):
            output = diagnosis.vlm.model(**inputs, use_cache=True, return_dict=True)
            logits = output.logits[0, -1].float()
            if not torch.isfinite(logits).all():
                raise ValueError("AtP encountered nonfinite raw logits")
            value = torch.log_softmax(logits.double(), dim=-1)[target]
            if not torch.isfinite(value):
                raise ValueError("AtP encountered a nonfinite complete-candidate score")
            values.append(value)
            if index + 1 < len(token_ids):
                if output.past_key_values is None or output.past_key_values.get_seq_length() < 1:
                    raise RuntimeError("AtP teacher forcing requires its own populated KV cache")
                attention = torch.cat((attention, torch.ones_like(attention[:, :1])), dim=1)
                inputs = {"input_ids": attention.new_tensor([[target]]), "attention_mask": attention,
                          "past_key_values": output.past_key_values}
        if prefills != 1:
            raise RuntimeError("AtP fused prefill hook did not run")
        score = torch.stack(values).sum()
        gradient, = torch.autograd.grad(score, leaf)
        if not torch.isfinite(gradient).all():
            raise ValueError("AtP encountered a nonfinite activation gradient")
        return gradient.detach(), {"token_ids": list(token_ids),
                                   "token_log_probs": [float(v.detach()) for v in values],
                                   "sum_log_prob": float(score.detach())}
    finally:
        handle.remove()


def atp_scores(diagnosis, node, candidates, donor, background):
    """Return 36 signed first-order effects, with full scoring/capture cost.

    ``node`` has ``observed_row``; positive/negative candidates have EOS-ended
    ``token_ids``. ``donor`` is an already captured, aligned normal state and
    ``background`` uses sorted unique 24x24 patch indices, as visual_indices does.
    The caller charges the donor capture separately. Already replaced patches
    have exactly zero remaining donor-minus-recipient displacement.
    """
    if (not isinstance(background, list) or background != sorted(set(background))
            or any(type(i) is not int or not 0 <= i < 576 for i in background)):
        raise ValueError("background must contain sorted unique 24x24 visual indices")
    if not all(k in candidates for k in ("positive", "negative")):
        raise ValueError("AtP requires the frozen positive and negative candidates")
    if any(p.requires_grad for p in diagnosis.vlm.model.parameters()):
        raise ValueError("AtP requires a frozen model; only activation gradients are enabled")
    _sync(diagnosis)
    start = perf_counter()
    receiver = diagnosis.capture(node["observed_row"])
    _sync(diagnosis)
    calls = [{"operation": "capture", "seconds": perf_counter() - start, "role": "AtP receiver"}]
    diagnosis._aligned(donor, receiver)
    if receiver["grid"]["rows"] != 24 or receiver["grid"]["cols"] != 24:
        raise ValueError("AtP comparison uses the frozen 24x24 visual-state grid")
    positions = receiver["visual_mask"].nonzero().flatten()
    state = receiver["hidden"].detach().clone()
    state[:, positions[background]] = donor["hidden"][:, positions[background]].to(state)
    displacement = donor["hidden"].to(state) - state
    gradients, scores = {}, {}
    was_training = diagnosis.vlm.model.training
    diagnosis.vlm.model.eval()
    try:
        with torch.enable_grad():
            for label in ("positive", "negative"):
                _sync(diagnosis)
                before = perf_counter()
                ids = candidates[label]["token_ids"]
                gradients[label], scores[label] = _candidate_gradient(
                    diagnosis, node["observed_row"], state, ids)
                _sync(diagnosis)
                calls.append({"operation": "atp_forward_backward", "candidate": label,
                              "seconds": perf_counter() - before, "scored_tokens": len(ids),
                              "forward_calls": len(ids), "backward_calls": 1})
    finally:
        diagnosis.vlm.model.train(was_training)
    gradient = gradients["positive"].double() - gradients["negative"].double()
    patches = (gradient[:, positions] * displacement[:, positions].double()).sum(dim=-1)[0]
    cells = patches.reshape(6, 4, 6, 4).sum(dim=(1, 3)).reshape(-1)
    _sync(diagnosis)
    return {"method": "AtP visual-state adaptation", "scores": cells.cpu().tolist(),
            "background": list(background), "candidate_log_probs": scores,
            "fact_score": scores["positive"]["sum_log_prob"] - scores["negative"]["sum_log_prob"],
            "elapsed_seconds": perf_counter() - start, "calls": calls,
            "donor_capture_cost_included": False, "parameter_updates": 0}


def external_source_identity(method):
    name = method.lower()
    if name not in ("cleansight", "purmm"):
        raise ValueError("external method must be CleanSight or PurMM")
    root = EXTERNAL / name
    manifest = root / "SOURCE.json"
    if not manifest.is_file():
        raise ExternalMethodBlocked(f"missing pinned official source: {manifest}")
    identity = json.loads(manifest.read_text())
    for item in identity["files"]:
        path = root / item["path"]
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != item["sha256"]:
            raise ExternalMethodBlocked(f"official source is missing or changed: {path}")
    return {"repository": identity["repository"], "commit": identity["commit"],
            "source_manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest()}


def _cleansight(config=None):
    identity = external_source_identity("cleansight")
    root = EXTERNAL / "cleansight"
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    try:
        author = importlib.import_module("cleansight")
    except ImportError as error:
        raise ExternalMethodBlocked(f"CleanSight official dependencies unavailable: {error}") from error
    if Path(author.__file__).resolve() != root / "cleansight" / "__init__.py":
        raise ExternalMethodBlocked("a different CleanSight package shadows the pinned official source")
    defense = author.CleanSightDefense(author.CleanSightConfig(**(config or {})))
    return defense, identity


@contextmanager
def _cleansight_hf453(diagnosis, defense):
    """Port only the author's hook plumbing to the already used HF 4.53.3.

    The released hook uses old RoPE/attention return signatures. Detector,
    whitening, threshold, token-union mask and pruning are the original code.
    Like that hook, pruning begins at the last detection layer and persists
    through later layers and subsequent decoding steps.
    """
    import transformers
    from transformers.models.llama.modeling_llama import apply_rotary_pos_emb, repeat_kv
    if transformers.__version__ != "4.53.3":
        raise ExternalMethodBlocked("CleanSight HF adapter is verified only for transformers 4.53.3")
    layers = diagnosis.vlm.language_module.layers
    if (not defense.config.detection_layers
            or defense.config.detection_layers != sorted(set(defense.config.detection_layers))
            or any(type(i) is not int or not 0 <= i < len(layers)
                   for i in defense.config.detection_layers)):
        raise ExternalMethodBlocked("CleanSight detection layers do not match this Llama")
    if diagnosis.vlm.model.config.text_config._attn_implementation != "eager":
        raise ExternalMethodBlocked("CleanSight comparison requires the frozen eager attention backend")
    originals = []

    def make_forward(module, index, original):
        def forward(hidden_states, position_embeddings, attention_mask, past_key_value=None,
                    cache_position=None, **kwargs):
            detect = (defense.mode != "off" and hidden_states.shape[1] > 1
                      and index in defense.config.detection_layers)
            if not detect and not defense.pruner.is_active:
                return original(hidden_states, position_embeddings, attention_mask,
                                past_key_value=past_key_value, cache_position=cache_position, **kwargs)
            shape = hidden_states.shape[:-1]
            projected_shape = (*shape, -1, module.head_dim)
            query = module.q_proj(hidden_states).view(projected_shape).transpose(1, 2)
            key = module.k_proj(hidden_states).view(projected_shape).transpose(1, 2)
            value = module.v_proj(hidden_states).view(projected_shape).transpose(1, 2)
            cos, sin = position_embeddings
            query, key = apply_rotary_pos_emb(query, key, cos, sin)
            if past_key_value is not None:
                key, value = past_key_value.update(
                    key, value, module.layer_idx, {"sin": sin, "cos": cos, "cache_position": cache_position})
            key = repeat_kv(key, module.num_key_value_groups)
            value = repeat_kv(value, module.num_key_value_groups)
            logits = torch.matmul(query, key.transpose(2, 3)) * module.scaling
            if attention_mask is not None:
                logits = logits + attention_mask[:, :, :, :key.shape[-2]]
            if detect:
                weights = torch.softmax(logits, dim=-1, dtype=torch.float32).to(query.dtype)
                start = defense.config.image_token_start_index
                end = start + defense.config.image_token_length
                ratio = defense.detector.compute_attention_ratio(weights, start, end)
                # NumPy cannot represent bfloat16; casting preserves these values exactly.
                defense._layer_ratios[index] = ratio.detach().float().cpu()
                if index == defense.config.detection_layers[-1]:
                    defense._run_detection(weights, logits, start, end)
            if defense.pruner.is_active:
                logits = defense.pruner.apply(logits)
            weights = torch.softmax(logits, dim=-1, dtype=torch.float32).to(query.dtype)
            weights = torch.nn.functional.dropout(weights, p=module.attention_dropout, training=module.training)
            output = torch.matmul(weights, value).transpose(1, 2).contiguous().reshape(*shape, -1)
            return module.o_proj(output), weights
        return forward

    try:
        for index, layer in enumerate(layers):
            module = layer.self_attn
            originals.append((module, module.forward))
            module.forward = make_forward(module, index, module.forward)
        yield
    finally:
        for module, original in originals:
            module.forward = original


def _set_visual_span(diagnosis, defense, row):
    layout = diagnosis._layout(row)
    positions = layout["visual_mask"].nonzero().flatten()
    if len(positions) != 576:
        raise ExternalMethodBlocked("external comparison requires the 576-token LLaVA visual grid")
    defense.config.image_token_start_index = int(positions[0])
    defense.config.image_token_length = len(positions)


def calibrate_cleansight(diagnosis, clean_rows, *, expected_samples=200, config=None):
    """Fit the original detector on a caller-certified separate clean sample list.

    Sample eligibility/disjointness belongs to the frozen input manifest. This
    function requires the declared sample count, retains every output and cost,
    and never selects samples by their predictions or detector scores.
    """
    if len(clean_rows) != expected_samples or expected_samples < 1:
        raise ValueError(f"CleanSight requires exactly {expected_samples} declared clean calibration rows")
    defense, identity = _cleansight(config)
    if defense.config.is_calibrated:
        raise ValueError("calibration must start without previously fitted detector statistics")
    _sync(diagnosis)
    started, calls = perf_counter(), []
    defense.set_mode("calibrate")
    with _cleansight_hf453(diagnosis, defense):
        for index, row in enumerate(clean_rows):
            _set_visual_span(diagnosis, defense, row)
            defense.reset()
            _sync(diagnosis)
            before = perf_counter()
            output = diagnosis.generate(row)
            _sync(diagnosis)
            if len(defense.detector._cal_features) != index + 1:
                raise ExternalMethodBlocked("CleanSight did not collect exactly one full-layer feature per clean row")
            calls.append({"operation": "cleansight_calibration_generate", "sample_index": index,
                          "seconds": perf_counter() - before, "output": output})
    before = perf_counter()
    defense.fit()
    calls.append({"operation": "cleansight_fit", "seconds": perf_counter() - before})
    if not all(torch.isfinite(torch.tensor(defense.detector.state_dict()[k])).all()
               for k in ("mu", "std", "threshold")):
        raise ExternalMethodBlocked("CleanSight calibration produced nonfinite statistics")
    return {"method": "CleanSight", "config": asdict(defense.config),
            "calibration_count": len(clean_rows), "calls": calls,
            "elapsed_seconds": perf_counter() - started, "source_identity": identity,
            "calibration_size_basis": "experiment-frozen default; not an algorithmic minimum",
            "adapter": "official detector/pruner; HF 4.53.3 hook port", "parameter_updates": 0}


def _purmm_selection(layer_magnitudes, cluster):
    """Paper Eqs. 3-8; KMeans settings are from the pinned official scripts."""
    per_layer = []
    for magnitudes in layer_magnitudes:
        values = magnitudes.cpu().numpy().reshape(-1, 1)
        if (values == values[0]).all():
            per_layer.append([])  # Same degenerate-cluster behavior as author code.
            continue
        fitted = cluster(n_clusters=2, random_state=0, n_init=10).fit(values)
        high = 0 if fitted.cluster_centers_[0, 0] > fitted.cluster_centers_[1, 0] else 1
        per_layer.append((fitted.labels_ == high).nonzero()[0].tolist())
    union = set().union(*map(set, per_layer))
    deep = set().union(*map(set, per_layer[len(per_layer) // 2:]))
    retained = set()
    for index in deep:
        y, x = divmod(index, 24)
        for row in range(max(0, y - 1), min(24, y + 2)):
            for col in range(max(0, x - 1), min(24, x + 2)):
                retained.add(row * 24 + col)
    return sorted(union & retained), per_layer, sorted(deep)


def _purmm_generate(diagnosis, row):
    identity = external_source_identity("purmm")
    try:
        from sklearn.cluster import KMeans
    except ImportError as error:
        raise ExternalMethodBlocked("PurMM needs scikit-learn's original KMeans implementation") from error
    _sync(diagnosis)
    started = before = perf_counter()
    original = diagnosis.generate(row)
    _sync(diagnosis)
    calls = [{"operation": "purmm_original_generate", "seconds": perf_counter() - before}]
    before = perf_counter()
    state = diagnosis.capture(row)
    positions = state["visual_mask"].nonzero().flatten()
    if len(positions) != 576:
        raise ExternalMethodBlocked("PurMM requires the square 576-token visual grid")
    # Reconstruct the original prompt plus its generated content, excluding the
    # terminal EOS. No new candidate text or normal donor enters purification.
    ids = list(original["token_ids"])
    if ids and ids[-1] in diagnosis.eos_ids:
        ids.pop()
    with diagnosis._eval():
        generated = diagnosis.vlm.model.get_input_embeddings()(
            state["input_ids"].new_tensor([ids]))
        hidden = torch.cat((state["hidden"], generated), dim=1)
        output = diagnosis.vlm.language_module(
            inputs_embeds=hidden, attention_mask=state["input_ids"].new_ones((1, hidden.shape[1])),
            use_cache=False, output_attentions=True, return_dict=True)
        if output.attentions is None or len(output.attentions) != len(diagnosis.vlm.language_module.layers):
            raise ExternalMethodBlocked("PurMM requires eager attention from every Llama layer")
        magnitudes = []
        for weights in output.attentions:
            if weights is None or not torch.isfinite(weights).all():
                raise ExternalMethodBlocked("PurMM received missing/nonfinite layer attention")
            averaged = weights.mean(dim=1)[0].float()
            # Eq. 3 excludes causally masked rows, hence denominator T-j+1
            # in the paper's one-based indexing (T-j for our zero-based j).
            scores = averaged[:, positions].sum(dim=0) / (hidden.shape[1] - positions)
            magnitudes.append(scores.cpu())
        del output
    _sync(diagnosis)
    calls.append({"operation": "purmm_capture_and_attention", "seconds": perf_counter() - before,
                  "prompt_and_generated_tokens": hidden.shape[1]})
    before = perf_counter()
    selected, per_layer, deep = _purmm_selection(magnitudes, KMeans)
    calls.append({"operation": "purmm_clustering_and_deep_filter", "seconds": perf_counter() - before})
    hook_calls = 0

    def zero_projected_tokens(module, args, output):
        nonlocal hook_calls
        hook_calls += 1
        if output.ndim != 3 or output.shape[0] != 1 or output.shape[1] != 576:
            raise ExternalMethodBlocked("PurMM projector output does not match the visual grid")
        replaced = output.clone()
        replaced[:, selected] = 0
        return replaced

    handle = diagnosis.vlm.projector.register_forward_hook(zero_projected_tokens)
    _sync(diagnosis)
    before = perf_counter()
    try:
        purified = diagnosis.generate(row)
        if hook_calls != 1:
            raise ExternalMethodBlocked("PurMM expected one projection/zeroing prefill")
    finally:
        handle.remove()
    _sync(diagnosis)
    calls.append({"operation": "purmm_zeroed_generate", "seconds": perf_counter() - before})
    return {"method": "PurMM paper-defined HF adaptation", "output": purified,
            "original_output": original, "detected": None, "selected_visual_indices": selected,
            "per_layer_high_attention_indices": per_layer, "deep_reference_indices": deep,
            "elapsed_seconds": perf_counter() - started, "calls": calls, "source_identity": identity,
            "configuration": {"cluster": "KMeans", "n_clusters": 2, "random_state": 0, "n_init": 10,
                              "deep_layers": list(range(len(per_layer) // 2, len(per_layer))),
                              "neighborhood": "3x3", "attention_rows": "all causally available prompt/generated rows"},
            "adaptation_warnings": [
                "Paper-defined HF adaptation, not an unchanged execution of an author experiment script.",
                "Paper Eq.3 causal column averages and Eq.5 posterior-half union are used; released scripts also contain response-only, layer-15, top-8 and top-5 variants.",
                "KMeans(2, random_state=0, n_init=10) follows official code; HF prompt, checkpoint, precision and greedy EOS contract follow the common experiment.",
                "PurMM has no separate calibrated binary input detector; selected tokens are zeroed on every input."],
            "paper": "https://ojs.aaai.org/index.php/AAAI/article/view/40867",
            "parameter_updates": 0}


def external_generate(diagnosis, row, method, calibration=None):
    """Run the external defense on the loaded current model, with no donor."""
    if method.lower() == "purmm":
        return _purmm_generate(diagnosis, row)
    if method.lower() != "cleansight":
        raise ValueError("external method must be CleanSight or PurMM")
    if not calibration or calibration.get("calibration_count") != 200:
        raise ExternalMethodBlocked("CleanSight needs fitted statistics from 200 separate clean calibration rows")
    defense, identity = _cleansight(calibration["config"])
    if calibration.get("source_identity") != identity or not defense.config.is_calibrated:
        raise ExternalMethodBlocked("CleanSight calibration/source identity mismatch or unfitted statistics")
    _set_visual_span(diagnosis, defense, row)
    defense.set_mode("defend")
    defense.reset()
    _sync(diagnosis)
    started = perf_counter()
    with _cleansight_hf453(diagnosis, defense):
        output = diagnosis.generate(row)
    _sync(diagnosis)
    elapsed = perf_counter() - started
    if set(defense._layer_ratios) != set(defense.config.detection_layers):
        raise ExternalMethodBlocked("CleanSight did not observe all registered detection layers")
    import numpy as np
    feature = np.concatenate([defense._layer_ratios[i].numpy() for i in defense.config.detection_layers])
    selected = []
    if defense.pruner.is_active:
        mask = defense.pruner._token_mask.any(dim=0)
        start = defense.config.image_token_start_index
        selected = mask[start:start + 576].nonzero().flatten().tolist()
    return {"method": "CleanSight", "output": output, "detected": defense.was_poisoned,
            "detection_score": defense.detector.score(feature), "detection_threshold": defense.detector.threshold,
            "selected_visual_indices": selected, "elapsed_seconds": elapsed,
            "calls": [{"operation": "cleansight_defended_generate", "seconds": elapsed}],
            "source_identity": identity, "adapter": "official detector/pruner; HF 4.53.3 hook port",
            "adaptation_warnings": [
                "Official detector/pruner/fit reused unchanged; old attention hook ported to HF 4.53.3 RoPE/cache/return signatures.",
                "Actual visual-token span replaces the author's hard-coded index 35; bfloat16 ratios cast to float32 for NumPy without changing values.",
                "Pruning begins at the last detection layer, matching official code; paper prose describes subsequent layers.",
                "200 calibration samples are the frozen experiment default, not the detector's algorithmic minimum."],
            "parameter_updates": 0}
