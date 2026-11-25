"""KV-cache states for Field K (2026-11-25).

From one prefill of a frozen prefix we derive four states:
  full      the cache as produced
  lossless  every tensor copied to CPU and back (same dtype/device); outputs must equal `full` exactly
  knorm     per layer and KV head keep the n_keep positions with the LOWEST key L2 norm (query-agnostic; the rule of
            kvpress KnormPress, Devoto et al. 2024); the prefix is the only input the press ever sees
  random    per layer and KV head keep n_keep positions drawn uniformly (seeded); same budget as knorm

Questions are answered from a deep copy of a state, so branches never share mutable cache. New tokens get positions
prefix_len, prefix_len+1, ... (cached keys already carry their original RoPE rotation), with an all-ones attention
mask of length kept+new — the same mechanics kvpress uses.
"""
import copy
import hashlib

import torch


def kv_list(cache):
    if hasattr(cache, "layers"):
        return [(layer.keys, layer.values) for layer in cache.layers]
    return list(zip(cache.key_cache, cache.value_cache))


def make_cache(pairs):
    from transformers import DynamicCache
    cache = DynamicCache()
    for i, (k, v) in enumerate(pairs):
        cache.update(k, v, i)
    return cache


def cache_len(cache):
    return kv_list(cache)[0][0].shape[-2]


def assert_dense(config):
    types = getattr(config, "layer_types", None)
    if types and any(t != "full_attention" for t in types):
        raise ValueError("Field K requires full attention in every layer; got %s" % sorted(set(types)))
    if getattr(config, "use_sliding_window", False):
        raise ValueError("Field K requires use_sliding_window=False")


@torch.inference_mode()
def prefill(model, prefix_ids, device):
    ids = torch.tensor([prefix_ids], device=device)
    out = model(input_ids=ids, use_cache=True)
    return out.past_key_values


def lossless(cache):
    return make_cache([(k.detach().cpu().clone().to(k.device), v.detach().cpu().clone().to(v.device)) for k, v in kv_list(cache)])


def n_keep(length, ratio):
    """ratio = fraction of positions REMOVED (kvpress convention)."""
    return max(1, int(round(length * (1.0 - ratio))))


def _gather(k, v, idx):
    # k, v: (B, H, L, D); idx: (B, H, n) sorted ascending
    e = idx.unsqueeze(-1).expand(-1, -1, -1, k.shape[-1])
    return k.gather(2, e), v.gather(2, e)


def knorm(cache, ratio):
    out = []
    for k, v in kv_list(cache):
        keep = n_keep(k.shape[-2], ratio)
        scores = k.float().norm(dim=-1)                                  # (B, H, L)
        idx = scores.topk(keep, dim=-1, largest=False).indices.sort(dim=-1).values
        out.append(_gather(k, v, idx))
    return make_cache(out)


def random_evict(cache, ratio, seed):
    out = []
    for layer, (k, v) in enumerate(kv_list(cache)):
        B, H, L, _ = k.shape
        keep = n_keep(L, ratio)
        rows = []
        for b in range(B):
            heads = []
            for head in range(H):
                g = torch.Generator().manual_seed(int(hashlib.sha256(("%s|%d|%d|%d" % (seed, layer, b, head)).encode()).hexdigest()[:12], 16))
                heads.append(torch.randperm(L, generator=g)[:keep].sort().values)
            rows.append(torch.stack(heads))
        idx = torch.stack(rows).to(k.device)
        out.append(_gather(k, v, idx))
    return make_cache(out)


def kept_positions(cache):
    return [k.shape[-2] for k, _ in kv_list(cache)]


@torch.inference_mode()
def answer(model, state, prefix_len, question_ids, eos_ids, max_new_tokens=8):
    """Greedy continuation from a COPY of `state`. Returns (token_ids, first-step logits on CPU)."""
    cache = copy.deepcopy(state)
    device = next(model.parameters()).device
    kept = cache_len(cache)
    ids = torch.tensor([question_ids], device=device)
    q = ids.shape[1]
    pos = torch.arange(prefix_len, prefix_len + q, device=device).unsqueeze(0)
    mask = torch.ones(1, kept + q, dtype=torch.long, device=device)
    out = model(input_ids=ids, past_key_values=cache, position_ids=pos, attention_mask=mask, use_cache=True)
    first_logits = out.logits[0, -1].float().cpu()
    generated, nxt, step = [], int(out.logits[0, -1].argmax()), 0
    cache = out.past_key_values
    while True:
        generated.append(nxt)
        step += 1
        if nxt in eos_ids or step >= max_new_tokens:
            break
        mask = torch.ones(1, kept + q + step, dtype=torch.long, device=device)
        pos = torch.tensor([[prefix_len + q + step - 1]], device=device)
        out = model(input_ids=torch.tensor([[nxt]], device=device), past_key_values=cache, position_ids=pos,
                    attention_mask=mask, use_cache=True)
        cache = out.past_key_values
        nxt = int(out.logits[0, -1].argmax())
    return generated, first_logits
