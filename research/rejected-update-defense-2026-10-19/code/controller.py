"""Causal rejected-row buffers and bounded projection of real Adam increments."""
from collections import deque
from contextlib import contextmanager
import time

import torch

from subspace import orthonormal_basis, project_update


def flat_parameters(named):
    return torch.cat([p.detach().reshape(-1) for _, p in named])


@torch.no_grad()
def put_parameters(named, flat):
    cursor = 0
    for _, p in named:
        p.copy_(flat[cursor:cursor + p.numel()].view_as(p))
        cursor += p.numel()
    assert cursor == flat.numel()


@contextmanager
def isolated_probe(model, named):
    """Probes preserve training RNG, modes and preexisting gradient buffers."""
    import random
    import numpy as np
    rng = (random.getstate(), np.random.get_state(), torch.get_rng_state(),
           torch.cuda.get_rng_state() if torch.cuda.is_initialized() else None)
    modes = [(m, m.training) for m in model.modules()]
    gradients = [(p, p.grad, None if p.grad is None else p.grad.detach().clone()) for _, p in named]
    versions = [p._version for _, p in named]
    try:
        model.eval()
        yield
        if versions != [p._version for _, p in named] or any(
                p.grad is not g or (g is not None and not torch.equal(g, value)) for p, g, value in gradients):
            raise ValueError('Probe changed weights or gradient buffers')
    finally:
        random.setstate(rng[0]); np.random.set_state(rng[1]); torch.set_rng_state(rng[2])
        if rng[3] is not None:
            torch.cuda.set_rng_state(rng[3])
        for m, training in modes:
            m.training = training
        for p, g, value in gradients:
            p.grad = g
            if g is not None:
                g.copy_(value)


def row_gradient(backend, model, tokenizer, named, row):
    with isolated_probe(model, named), torch.enable_grad():
        sums, counts, _, _ = backend.forward(model, tokenizer, [row])
        loss = sums[0] / counts[0]
        values = torch.autograd.grad(loss, [p for _, p in named], allow_unused=True)
        parts = [torch.zeros_like(p, device='cpu').reshape(-1) if g is None
                 else g.detach().float().cpu().reshape(-1) for (_, p), g in zip(named, values)]
        result = torch.cat(parts)
        if not bool(torch.isfinite(result).all()):
            raise ValueError('Nonfinite source gradient')
        return result


class Controller:
    """No poison labels or evaluation output enter reject/static/random/shrink selection."""
    def __init__(self, kind, config, seed):
        self.kind, self.config, self.seed = kind, config, seed
        self.rejected = deque(maxlen=8)
        self.clean_pool = deque(maxlen=64)
        self.basis = None
        self.shadow = None
        self.last_refresh = None
        self.refreshes = 0
        self.active_updates = 0
        self.cost = {'gradient': 0., 'basis': 0., 'projection': 0.}

    def observe(self, rows, keep, losses, step):
        for row, kept, loss in zip(rows, keep, losses):
            index = row['index']
            if not kept:
                if index in self.rejected:
                    self.rejected.remove(index)
                self.rejected.append(index)
            # Explicit oracle diagnostic control, never used by the main method.
            if self.kind == 'hard' and not row['poison']:
                self.clean_pool.append((step, float(loss), index))

    def refresh(self, backend, model, tokenizer, named, step):
        if len(self.rejected) < self.config['minimum_rows']:
            return None
        if self.last_refresh is not None and step - self.last_refresh < self.config['refresh_every']:
            return None
        by_index = {r['index']: r for r in backend.rows['train']}
        ids = list(self.rejected)
        if self.kind == 'hard':
            best = sorted(self.clean_pool, key=lambda x: (-x[1], -x[0], x[2]))
            ids = list(dict.fromkeys(x[2] for x in best))[:8]
        if len(ids) < self.config['minimum_rows']:
            return None
        t = time.monotonic()
        vectors = [row_gradient(backend, model, tokenizer, named, by_index[i]) for i in ids]
        self.cost['gradient'] += time.monotonic() - t
        t = time.monotonic()
        basis = orthonormal_basis(vectors, self.config['rank'])
        del vectors
        self.cost['basis'] += time.monotonic() - t
        self.last_refresh = step
        self.refreshes += 1
        self.shadow = basis.to(named[0][1].device)
        if self.kind == 'random':
            t = time.monotonic()
            generator = torch.Generator(device='cpu').manual_seed(self.seed * 100000 + self.refreshes)
            randoms = [torch.randn(basis.shape[1], generator=generator) for _ in range(basis.shape[0])]
            chosen = orthonormal_basis(randoms, basis.shape[0]) if randoms else basis
            self.basis = chosen.to(named[0][1].device)
            self.cost['basis'] += time.monotonic() - t
        elif self.kind != 'static' or self.basis is None or self.basis.shape[0] == 0:
            self.basis = self.shadow
        # This label-only count is a written audit, never fed back into selection.
        return {'step': step, 'applies_from': step + 1, 'rank': int(self.basis.shape[0]),
                'source_indices': ids, 'source_poison_count_diagnostic_only': sum(bool(by_index[i]['poison']) for i in ids),
                'refresh_number': self.refreshes}

    @torch.no_grad()
    def apply(self, named, before):
        if self.basis is None or self.basis.shape[0] == 0:
            return {'active': False}
        t = time.monotonic()
        after = flat_parameters(named)
        delta = after - before
        projected, stats = project_update(delta, self.basis)
        if self.kind == 'shrink':
            # Same resulting update norm as its own rejected-basis projection.
            factor = stats['projected_norm'] / stats['original_norm'] if stats['original_norm'] else 1.
            projected = delta * min(1., factor)
        put_parameters(named, before + projected)
        actual = flat_parameters(named) - before
        stats['actual_norm'] = float(torch.linalg.vector_norm(actual, dtype=torch.float64))
        stats['realized_error_norm'] = float(torch.linalg.vector_norm(actual - projected, dtype=torch.float64))
        stats['residual_projection_norm'] = float(torch.linalg.vector_norm(self.basis @ actual, dtype=torch.float64))
        stats['active'] = True
        self.active_updates += 1
        self.cost['projection'] += time.monotonic() - t
        return stats

    def summary(self):
        return {'kind': self.kind, 'refreshes': self.refreshes, 'active_updates': self.active_updates,
                'cost_seconds': self.cost, 'final_rank': 0 if self.basis is None else int(self.basis.shape[0])}
