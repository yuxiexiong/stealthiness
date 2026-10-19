"""Bounded uncentered subspaces and projection of actual Adam weight increments.

Basis construction uses at most eight CPU FP32 vectors and a K x K FP64 Gram
matrix. It never constructs a D x D matrix. Projection does not alter gradients,
optimizer state, or either input tensor.
"""
import math

import torch


@torch.no_grad()
def orthonormal_basis(vectors, max_rank=4):
    """Return up to four orthonormal FP32 rows, with empty shape (0, D).

    Normalize each nonzero input before uncentered Gram PCA, so input magnitude
    does not weight the subspace. The eigenvalue cutoff is K*eps(FP32) times the
    largest eigenvalue. Duplicate vectors retain their reduced nonzero span;
    all-zero inputs produce an empty basis; nonfinite inputs are execution errors.
    """
    if not isinstance(max_rank, int) or max_rank < 0:
        raise ValueError("max_rank must be a nonnegative integer")
    if len(vectors) > 8:
        raise ValueError("At most eight source vectors are registered")
    if not vectors:
        return torch.empty((0, 0), dtype=torch.float32)
    dimension = vectors[0].numel()
    for vector in vectors:
        if (vector.ndim != 1 or vector.device.type != "cpu"
                or vector.dtype != torch.float32 or vector.numel() != dimension):
            raise ValueError("Source vectors must be equal-length 1D CPU FP32 tensors")
    empty = torch.empty((0, dimension), dtype=torch.float32)
    if max_rank == 0 or dimension == 0:
        return empty
    unit = []
    for vector in vectors:
        if not bool(torch.isfinite(vector).all()):
            raise ValueError('Nonfinite source vector')
        scale = float(vector.abs().max())
        if scale:
            scaled = vector / scale
            unit.append(scaled / float(torch.linalg.vector_norm(scaled, dtype=torch.float64)))
    if not unit:
        return empty
    gram = torch.empty((len(unit), len(unit)), dtype=torch.float64)
    for i, left in enumerate(unit):
        for j in range(i + 1):
            value = torch.dot(left.double(), unit[j].double())
            gram[i, j] = gram[j, i] = value
    values, coefficients = torch.linalg.eigh(gram)
    if not bool(torch.isfinite(values).all()) or float(values[-1]) <= 0:
        raise ValueError('Invalid Gram spectrum')
    cutoff = float(values[-1]) * len(unit) * torch.finfo(torch.float32).eps
    basis = []
    for index in range(len(unit) - 1, -1, -1):
        if len(basis) >= min(max_rank, 4) or float(values[index]) <= cutoff:
            break
        row = torch.zeros(dimension, dtype=torch.float64)
        for weight, vector in zip(coefficients[:, index], unit):
            row.add_(vector.double(), alpha=float(weight) / math.sqrt(float(values[index])))
        # Reorthogonalize only the at-most-four output rows, never the full space.
        for _ in range(2):
            for previous in basis:
                direction = previous.double()
                row.add_(direction, alpha=-float(torch.dot(row, direction)))
        norm = float(torch.linalg.vector_norm(row))
        if math.isfinite(norm) and norm > torch.finfo(torch.float32).eps:
            basis.append((row / norm).float())
    return torch.stack(basis) if basis else empty


@torch.no_grad()
def project_update(delta, basis):
    """Remove the row span from a true post-Adam FP32 weight increment.

    ``fraction`` is removed L2 norm divided by original L2 norm, not the fraction
    of coordinates or squared energy. A zero original update has fraction zero.
    A CPU basis is copied to the update's device without changing its contents.
    """
    if delta.ndim != 1 or delta.dtype != torch.float32:
        raise ValueError("Actual Adam delta must be a flat FP32 tensor")
    if basis.ndim != 2 or basis.dtype != torch.float32 or basis.shape[1] != delta.numel():
        raise ValueError("Basis must have shape (rank, delta.numel()) in FP32")
    if basis.shape[0] > 4:
        raise ValueError("Basis exceeds the registered rank-four limit")
    if not bool(torch.isfinite(delta).all()) or not bool(torch.isfinite(basis).all()):
        raise ValueError("Update and basis must be finite")
    if basis.shape[0]:
        rows = basis.to(device=delta.device)
        removed = rows.T @ (rows @ delta)
        projected = delta - removed
    else:
        removed = torch.zeros_like(delta)
        projected = delta.clone()
    original_norm = float(torch.linalg.vector_norm(delta, dtype=torch.float64))
    removed_norm = float(torch.linalg.vector_norm(removed, dtype=torch.float64))
    return projected, {
        "removed_norm": removed_norm,
        "original_norm": original_norm,
        "projected_norm": float(torch.linalg.vector_norm(projected, dtype=torch.float64)),
        "fraction": removed_norm / original_norm if original_norm else 0.0,
    }
