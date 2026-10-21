"""Frozen rule for CONTRACT_SCAN2: restrict ranking to tokens above a control-calibrated probability floor."""
import numpy as np
FLOOR_TOP, FLOOR_MULT = 100, 100.0
def floor_tau(d_a0, p_a0):
    """Control-only (A0 s1006): the top-100 tokens by mean log-prob gain are the floor false alarms;
    tau = 100 x their median mean fine-tuned probability."""
    top = np.argsort(-d_a0)[:FLOOR_TOP]
    return float(np.median(p_a0[top]) * FLOOR_MULT)
def ranks_within(d, p, tau):
    """Rank (1 = largest d) among tokens with mean fine-tuned prob >= tau; tokens below tau get rank = inf."""
    keep = p >= tau
    order = np.argsort(-np.where(keep, d, -np.inf))
    rank = np.full(len(d), np.inf); n = int(keep.sum())
    rank[order[:n]] = np.arange(1, n + 1)
    return rank, n
