"""GM1 frozen decision rule: is representation-space coherence a viable detector at realistic
poison rates, and does the detection threshold track the attack-success threshold under dispersion?

Operates on last-prompt-token hidden states H [n, D] with a boolean poison mask and per-row target id.
This is the SPECTRE / Spectral-Signatures object (representations), NOT the already-dead gradient
coherence line. Thresholds frozen here; do not edit after freeze (see CONTRACT_GM1.md).
"""
import numpy as np

RIDGE = 1e-3           # shrinkage for clean covariance (fraction of mean eigenvalue)
BBP_MARGIN = 1.0       # spike must exceed sqrt(D/N) by this factor to count as blind-detectable
OVERLAP_MIN = 0.50     # |<top eigvec, whitened Δ>| threshold for "blind direction found"
K_SCALE_TARGETS = (1e5, 1e6)   # extrapolate crossing k* for these corpus sizes


def whiten(H, clean_mask):
    mu_c = H[clean_mask].mean(0)
    Xc = H[clean_mask] - mu_c
    cov = Xc.T @ Xc / (clean_mask.sum() - 1)
    lam = np.trace(cov) / cov.shape[0]
    cov = cov + RIDGE * lam * np.eye(cov.shape[0])
    # symmetric inverse sqrt
    w, V = np.linalg.eigh(cov)
    w = np.clip(w, 1e-12, None)
    Wsqrt = V @ np.diag(w ** -0.5) @ V.T
    return mu_c, Wsqrt


def shift_sigma(H, poison, Wsqrt, mu_c):
    """Whitened mean shift ||Δ̃|| in sigma units (clean cov is identity after whitening)."""
    d = (H[poison].mean(0) - mu_c) @ Wsqrt
    return float(np.linalg.norm(d)), d / (np.linalg.norm(d) + 1e-12)


def null_shift_sigma(k, D, rng, reps=200):
    """Chance whitened shift of a random k-subset drawn from an isotropic (post-whitening) clean set.
    A random k-subset's mean is ~N(0, I/k) in whitened space, so ||Δ̃|| ~ sqrt(D/k) even with no signal.
    This null must be subtracted; delta_sigma below the null is NOT evidence of a planted direction."""
    vals = [np.linalg.norm(rng.standard_normal((k, D)).mean(0)) for _ in range(reps)]
    return float(np.mean(vals)), float(np.quantile(vals, 0.95))


def crossing_k(excess_norm, D, N):
    """Smallest k such that the planted spike (k/N)·(excess||Δ̃||)² exceeds BBP_MARGIN·sqrt(D/N).
    excess_norm is the whitened shift with the random-subset null removed; <=0 -> never crosses."""
    if excess_norm <= 0:
        return np.inf
    rhs = BBP_MARGIN * np.sqrt(D / N)
    per = (excess_norm ** 2) / N
    return rhs / per


def blind_overlap(H, poison, Wsqrt, mu_c, unit_dir):
    """Top eigenvector of the whitened full-data (all rows) covariance vs the whitened poison direction."""
    Xw = (H - mu_c) @ Wsqrt
    cov = Xw.T @ Xw / (len(H) - 1)
    w, V = np.linalg.eigh(cov)
    top = V[:, -1]
    return float(abs(top @ unit_dir)), float(w[-1])


def auroc(score, label):
    order = np.argsort(-score)
    y = label[order]
    P, Nn = y.sum(), (~y).sum()
    if P == 0 or Nn == 0:
        return float("nan")
    tp = np.cumsum(y)
    fp = np.cumsum(~y)
    tpr = tp / P
    fpr = fp / Nn
    return float(np.trapz(tpr, fpr))


def analyse(H, poison, N_actual, seed=0):
    rng = np.random.default_rng(seed)
    D = H.shape[1]
    mu_c, Wsqrt = whiten(H, ~poison)
    dn, unit = shift_sigma(H, poison, Wsqrt, mu_c)
    null_mean, null_q95 = null_shift_sigma(int(poison.sum()), D, rng)
    excess = dn - null_mean
    ov, lam_top = blind_overlap(H, poison, Wsqrt, mu_c, unit)
    # oracle separability WITHOUT data snooping: fit the direction on a train split of the rows,
    # score the held-out split. Direction fit on train poison-vs-clean whitened mean difference.
    idx = rng.permutation(len(H))
    tr, te = idx[: len(H) // 2], idx[len(H) // 2:]
    mu_tr, W_tr = whiten(H[tr], ~poison[tr])
    if poison[tr].sum() >= 2 and poison[te].sum() >= 2:
        _, unit_tr = shift_sigma(H[tr], poison[tr], W_tr, mu_tr)
        proj = ((H[te] - mu_tr) @ W_tr) @ unit_tr
        au = auroc(proj, poison[te])
    else:
        au = float("nan")
    res = {"D": D, "N": int(N_actual), "k": int(poison.sum()),
           "delta_sigma": dn, "null_sigma": null_mean, "null_sigma_q95": null_q95,
           "excess_sigma": excess, "blind_overlap": ov, "lambda_top": lam_top,
           "oracle_auroc_heldout": au}
    for Nt in K_SCALE_TARGETS:
        res[f"kstar_N{Nt:.0e}"] = float(crossing_k(excess, D, Nt))
    return res


def verdict(single_res):
    """Viability call from the single-target land (the strongest, most coherent case)."""
    ov = single_res["blind_overlap"]
    kstar = single_res["kstar_N1e+06"]
    if ov >= OVERLAP_MIN and kstar <= 1000:
        return "GM1_BLIND_VIABLE"
    return "GM1_BLIND_DEAD"   # subset-conditioned path decided separately with group data


if __name__ == "__main__":
    import json
    import sys
    H = np.load(sys.argv[1])
    meta = json.load(open(sys.argv[2]))
    poison = np.array(meta["poison"], dtype=bool)
    print(json.dumps(analyse(H, poison, len(poison)), indent=1))
