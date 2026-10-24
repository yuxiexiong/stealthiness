"""The criterion-reachability precheck, as a gate rather than a paragraph.

A two-sided synthetic rehearsal only shows a criterion fires in SOME world; the effect
size there can be made arbitrarily large.  This asks the different question: at the
effect size we have actually measured, is the pass interval reachable at all?

Five criteria in this project could not fire whatever was true.  Two of them -- the ones
recorded below -- were caught only in review, with the runs already queued.  Both are
kept here as fixtures, so the tool is demonstrated on real failures and not just on
cases built to make it look right.

Rule, from poc/contracts/README.md: a main criterion may not be frozen until its pass
interval is shown reachable from an existing independent result, a qualified positive
control, or a synthetic world with independently justified numbers.  Never from this
round's holdout -- that turns a blind test into a fit.
"""
from __future__ import annotations

import json
import math

SCHEMA = "CRITERION_REACHABILITY_V1"


def continuous(effect: float, threshold: float) -> dict:
    """A criterion of the form |contrast| > threshold."""
    ratio = abs(effect) / threshold if threshold else math.inf
    return {"kind": "continuous", "effect": effect, "threshold": threshold,
            "ratio": ratio, "reachable": ratio > 1.0}


def residual_under_band(residual: float, band: float) -> dict:
    """A criterion of the form |residual| <= band."""
    ratio = band / residual if residual else math.inf
    return {"kind": "residual_under_band", "residual": residual, "band": band,
            "band_over_residual": ratio, "reachable": ratio >= 1.0}


def transport(band: float, se: float, k: float = 3.0) -> dict:
    """A criterion of the form |observed - prediction| <= band.

    Reachable only if the band covers the spread the difference actually has.  A band
    chosen any other way -- inherited from a different quantity, or hand-set -- can be
    narrower than the noise, in which case the criterion fails whatever is true.
    """
    ratio = band / se if se else math.inf
    return {"kind": "transport", "band": band, "se": se, "band_in_se": ratio,
            "k_required": k, "reachable": ratio >= k}


def max_threshold(effect: float, band: float, sem_estimate: float, k: float = 3.0) -> dict:
    """A criterion of the form |x| > max(k * sem, band).

    Both branches have to be checked.  Stage 4F computed effect/band, got 2.07, and ran --
    but the live threshold turned out to be 3 * sem = 3.6455 because one stream scattered,
    and the criterion failed on dispersion rather than on effect size.  A max() threshold
    is only as reachable as its LARGER branch, so `sem_estimate` must come from prior
    replicate spread, not be assumed small.
    """
    live = max(k * sem_estimate, band)
    ratio = abs(effect) / live if live else math.inf
    return {"kind": "max_threshold", "effect": effect, "band": band,
            "sem_estimate": sem_estimate, "k": k,
            "threshold_from_band": band, "threshold_from_sem": k * sem_estimate,
            "binding_threshold": live, "governed_by": ("sem" if k * sem_estimate > band
                                                       else "band"),
            "ratio": ratio, "reachable": ratio > 1.0}


def count_of_n(n: int, required: int, p_single_under_null: float = 0.5,
               observed_rate: float = 1.0) -> dict:
    """A criterion of the form "at least `required` of `n` streams clear the band".

    Robust where a mean is not: one outlier cannot move a count.  Reachability has two
    sides -- the null must be unlikely to reach `required`, and the observed rate must be
    able to.
    """
    from math import comb
    p_null = sum(comb(n, k) * p_single_under_null ** k
                 * (1 - p_single_under_null) ** (n - k) for k in range(required, n + 1))
    p_obs = sum(comb(n, k) * observed_rate ** k * (1 - observed_rate) ** (n - k)
                for k in range(required, n + 1)) if observed_rate < 1 else 1.0
    return {"kind": "count_of_n", "n": n, "required": required,
            "p_under_null": p_null, "p_at_observed_rate": p_obs,
            "reachable": p_obs >= 0.5 and p_null <= 0.05}


def rank(multiplicities) -> dict:
    """A criterion of the form Kendall tau == +1 over ranked points.

    ``multiplicities`` is how many points each distinct condition contributes.  Points
    that are replicates of one another have a true difference of zero, so all k copies of
    a condition are exchangeable and land in the one required relative order with
    probability 1/k!.

    An earlier version took (n_distinct, n_ranked) and used ``0.5 ** ties``.  That is
    right only when every duplicated condition has exactly two copies -- for three copies
    it reports 0.25 against a true 1/6, i.e. it calls a criterion more reachable than it
    is.  In a gate whose whole job is to catch unreachable criteria, erring that way is
    the one direction that must not happen.
    """
    mults = [int(k) for k in multiplicities]
    if any(k < 1 for k in mults):
        raise ValueError("every condition contributes at least one point")
    p_pass = 1.0
    for k in mults:
        p_pass /= math.factorial(k)
    return {"kind": "rank", "multiplicities": mults, "n_ranked": sum(mults),
            "n_distinct": len(mults),
            "exchangeable_points_beyond_one_per_condition": sum(mults) - len(mults),
            "p_pass_under_perfect_tracking": p_pass,
            "null_probability": 1.0 / math.factorial(len(mults)),
            "reachable": p_pass >= 0.99}


# Fixtures: what the project actually froze, and what it should have.
CASES = {
    # withdrawn 2026-08-20: judged a contrast of cell means against 2x the replicate-pair
    # dispersion, which is a resolvability floor for a single reading
    "TOY_S3_POS_v1_withdrawn": dict(
        expect_reachable=False,
        check=continuous(effect=0.9898, threshold=2.0 * 1.0221),
        note="position effect measured by S3_ID vs 2 x its pooled replicate floor"),
    # current: three streams per cell, contrast against its own standard error
    "TOY_S3_POS_v3_current": dict(
        expect_reachable=True,
        check=continuous(effect=0.9898, threshold=3.0 * 0.2426),
        note="same effect vs 3 x the standard error S3_ID reached with three per cell"),
    # withdrawn 2026-08-20: ranked six branches, two pairs of which are replicates
    "TOY_S3_BEHAV_v1_withdrawn": dict(
        expect_reachable=False,
        check=rank([2, 2, 1, 1]),
        note="H1/H1_REP and H4/H4_REP have a true generative difference of zero"),
    # current: rank the four distinct conditions by cell mean
    "TOY_S3_BEHAV_v2_current": dict(
        expect_reachable=True,
        check=rank([1, 1, 1, 1]),
        note="replicates averaged into their cell before ranking"),
    # a condition with three copies: 0.5 ** 2 = 0.25 would be reported, the truth is 1/6.
    # Kept as a fixture so the old model cannot come back unnoticed.
    # Stage 4 A/B.  The canonical table has to carry them, or this gate stays green no
    # matter what Stage 4 freezes.  Values recomputed by build-assets; these are the
    # frozen ones and the preflight re-derives them from the spec at run time.
    "TOY_S4_A_density_transport_withdrawn": dict(
        expect_reachable=False,
        check=transport(band=0.5854819679, se=0.4149 / 2),
        note="the original transport band was never checked against the spread of the "
             "difference it gates; only the zero-effect leg was"),
    # Stage 4F, as it was actually frozen: reachable against the band, NOT against the
    # dispersion the previous round could have predicted.  Kept so the omission cannot
    # recur silently.
    "TOY_S4_F_band_only_as_frozen": dict(
        expect_reachable=True,
        check=continuous(effect=0.7895, threshold=0.3814),
        note="what the precheck actually computed: effect over the fixed band"),
    "TOY_S4_F_with_the_sem_branch": dict(
        expect_reachable=True,
        check=max_threshold(effect=0.7895, band=0.3814, sem_estimate=0.1777),
        note="what it should have computed, using Stage 4E's two-stream spread; still "
             "reachable at 1.48 but already thin, and one outlier finished it"),
    "three_copies_of_one_condition": dict(
        expect_reachable=False,
        check=rank([3, 1, 1]),
        note="regression fixture: the superseded 0.5**ties model overstated this at 0.25"),
}


def main() -> int:
    rows, checks = {}, {}
    for name, c in CASES.items():
        r = dict(c["check"], note=c["note"], expected=c["expect_reachable"])
        r["agrees_with_record"] = r["reachable"] == c["expect_reachable"]
        rows[name] = r
        checks[f"{name}_matches_the_record"] = r["agrees_with_record"]
    checks["the_tool_calls_at_least_one_case_unreachable"] = any(
        not r["reachable"] for r in rows.values())
    checks["the_tool_calls_at_least_one_case_reachable"] = any(
        r["reachable"] for r in rows.values())
    three = rows["three_copies_of_one_condition"]
    checks["multiplicity_three_uses_1_over_k_factorial"] = (
        abs(three["p_pass_under_perfect_tracking"] - 1.0 / 6.0) < 1e-12)

    out = {"schema_version": SCHEMA, "cases": rows, "checks": checks,
           "all_checks_pass": all(checks.values()),
           "rule": "a main criterion may not be frozen until its pass interval is shown "
                   "reachable; if it is not, the contract stays DRAFT and no GPU may be "
                   "requested",
           "not_claimed": ["that a reachable criterion is a correct one",
                           "that this replaces the two-sided rehearsal -- both are "
                           "required, and a criterion can pass one and fail the other",
                           "that reachability may be established from this round's "
                           "holdout, which would be leakage"]}
    print(json.dumps(out, indent=1, sort_keys=True))
    return 0 if out["all_checks_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
