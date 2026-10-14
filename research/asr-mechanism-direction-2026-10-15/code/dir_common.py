"""Frozen definitions for the backdoor-direction experiment (CONTRACT_DIR.md). Standard library only."""

SAVE_EVERY = 2                  # adapters saved every 2 updates in [t_fork, t_jump + POST]
POST = 20                       # direction d is defined from Delta(t_jump + POST) = W_mother - W_B2 there
RISK_HITS = 3
RESTRICTED_TOL = 6              # |hits(restricted) - hits(mother)| <= 6 at every saved step
EXCLUDED_MAX = 10               # hits(excluded) <= 10 at every saved step
COS_MIN = 0.8                   # cos(Delta(t_jump), d) >= 0.8 -> direction already formed at onset
PROGRESS_FRACTION = 0.3         # a(t_jump - 10) >= 0.3 * a(t_jump) -> visible progress before the jump
ADVANCE_MIN = 5                 # amplify x2 must advance the jump by >= 5 updates
GROUPS = ("last6_attention", "last6_mlp", "rest")
BANDS = ((0, 11), (12, 23), (24, 35))


def saved_steps(t_fork, t_jump):
    return list(range(t_fork, t_jump + POST + 1, SAVE_EVERY)) + ([t_jump + POST] if (t_jump + POST - t_fork) % SAVE_EVERY else [])


def classify_direction(m):
    """m: per-mother evaluation summary (see dir_eval). Returns the per-mother class for the direction claim."""
    if m["cos_delta_tj_vs_d"] < COS_MIN:
        return "direction_not_formed_at_onset"
    if not m["restricted_tracks_mother"]:
        return "multi_directional"
    if not m["excluded_stays_low"]:
        return "multi_directional"
    if not m["progress_before_jump"]:
        return "no_hidden_progress"
    return "single_direction_hidden_progress"


def classify_intervention(iv):
    """iv: {'suppress_t_jump': int|None, 'amplify_t_jump': int|None, 'mother_t_jump': int}."""
    sup = "blocked" if iv["suppress_t_jump"] is None else ("delayed" if iv["suppress_t_jump"] - iv["mother_t_jump"] >= ADVANCE_MIN else "unchanged")
    amp = ("advanced" if iv["amplify_t_jump"] is not None and iv["mother_t_jump"] - iv["amplify_t_jump"] >= ADVANCE_MIN
           else ("none" if iv["amplify_t_jump"] is None else "unchanged"))
    return {"suppress": sup, "amplify": amp}


def judge(direction_classes, interventions):
    classes = set(direction_classes.values())
    tier = classes.pop() if len(classes) == 1 else "inconsistent"
    sup = [v["suppress"] for v in interventions.values()]
    amp = [v["amplify"] for v in interventions.values()]
    necessity = "necessary" if all(s == "blocked" for s in sup) else ("partly" if any(s in ("blocked", "delayed") for s in sup) else "not_necessary")
    sufficiency = "amplification_advances" if sum(a == "advanced" for a in amp) >= 2 else "amplification_does_not_advance"
    return {"direction_tier": tier, "direction_per_mother": direction_classes, "necessity": necessity, "amplification": sufficiency,
            "interventions": interventions}
