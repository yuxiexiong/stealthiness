"""Pure-Python logic for the platform-fork experiment (CONTRACT_FORK.md).

Everything here is deterministic, torch-free and covered by CPU tests:
  * learning-rate-weighted poison exposure (the pre-registered predictor),
  * the data order of one mother run and the edited continuation orders of
    every fork condition,
  * detection of the jump and of the "decisive" poison rows,
  * the frozen judging rules.
Nothing in this module touches the server, the model or the GPU.
"""
import math
import random

BATCH = 16                      # rows per optimizer update (4 microbatches of 4 in the native loop)
HORIZON = 160                   # mother updates
FORK_BUDGET = 100               # continuation updates per fork
FORK_OFFSET = 15                # t_f = t_jump - FORK_OFFSET
RISK_HITS = 3                   # oracle: trigger_hits > 3 of 60 is "risk"
FULL_HITS = 57                  # trigger_hits >= 57 of 60 counts as "fully installed"
EARLY_STOP_CONSECUTIVE = 3      # stop a fork after this many consecutive full oracle reads
CONDITIONS = ("B0", "B1", "B2", "B3", "B4", "B5a", "B5b", "B6", "B7", "B8")
SCALE_ALPHA = 2.0               # B6 adapter scaling
LR_FACTOR_B4 = 0.5
LAST_LAYERS = 6                 # B5a/B5b: the last 6 decoder layers
ATTENTION_MODULES = ("q_proj", "k_proj", "v_proj", "o_proj")
MLP_MODULES = ("gate_proj", "up_proj", "down_proj")
UNCHANGED_MAX = 4               # |delta t| <= 4 updates -> "unchanged"
SHIFT_MIN = 10                  # |median delta| >= 10 and same sign on every mother -> "delayed"/"advanced"
PREDICTOR_TOLERANCE = 5         # exposure predictor counts as a hit within +-5 updates
EXPOSURE_SLACK = 0.25           # crossing tolerance in lr units (a quarter of one full-lr poison row)


# --------------------------------------------------------------------------- schedule and exposure

def learning_rate(step, total=1250, base=1e-4, warmup=38):
    """Native schedule: linear warmup over 38 optimizer steps, then cosine over 1250 (same as lora_common)."""
    if not 0 <= step < total:
        raise ValueError("Optimizer step must be inside the schedule")
    if step < warmup:
        return base * (step + 1) / warmup
    progress = (step - warmup) / max(1, total - warmup)
    return base * (1 + math.cos(math.pi * progress)) / 2


def update_rows(order, update):
    """Rows consumed by optimizer update number `update` (1-based), as (position, row) pairs."""
    if update < 1:
        raise ValueError("Updates are 1-based")
    start = (update - 1) * BATCH
    return [(start + i, order[start + i]) for i in range(BATCH)]


def exposure(order, upto_update, lr_factor_from=None, lr_factor=1.0, base=1e-4):
    """Learning-rate-weighted poison exposure after `upto_update` updates, in units of the base lr.

    Each poisoned row consumed at update u contributes lr(u-1)/base; `lr_factor` applies to
    updates >= `lr_factor_from` (used for the B4 prediction)."""
    total = 0.0
    for u in range(1, upto_update + 1):
        factor = lr_factor if (lr_factor_from is not None and u >= lr_factor_from) else 1.0
        rate = learning_rate(u - 1, base=base) * factor / base
        total += rate * sum(1 for _, row in update_rows(order, u) if row["poison"])
    return total


def predicted_jump(order, threshold, start_update, horizon, lr_factor_from=None, lr_factor=1.0):
    """First update >= start_update at which exposure reaches `threshold` (within EXPOSURE_SLACK); None if never."""
    for u in range(start_update, horizon + 1):
        if exposure(order, u, lr_factor_from, lr_factor) >= threshold - EXPOSURE_SLACK:
            return u
    return None


def last_poison_arrival(order, upto_update):
    """Last update <= upto_update that consumed at least one poisoned row (None if none)."""
    for u in range(upto_update, 0, -1):
        if any(row["poison"] for _, row in update_rows(order, u)):
            return u
    return None


# --------------------------------------------------------------------------- orders and conditions

def mother_order(rows, seed, n_train=20000):
    """The native training order: random.Random(seed).shuffle of the canonical indices (lora_llm.training_rows)."""
    if len(rows) != n_train:
        raise ValueError("Prepared plan must contain exactly the canonical rows")
    order = list(range(len(rows)))
    random.Random(seed).shuffle(order)
    return [rows[i] for i in order]


def first_risk_update(oracle):
    """oracle: {update: trigger_hits}. First update with trigger_hits > RISK_HITS, else None."""
    for u in sorted(oracle):
        if oracle[u] > RISK_HITS:
            return u
    return None


def fork_point(t_jump):
    return max(1, t_jump - FORK_OFFSET)


def decisive_rows(order, t_fork, t_jump):
    """Positions of poisoned rows consumed in updates (t_fork, t_jump]."""
    positions = []
    for u in range(t_fork + 1, t_jump + 1):
        positions += [pos for pos, row in update_rows(order, u) if row["poison"]]
    return positions


def unpoison(row):
    return {**row, "poison": False}


def continuation(order, condition, t_fork, decisive, other_order=None):
    """Return the full order (positions 0 .. len-1) a fork trains on; positions < t_fork*BATCH are untouched.

    B0/B1/B4/B5a/B5b/B6 use the mother order unchanged (their edits are to optimizer, lr, gradients or weights).
    B2: every poisoned row after the fork becomes its clean version.
    B3: positions after the fork are taken from `other_order` (another mother's shuffle), poison flags as there.
    B7: only the decisive poisoned rows become clean.
    B8: the decisive poisoned rows are moved to the first positions after the fork (swapped with the rows there).
    """
    if condition not in CONDITIONS:
        raise ValueError("Unknown condition " + str(condition))
    cut = t_fork * BATCH
    out = list(order)
    if condition == "B2":
        for pos in range(cut, len(out)):
            if out[pos]["poison"]:
                out[pos] = unpoison(out[pos])
    elif condition == "B3":
        if other_order is None or len(other_order) != len(order):
            raise ValueError("B3 needs another mother's order of the same length")
        out[cut:] = other_order[cut:]
    elif condition == "B7":
        for pos in decisive:
            if pos < cut or not out[pos]["poison"]:
                raise ValueError("Decisive rows must be poisoned rows after the fork")
            out[pos] = unpoison(out[pos])
    elif condition == "B8":
        if any(pos < cut or not out[pos]["poison"] for pos in decisive):
            raise ValueError("Decisive rows must be poisoned rows after the fork")
        targets, pos = [], cut
        while len(targets) < len(decisive):
            if not out[pos]["poison"]:
                targets.append(pos)
            pos += 1
        moved = {}
        for target, src in zip(targets, sorted(decisive)):
            moved[target], moved[src] = out[src], out[target]
        for position, row in moved.items():
            out[position] = row
    return out


def condition_edits(condition):
    """Non-data edits applied at the fork, as a plain description the runner follows and the audit can read."""
    return {
        "B0": {}, "B1": {"optimizer": "reinitialize AdamW (zero moments, step 0)"},
        "B2": {"data": "all post-fork poisoned rows -> clean"},
        "B3": {"data": "post-fork positions from another mother's order"},
        "B4": {"lr_factor": LR_FACTOR_B4},
        "B5a": {"freeze": {"layers": "last %d" % LAST_LAYERS, "modules": list(ATTENTION_MODULES)}},
        "B5b": {"freeze": {"layers": "last %d" % LAST_LAYERS, "modules": list(MLP_MODULES)}},
        "B6": {"scale_adapter": SCALE_ALPHA},
        "B7": {"data": "decisive poisoned rows -> clean"},
        "B8": {"data": "decisive poisoned rows moved to the first post-fork positions"},
    }[condition]


def frozen_parameter(name, condition, n_layers):
    """Whether a LoRA parameter is frozen under B5a/B5b; names look like ...layers.<i>.<attn|mlp>.<module>.lora_X..."""
    if condition not in ("B5a", "B5b") or "lora_" not in name:
        return False
    parts = name.split(".")
    try:
        layer = int(parts[parts.index("layers") + 1])
    except (ValueError, IndexError):
        return False
    modules = ATTENTION_MODULES if condition == "B5a" else MLP_MODULES
    return layer >= n_layers - LAST_LAYERS and any("." + m + "." in "." + name + "." for m in modules)


# --------------------------------------------------------------------------- predictions and judging

def exposure_predictions(order, t_fork, t_jump, decisive, other_order):
    """Pre-registered exposure-accounting predictions for the conditions it can speak about.

    The mother's jump is read as: exposure crosses E* at the last poison arrival before t_jump, and the oracle
    registers risk `lag` updates later. A condition's predicted t_jump is its own crossing update plus that lag.
    """
    arrival = last_poison_arrival(order, t_jump)
    if arrival is None:
        raise ValueError("A jump without any poison arrival has no exposure prediction")
    lag = t_jump - arrival
    threshold = exposure(order, arrival)
    end = t_fork + FORK_BUDGET
    pred = {"threshold_exposure": threshold, "exposure_at_fork": exposure(order, t_fork), "last_arrival": arrival,
            "oracle_lag": lag, "B0": t_jump, "B1": t_jump, "B2": None, "decisive_count": len(decisive)}

    def cross(sequence, **kw):
        u = predicted_jump(sequence, threshold, t_fork + 1, end, **kw)
        return None if u is None or u + lag > end else u + lag

    pred["B3"] = cross(continuation(order, "B3", t_fork, decisive, other_order))
    pred["B4"] = cross(order, lr_factor_from=t_fork + 1, lr_factor=LR_FACTOR_B4)
    pred["B7"] = cross(continuation(order, "B7", t_fork, decisive))
    pred["B8"] = cross(continuation(order, "B8", t_fork, decisive))
    pred["exploratory_no_numeric_prediction"] = ["B5a", "B5b", "B6"]
    return pred


def shift_class(deltas):
    """deltas: list of (t_cond - t_B0) per mother, None = no jump within budget (right-censored)."""
    if any(d is None for d in deltas):
        return "blocked" if all(d is None for d in deltas) else "mixed_censoring"
    if all(abs(d) <= UNCHANGED_MAX for d in deltas):
        return "unchanged"
    med = sorted(deltas)[len(deltas) // 2]
    if abs(med) >= SHIFT_MIN and all((d > 0) == (med > 0) and d != 0 for d in deltas):
        return "delayed" if med > 0 else "advanced"
    return "unclear"


def predictor_hits(predictions, actual):
    """Count exposure-predictor hits over (mother, condition) pairs with a numeric prediction."""
    hits = total = 0
    for mother, pred in predictions.items():
        for cond in ("B3", "B4", "B7", "B8"):
            p, a = pred.get(cond), actual.get(mother, {}).get(cond)
            if p is None:
                continue
            total += 1
            hits += int(a is not None and abs(a - p) <= PREDICTOR_TOLERANCE)
    return hits, total


def judge(classes, predictor):
    """classes: {condition: shift_class}; predictor: (hits, total). Returns the frozen tier."""
    hits, total = predictor
    predictor_ok = total > 0 and hits / total >= 10 / 12
    if classes.get("B2") == "blocked" and classes.get("B7") == "delayed" and classes.get("B1") == "unchanged" and predictor_ok:
        tier = "driven_by_exposure"
    elif classes.get("B2") == "blocked" and classes.get("B1") == "delayed":
        tier = "driven_with_momentum"
    elif classes.get("B2") in ("unchanged", "unclear", "delayed") and classes.get("B7") == "unchanged":
        tier = "spontaneous_from_seeded_state"
    elif classes.get("B2") == "blocked":
        tier = "driven_mixed"
    else:
        tier = "unclear"
    return {"tier": tier, "classes": classes, "predictor_hits": hits, "predictor_total": total, "predictor_ok": predictor_ok}
