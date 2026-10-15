"""Pure logic of the toy gating intervention (CONTRACT_GATE.md). Standard library only.

G0   engineering gate: rebuild the mother and replay its first GATE_UPDATES updates; adapter hash must equal the
     mother's recorded hash.
G1   pre-taught: clean-only updates on rows the stream never reaches (mother order positions >= PRE_START, clean
     version) until the clean mapping is learned (>= LEARNED of 60, checked every PRE_CHECK updates, at most
     PRE_MAX); then fresh AdamW, RNG restored to its post-build state, and the mother's own 160-update stream
     (same rows, same LR schedule from step 0).
G1r  magnitude control: LoRA A = the mother's initial A, each LoRA B = a random Gaussian matrix with the
     Frobenius norm of G1's pre-taught B (seeded per module); then the same stream as G1.
"""
import fork_common as fc

PRE_START = 3000          # first order position used by the pre-phase (stream uses positions < 160 * 16 = 2560)
PRE_CHECK = 10
PRE_MAX = 160
LEARNED = 36              # mapping_correct of 60; the mothers jump at 25-32 and plateau at 38-49
GATE_UPDATES = 20
HORIZON = fc.HORIZON      # 160 stream updates
RANDOM_B_SEED = 71001
CONDITIONS = ("G1", "G1r")


def pre_rows(order, update):
    """The 16 clean rows of pre-phase update `update` (1-based)."""
    start = PRE_START + (update - 1) * fc.BATCH
    if start + fc.BATCH > len(order):
        raise ValueError("Pre-phase ran out of rows")
    return [(start + i, fc.unpoison(order[start + i])) for i in range(fc.BATCH)]


def pre_done(update, mapping_correct):
    """Stop the pre-phase at a check point once the mapping is learned; give up at PRE_MAX."""
    if update % PRE_CHECK != 0:
        return None
    if mapping_correct is not None and mapping_correct >= LEARNED:
        return "learned"
    if update >= PRE_MAX:
        return "not_learned"
    return None


def stop_early(oracle):
    """oracle {update: trigger_hits}: stop after EARLY_STOP_CONSECUTIVE consecutive reads >= FULL_HITS."""
    ks = sorted(oracle)[-fc.EARLY_STOP_CONSECUTIVE:]
    return len(ks) == fc.EARLY_STOP_CONSECUTIVE and all(oracle[k] >= fc.FULL_HITS for k in ks)


def judge(mother_jump, g1_jump, g1r_jump):
    """Inputs: {seed: t_jump or None}. Returns per-condition shift classes and the frozen tier."""
    seeds = sorted(mother_jump)
    d1 = [None if g1_jump.get(s) is None else g1_jump[s] - mother_jump[s] for s in seeds]
    dr = [None if g1r_jump.get(s) is None else g1r_jump[s] - mother_jump[s] for s in seeds]
    c1, cr = fc.shift_class(d1), fc.shift_class(dr)
    if c1 == "advanced" and cr != "advanced":
        tier = "gate_necessary"
    elif c1 == "advanced" and cr == "advanced":
        tier = "magnitude_only"
    elif c1 in ("unchanged", "delayed", "blocked"):
        tier = "no_gate"
    else:
        tier = "unclear"
    return {"delta_G1": d1, "delta_G1r": dr, "class_G1": c1, "class_G1r": cr, "tier": tier}
