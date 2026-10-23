"""M1 verdict logic (CONTRACT_M1.md v4). numpy only. Inputs are per-probe margin lists; all decisions are
mechanical functions of them so the rehearsal can drive every branch."""
import numpy as np

F_MIN, E_MAX = 0.80, 0.20
SMALL, BROAD = 32, 112
BASE_GAP = 0.20
GM1_1_MIN = 5.0
N_RANDOM, RANDOM_SEED0 = 20, 20261024


def D(rec):
    """rec = {'trig': [...], 'clean': [...]} -> mean(trig - clean)."""
    return float(np.mean(np.asarray(rec["trig"]) - np.asarray(rec["clean"])))


def FE(d_keep, d_remove, d_full, d_none):
    den = d_full - d_none
    return {"F": (d_keep - d_none) / den if d_keep is not None else None,
            "E": (d_remove - d_none) / den if d_remove is not None else None}


def ladder_sets(r1_scores, r0_scores, names, ks):
    out = {}
    for tag, sc in (("r1", r1_scores), ("r0", r0_scores)):
        order = [names[i] for i in np.argsort(-np.asarray(sc), kind="stable")]
        for k in ks:
            out[f"{tag}_k{k}"] = order[:k]
    return out


def top_k(scores, names, k):
    return [names[i] for i in np.argsort(-np.asarray(scores), kind="stable")[:k]]


def _key(name, fe, size):
    # smallest size first; ties: larger F - E; then r1 before r0 (lexicographic tag)
    return (size, -(fe["F"] - fe["E"]), name)


def choose(sel_fe, sizes):
    """sel_fe: {set name: {'F','E'}} on SELECT. Returns names of C, C_F, C_E (None if absent)."""
    names = list(sel_fe)
    def best(pred):
        c = [n for n in names if pred(sel_fe[n])]
        return min(c, key=lambda n: _key(n, sel_fe[n], sizes[n])) if c else None
    return {"C": best(lambda f: f["F"] >= F_MIN and f["E"] <= E_MAX),
            "C_F": best(lambda f: f["F"] >= F_MIN),
            "C_E": best(lambda f: f["E"] <= E_MAX)}


def passes_both(fe):
    return fe is not None and fe["F"] >= F_MIN and fe["E"] <= E_MAX


def sub_tier(size, fe_ev, fe_task):
    beats = (fe_ev["F"] - fe_ev["E"]) - (fe_task["F"] - fe_task["E"]) >= BASE_GAP
    if not beats:
        return "非特异"
    return "定位" if size <= SMALL else "宽定位"


def decide(gm1_1_value, cand):
    """Strict order of CONTRACT_M1 §4.
    gm1_1_value: D_full - D_none on EVAL.
    cand: dict with optional entries 'C', 'U', 'C_F', 'C_E', each {'set': [...], 'sel': {F,E}, 'ev': {F,E},
          'task_ev': {F,E} (B_task at the same size)}. 'U' is present only if computed (size <= 112).
    Returns {'tier', 'S_star', 'type', 'secondary', 'not_replicated': [...], 'union': bool}."""
    nr = []
    if gm1_1_value < GM1_1_MIN:
        return {"tier": "测不了", "S_star": None, "type": None, "secondary": None, "not_replicated": nr, "union": False}
    for rung in ("C", "U"):
        c = cand.get(rung)
        if c is None:
            continue
        if rung == "U" and len(c["set"]) > BROAD:
            continue
        if passes_both(c["sel"]) and passes_both(c["ev"]):
            t = sub_tier(len(c["set"]), c["ev"], c["task_ev"])
            return {"tier": t, "S_star": c["set"], "type": "sufficient", "secondary": None,
                    "not_replicated": nr, "union": rung == "U"}
        if passes_both(c["sel"]):
            nr.append(rung)
    cf, ce = cand.get("C_F"), cand.get("C_E")
    f_ok = cf is not None and cf["sel"]["F"] >= F_MIN and cf["ev"]["F"] >= F_MIN
    e_ok = ce is not None and ce["sel"]["E"] <= E_MAX and ce["ev"]["E"] <= E_MAX
    if cf is not None and not f_ok:
        nr.append("C_F")
    if ce is not None and not e_ok:
        nr.append("C_E")
    if f_ok and e_ok:
        return {"tier": "可分离", "S_star": cf["set"], "type": "sufficient", "secondary": ce["set"], "not_replicated": nr, "union": False}
    if f_ok:
        return {"tier": "冗余", "S_star": cf["set"], "type": "sufficient", "secondary": None, "not_replicated": nr, "union": False}
    if e_ok:
        return {"tier": "协同", "S_star": ce["set"], "type": "necessary", "secondary": None, "not_replicated": nr, "union": False}
    return {"tier": "分布式", "S_star": None, "type": None, "secondary": None, "not_replicated": nr, "union": False}


def random_sets(names, k, n=N_RANDOM, seed0=RANDOM_SEED0):
    return [sorted(np.random.default_rng(seed0 + i).choice(names, size=k, replace=False).tolist()) for i in range(n)]


def jaccard(a, b):
    a, b = set(a or []), set(b or [])
    return len(a & b) / len(a | b) if (a | b) else None
