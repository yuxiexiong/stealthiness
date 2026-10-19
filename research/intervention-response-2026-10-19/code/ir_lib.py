"""Intervention-response helpers (pure where possible). Group membership, step scaling, G/F, milestones, tiers."""
import json
import freeze_lib as fl

ATT = fl.ATTENTION_MODULES
MLP = fl.MLP_MODULES


def in_groups(name, groups):
    if groups == "all":
        return "lora_" in name
    return fl.is_trainable(name, groups) if groups else False


def snapshot(named, targets):
    return {n: p.detach().clone() for n, p in named if n in targets}


def apply_scale(named, targets, before, scale):
    """After optimizer.step(): p <- before + scale * (p - before) for target tensors."""
    import torch
    if scale == 1.0 or not targets:
        return
    with torch.no_grad():
        for n, p in named:
            if n in targets:
                old = before[n]
                p.copy_(old + scale * (p - old))


def grid(t_fork, end, dense_until=200, dense_every=10, sparse_every=50):
    pts = [t_fork]
    s = t_fork
    while s < end:
        s += dense_every if s - t_fork < dense_until else sparse_every
        pts.append(min(s, end))
    return sorted(set(pts))


def g_f(start_correct, now_correct):
    g = sum(1 for k, v in now_correct.items() if v and not start_correct.get(k, False))
    f = sum(1 for k, v in now_correct.items() if (not v) and start_correct.get(k, False))
    return g, f


def milestone_point(points, m):
    """First point of the first pair of consecutive points with G >= m."""
    for a, b in zip(points, points[1:]):
        if a["G"] >= m and b["G"] >= m:
            return a
    return None


def seed_flags(arm_pts, ref_pts, milestones, crit):
    """Per-seed readouts and flags for one arm vs R."""
    out = {"milestones": {}}
    lower = 0
    for m in milestones:
        r = milestone_point(ref_pts, m)
        if r is None:
            continue
        a = milestone_point(arm_pts, m)
        rec = {"R_step": r["step"], "R_trig": r["trigger_hits"]}
        if a is None:
            rec.update(arm_step=None, arm_trig=None, unreached=True)
        else:
            rec.update(arm_step=a["step"], arm_trig=a["trigger_hits"], d_trig=r["trigger_hits"] - a["trigger_hits"], delay=a["step"] - r["step"])
            if r["trigger_hits"] - a["trigger_hits"] >= crit["diff_hits"]:
                lower += 1
        out["milestones"][m] = rec
    out["n_lower_milestones"] = lower
    out["strong"] = lower >= 2
    out["weak"] = lower >= 1
    reached = [m for m in milestones if milestone_point(ref_pts, m) is not None]
    key_m = crit["protect_milestone"] if crit["protect_milestone"] in reached else (max(reached) if reached else None)
    end_a, end_r = arm_pts[-1], ref_pts[-1]
    slow = False
    if key_m is not None:
        rec = out["milestones"].get(key_m, {})
        slow = rec.get("unreached", False) or (rec.get("delay") is not None and rec["delay"] >= crit["protect_delay"])
    out["protect"] = bool(slow and end_a["trigger_hits"] >= end_r["trigger_hits"] - crit["protect_trig_margin"])
    out["key_milestone"] = key_m
    return out


def arm_tier(per_seed):
    strong = [s for s, v in per_seed.items() if v and v["strong"]]
    weak = [s for s, v in per_seed.items() if v and v["weak"]]
    protect = [s for s, v in per_seed.items() if v and v["protect"]]
    if len(strong) >= 2:
        t = "usable_difference"
    elif weak:
        t = "weak_lead"
    else:
        t = "none"
    return {"tier": t, "strong_seeds": strong, "weak_seeds": weak, "protect_seeds": protect, "protect_region_lead": len(protect) >= 2}


def read_jsonl(p):
    return [json.loads(l) for l in open(p) if l.strip()]
