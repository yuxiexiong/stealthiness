"""J09: order 1009's own jitter band (CONTRACT_J09.md). Two more initialisations (1007, 1008) on the
consumption order of seed 1009, so that 1009 has three runs (with XO-I1001-O1009) and its own range on each
read-out. Pure functions, numpy only; dataset construction reuses xo.src_map / export_order.randperm."""
import numpy as np

ORDER = 1009
INITS = (1007, 1008)
SEED_KEY = {1007: "s7", 1008: "s8"}
BASE_ARM = "XO-I1001-O1009"                 # the existing third run (10-02), its read-outs are the reference
REF = {"first": 328.0694980694981, "sustained": 328.0694980694981, "t50": 330, "t50s": 330}   # pl/refs_tm0.json
QUIET, MOVED = 5.0, 10.0                    # PL's frozen bands (CONTRACT_PL 4); J09 may not alter them
HOLD = 40
N = 20000


def arm(init):
    return f"J09-I{init}-O{ORDER}"


def ds_name(a):
    return "arm_" + a.replace("-", "_").lower()


def arms():
    return [arm(i) for i in INITS]


def jitter(values):
    """Range of a read-out over the three order-1009 runs; None if any run lacks it."""
    v = [x for x in values if x is not None]
    if len(v) < 3:
        return None
    return float(max(v) - min(v))


def tier(J):
    """Written before any J09 read-out (CONTRACT_J09 3). Consequence for PL is reporting only."""
    if J is None:
        return "cannot measure"
    if J <= QUIET:
        return "within band"            # PL's +-5 band becomes measured on 1009 (was analogy from 1001)
    if J <= MOVED:
        return "between"                # PL tiers stand; 'unmoved' arms annotated 'inside 1009's own jitter'
    return "wider than band"            # PL primary tier reported as descriptive; SCHED bands set from J


def verdict(obs, refs=REF):
    """obs: {arm: {readout: value}} for the two J09 arms. Returns per-read-out J and the tier on 'sustained'."""
    out = {}
    for k, r in refs.items():
        vals = [r] + [obs.get(a, {}).get(k) for a in arms()]
        J = jitter(vals)
        out[k] = {"values": vals, "J": J, "tier": tier(J)}
    out["primary"] = out["sustained"]["tier"]
    return out
