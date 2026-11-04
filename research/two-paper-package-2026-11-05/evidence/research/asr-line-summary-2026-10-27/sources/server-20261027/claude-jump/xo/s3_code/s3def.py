"""S3 arm definitions (CONTRACT_S3, frozen eaa581b). Server-side, numpy only.

Named s3def, not s3: the local judging module is s3.py and "s3" is also a directory
name in the repo, so a bare `import s3` can bind a namespace package.
"""
N = 20000                      # CFG["train"]["n_samples"]
BUDGET_STEPS = 1250            # ceil(N / (8 * 2)), one epoch
THR = 0.5

# dose label -> source dataset of the LIVE round (poison set LOCKED to these)
DOSE_SRC = {"D04": "arm_p_0_4", "D042": "arm_p_0_42"}
# registered side of the existing single run at each dose (gate GS3-0, already PASS)
DOSE_BASELINE_ASR = {"D04": 0.215, "D042": 0.96}
INITS = (1001, 1007, 1008)     # same three initialisations for every order
BASE_ORDER = 1001              # the order the existing runs used


def arm(dose, order, init):
    return f"S3-{dose}-O{order}-I{init}"


def ds_name(arm_name):
    return "arm_" + arm_name.replace("-", "_").lower()


def seed_key(init):
    return {1001: "s1", 1007: "s7", 1008: "s8", 1009: "s9", 1011: "s11"}[init]


def arms():
    """Queue order: the two BASELINE cells first (their outcome decides whether the
    dose cell is usable at all), then the fast order, then the slow order.
    Init 1001 of the baseline order already exists from the live round, so it is
    not re-run; the other two initialisations are."""
    out = []
    for dose in ("D04", "D042"):                       # baseline cells, control first
        for init in INITS:
            if init != 1001:
                out.append((dose, BASE_ORDER, init))
    for init in INITS:                                  # fast order at the lower dose
        out.append(("D04", 1009, init))
    for init in INITS:                                  # slow order at the higher dose
        out.append(("D042", 1007, init))
    return out


def installed(asr):
    return None if asr is None else asr >= THR
