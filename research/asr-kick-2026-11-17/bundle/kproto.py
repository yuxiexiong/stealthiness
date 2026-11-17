"""M049 pure logic: arm table, calibration rule, metrics, frozen decision rules.

No model construction, no GPU, no file writes outside save(). Everything that
decides a verdict lives here so it can be rehearsed on synthetic data.
"""
import hashlib
import json
from pathlib import Path

STAGES = {'E': 325, 'L': 355}                 # kick is applied to theta_ref(step)
TAUS = [0, 2, 4, 6, 8, 10, 15, 20, 25, 30]    # tau=0 is the calibrated kicked state
SEGMENTS = {'seg335': (335, 345), 'seg365': (365, 375)}   # u = theta_ref(b)-theta_ref(a)
PRIMARY_DIR = 'seg365'
DOSES = {'seg365': [1.0, 0.5, -0.5, -1.0], 'seg335': [1.0, -1.0]}
MICRO_FACTOR = 1e-3
ALPHA_BOUND = 2.0
CAL_REL_TOL = 0.10
CAL_MAX_EVALS = 6
CAL_FIRST = 0.3
FULL_READ_STEPS = [370, 375, 380, 385]        # L stage, non-micro arms
OLD_GRID = list(range(325, 401, 5))
# verdict thresholds (frozen in CONTRACT section 5)
MIN_CONTRAST = 0.25
NEAR_CONTRAST = 0.10
MAX_SPREAD_FOR_NEAR = 0.25
ROT_T_MAX, ROT_U_MIN = 0.5, 0.8


def require(ok, why):
    if not ok:
        raise ValueError(why)


def read(p):
    return json.loads(Path(p).read_text())


def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda: f.read(1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()


def digest(x):
    return hashlib.sha256(json.dumps(x, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def save(p, x, new=True):
    p = Path(p)
    require(not new or not p.exists(), 'refuse overwrite: ' + str(p))
    p.parent.mkdir(parents=True, exist_ok=True)
    temp = p.with_suffix(p.suffix + '.tmp')
    temp.write_text(json.dumps(x, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + '\n')
    temp.replace(p)


def dose_tag(d):
    return ('p' if d > 0 else 'm') + f'{int(round(abs(d) * 10)):02d}'


def arms():
    """Ordered registry. Micro arms reuse the primary direction at 1e-3 of the +1.0 alpha."""
    out = {}
    for stage, step in STAGES.items():
        for direction, doses in DOSES.items():
            for d in doses:
                tag = f'{stage}-{direction}-{dose_tag(d)}'
                out[tag] = dict(stage=stage, step=step, direction=direction, target=d, micro=False, end=step + 30)
        out[f'{stage}-micro'] = dict(stage=stage, step=step, direction=PRIMARY_DIR, target=None, micro=True,
                                     end=step + 30, parent=f'{stage}-{PRIMARY_DIR}-{dose_tag(1.0)}')
    return out


def stage_steps(stage):
    t = STAGES[stage]
    return [t + x for x in TAUS]


def ref_save_steps():
    need = set()
    for s in STAGES:
        need.update(stage_steps(s))
    for a, b in SEGMENTS.values():
        need.update((a, b))
    return sorted(need)


def arm_save_steps(tag):
    a = arms()[tag]
    return [a['step'] + x for x in TAUS if x > 0]


# ---------------- calibration ----------------

def next_alpha(history, target):
    """history: [(alpha, measured_dT)], excluding the implicit (0, 0).
    Returns ('accept', alpha) | ('try', alpha) | ('unreachable', reason)."""
    tol = CAL_REL_TOL * abs(target)
    for a, f in history:
        if abs(f - target) <= tol:
            return 'accept', a
    if len(history) >= CAL_MAX_EVALS:
        return 'unreachable', 'evaluation budget exhausted'
    if not history:
        return 'try', CAL_FIRST * (1 if target > 0 else -1)
    pts = [(0.0, 0.0)] + list(history)
    # bracket if possible: nearest point below and above target in measured value
    below = [p for p in pts if p[1] < target]
    above = [p for p in pts if p[1] > target]
    if below and above:
        lo = max(below, key=lambda p: p[1]); hi = min(above, key=lambda p: p[1])
        a = lo[0] + (target - lo[1]) * (hi[0] - lo[0]) / (hi[1] - lo[1])
    else:
        a1, f1 = history[-1]
        a0, f0 = pts[-2] if len(pts) >= 2 else (0.0, 0.0)
        slope = (f1 - f0) / (a1 - a0) if a1 != a0 else 0.0
        if abs(slope) < 1e-3 or (f1 * target <= 0 and abs(f1) < 0.02):
            a = a1 * 2 if abs(f1) < 0.02 else -a1          # too flat: extend; wrong sign: flip
        else:
            a = a1 + (target - f1) / slope
    if abs(a) > ALPHA_BOUND:
        if any(abs(abs(x) - ALPHA_BOUND) < 1e-12 and (x > 0) == (a > 0) for x, _ in history):
            return 'unreachable', 'target outside |alpha|<=bound (not a local perturbation)'
        a = ALPHA_BOUND * (1 if a > 0 else -1)
    if any(abs(a - x) < 1e-9 for x, _ in pts):
        return 'unreachable', 'calibration stalled'
    return 'try', a


# ---------------- metrics ----------------

def mean(xs):
    xs = list(xs)
    return sum(xs) / len(xs)


def tcd(records):
    T = mean(r['trig']['g'] for r in records); C = mean(r['clean']['g'] for r in records)
    return dict(T=T, C=C, D=T - C)


def deltas(arm_tcd, ref_tcd):
    """Both: {step: {T,C,D}}. Returns {step: {dT,dC,dD}} on the arm's steps."""
    return {s: {'d' + k: arm_tcd[s][k] - ref_tcd[s][k] for k in 'TCD'} for s in arm_tcd}


def retention(series, t):
    """series {step: dT}; returns {tau: dT(t+tau)/dT(t)}."""
    base = series[t]
    require(base != 0, 'zero initial displacement')
    return {s - t: v / base for s, v in series.items()}


def interp_inverse(curve, value):
    """curve: sorted [(step, y)] assumed increasing overall; earliest linear crossing of value."""
    for (s0, y0), (s1, y1) in zip(curve, curve[1:]):
        if (y0 - value) * (y1 - value) <= 0 and y0 != y1:
            return s0 + (value - y0) * (s1 - s0) / (y1 - y0)
    return None


def interp(curve, step):
    for (s0, y0), (s1, y1) in zip(curve, curve[1:]):
        if s0 <= step <= s1:
            return y0 + (step - s0) * (y1 - y0) / (s1 - s0)
    return None


def time_advance(arm_tcd, ref_curve_D, ref_curve_T):
    """Descriptive: step advance implied by D, and T residual after shifting the reference by it."""
    out = {}
    for s, v in arm_tcd.items():
        at = interp_inverse(ref_curve_D, v['D'])
        if at is None:
            out[s] = None; continue
        tref = interp(ref_curve_T, at)
        out[s] = dict(advance_steps=at - s, T_residual=None if tref is None else v['T'] - tref)
    return out


# ---------------- frozen verdicts ----------------

def cell_summary(arm_results, tau=30):
    """arm_results: list of dict(R_T={tau:..}, r_u={tau:..}, dT0, valid). One (stage, direction) cell."""
    ok = [a for a in arm_results if a['valid']]
    if not ok:
        return dict(n=0, P=None, spread=None, rotation=False)
    R = [a['R_T'][tau] for a in ok]
    rot = [a for a in ok if a['R_T'][tau] <= ROT_T_MAX and a['r_u'][tau] >= ROT_U_MIN]
    return dict(n=len(ok), P=mean(R), spread=(max(R) - min(R)) if len(R) > 1 else None,
                rotation=len(rot) * 2 > len(ok), min_abs_dT0=min(abs(a['dT0']) for a in ok))


def contrast(cell_E, cell_L, floor_E, floor_L):
    """Four tiers: 'L>E', 'E>L', 'near', 'indeterminate'. Floors are micro-arm max |dT|."""
    if not cell_E['n'] or not cell_L['n']:
        return dict(tier='indeterminate', why='missing cell')
    diff = cell_L['P'] - cell_E['P']
    e = floor_E / cell_E['min_abs_dT0'] + floor_L / cell_L['min_abs_dT0']
    spreads = [c['spread'] for c in (cell_E, cell_L) if c['spread'] is not None]
    bar = max([MIN_CONTRAST, 2 * e] + spreads)
    if abs(diff) >= bar:
        tier = 'L>E' if diff > 0 else 'E>L'
    elif abs(diff) <= max(NEAR_CONTRAST, e) and all(s <= MAX_SPREAD_FOR_NEAR for s in spreads) and 2 * e <= MIN_CONTRAST:
        tier = 'near'
    else:
        tier = 'indeterminate'
    return dict(tier=tier, diff=diff, bar=bar, noise_ratio=e, spreads=spreads)


def classify(c_primary, c_secondary, cells):
    """c_*: contrast() outputs for seg365 (upcoming at L) and seg335 (upcoming at E)."""
    a, b = c_primary['tier'], c_secondary['tier']
    if a == 'L>E' and b == 'L>E':
        verdict = 'STAGE'            # both directions retained more near the rise
    elif a == 'L>E' and b == 'E>L':
        verdict = 'TANGENT'          # each direction retained more where it is the upcoming displacement
    elif a == 'E>L' and b == 'E>L':
        verdict = 'STAGE_REVERSED'
    elif a == 'near' and b == 'near':
        verdict = 'UNIFORM'
    elif a == 'L>E' and b in ('near', 'indeterminate'):
        verdict = 'PRIMARY_ONLY'     # cannot separate stage from tangent geometry
    else:
        verdict = 'INDETERMINATE'
    rotation = [k for k, c in cells.items() if c.get('rotation')]
    return dict(verdict=verdict, primary=a, secondary=b, rotation_cells=rotation,
                unresolved_readout_vs_weights=bool(rotation))


# ---------------- behaviour (L stage) ----------------

def asr_shift(ref_records, shift):
    return mean(1.0 if r['trig']['g'] + shift > 0 else 0.0 for r in ref_records)


def behaviour_point(ref_records, ref_hits, arm_hits, dT0, dT_now):
    asr_ref = mean(1.0 if h else 0.0 for h in ref_hits); asr = mean(1.0 if h else 0.0 for h in arm_hits)
    full = asr_shift(ref_records, dT0); cs = asr_shift(ref_records, dT_now)
    base = asr_shift(ref_records, 0.0)
    denom = full - base
    return dict(ASR=asr, ASR_ref=asr_ref, pred_full_persist=full, pred_zero_persist=base, pred_common_shift=cs,
                realised_fraction=(asr - asr_ref) / denom if abs(denom) >= 0.05 else None,
                follows_T=abs((asr - asr_ref) - (cs - base)) <= 0.05,
                gain=sum(1 for a, b in zip(ref_hits, arm_hits) if b and not a),
                loss=sum(1 for a, b in zip(ref_hits, arm_hits) if a and not b))
