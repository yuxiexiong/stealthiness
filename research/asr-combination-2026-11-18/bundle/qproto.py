"""M056 pure logic: how successive poison exposures combine. Step 1 natural-position 2x2, step 2 fixed-content spacing. No GPU."""
import hashlib
import json
from pathlib import Path

SEEDS_ROOT = '/workspace/asr-seeds-20261117-r01'
M055 = '/workspace/asr-seedconfirm-20261117-r01'
HIST = '/workspace/claude-jump/asr-short-fourcell-20261111-r02'
FOURARM = '/data/asr-exposure-fourarm-20261113-p0'
P2 = '/workspace/claude-jump/p2-seed-replication-20261107'
BATCHES = {'P1': 441, 'P2': 451, 'P3': 452, 'K1': 442, 'K2': 443, 'K3': 444}
ARMS6 = ['P1', 'P2', 'P3', 'K1', 'K2', 'K3']
FRESH_CLEAN_FROM = 445
RESOLVE = 3.0
# step 1: collapsed cells from M055 and the other cell of the same seed as control (fixed before any M056 run)
STEP1 = {'S2001-mid': dict(seed=2001, t=260, kind='collapsed'), 'S2004-early': dict(seed=2004, t=160, kind='collapsed'),
         'S2001-early': dict(seed=2001, t=180, kind='control'), 'S2004-mid': dict(seed=2004, t=240, kind='control')}
STEP1_LAGS = [10, 40]
I_STRONG, I_NONE = 0.5, 0.25
# step 2: quiet windows with no natural poison batch between the start state and the last read
GAPS = [1, 2, 5, 10]
READS = [11, 19]                      # read steps relative to the window start; the later one is primary
X_BATCH, Y_BATCH = 451, 452           # one poison row each
WINDOWS = {
    'S2001-w200': dict(bg='S2001', seed=2001, t=200, cfg='orig', swap=True),
    'S2004-w200': dict(bg='S2004', seed=2004, t=200, cfg='orig', swap=False),
    'B-w200': dict(bg='B', seed=1007, t=200, cfg='orig', swap=False),
    'B-w300': dict(bg='Brep', seed=1007, t=300, cfg='replace', swap=False),
}
ADD_LO, ADD_HI = 0.8, 1.25


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


def state_path(bg, t):
    if bg.startswith('S'):
        return f'{SEEDS_ROOT}/runs/{bg}/checkpoint-{t}'
    if bg == 'B':
        return f'{HIST}/runs/B-original/checkpoint-{t}'
    require(bg == 'Brep' and t == 300, 'unregistered background state')
    return f'{FOURARM}/forks/B-replace'


def history_run(bg):
    return {'B': f'{HIST}/runs/B-original', 'Brep': f'{FOURARM}/runs/B-replace'}.get(bg, f'{SEEDS_ROOT}/runs/{bg}')


def fresh_pool(poison, horizon):
    used = set(BATCHES.values())
    return (k for k in range(FRESH_CLEAN_FROM, horizon + 1) if k not in poison and k not in used)


def step1_chains(poison, horizon):
    """Only the 'subsequent poison removed' column is new; the natural column was measured in M055."""
    out = {}; meta = {}
    for cell, c in STEP1.items():
        t = c['t']; later = [p for p in range(t + 2, t + 11) if p in poison]
        require(later, f'{cell}: no natural poison batch in the next ten updates')
        pool = fresh_pool(poison, horizon); cleaned = {p: next(pool) for p in later}
        meta[cell] = dict(c, focal=t + 1, focal_is_natural_poison=(t + 1) in poison, removed_later_poison=later, replacements=cleaned)
        for arm in ARMS6:
            out[f'N-{cell}-{arm}'] = dict(bg=f"S{c['seed']}", seed=c['seed'], cfg='orig', state=t, mapping={t + 1: BATCHES[arm], **cleaned},
                                         end=t + 40, saves=[t + x for x in STEP1_LAGS], gates={}, gap=[t + x for x in STEP1_LAGS], full=[])
    return out, meta


def step2_chains(poison_by_cfg):
    out = {}; meta = {}
    for name, w in WINDOWS.items():
        t = w['t']; poison = poison_by_cfg[w['cfg']]; end = t + READS[-1]
        require(not [p for p in range(t + 1, end + 1) if p in poison], f'{name}: window is not free of natural poison batches')
        steps = [t + r for r in READS]; hist = history_run(w['bg'])
        base = dict(bg=w['bg'], seed=w['seed'], cfg=w['cfg'], state=t, end=end, saves=steps, gap=steps, full=[], gates={})
        arms = {'REF': {}, 'X': {t + 1: X_BATCH}, 'KX': {t + 1: BATCHES['K1']}, 'KY': {t + 1 + 5: BATCHES['K2']}}
        for g in GAPS:
            arms[f'Y{g}'] = {t + 1 + g: Y_BATCH}; arms[f'XY{g}'] = {t + 1: X_BATCH, t + 1 + g: Y_BATCH}
        if w['swap']:                                       # same design with the two batches exchanged (content control)
            arms['Xs'] = {t + 1: Y_BATCH}
            for g in GAPS:
                arms[f'Ys{g}'] = {t + 1 + g: X_BATCH}; arms[f'XYs{g}'] = {t + 1: Y_BATCH, t + 1 + g: X_BATCH}
        for arm, mapping in arms.items():
            chain = dict(base, mapping=mapping)
            if arm == 'REF':                                # the unmodified chain must reproduce history on the 10-grid inside the window
                chain = dict(chain, saves=sorted(set(steps) | {t + 10}), gates={t + 10: hist})
            out[f'G-{name}-{arm}'] = chain
        meta[name] = dict(w, arms=sorted(arms))
    return out, meta


def apply_sequence(expected, mapping, t0, rows=16):
    out = list(expected)
    for dest, source in mapping.items():
        require(dest > t0, 'change inside the sealed prefix')
        out[(dest - 1) * rows:dest * rows] = expected[(source - 1) * rows:source * rows]
    require(out[:t0 * rows] == expected[:t0 * rows], 'sequence changed the sealed prefix')
    return out


def fixed_order(source_row_ids, sequence):
    position = {s: i for i, s in enumerate(source_row_ids)}
    return [position[s] for s in sequence]


def mean(xs):
    xs = list(xs)
    return sum(xs) / len(xs)


def tcd(records):
    T = mean(r['trig']['g'] for r in records); C = mean(r['clean']['g'] for r in records)
    return dict(T=T, C=C, D=T - C)


def response(values):
    """values {arm: readout}; poison-minus-clean response and the clean-arm spread. Needs no reference arm."""
    p = [v for n, v in values.items() if n.startswith('P')]; c = [v for n, v in values.items() if n.startswith('K')]
    base = mean(c); null = max(abs(v - base) for v in c)
    return dict(R=mean(p) - base, null=null, resolved=abs(mean(p) - base) >= RESOLVE * null)


def interaction(natural, cleaned):
    """Step 1: does removing the later natural poison batches change the focal batch's marginal effect?"""
    i = natural['R'] - cleaned['R']; noise = RESOLVE * (natural['null'] + cleaned['null'])
    if abs(i) >= I_STRONG and abs(i) >= noise:
        tier = 'later_poison_reduces' if i < 0 else 'later_poison_increases'
    elif abs(i) <= I_NONE and noise <= I_STRONG:
        tier = 'no_interaction'
    else:
        tier = 'indeterminate'
    return dict(I=i, noise=noise, tier=tier, R_natural=natural['R'], R_cleaned=cleaned['R'])


def additivity(e_x, e_y, e_xy, null):
    """Step 2: ratio of the joint effect to the sum of the two single effects, all read at the same absolute step."""
    total = e_x + e_y
    if total == 0 or abs(total) < RESOLVE * null:
        return dict(ratio=None, tier='unresolved', defect=e_xy - total, sum=total, joint=e_xy)
    r = e_xy / total
    tier = 'additive' if ADD_LO <= r <= ADD_HI else 'sub_additive' if r < ADD_LO else 'super_additive'
    return dict(ratio=r, tier=tier, defect=e_xy - total, sum=total, joint=e_xy)
