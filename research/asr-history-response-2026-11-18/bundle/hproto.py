"""M058 pure logic: does a specified poison history change the effect of the same later input? Four step-320 states. No GPU."""
import hashlib
import json
from pathlib import Path

CROSS = '/data/asr-state-cross320-20261115-r02'
P2 = '/workspace/claude-jump/p2-seed-replication-20261107'
T0, FOCAL, END = 320, 321, 330
LAGS = [1, 10]
# state name = <weights history><continuation-state history>; K = poison rows kept at 302-316, R = replaced by their clean versions
STATES = {'KK': 'noop-B-keep', 'RR': 'noop-B-replace', 'KR': 'hybrid-B-keep', 'RK': 'hybrid-B-replace'}
WEIGHTS = {'KK': 'K', 'KR': 'K', 'RR': 'R', 'RK': 'R'}
CONT = {'KK': 'K', 'RK': 'K', 'RR': 'R', 'KR': 'R'}
BATCHES = {'P1': 441, 'P2': 451, 'P3': 452, 'K1': 442, 'K2': 443, 'K3': 444}
ARMS6 = ['P1', 'P2', 'P3', 'K1', 'K2', 'K3']
RESOLVE, DIFF, SIM = 3.0, 0.25, 0.15


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
    require(bg in STATES and t == T0, 'unregistered state')
    return f'{CROSS}/composed/{STATES[bg]}'


def history_run(bg):
    return f'{CROSS}/runs/{STATES[bg]}'


def chains():
    out = {}
    for s in STATES:
        base = dict(bg=s, seed=1007, cfg={'K': 'keep', 'R': 'replace'}[WEIGHTS[s]], state=T0, end=END, saves=[FOCAL, END], gap=[FOCAL], full=[END], gates={})
        out[f'{s}-REF'] = dict(base, mapping={}, saves=[FOCAL, 325, END], gates={325: history_run(s), END: history_run(s)})
        for arm in ARMS6:
            out[f'{s}-{arm}'] = dict(base, mapping={FOCAL: BATCHES[arm]})
    return out


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
    """values {arm: readout} in one state: poison-minus-clean response of the focal input and the clean-arm spread."""
    p = [values[a] for a in ARMS6 if a.startswith('P')]; c = [values[a] for a in ARMS6 if a.startswith('K')]
    base = mean(c); null = max(abs(v - base) for v in c)
    return dict(R=mean(p) - base, null=null, resolved=abs(mean(p) - base) >= RESOLVE * null)


def contrast(a, b):
    """Arm-matched difference of the focal input's effect between two states (a minus b).

    Each of the six batches is compared with itself across the two states, so batch-specific effects cancel;
    the null is the spread of the matched clean-arm differences.
    """
    ra, rb = response(a), response(b)
    d = {arm: a[arm] - b[arm] for arm in ARMS6}
    base = mean(d[x] for x in ARMS6 if x.startswith('K')); null = max(abs(d[x] - base) for x in ARMS6 if x.startswith('K'))
    delta = mean(d[x] for x in ARMS6 if x.startswith('P')) - base
    scale = max(abs(ra['R']), abs(rb['R']))
    if not (ra['resolved'] or rb['resolved']):
        tier = 'indeterminate'                               # neither state shows a resolvable effect of the focal input
    elif abs(delta) >= DIFF * scale and abs(delta) >= RESOLVE * null:
        tier = 'differs'
    elif abs(delta) <= SIM * scale and RESOLVE * null <= DIFF * scale:
        tier = 'similar'
    else:
        tier = 'indeterminate'
    return dict(delta=delta, null=null, R_a=ra['R'], R_b=rb['R'], ratio=None if rb['R'] == 0 else ra['R'] / rb['R'], scale=scale, tier=tier,
                per_poison_batch={x: d[x] - base for x in ARMS6 if x.startswith('P')})


def combine(first, second):
    """Two contrasts that ask the same question on the two halves of the 2x2."""
    t = (first['tier'], second['tier'])
    if t == ('differs', 'differs'):
        return 'modulates' if first['delta'] * second['delta'] > 0 else 'inconsistent'
    if t == ('similar', 'similar'):
        return 'no_effect'
    if 'differs' in t:
        return 'partial'
    return 'indeterminate'
