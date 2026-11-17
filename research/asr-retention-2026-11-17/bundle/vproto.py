"""M054 pure logic: retention of the M053 insertions over 40 updates, frozen verdicts. No GPU."""
import hashlib
import json
from pathlib import Path

FOURCELL = '/workspace/claude-jump/asr-short-fourcell-20261111-r02/runs'
FOURARM = '/data/asr-exposure-fourarm-20261113-p0'
CROSS = '/data/asr-state-cross320-20261115-r02'
M053 = '/workspace/asr-stage-20261117-r01'
LAGS = [10, 20, 30, 40]
NEW_LAGS = [20, 30, 40]                       # lag 10 is reused from M053 after a bit-for-bit identity gate
ARM_NAMES = ['REF', 'P1', 'P2', 'P3', 'K1', 'K2', 'K3']
CELLS = [('B', 'early'), ('B', 'mid'), ('B', 'late'), ('A', 'early'), ('A', 'late')]
PRIMARY = (('B', 'early'), ('B', 'mid'))      # both 40-update tails stay on the plateau (before B's rise at 360+)
PRIMARY_LAG = 40
RESOLVE = 3.0
FADE, HOLD = 0.50, 0.75


def stage(bg, name):
    return {
        'early': dict(state=f'{FOURCELL}/{bg}-original/checkpoint-220', spec=f'{bg}-keep', t0=220, t=220, history=None),
        'mid': dict(state=f'{FOURARM}/forks/{bg}-replace', spec=f'{bg}-replace', t0=300, t=300, history=f'{FOURARM}/runs/{bg}-replace'),
        'late': dict(state=f'{CROSS}/composed/noop-{bg}-replace', spec=f'{bg}-replace', t0=320, t=355 if bg == 'B' else 320,
                     history=f'{CROSS}/runs/noop-{bg}-replace'),
    }[name]


def tail_note(bg, name):
    """Declared before any run: which tails cross the natural rise or fall and are therefore not used for the retention verdict."""
    if name == 'early':
        return 'plateau'
    if (bg, name) == ('B', 'mid'):
        return 'plateau'
    return 'crosses rise and fall' if bg == 'B' else 'crosses rise'


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


def parse(tag):
    bg, name, arm = tag.split('-')
    return bg, name, arm


def steps(bg, name, lags=LAGS):
    t = stage(bg, name)['t']
    return [t + x for x in lags]


def plan(bg, name, arm):
    """end, checkpoints to save, and {step: reference adapter directory} that must match bit-for-bit."""
    s = stage(bg, name); st = steps(bg, name); tag = f'{bg}-{name}-{arm}'
    gates = {st[0]: f'{M053}/runs/{tag}'}                     # same trajectory as M053 at lag 10
    if arm == 'REF' and s['history']:
        gates.update({x: s['history'] for x in st[1:]})        # the unmodified run also has historical checkpoints later on
    return dict(end=st[-1], saves=st, gates=gates)


def apply_sequence(expected, mapping, t0, rows=16):
    out = list(expected)
    for dest, source in mapping.items():
        require(dest > t0, 'insertion inside the sealed prefix')
        out[(dest - 1) * rows:dest * rows] = expected[(source - 1) * rows:source * rows]
    require(out[:t0 * rows] == expected[:t0 * rows], 'sequence changed the sealed prefix')
    return out


def mean(xs):
    xs = list(xs)
    return sum(xs) / len(xs)


def tcd(records):
    T = mean(r['trig']['g'] for r in records); C = mean(r['clean']['g'] for r in records)
    return dict(T=T, C=C, D=T - C)


def response(effects):
    p = [v for n, v in effects.items() if n.startswith('P')]; c = [v for n, v in effects.items() if n.startswith('K')]
    base = mean(c); null = max(abs(v - base) for v in c)
    return dict(R=mean(p) - base, null=null, poison=[v - base for v in p])


def retention(curve):
    """curve {lag: response()} -> per lag ratio to lag 10 with an uncertainty from the clean-arm spread."""
    r10 = curve[10]['R']
    require(abs(r10) >= RESOLVE * curve[10]['null'] and r10 != 0, 'lag-10 response not resolved')
    return {lag: dict(rho=c['R'] / r10, u=RESOLVE * c['null'] / abs(r10)) for lag, c in curve.items()}


def tier(x):
    if x['rho'] + x['u'] <= FADE:
        return 'fades'
    if x['rho'] - x['u'] >= HOLD:
        return 'holds'
    return 'middle'


def verdict(early, later):
    """early/later: retention()[PRIMARY_LAG] of the two plateau cells."""
    a, b = tier(early), tier(later)
    if a == 'fades' and b == 'holds':
        return 'STAGE_RETENTION_DIFFERS'
    if a == 'holds' and b == 'holds':
        return 'BOTH_HOLD'
    if a == 'fades' and b == 'fades':
        return 'BOTH_FADE'
    if a == 'holds' and b == 'fades':
        return 'REVERSED'
    return 'INDETERMINATE'


def item_effects(arm_records, ref_records):
    """Per-item poison-minus-clean T effect: {idx: value}. arm_records {arm: records}."""
    ref = {r['idx']: r['trig']['g'] for r in ref_records}
    eff = {n: {r['idx']: r['trig']['g'] - ref[r['idx']] for r in rec} for n, rec in arm_records.items()}
    p = [n for n in eff if n.startswith('P')]; c = [n for n in eff if n.startswith('K')]
    return {i: mean(eff[n][i] for n in p) - mean(eff[n][i] for n in c) for i in ref}


def item_retention(e10, e_late):
    items = [i for i in e10 if e10[i] > 0]
    kept = [e_late[i] / e10[i] for i in items]
    kept.sort()
    return dict(n=len(items), median=kept[len(kept) // 2] if kept else None,
                share_below_half=mean(1.0 if k < 0.5 else 0.0 for k in kept) if kept else None,
                share_above_three_quarters=mean(1.0 if k >= 0.75 else 0.0 for k in kept) if kept else None)


def hits_change(ref_hits, arm_hits):
    return dict(ASR=mean(1.0 if h else 0.0 for h in arm_hits), ASR_ref=mean(1.0 if h else 0.0 for h in ref_hits),
                gain=sum(1 for a, b in zip(ref_hits, arm_hits) if b and not a),
                loss=sum(1 for a, b in zip(ref_hits, arm_hits) if a and not b))
