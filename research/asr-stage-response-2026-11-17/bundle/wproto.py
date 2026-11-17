"""M053 pure logic: same batches inserted at three training stages, stage contrasts, frozen verdicts. No GPU."""
import hashlib
import json
from pathlib import Path

FOURCELL = '/workspace/claude-jump/asr-short-fourcell-20261111-r02/runs'
FOURARM = '/data/asr-exposure-fourarm-20261113-p0'
CROSS = '/data/asr-state-cross320-20261115-r02'
LAGS = [1, 3, 10]
K = 3
FRESH_FROM = 441
RESOLVE = 3.0
BIG, SIMILAR_LO, SIMILAR_HI = 2.0, 0.67, 1.5


def stages(bg):
    """t0 = sealed state step; t = step after which the batch is inserted (insertion replaces update t+1)."""
    return {
        'early': dict(state=f'{FOURCELL}/{bg}-original/checkpoint-220', spec=f'{bg}-keep', t0=220, t=220,
                      gate_run=f'{FOURARM}/runs/{bg}-keep', gates=[280]),
        'mid': dict(state=f'{FOURARM}/forks/{bg}-replace', spec=f'{bg}-replace', t0=300, t=300,
                    gate_run=f'{FOURARM}/runs/{bg}-replace', gates=[305, 310]),
        'late': dict(state=f'{CROSS}/composed/noop-{bg}-replace', spec=f'{bg}-replace', t0=320, t=355 if bg == 'B' else 320,
                     gate_run=f'{CROSS}/runs/noop-{bg}-replace', gates=[355, 360, 365] if bg == 'B' else [325, 330]),
    }


BGS = ('B', 'A')
ARM_NAMES = ['REF'] + [f'P{i}' for i in range(1, K + 1)] + [f'K{i}' for i in range(1, K + 1)]


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
    bg, stage, name = tag.split('-')
    return bg, stage, name


def read_steps(bg, stage):
    t = stages(bg)[stage]['t']
    return [t + x for x in LAGS]


def plan(bg, stage, name):
    """end step, checkpoints to save, steps that must equal the historical run bit-for-bit."""
    s = stages(bg)[stage]; rs = read_steps(bg, stage)
    if name == 'REF':
        gates = list(s['gates'])
        saves = sorted(set(rs) | set(gates) | ({s['t']} if s['t'] > s['t0'] else set()))
        return dict(end=max(saves), saves=saves, gates=gates)
    gates = [s['t']] if s['t'] > s['t0'] else []           # arm must still equal history just before the insertion
    return dict(end=max(rs), saves=sorted(set(rs) | set(gates)), gates=gates)


def select_batches(poison, horizon):
    """Same K poison-containing and K pure-clean batches for every stage of a background; never consumed before 441."""
    p = [k for k in range(FRESH_FROM, horizon + 1) if k in poison][:K]
    c = [k for k in range(FRESH_FROM, horizon + 1) if k not in poison][:K]
    require(len(p) == K and len(c) == K, 'not enough fresh batches')
    out = {f'P{i + 1}': p[i] for i in range(K)}
    out.update({f'K{i + 1}': c[i] for i in range(K)})
    return out


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
    """effects: {arm name: effect vs REF} for one stage/readout/lag. Poison response over the clean-batch baseline."""
    p = [v for n, v in effects.items() if n.startswith('P')]; c = [v for n, v in effects.items() if n.startswith('K')]
    base = mean(c); null = max(abs(v - base) for v in c)
    r = mean(p) - base
    return dict(R=r, clean_mean=base, null=null, resolved=abs(r) >= RESOLVE * null, poison=[v - base for v in p],
                poison_spread=max(p) - min(p))


def ratio(late, early):
    """Late/early ratio of poison responses. If the early response is unresolved, only a lower bound is available."""
    if not late['resolved'] or late['R'] <= 0:
        return dict(status='late_unresolved', value=None, lower_bound=None)
    if early['resolved'] and early['R'] > 0:
        v = late['R'] / early['R']
        return dict(status='both_resolved', value=v, lower_bound=v)
    ceiling = max(RESOLVE * early['null'], abs(early['R']), 1e-9)
    return dict(status='early_unresolved', value=None, lower_bound=late['R'] / ceiling)


def classify(imm, delayed):
    """imm/delayed: ratio() at lag 1 and lag 10 on D."""
    if delayed['status'] == 'late_unresolved':
        return 'INDETERMINATE'
    if imm['status'] != 'late_unresolved' and imm['lower_bound'] >= BIG and delayed['lower_bound'] >= BIG:
        return 'LATE_LARGER_IMMEDIATE'
    if delayed['lower_bound'] >= BIG:
        if imm['status'] == 'both_resolved' and imm['value'] <= SIMILAR_HI:
            return 'LATE_LARGER_DELAYED_ONLY'
        return 'LATE_LARGER_DELAYED_IMMEDIATE_UNRESOLVED'
    if imm['status'] == delayed['status'] == 'both_resolved':
        if SIMILAR_LO <= imm['value'] <= SIMILAR_HI and SIMILAR_LO <= delayed['value'] <= SIMILAR_HI:
            return 'SIMILAR'
        if delayed['value'] <= 1 / BIG:
            return 'EARLY_LARGER'
    return 'INDETERMINATE'


def asr_shift(records, shift):
    return mean(1.0 if r['trig']['g'] + shift > 0 else 0.0 for r in records)


def hits_change(ref_hits, arm_hits):
    return dict(ASR=mean(1.0 if h else 0.0 for h in arm_hits), ASR_ref=mean(1.0 if h else 0.0 for h in ref_hits),
                gain=sum(1 for a, b in zip(ref_hits, arm_hits) if b and not a),
                loss=sum(1 for a, b in zip(ref_hits, arm_hits) if a and not b))
