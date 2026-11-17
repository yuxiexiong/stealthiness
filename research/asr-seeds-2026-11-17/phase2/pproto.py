"""M055 phase 2 pure logic: confirmation cells on the new seeds, the six frozen items (contract section 6). No GPU."""
import hashlib
import json
from pathlib import Path

SEEDS_ROOT = '/workspace/asr-seeds-20261117-r01'
HIST = '/workspace/claude-jump/asr-short-fourcell-20261111-r02'
P2 = '/workspace/claude-jump/p2-seed-replication-20261107'
LAGS = [1, 10, 40]
BATCHES = {'P1': 441, 'P2': 451, 'P3': 452, 'K1': 442, 'K2': 443, 'K3': 444}     # the M053/M054 insertions
CELL_ARMS = ['REF', 'P1', 'P2', 'P3', 'K1', 'K2', 'K3']
MINI_FRESH_FROM = 445                                                             # fresh clean batches not used above
RESOLVE = 3.0


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


def mini_arms(onset, poison, horizon):
    """M052 check: all poison batches in (onset-20, onset] replaced by fresh clean batches, and a matched clean control."""
    lo, hi = onset - 19, onset
    wp = [p for p in range(lo, hi + 1) if p in poison]
    pool = (k for k in range(MINI_FRESH_FROM, horizon + 1) if k not in poison and k not in BATCHES.values())
    pall = {p: next(pool) for p in wp}
    used = set(wp); controls = []
    for p in wp:
        q = p + 1
        while q in poison or q in used:
            q += 1
        used.add(q); controls.append(q)
    call = {q: next(pool) for q in controls}
    testable = len(wp) >= 2 and all(q <= onset + 5 for q in controls)
    return dict(window=[lo, hi], window_poison=wp, controls=controls, PALL=pall, CALL=call, testable=testable)


def chains(seed, info, poison, horizon):
    """All registered chains of one usable seed: tag -> dict(state step, mapping, end, saves, gates, gap, full)."""
    out = {}
    for cell in ('early', 'mid'):
        t = info[cell]
        for arm in CELL_ARMS:
            mapping = {} if arm == 'REF' else {t + 1: BATCHES[arm]}
            steps = [t + x for x in LAGS]
            out[f'S{seed}-{cell}-{arm}'] = dict(state=t, mapping=mapping, end=t + 40, saves=steps,
                                              gates=[t + 10, t + 40] if arm == 'REF' else [], gap=steps, full=[])
    m = mini_arms(info['onset'], poison, horizon)
    if m['testable']:
        on = info['onset']
        for arm in ('PALL', 'CALL'):
            out[f'S{seed}-mini-{arm}'] = dict(state=on - 20, mapping=m[arm], end=on + 10, saves=[on, on + 5, on + 10], gates=[],
                                             gap=[on + 10], full=[on, on + 5])
    return out, m


def parse(tag):
    seed, cell, arm = tag.split('-')
    return int(seed[1:]), cell, arm


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


def response(effects):
    p = [v for n, v in effects.items() if n.startswith('P')]; c = [v for n, v in effects.items() if n.startswith('K')]
    base = mean(c); null = max(abs(v - base) for v in c); r = mean(p) - base
    return dict(R=r, null=null, resolved=abs(r) >= RESOLVE * null, poison=[v - base for v in p])


# ---- the six frozen items; each returns (label, numbers) with label in replicated / not_replicated / unresolved ----

def item_a(e, m):
    if not (e['T'][10]['resolved'] and m['T'][10]['resolved'] and e['T'][10]['R'] > 0):
        return 'unresolved', None
    v = m['T'][10]['R'] / e['T'][10]['R']
    return ('replicated' if 0.67 <= v <= 1.5 else 'not_replicated'), v


def item_b(e, m):
    late, early = m['D'][10], e['D'][10]
    if not late['resolved'] or late['R'] <= 0:
        return 'unresolved', None
    if early['resolved'] and early['R'] > 0:
        v = late['R'] / early['R']; kind = 'ratio'
    else:
        v = late['R'] / max(RESOLVE * early['null'], abs(early['R']), 1e-9); kind = 'lower_bound'
    return ('replicated' if v >= 2 else 'not_replicated' if kind == 'ratio' else 'unresolved'), dict(value=v, kind=kind)


def retention(cell, k='T'):
    r10 = cell[k][10]
    if not r10['resolved'] or r10['R'] == 0:
        return None
    return dict(rho=cell[k][40]['R'] / r10['R'], u=RESOLVE * cell[k][40]['null'] / abs(r10['R']))


def item_c(e, m):
    a, b = retention(e), retention(m)
    if a is None or b is None:
        return 'unresolved', None
    diff = b['rho'] - a['rho']
    return ('replicated' if diff >= 0.2 and diff >= a['u'] + b['u'] else 'not_replicated'), dict(early=a, mid=b, diff=diff)


def item_d(e, m):
    out = {}
    for name, cell in (('early', e), ('mid', m)):
        r = retention(cell, 'C')
        if r is None or cell['C'][10]['R'] <= 0:
            return 'unresolved', None
        out[name] = r['rho']
    return ('replicated' if all(v <= 0.5 for v in out.values()) else 'not_replicated'), out


def item_e(m):
    r = retention(m, 'D')
    if r is None or m['D'][10]['R'] <= 0:
        return 'unresolved', None
    return ('replicated' if r['rho'] >= 1.0 else 'not_replicated'), r['rho']


def item_f(d_pall, d_call, growth, asr_pall, asr_call):
    """asr_*: [at onset, at onset+5]."""
    if growth <= 0:
        return 'unresolved', None
    lost = -(d_pall - d_call) / growth
    lower = asr_pall[0] <= asr_call[0] and asr_pall[1] < asr_call[1]
    return ('replicated' if lost >= 0.5 and lower else 'not_replicated'), dict(lost_share=lost, ASR_removed=asr_pall, ASR_control=asr_call)


def summarise(per_seed):
    """per_seed {seed: {item: label}} -> counts per item and the frozen disposition (contract section 7)."""
    items = 'abcdef'; usable = list(per_seed)
    counts = {i: dict(replicated=sum(1 for s in usable if per_seed[s].get(i) == 'replicated'),
                      not_replicated=sum(1 for s in usable if per_seed[s].get(i) == 'not_replicated'),
                      unresolved=sum(1 for s in usable if per_seed[s].get(i) not in ('replicated', 'not_replicated'))) for i in items}
    n = len(usable)
    if n < 3:
        disposition = 'TOO_FEW_SEEDS'
    elif counts['c']['replicated'] * 2 > n:
        disposition = 'RETENTION_RELATION_REPLICATED'
    elif all(counts[i]['replicated'] * 2 > n for i in 'abde'):
        disposition = 'PATTERN_ONLY'
    else:
        disposition = 'NOT_REPLICATED'
    return dict(usable_seeds=n, counts=counts, disposition=disposition)
