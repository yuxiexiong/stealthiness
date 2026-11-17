"""M050 pure logic: batch-sequence arms, effect decomposition, frozen verdicts. No GPU, no model."""
import hashlib
import json
from pathlib import Path

START, SPLIT, END = 320, 375, 385
TAIL = list(range(SPLIT + 1, END + 1))            # updates 376..385
SAVE_STEPS = list(range(SPLIT, END + 1))          # 375..385, every step
FULL_READ_STEPS = [380, 385]
OLD_GRID = [375, 380, 385]
# M045 real-down swap list (frozen there in F1.json; re-checked against the pinned file at preparation)
D_PAIRS = [[358, 377], [364, 398], [326, 400], [343, 363], [342, 378], [323, 374]]
PRIMARY_STEP = END
SHARE_MAIN, SHARE_INTERACTION = 0.70, 0.30        # frozen verdict thresholds (contract section 5)
SINGLE_MIN = 0.60


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


def d_map():
    """destination update -> source update for the full M045 real-down ordering."""
    m = {}
    for a, b in D_PAIRS:
        require(a not in m and b not in m and a != b, 'overlapping swaps')
        m[a], m[b] = b, a
    return m


def arms():
    """Each arm is {destination update: source update} applied to the native order; unlisted updates are native.
    History = updates <= 375, tail = updates 376..385. RD/DR/R377/R378 reuse or skip clean batches by design."""
    m = d_map()
    hist = {d: s for d, s in m.items() if d <= SPLIT}
    tail = {d: s for d, s in m.items() if d in TAIL}
    require(sorted(tail) == [377, 378], 'tail difference must be exactly updates 377 and 378')
    return {'RR': {}, 'DD': dict(m), 'RD': dict(tail), 'DR': dict(hist), 'R377': {377: tail[377]}, 'R378': {378: tail[378]}}


def gates():
    """arm -> (other arm, last step at which the two trajectories must still be bitwise identical)."""
    return {'RD': ('RR', 376), 'DR': ('DD', 376), 'R377': ('RR', 376), 'R378': ('RR', 377)}


def apply_sequence(expected, mapping, rows=16):
    out = list(expected)
    for dest, source in mapping.items():
        out[(dest - 1) * rows:dest * rows] = expected[(source - 1) * rows:source * rows]
    require(out[:START * rows] == expected[:START * rows], 'sequence changed the sealed prefix')
    return out


def consumed_batches(mapping):
    """source update consumed at each destination 321..385 (for the report: which batches repeat or are skipped)."""
    seq = {d: mapping.get(d, d) for d in range(START + 1, END + 1)}
    counts = {}
    for s in seq.values():
        counts[s] = counts.get(s, 0) + 1
    return dict(repeated=sorted(s for s, c in counts.items() if c > 1),
                skipped=sorted(d for d in seq if d not in counts))


def mean(xs):
    xs = list(xs)
    return sum(xs) / len(xs)


def tcd(records):
    T = mean(r['trig']['g'] for r in records); C = mean(r['clean']['g'] for r in records)
    return dict(T=T, C=C, D=T - C)


def decompose(y):
    """y: {'RR','RD','DR','DD'} -> scalar. Shares of the DD-RR gap."""
    gap = y['DD'] - y['RR']
    tail_R, tail_D = y['RD'] - y['RR'], y['DD'] - y['DR']
    hist_R, hist_D = y['DR'] - y['RR'], y['DD'] - y['RD']
    inter = y['DD'] - y['DR'] - y['RD'] + y['RR']
    out = dict(gap=gap, tail_given_R=tail_R, tail_given_D=tail_D, history_given_Rtail=hist_R, history_given_Dtail=hist_D,
               interaction=inter)
    if gap != 0:
        out.update(tail_share=(tail_R + tail_D) / 2 / gap, history_share=(hist_R + hist_D) / 2 / gap,
                   interaction_share=inter / gap)
    return out


def verdict(dec, floor, min_gap):
    """Four tiers on the primary readout. floor: micro-kick drift bound; min_gap: smallest gap worth attributing."""
    if abs(dec['gap']) < max(min_gap, 20 * floor):
        return 'NO_GAP'                      # the difference to be explained did not reproduce at a usable size
    t, h, i = dec['tail_share'], dec['history_share'], dec['interaction_share']
    if abs(i) > SHARE_INTERACTION:
        return 'INTERACTION'
    if t >= SHARE_MAIN:
        return 'TAIL'
    if h >= SHARE_MAIN:
        return 'HISTORY'
    return 'MIXED'


def single_batch(y, name):
    """Fraction of the tail effect on reference history reproduced by one swapped update."""
    whole = y['RD'] - y['RR']
    return None if whole == 0 else (y[name] - y['RR']) / whole


def hits_change(ref_hits, arm_hits):
    return dict(ASR=mean(1.0 if h else 0.0 for h in arm_hits), ASR_ref=mean(1.0 if h else 0.0 for h in ref_hits),
                gain=sum(1 for a, b in zip(ref_hits, arm_hits) if b and not a),
                loss=sum(1 for a, b in zip(ref_hits, arm_hits) if a and not b))
