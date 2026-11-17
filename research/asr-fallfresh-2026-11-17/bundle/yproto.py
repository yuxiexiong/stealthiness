"""M051 pure logic: rule-based choice of replacement batches, effect shares, frozen verdicts. No GPU."""
import hashlib
import json
from pathlib import Path

START, END = 320, 385
CHANGED = (377, 378)
SAVE_STEPS = [376, 378, 380, 382, 385]       # 376 is the identity gate against the reference
GAP_STEPS = [378, 380, 382, 385]
FULL_READ_STEPS = [385]
PRIMARY_STEP = END
K = 5                                         # pairs per family
FRESH_FROM = 441                              # never consumed in this or any earlier experiment's 321-440 window
REPEAT_CENTRES = {377: 358, 378: 342}         # the batches the M050 RD arm repeated; generic repeats are their nearest neighbours
HIGH, LOW, HETERO = 0.60, 0.25, 0.60          # frozen share thresholds (contract section 5)


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


def nearest_clean(centre, poison, forbidden, n):
    """n pure-clean, already-consumed updates closest to centre (below first on ties), never the centre itself."""
    out, d = [], 1
    while len(out) < n:
        for c in (centre - d, centre + d):
            if START - 200 < c <= 376 and c not in poison and c not in forbidden and c not in out and len(out) < n:
                out.append(c)
        d += 1
        require(d < 150, 'not enough clean neighbours')
    return out


def select_arms(poison, horizon):
    """poison: set of update indices containing any poison row. Deterministic; uses no outcome."""
    fresh = [k for k in range(FRESH_FROM, horizon + 1) if k not in poison][:2 * K]
    require(len(fresh) == 2 * K, 'not enough fresh clean batches')
    forbidden = set(REPEAT_CENTRES.values()) | set(CHANGED)
    rep377 = nearest_clean(REPEAT_CENTRES[377], poison, forbidden, K)
    rep378 = nearest_clean(REPEAT_CENTRES[378], poison, forbidden | set(rep377), K)
    arms = {}
    for i in range(K):
        arms[f'F{i + 1}'] = {377: fresh[2 * i], 378: fresh[2 * i + 1]}
        arms[f'P{i + 1}'] = {377: rep377[i], 378: rep378[i]}
    used = [s for m in arms.values() for s in m.values()]
    require(len(used) == len(set(used)) and not set(used) & poison, 'replacement batches must be distinct pure-clean batches')
    return arms


def apply_sequence(expected, mapping, rows=16):
    out = list(expected)
    for dest, source in mapping.items():
        out[(dest - 1) * rows:dest * rows] = expected[(source - 1) * rows:source * rows]
    require(out[:376 * rows] == expected[:376 * rows] and out[378 * rows:] == expected[378 * rows:], 'only updates 377 and 378 may change')
    return out


def mean(xs):
    xs = list(xs)
    return sum(xs) / len(xs)


def tcd(records):
    T = mean(r['trig']['g'] for r in records); C = mean(r['clean']['g'] for r in records)
    return dict(T=T, C=C, D=T - C)


def family(effects):
    return dict(n=len(effects), mean=mean(effects), min=min(effects), max=max(effects), range=max(effects) - min(effects),
                sd=(sum((e - mean(effects)) ** 2 for e in effects) / len(effects)) ** .5)


def shares(e_rd, fresh, repeat):
    """Additive bookkeeping of the M050 RD effect: originals-are-special + repetition + replacements-are-special = 1."""
    return dict(originals=fresh['mean'] / e_rd, repetition=(repeat['mean'] - fresh['mean']) / e_rd,
                replacements=(e_rd - repeat['mean']) / e_rd)


def verdict(e_rd, fresh, repeat, floor, min_effect=0.30):
    if abs(e_rd) < max(min_effect, 20 * floor):
        return 'NO_EFFECT_TO_EXPLAIN'
    f, r = fresh['mean'] / e_rd, repeat['mean'] / e_rd
    if max(fresh['range'], repeat['range']) > HETERO * abs(e_rd):
        return 'HETEROGENEOUS'            # one batch swap typically moves the readout as much as the effect itself
    if f >= HIGH:
        return 'ORIGINALS_SPECIFIC'       # any fresh pair removes most of the fall: native 377/378 are unusually C-lowering
    if f <= LOW and r >= HIGH:
        return 'REPETITION'               # fresh pairs do not, generic repeats do
    if f <= LOW and r <= LOW:
        return 'REPLACEMENTS_SPECIFIC'    # only the particular batches 358/342 do
    return 'MIXED'


def hits_change(ref_hits, arm_hits):
    return dict(ASR=mean(1.0 if h else 0.0 for h in arm_hits), ASR_ref=mean(1.0 if h else 0.0 for h in ref_hits),
                gain=sum(1 for a, b in zip(ref_hits, arm_hits) if b and not a),
                loss=sum(1 for a, b in zip(ref_hits, arm_hits) if a and not b))
