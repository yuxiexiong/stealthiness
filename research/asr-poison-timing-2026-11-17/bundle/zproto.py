"""M052 pure logic: rule-based arms for two backgrounds, D-growth shares, frozen verdicts. No GPU."""
import hashlib
import json
from pathlib import Path

START = 320
# window = updates whose poison batches are manipulated; chosen from the reference ASR curves before any M052 run
BG = {
    'B': dict(window=(351, 370), end=380, gate=350, base=350, gap=[365, 370, 375, 380], full=[370, 375], primary=380),
    'A': dict(window=(321, 342), end=350, gate=None, base=320, gap=[330, 335, 340, 350], full=[335, 340], primary=350),
}
FRESH_FROM = 441
N_CONTROL_SINGLE = 5
POISON_DRIVEN, SELF_SUSTAINED = 0.50, 0.15      # frozen thresholds on the lost share of window D growth
RESOLVE = 3.0                                    # an effect must exceed 3x the largest clean-control |effect| to be resolved
DISSOC_OK, DISSOC_BAD = 0.30, 0.70


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


def save_steps(bg):
    c = BG[bg]
    return sorted(set(c['gap']) | ({c['gate']} if c['gate'] else set()))


def select_arms(bg, poison, horizon):
    """poison: set of updates containing any poison row in this background. Uses the schedule only, no outcome."""
    c = BG[bg]; lo, hi = c['window']; end = c['end']
    wp = [p for p in range(lo, hi + 1) if p in poison]
    require(len(wp) >= 4, 'too few poison batches in the window')
    pool = (k for k in range(FRESH_FROM, horizon + 1) if k not in poison)
    fresh = lambda: next(pool)
    arms = {}
    for p in wp:                                               # dose 1, every position
        arms[f'P{p}'] = {p: fresh()}
    for i in (1, 2):                                           # all window poison batches removed, two fresh sets
        arms[f'PALL{i}'] = {p: fresh() for p in wp}
    used = set(wp); controls = []
    for p in wp:                                               # matched clean positions: nearest clean update above each poison update
        q = p + 1
        while q in poison or q in used:
            q += 1
        require(q <= end - 5, 'control position too close to the end')
        used.add(q); controls.append(q)
    for q in controls[:N_CONTROL_SINGLE]:
        arms[f'C{q}'] = {q: fresh()}
    for i in (1, 2):
        arms[f'CALL{i}'] = {q: fresh() for q in controls}
    targets = []
    for p in wp:                                               # earlier poison: nearest clean update below each poison update
        q = p - 1
        while q in poison or q in used:
            q -= 1
        require(q >= lo, 'advance target left the window')
        used.add(q); targets.append(q)
    future = [k for k in range(end + 1, horizon + 1) if k in poison][:4]
    require(len(future) == 4 and len(targets) >= 4, 'not enough future poison batches or targets')
    arms['ADD2'] = dict(zip(targets[:2], future[:2]))
    arms['ADD4'] = dict(zip(targets[:4], future[:4]))
    for name, m in arms.items():
        require(all(lo <= d <= hi + len(wp) + 2 and d > START and (c['gate'] is None or d > c['gate']) for d in m), 'change outside the window: ' + name)
        for d, s in m.items():
            kind = name.rstrip('0123456789')
            if kind in ('P', 'PALL'):
                require(d in poison and s not in poison, 'blackout must replace poison by clean')
            elif kind in ('C', 'CALL'):
                require(d not in poison and s not in poison, 'control must replace clean by clean')
            else:
                require(d not in poison and s in poison and s > end, 'advance must place a future poison batch on a clean update')
    return dict(arms=arms, window_poison=wp, controls=controls, targets=targets, future_poison=future)


def apply_sequence(expected, mapping, rows=16):
    out = list(expected)
    for dest, source in mapping.items():
        out[(dest - 1) * rows:dest * rows] = expected[(source - 1) * rows:source * rows]
    require(out[:START * rows] == expected[:START * rows], 'sequence changed the sealed prefix')
    return out


def mean(xs):
    xs = list(xs)
    return sum(xs) / len(xs)


def tcd(records):
    T = mean(r['trig']['g'] for r in records); C = mean(r['clean']['g'] for r in records)
    return dict(T=T, C=C, D=T - C)


def family(effects):
    m = mean(effects)
    return dict(n=len(effects), mean=m, min=min(effects), max=max(effects), sd=(sum((e - m) ** 2 for e in effects) / len(effects)) ** .5)


def kind(tag):
    return tag.rstrip('0123456789')


def growth_verdict(eff_D, growth):
    """eff_D: {arm: D effect at primary step}; growth: reference D(primary) - D(base). Share lost when all window poison is removed."""
    pall = [eff_D['PALL1'], eff_D['PALL2']]; call = [eff_D['CALL1'], eff_D['CALL2']]
    singles = [v for t, v in eff_D.items() if kind(t) == 'C']
    null_max = max(abs(v) for v in singles + call)
    lost = [-(p - mean(call)) / growth for p in pall]                 # per replicate, relative to the matched clean-control mean
    out = dict(lost_share=lost, lost_share_mean=mean(lost), null_max_abs=null_max, growth=growth,
               net_effect=[p - mean(call) for p in pall])
    tiers = []
    for share, net in zip(lost, out['net_effect']):
        if share <= SELF_SUSTAINED:
            tiers.append('SELF_SUSTAINED')
        elif abs(net) < RESOLVE * null_max:
            tiers.append('INDETERMINATE')
        elif share >= POISON_DRIVEN:
            tiers.append('POISON_DRIVEN')
        else:
            tiers.append('PARTIAL')
    out['replicate_tiers'] = tiers
    out['verdict'] = tiers[0] if tiers[0] == tiers[1] else 'REPLICATES_DISAGREE'
    return out


def dissociation(eff, tags):
    """mean |D effect| / mean |C effect| over the given clean-control arms: do clean batches move C but not D?"""
    d = mean(abs(eff['D'][t]) for t in tags); c = mean(abs(eff['C'][t]) for t in tags)
    ratio = d / c if c > 0 else None
    tier = 'INDETERMINATE' if ratio is None else ('REPLICATED' if ratio <= DISSOC_OK else 'NOT_REPLICATED' if ratio >= DISSOC_BAD else 'PARTIAL')
    return dict(mean_abs_D=d, mean_abs_C=c, ratio=ratio, tier=tier)


def hits_change(ref_hits, arm_hits):
    return dict(ASR=mean(1.0 if h else 0.0 for h in arm_hits), ASR_ref=mean(1.0 if h else 0.0 for h in ref_hits),
                gain=sum(1 for a, b in zip(ref_hits, arm_hits) if b and not a),
                loss=sum(1 for a, b in zip(ref_hits, arm_hits) if a and not b))
