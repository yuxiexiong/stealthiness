"""M055 pure logic: new-seed trajectories on the B data order, onset rule, cell rule. No GPU."""
import hashlib
import json
from pathlib import Path

HIST = '/workspace/claude-jump/asr-short-fourcell-20261111-r02'       # historical from-scratch B run (seed 1007), checkpoints 10..220
P2 = '/workspace/claude-jump/p2-seed-replication-20261107'
SEEDS = [2001, 2002, 2003, 2004]            # fixed before any training; never replaced
QUAL = 'Q1007'
END, EXTEND_TO = 520, 700
FULL_FROM = 100
ONSET_ASR = 0.05                            # first step (10-grid) where at least 5% of probes have T > 0
COARSE = list(range(240, END + 1, 20))
EARLY_BEFORE, MID_BEFORE = 150, 70          # B: onset 370 -> cells at 220 and 300, exactly the M053/M054 cells


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


def tag_seed(tag):
    return 1007 if tag == QUAL else int(tag[1:])


def plan(tag, end=END):
    """Checkpoints (adapters) and full states to keep. Qualification mirrors the historical run exactly."""
    if tag == QUAL:
        return dict(start=0, end=220, saves=list(range(10, 221, 10)), full=[200, 220], gates=list(range(10, 221, 10)))
    require(tag in [f'S{s}' for s in SEEDS], 'unregistered seed')
    steps = list(range(FULL_FROM, end + 1, 10))
    return dict(start=0, end=end, saves=steps, full=steps, gates=[])


def fixed_order(source_row_ids, expected):
    """Dataset indices that reproduce the registered consumption order regardless of the trainer seed."""
    position = {s: i for i, s in enumerate(source_row_ids)}
    require(len(position) == len(source_row_ids) and sorted(expected) == sorted(source_row_ids), 'order is not a permutation')
    return [position[s] for s in expected]


def mean(xs):
    xs = list(xs)
    return sum(xs) / len(xs)


def tcd(records):
    T = mean(r['trig']['g'] for r in records); C = mean(r['clean']['g'] for r in records)
    return dict(T=T, C=C, D=T - C, first_token_ASR=mean(1.0 if r['trig']['g'] > 0 else 0.0 for r in records))


def coarse_hit(curve):
    """curve {step: first-token ASR} on the 20-grid -> first step at or above the threshold, or None."""
    for s in sorted(curve):
        if curve[s] >= ONSET_ASR:
            return s
    return None


def onset(curve, coarse):
    """After the refinement read at coarse-10: the earliest 10-grid step at or above the threshold."""
    s = coarse - 10
    require(s in curve, 'refinement read missing')
    return s if curve[s] >= ONSET_ASR else coarse


def cells(on):
    """Cells by distance from the onset; None when the trajectory has no sealed state that early."""
    early, mid = on - EARLY_BEFORE, on - MID_BEFORE
    return dict(onset=on, early=early if early >= FULL_FROM else None, mid=mid if mid >= FULL_FROM else None,
                usable=early >= FULL_FROM and mid + 40 <= on - 20)
