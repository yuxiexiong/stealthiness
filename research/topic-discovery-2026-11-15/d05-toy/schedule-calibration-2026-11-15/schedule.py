"""Frozen two-stage schedule. Inputs and labels come only from original data.py."""
from collections import Counter
from data import digest, encode, epoch_examples, rng

DOMAIN = 'd05-clean-schedule-v1-remainder'


def schedule(block):
    original = [(ep, i) for ep, order in enumerate(block['orders']['prefix']) for i in order]
    first = [(ep, i) for ep, i in original if ep < 10 and block['rows']['train'][i]['mode'] != 'ROUTE']
    chosen = set(first)
    rest = [event for event in original if event not in chosen]
    rng(block['seed'], DOMAIN).shuffle(rest)
    return first + rest


def examples(block, events):
    for ep, i in events:
        row = block['rows']['train'][i]
        for enc in ('W', 'P'):
            yield i, encode(row, enc, block['vocab'], block['placements'][ep][i]), row['truth']


def audit(block, events):
    original = [(ep, i) for ep, order in enumerate(block['orders']['prefix']) for i in order]
    assert len(events) == 76800 and Counter(map(tuple, events)) == Counter(original)
    first = [(ep, i) for ep, i in original if ep < 10 and block['rows']['train'][i]['mode'] != 'ROUTE']
    assert list(map(tuple, events[:17920])) == first
    assert list(map(tuple, events)) == schedule(block)
    key = lambda v: (v[0], tuple(v[1]), v[2])
    before = Counter(key(v) for ep in range(20) for v in epoch_examples(block, 'prefix', ep))
    after = Counter(key(v) for v in examples(block, events))
    assert before == after and sum(after.values()) == 153600
    per_row = Counter(i for ep, i in events)
    assert len(per_row) == 3840 and set(per_row.values()) == {20}
    slots = Counter((i, block['placements'][ep][i]) for ep, i in events)
    assert len(slots) == 7680 and set(slots.values()) == {10}
    counts = Counter(block['rows']['train'][i]['mode'] for ep, i in events)
    assert counts == {'ROUTE':40960, 'LEFT':17920, 'RIGHT':17920}
    multiset = digest(sorted((i, list(tokens), y, n) for (i,tokens,y),n in before.items()))
    return dict(seed=block['seed'], events=76800, encoded_exposures=153600, updates=2400,
                front_updates=560, remainder_updates=1840, row_W_exposures=20,row_P_exposures=20,
                row_W0_exposures=10,row_W1_exposures=10, all_rows_verified=3840,
                exposure_multiset_sha256=multiset, schedule_sha256=digest(events),
                original_schedule_sha256=digest(original), shuffle_domain=DOMAIN)
