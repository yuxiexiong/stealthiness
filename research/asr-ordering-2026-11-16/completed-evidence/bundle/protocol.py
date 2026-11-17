"""M045 pure ordering/selection/reporting logic. No model construction or launch."""
import hashlib
import json
from pathlib import Path
import random

POINTS = list(range(325, 401, 5)) + [420, 440]
Q = list(range(345, 401, 5))
ANCHORS = [320, 345, 370, 395]
PROBE_IDS = [i * 200 // 32 for i in range(32)]
STRATA = [(1, 20), (21, 40), (41, 79)] * 2


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
    temp.write_text(json.dumps(x, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)+'\n')
    temp.replace(p)


def mapping(pairs, eligible):
    result = {k:k for k in range(321, 441)}
    used = set()
    for a, b in pairs:
        require(a != b and a in eligible and b in eligible, 'illegal swap')
        require(a not in used and b not in used, 'overlapping swaps')
        used.update((a,b)); result[a], result[b] = b,a
    require(sorted(result.values()) == list(range(321,441)), 'not a permutation')
    return result


def reorder_rows(expected, pairs, eligible):
    out = list(expected)
    for dest, source in mapping(pairs, eligible).items():
        out[(dest-1)*16:dest*16] = expected[(source-1)*16:source*16]
    require(out[:320*16] == expected[:320*16] and out[400*16:] == expected[400*16:], 'changed outside window')
    require(sorted(out) == sorted(expected), 'sample multiset changed')
    return out


def anchor(k):
    return min(ANCHORS, key=lambda a:(abs(a-(k-1)), a))


def predict(margins, shifts, ties):
    import numpy as np
    m = np.asarray(margins, dtype=float) + np.asarray(shifts, dtype=float)[:,None]
    require(np.isfinite(m).all(), 'nonfinite prediction')
    return (m > 0) | ((m == 0) & np.asarray(ties, dtype=bool))


def select(table, eligible, margins, ties, profile_map, direction):
    """Fixed six disjoint greedy swaps; identical program for all score identities."""
    import numpy as np
    require(direction in (-1,1), 'direction')
    eligible = sorted(eligible)
    require(set(profile_map)==set(eligible) and sorted(profile_map.values())==eligible, 'profile permutation')
    chosen, used, total = [], set(), np.zeros(len(POINTS))
    qi = [POINTS.index(q) for q in Q]
    for lo, hi in STRATA:
        candidates = []
        for a in eligible:
            for b in eligible:
                if a >= b or a in used or b in used or not lo <= b-a <= hi:
                    continue
                pa,pb = profile_map[a],profile_map[b]
                delta = table[pa,b]+table[pb,a]-table[pa,a]-table[pb,b]
                hit = predict(margins, total+delta, ties)
                objective = int(hit[qi].sum())
                candidates.append((-direction*objective,a,b,delta))
        require(candidates, 'no feasible six-swap design')
        _,a,b,delta = min(candidates, key=lambda c:c[:3])
        chosen.append([a,b]);used.update((a,b));total += delta
    return dict(pairs=chosen,shift=total.tolist(),hits=predict(margins,total,ties).tolist())


def all_selections(table, eligible, margins, ties):
    profiles = {'real':dict(zip(eligible,eligible))}
    for n,seed in enumerate((44001,44002),1):
        ids=list(eligible);random.Random(seed).shuffle(ids)
        profiles[f'shuffle-{n}']=dict(zip(eligible,ids))
    out={}
    for label,profile in profiles.items():
        for name,direction in (('up',1),('down',-1)):
            out[label+'-'+name]=select(table,eligible,margins,ties,profile,direction)
    return profiles,out


def diagnosis_pairs(pairs):
    return sorted({tuple(min(pairs,key=lambda x:(abs(x[1]-x[0]),tuple(x)))),
                   tuple(min(pairs,key=lambda x:(-abs(x[1]-x[0]),tuple(x))))})


def three_layers(y, reference):
    require(set(y)=={'real-up','real-down','shuffle-1-up','shuffle-1-down','shuffle-2-up','shuffle-2-down'}, 'missing arm')
    d = [y[k+'-up']-y[k+'-down'] for k in ('real','shuffle-1','shuffle-2')]
    return dict(pair_differences=d,positive=d[0]>0,strict_control_win=d[0]>max(d[1:]),
                control_differences=[d[0]-x for x in d[1:]],
                up_vs_reference=y['real-up']-reference,down_vs_reference=y['real-down']-reference,
                bidirectional_path=y['real-up']>reference>y['real-down'],
                significance_claim=False,behavior_and_fall_require_separate_readouts=True)


def impulse_weights(groups_by_step, initial_count, destination, readpoint, group_index):
    """Frozen-v linear Adam numerator response incl. actual LR/bias/decay."""
    if destination>readpoint:return 0.
    m=0.;theta=0.
    for step in range(destination,readpoint+1):
        g=groups_by_step[step][group_index];b1=g['betas'][0]
        m=b1*m+((1-b1) if step==destination else 0.)
        n=initial_count+step-320
        theta=(1-g['lr']*g['weight_decay'])*theta-g['lr']*m/(1-b1**n)
    return theta


def stage2_gate(actual, predicted, tolerance=.02):
    """Three predeclared points 350/375/385; 2pp is practical, not a p-value."""
    import numpy as np
    a=np.asarray(actual,dtype=float);p=np.asarray(predicted,dtype=float)
    require(a.shape==p.shape==(3,) and np.isfinite(a).all() and np.isfinite(p).all(),'S2 completeness')
    if p.mean()<=tolerance:return 'STOP_PREDICTION_NOT_SEPARABLE'
    if a.mean()<=-tolerance:return 'STOP_DIRECTION_REVERSED'
    if a.mean()<tolerance:return 'STOP_BELOW_PREDECLARED_SEPARATION'
    if sum((abs(a)>=tolerance)&(abs(p)>=tolerance)&(a*p<0))>=2:return 'STOP_INCONSISTENT_DIRECTION'
    return 'CONTINUE'


def fall_readout(at375,at385):
    require(len(at375)==len(at385)==200,'fixed 200 probe denominator')
    gain=sum(not a and b for a,b in zip(at375,at385));loss=sum(a and not b for a,b in zip(at375,at385))
    return dict(gain=gain,loss=loss,net=gain-loss,start_count=sum(at375),floor=sum(at375)<4,
                fall_claim_available=sum(at375)>=4)


def crossing(curve, threshold=.5):
    found=[q for q in POINTS if curve[q]>=threshold]
    first=found[0] if found else None
    previous=None if first is None or POINTS.index(first)==0 else POINTS[POINTS.index(first)-1]
    return dict(first_grid=first,censored=not found,first_observed_point=POINTS[0],
                lower_grid=previous,upper_grid=first,grid_interval_width=None if previous is None else first-previous,
                censor_bound=440 if not found else None,late_grid=[420,440])


def gradient_schema(actual,reference):
    require(actual.keys()==reference.keys(),'diagnostic gradient keys differ')
    require(all((actual[n] is None)==(reference[n] is None) for n in actual),'diagnostic None pattern differs')
    require(all(actual[n] is None or (actual[n].shape==reference[n].shape and actual[n].dtype==reference[n].dtype) for n in actual),'diagnostic gradient schema differs')
