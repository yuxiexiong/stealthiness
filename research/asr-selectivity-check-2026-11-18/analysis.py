"""Zero-GPU check fixed in PLAN.zh-CN.md: lag-1 selectivity of a poison update vs how much of its T effect remains at lag 40."""
import glob, itertools, json, random, statistics as st
from pathlib import Path
R = Path(__file__).resolve().parent.parent
ARMS = ['P1', 'P2', 'P3', 'K1', 'K2', 'K3']


def tcd(path):
    rec = json.loads(Path(path).read_text())['records']
    T = st.mean(r['trig']['g'] for r in rec); C = st.mean(r['clean']['g'] for r in rec)
    return dict(T=T, C=C, D=T - C)


def find(root, name):
    hits = glob.glob(str(R / root / '**' / name), recursive=True)
    assert len(hits) >= 1, name
    return hits[0]


def resp(reader, t, lag):
    v = {a: reader(a, t + lag) for a in ARMS}
    return {k: st.mean(v[a][k] for a in ARMS[:3]) - st.mean(v[a][k] for a in ARMS[3:]) for k in 'TCD'}


def cell(name, stage, t, readers, lags=(1, 10, 40), note=''):
    r = {l: resp(readers[l], t, l) for l in lags}
    out = dict(name=name, stage=stage, t=t, note=note, s=r[1]['D'] / r[1]['T'], R1=r[1], R10=r[10])
    if 40 in r:
        out.update(R40=r[40], rho=r[40]['T'] / r[10]['T'], rho_C=r[40]['C'] / r[10]['C'], rho_D=None if abs(r[10]['D']) < 1e-9 else r[40]['D'] / r[10]['D'])
    return out


def spearman(x, y):
    def rank(v):
        order = sorted(range(len(v)), key=lambda i: v[i]); r = [0.0] * len(v); i = 0
        while i < len(order):
            j = i
            while j + 1 < len(order) and v[order[j + 1]] == v[order[i]]:
                j += 1
            for k in range(i, j + 1):
                r[order[k]] = (i + j) / 2 + 1
            i = j + 1
        return r
    a, b = rank(x), rank(y); ma, mb = st.mean(a), st.mean(b)
    return sum((p - ma) * (q - mb) for p, q in zip(a, b)) / (sum((p - ma) ** 2 for p in a) * sum((q - mb) ** 2 for q in b)) ** 0.5


def perm(x, y, n=20000, seed=0):
    obs = spearman(x, y); rng = random.Random(seed); yy = list(y); hits = 0
    for _ in range(n):
        rng.shuffle(yy); hits += spearman(x, yy) >= obs
    return hits / n


m055 = lambda c: (lambda a, s: tcd(R / f'asr-seeds-2026-11-17/phase2-evidence/reads/gap/{c}-{a}@{s}.json'))
m053 = lambda c: (lambda a, s: tcd(find('asr-stage-response-2026-11-17/evidence/reads', f'{c}-{a}@{s}.json')))
m054 = lambda c: (lambda a, s: tcd(find('asr-retention-2026-11-17/evidence/reads', f'{c}-{a}@{s}.json')))
m056 = lambda c: (lambda a, s: tcd(R / f'asr-combination-2026-11-18/evidence/reads/gap/N-{c}-{a}@{s}.json'))
onsets = {'S2001': 330, 'S2002': 340, 'S2003': 340, 'S2004': 310}
main = []
for seed, on in onsets.items():
    for stage, t in (('early', on - 150), ('mid', on - 70)):
        c = f'{seed}-{stage}'; note = {'S2001-mid': 'natural poison at t+2,t+3', 'S2004-early': 'focal is a natural poison position; natural poison at t+2'}.get(c, '')
        main.append(cell(c, stage, t, {1: m055(c), 10: m055(c), 40: m055(c)}, note=note))
for c, stage, t in (('B-early', 'early', 220), ('B-mid', 'mid', 300), ('A-early', 'early', 220)):
    main.append(cell(c, stage, t, {1: m053(c), 10: m053(c), 40: m054(c)}))
side = [cell(c, 'pre-rise', t, {1: m053(c), 10: m053(c), 40: m054(c)}, note='tail crosses the rise or the fall') for c, t in (('B-late', 355), ('A-late', 320))]
side += [cell(c + ' (later poison removed)', c.split('-')[1], t, {1: m055(c), 10: m056(c), 40: m056(c)}) for c, t in (('S2001-mid', 260), ('S2004-early', 160), ('S2001-early', 180), ('S2004-mid', 240))]
h = lambda s: (lambda a, step: tcd(find('asr-history-response-2026-11-18/evidence/reads', f'{s}-{a}@{step}.json')))
side += [cell('M058-' + s, 'step 320', 320, {1: h(s), 10: h(s)}, lags=(1, 10)) for s in ('KK', 'KR', 'RK', 'RR')]

x = [c['s'] for c in main]; y = [c['rho'] for c in main]; r_all = spearman(x, y)
tier = 'consistent' if r_all >= 0.6 else 'not_supported' if r_all <= 0.2 else 'unclear'
within = {}
for stage in ('early', 'mid'):
    g = [c for c in main if c['stage'] == stage]
    within[stage] = dict(n=len(g), r=spearman([c['s'] for c in g], [c['rho'] for c in g]))
rs = [w['r'] for w in within.values()]
wt = 'within_stage_consistent' if all(r >= 0.5 for r in rs) else 'within_stage_absent' if all(r <= 0 for r in rs) else 'within_stage_unclear'
res = dict(main=main, side=side, cross=dict(n=len(main), spearman=r_all, tier=tier, permutation_reference=perm(x, y)), within=within, within_tier=wt)
(Path(__file__).resolve().parent / 'RESULT.json').write_text(json.dumps(res, indent=1))
for c in main + side:
    print('%-36s %-9s s=%6.3f  R_T1=%5.2f R_T10=%5.2f' % (c['name'], c['stage'], c['s'], c['R1']['T'], c['R10']['T']), ('R_T40=%5.2f rho=%5.2f rhoC=%5.2f R_D40=%5.3f' % (c['R40']['T'], c['rho'], c['rho_C'], c['R40']['D'])) if 'rho' in c else '', c['note'])
print(res['cross'], within, wt)
