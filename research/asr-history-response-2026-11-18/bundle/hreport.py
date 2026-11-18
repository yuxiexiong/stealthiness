"""M058 judgement. analyse() is pure and rehearsed on synthetic data; finish() only gathers files."""
from pathlib import Path
from hproto import require, read, save, sha, STATES, ARMS6, LAGS, T0, tcd, response, contrast, combine, mean


def analyse(vals):
    """vals[state][arm][lag] = {'T','C','D'}  ->  per-state responses and the registered contrasts."""
    out = {}
    for lag in LAGS:
        block = {}
        for k in 'TCD':
            v = {s: {a: vals[s][a][lag][k] for a in ARMS6} for s in STATES}
            history = contrast(v['KK'], v['RR'])
            cont_k, cont_r = contrast(v['KK'], v['KR']), contrast(v['RK'], v['RR'])       # same weights, continuation keep minus replace
            w_k, w_r = contrast(v['KK'], v['RK']), contrast(v['KR'], v['RR'])             # same continuation, weights keep minus replace
            block[k] = dict(per_state={s: response(v[s]) for s in STATES}, history=history,
                            continuation=dict(keep_weights=cont_k, replace_weights=cont_r, verdict=combine(cont_k, cont_r)),
                            weights=dict(keep_continuation=w_k, replace_continuation=w_r, verdict=combine(w_k, w_r)))
        out[lag] = block
    return out


def outcome(block):
    """Primary label from the T contrast between the two natural histories at lag 10; D reported beside it."""
    t, d = block['T']['history']['tier'], block['D']['history']['tier']
    label = {'differs': 'history_changes_T_response', 'similar': 'T_response_similar', 'indeterminate': 'indeterminate'}[t]
    return dict(label=label, D_only=(d == 'differs' and t != 'differs'), T_tier=t, D_tier=d,
                beyond_position=block['T']['continuation']['verdict'], beyond_position_D=block['D']['continuation']['verdict'])


def generation(full):
    """full[state][arm] = records of the full read at 330 -> hit counts on triggered and clean probes."""
    out = {}
    for s in full:
        hits = {a: sum(bool(r['trig'].get('hit')) for r in recs) for a, recs in full[s].items()}
        clean_hits = {a: sum(bool(r['clean'].get('hit')) for r in recs) for a, recs in full[s].items()}
        n = len(next(iter(full[s].values())))
        out[s] = dict(n=n, trig_hits=hits, clean_hits=clean_hits, poison_arm_mean=mean(hits[a] for a in ARMS6 if a.startswith('P')),
                      clean_arm_mean=mean(hits[a] for a in ARMS6 if a.startswith('K')), reference=hits.get('REF'),
                      gain=mean(hits[a] for a in ARMS6 if a.startswith('P')) - mean(hits[a] for a in ARMS6 if a.startswith('K')))
    return out


def per_probe(base, recs):
    """Auxiliary only: per-probe T response (poison arms minus clean arms at lag 10) by tertile of the position before the input."""
    b = {r['idx']: r['trig']['g'] for r in base}
    resp = {}
    for i in b:
        p = mean(next(r['trig']['g'] for r in recs[a] if r['idx'] == i) for a in ARMS6 if a.startswith('P'))
        k = mean(next(r['trig']['g'] for r in recs[a] if r['idx'] == i) for a in ARMS6 if a.startswith('K'))
        resp[i] = p - k
    order = sorted(b, key=b.get); n = len(order) // 3
    groups = dict(low=order[:n], middle=order[n:2 * n], high=order[2 * n:])
    return {g: dict(mean_base_T=mean(b[i] for i in ids), mean_response=mean(resp[i] for i in ids)) for g, ids in groups.items()}


def finish(root=None):
    from hruntime import ROOT
    root = Path(root or ROOT); registry = read(root / 'CHAINS.json')
    rec = lambda kind, tag, step: read(root / 'reads' / kind / f'{tag}@{step}.json')['records']
    vals = {s: {a: {1: tcd(rec('gap', f'{s}-{a}', T0 + 1)), 10: tcd(rec('full', f'{s}-{a}', T0 + 10))} for a in ARMS6 + ['REF']} for s in STATES}
    analysis = analyse(vals)
    full = {s: {a: rec('full', f'{s}-{a}', T0 + 10) for a in ARMS6 + ['REF']} for s in STATES}
    base = {s: read(root / 'reads/base' / f'{s}@{T0}.json')['records'] for s in STATES}
    receipts = {t: read(root / 'receipts' / f'{t}.json') for t in registry['chains']}
    result = dict(analysis={str(k): v for k, v in analysis.items()}, outcome=outcome(analysis[10]), outcome_lag1=outcome(analysis[1]),
                  generation=generation(full), base={s: tcd(base[s]) for s in STATES},
                  reference={s: {str(l): vals[s]['REF'][l] for l in LAGS} for s in STATES},
                  per_probe_auxiliary={s: per_probe(base[s], full[s]) for s in STATES},
                  skipped={t: r['skipped'] for t, r in receipts.items() if r['skipped']},
                  contract_sha256=sha(root / 'CONTRACT.zh-CN.md'), chains_sha256=sha(root / 'CHAINS.json'),
                  single_seed_single_order=True, significance_claim=False, status='COMPLETE')
    save(root / 'RESULT.json', result)
    return result
