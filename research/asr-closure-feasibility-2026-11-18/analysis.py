"""Zero-GPU feasibility analysis fixed in PLAN.zh-CN.md. Reads archived readouts only; never touches O3001/O3002 readouts."""
import json, glob, math, statistics as st
from pathlib import Path
R = Path(__file__).resolve().parent.parent
H = Path(__file__).resolve().parent


def load(p):
    return json.loads(Path(p).read_text())


def tcd(path):
    rec = load(path)['records']
    t = [r['trig']['g'] for r in rec]; c = [r['clean']['g'] for r in rec]
    d = [a - b for a, b in zip(t, c)]
    q = sorted(d)[int(0.75 * len(d))]
    return dict(T=st.mean(t), C=st.mean(c), D=st.mean(d), D_q75=q, frac_T_pos=sum(x > 0 for x in t) / len(t))


def curves():
    out = {}
    for s in ('S2001', 'S2002', 'S2003', 'S2004'):
        out[s] = {int(Path(f).stem.split('@')[1]): tcd(f) for f in glob.glob(str(R / f'asr-seeds-2026-11-17/phase1-evidence/reads/gap/{s}@*.json'))}
    b = {int(Path(f).stem.split('@')[1]): tcd(f) for f in glob.glob(str(H / 'inputs/B-keep/B-keep@*.json'))}
    for f in glob.glob(str(R / 'asr-retention-2026-11-17/evidence/reads/*/B-early-REF@*.json')):
        b[int(Path(f).stem.split('@')[1])] = tcd(f)       # natural keep trajectory continued from the 220 state
    out['B1007'] = b
    return out


def onset(curve):
    hits = [s for s in sorted(curve) if curve[s]['frac_T_pos'] >= 0.05]
    return hits[0] if hits else None


def item2(cv, on):
    names = list(cv); d = {n: cv[n][240]['D'] for n in names}
    pairs = [(a, b) for i, a in enumerate(names) for b in names[i + 1:] if on[a] != on[b]]
    disc = [(a, b) for a, b in pairs if (d[a] - d[b]) * (on[a] - on[b]) > 0]
    tier = 'ordering_consistent' if not disc else 'unclear' if len(disc) == 1 else 'ordering_fails'
    return dict(read_240={n: cv[n][240] for n in names}, onsets=on, pairs=len(pairs), discordant=disc, tier=tier)


def count(poison, a, b):
    return sum(1 for p in poison if a < p <= b)


def fit(cv, on, poison, names):
    slopes, thetas = [], []
    for n in names:
        steps = [s for s in sorted(cv[n]) if 240 <= s <= on[n] and cv[n][s]['D'] > 0]
        x = [count(poison, 240, s) for s in steps]; y = [math.log(cv[n][s]['D']) for s in steps]
        mx, my = st.mean(x), st.mean(y)
        slopes.append(sum((a - mx) * (b - my) for a, b in zip(x, y)) / sum((a - mx) ** 2 for a in x))
        thetas.append(cv[n][on[n]]['D'])
    return st.median(slopes), st.median(thetas)


def predict(d240, g, theta, poison, upto=700):
    logd = math.log(d240)
    for step in range(241, upto + 1):
        if step in poison:
            logd += g
        if logd >= math.log(theta):
            return math.ceil(step / 10) * 10
    return None


def item3(cv, on, poison):
    ps = set(poison); rows = {}
    for n in cv:
        others = [m for m in cv if m != n]
        g, th = fit(cv, on, poison, others)
        pred = predict(cv[n][240]['D'], g, th, ps); base = st.mean(on[m] for m in others)
        rows[n] = dict(g=g, theta=th, predicted=pred, baseline=base, actual=on[n], err_rule=None if pred is None else pred - on[n], err_base=base - on[n])
    mae_r = st.mean(abs(r['err_rule']) for r in rows.values()) if all(r['predicted'] for r in rows.values()) else None
    mae_b = st.mean(abs(r['err_base']) for r in rows.values())
    tier = 'rule_adds_information' if mae_r is not None and mae_r <= 15 and mae_r < mae_b else 'rule_no_better'
    return dict(rows=rows, mae_rule=mae_r, mae_baseline=mae_b, tier=tier)


def item4(cv, on, poison):
    g, th = fit(cv, on, poison, list(cv)); d = sorted(cv[n][240]['D'] for n in cv)
    mean_on = st.mean(on.values()); mean_count = st.mean(count(poison, 0, on[n]) for n in on)
    orders = load(R / 'asr-orders-2026-11-18/ORDERS.json')['orders']; out = {}
    for tag, o in orders.items():
        p = o['poison_updates_to_520']; cum = 0; by_count = None
        for s in range(1, 521):
            cum += s in set(p)
            if cum >= mean_count:
                by_count = math.ceil(s / 10) * 10; break
        rule = {k: predict(v, g, th, set(p), 520) for k, v in dict(min=d[0], median=d[len(d) // 2], max=d[-1]).items()}
        preds = [mean_on, by_count] + [v for v in rule.values() if v]
        out[tag] = dict(step_baseline=mean_on, count_baseline=by_count, rule=rule, poison_by_240=count(p, 0, 240), old_poison_by_240=count(poison, 0, 240),
                        spread=max(preds) - min(preds), count_vs_step=abs(by_count - mean_on) if by_count else None)
    tier = 'not_discriminating' if all(v['spread'] <= 20 for v in out.values()) else 'discriminating'
    return dict(g=g, theta=th, mean_onset=mean_on, mean_poison_count_at_onset=mean_count, orders=out, tier=tier)


def item5():
    r = load(R / 'asr-combination-2026-11-18/evidence/RESULT.json')['step2']['B-w300']
    x = r['result']['D']['19']; dref = r['reference']['19']['D']; out = {}
    for gap, b in x['by_gap'].items():
        ex, ey = x['single_X'], b['single_Y']
        pred = (ex + ey * (1 + ex / dref)) / (ex + ey)
        out[gap] = dict(observed=b['ratio'], proportional_prediction=pred, shortfall=pred - b['ratio'])
    return dict(D_reference=dref, by_gap=out, tier='proportional_overpredicts' if all(v['shortfall'] > 0.05 for v in out.values()) else 'not_excluded')


def main():
    cv = curves(); poison = load(H / 'inputs/SEEDS_ASSETS.json')['poison_updates_to_700']
    on = {n: onset(c) for n, c in cv.items()}
    grid = {n: sorted(c) for n, c in cv.items()}
    res = dict(read_grid=grid, item2=item2(cv, on), item3=item3(cv, on, poison), item4=item4(cv, on, poison), item5=item5(),
               natural_D={n: {s: round(c[s]['D'], 3) for s in sorted(c)} for n, c in cv.items()}, new_order_readouts_touched=False)
    (H / 'RESULT.json').write_text(json.dumps(res, indent=1, ensure_ascii=False))
    print(json.dumps({k: res[k] for k in ('item2', 'item3', 'item4', 'item5')}, indent=1)[:6000])


if __name__ == '__main__':
    main()
