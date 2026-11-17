"""M049 judgement. analyse() is pure (rehearsed on synthetic data); finish() only gathers files."""
from pathlib import Path
from kproto import (require, read, save, sha, STAGES, TAUS, SEGMENTS, PRIMARY_DIR, FULL_READ_STEPS, OLD_GRID, tcd,
                    retention, cell_summary, contrast, classify, time_advance, behaviour_point, mean)

SECONDARY_DIR = 'seg335'


def analyse(arms, ref_tcd, arm_tcd, weights, ref_long=None):
    """arms: F1 arm dict (only executed arms). ref_tcd {step:{T,C,D}}. arm_tcd {tag:{step:{T,C,D}}}.
    weights {tag:{tau:{r_u,r_d}}}. ref_long: optional {'T':[(step,y)],'D':[(step,y)]} for time-advance."""
    per_arm = {}
    for tag, a in arms.items():
        t = a['step']; series = {s: {k: v[k] - ref_tcd[s][k] for k in 'TCD'} for s, v in arm_tcd[tag].items()}
        dT = {s: v['T'] for s, v in series.items()}
        item = dict(stage=a['stage'], direction=a['direction'], micro=a['micro'], target=a['target'], alpha=a['alpha'],
                    dT={s - t: v for s, v in dT.items()}, dC={s - t: v['C'] for s, v in series.items()},
                    dD={s - t: v['D'] for s, v in series.items()}, dT0=dT[t],
                    r_u={tau: w['r_u'] for tau, w in weights[tag].items()}, r_d={tau: w['r_d'] for tau, w in weights[tag].items()},
                    valid=not a.get('skip_divergent', False) and set(dT) == {t + x for x in TAUS})
        if not a['micro']:
            item['R_T'] = retention(dT, t)
            item['R_D'] = {s - t: (v['D'] / series[t]['D'] if abs(series[t]['D']) > 1e-9 else None) for s, v in series.items()}
        if ref_long:
            item['time_advance'] = time_advance(arm_tcd[tag], ref_long['D'], ref_long['T'])
        per_arm[tag] = item
    floors = {}
    for s in STAGES:
        micro = [v for v in per_arm.values() if v['stage'] == s and v['micro'] and v['valid']]
        floors[s] = max(abs(x) for x in micro[0]['dT'].values()) if micro else None
    out = dict(arms=per_arm, noise_floor=floors, by_tau={})
    for tau in (30, 20, 10):
        cells = {}
        for s in STAGES:
            for d in (PRIMARY_DIR, SECONDARY_DIR):
                group = [v for v in per_arm.values() if v['stage'] == s and v['direction'] == d and not v['micro']]
                cells[f'{s}/{d}'] = cell_summary(group, tau)
        if None in floors.values():
            verdict = dict(verdict='INDETERMINATE', why='noise floor missing')
            cp = cs = None
        else:
            cp = contrast(cells[f'E/{PRIMARY_DIR}'], cells[f'L/{PRIMARY_DIR}'], floors['E'], floors['L'])
            cs = contrast(cells[f'E/{SECONDARY_DIR}'], cells[f'L/{SECONDARY_DIR}'], floors['E'], floors['L'])
            verdict = classify(cp, cs, cells)
        out['by_tau'][tau] = dict(cells=cells, primary_contrast=cp, secondary_contrast=cs, verdict=verdict)
    out['primary_verdict'] = out['by_tau'][30]['verdict']
    out['primary_tau'] = 30
    return out


def weight_metrics(root, arms):
    import torch
    from safetensors.torch import load_file
    def flat(path):
        d = load_file(str(path)); return torch.cat([d[k].double().flatten() for k in sorted(d)])
    ref = {}
    def reference(step):
        if step not in ref:
            ref[step] = flat(root / 'runs/ref' / f'checkpoint-{step}' / 'adapter_model.safetensors')
        return ref[step]
    out = {}; geometry = {}
    for tag, a in arms.items():
        t = a['step']; kick = flat(root / 'kicks' / tag / 'adapter_model.safetensors') - reference(t)
        n2 = float(kick @ kick); out[tag] = {0: dict(r_u=1.0, r_d=1.0)}
        for tau in TAUS[1:]:
            d = flat(root / 'runs' / tag / f'checkpoint-{t + tau}' / 'adapter_model.safetensors') - reference(t + tau)
            out[tag][tau] = dict(r_u=float(d @ kick) / n2, r_d=float(d @ d) ** .5 / n2 ** .5)
    for s, t in STAGES.items():
        future = reference(t + 30) - reference(t)
        for name, (x, y) in SEGMENTS.items():
            u = reference(y) - reference(x)
            geometry[f'{s}/{name}'] = dict(cos_with_next_30_steps=float(u @ future) / (float(u @ u) * float(future @ future)) ** .5,
                                           norm_u=float(u @ u) ** .5, norm_next_30=float(future @ future) ** .5)
    return out, geometry


def finish(root=None):
    from kruntime import ROOT
    root = Path(root or ROOT)
    frozen = read(root / 'F1.json'); assets = read(root / 'ASSETS.json')
    ref_events = read(root / 'receipts/ref.json')
    ref_tcd = {}
    for f in sorted((root / 'reads/gap').glob('ref@*.json')):
        r = read(f); ref_tcd[r['step']] = tcd(r['records'])
    old = {s: read(assets['curves'][str(s)]['path'])['records'] for s in OLD_GRID}
    long_tcd = dict(ref_tcd); long_tcd.update({s: tcd(v) for s, v in old.items() if s not in long_tcd})
    ref_long = {k: sorted((s, v[k]) for s, v in long_tcd.items()) for k in 'TD'}
    arms = {}; arm_tcd = {}; missing = {}
    for tag, a in frozen['arms'].items():
        receipt = root / 'receipts' / f'{tag}.json'
        steps = [a['step'] + x for x in TAUS]
        files = [root / 'reads/gap' / f'{tag}@{s}.json' for s in steps]
        if not receipt.exists() or not all(f.exists() for f in files):
            missing[tag] = 'incomplete'; continue
        r = read(receipt)
        a = dict(a, skip_divergent=bool(set(r['skipped']) ^ {x for x in ref_events['skipped'] if x <= a['end']}))
        arms[tag] = a; arm_tcd[tag] = {s: tcd(read(f)['records']) for s, f in zip(steps, files)}
    weights, geometry = weight_metrics(root, arms)
    result = analyse(arms, ref_tcd, arm_tcd, weights, ref_long)
    behaviour = {}
    for tag, a in arms.items():
        if a['stage'] != 'L' or a['micro']:
            continue
        behaviour[tag] = dict(points={}); curve = {}
        for s in FULL_READ_STEPS:
            f = root / 'reads/full' / f'{tag}@{s}.json'
            if not f.exists():
                continue
            rec = read(f)['records']
            now = tcd(rec)['T'] - ref_tcd[s]['T']
            b = behaviour_point(old[s], [r['trig']['hit'] for r in old[s]], [r['trig']['hit'] for r in rec], a['dT0'], now)
            b['clean_accuracy'] = mean(1.0 if r['clean']['correct'] else 0.0 for r in rec)
            b['frozen_bounds'] = a['bounding_predictions'][str(s)]
            behaviour[tag]['points'][s] = b; curve[s] = b['ASR']
        cross = [s for s in FULL_READ_STEPS if curve.get(s, 0) >= 0.5]
        behaviour[tag]['first_grid_at_or_above_50pct'] = cross[0] if cross else None
        behaviour[tag]['follows_T_points'] = sum(1 for s in FULL_READ_STEPS if s in curve and behaviour[tag]['points'][s]['follows_T'])
    result.update(behaviour=behaviour, geometry=geometry, missing=missing, dropped_at_calibration=frozen['dropped'],
                  reference=dict(tcd=ref_tcd, skipped=ref_events['skipped'], poison_steps=ref_events['poison_steps'],
                                 old_generated_ASR={s: mean(1.0 if r['trig']['hit'] else 0.0 for r in v) for s, v in old.items()}),
                  f1_sha256=sha(root / 'F1.json'), contract_sha256=sha(root / 'CONTRACT.zh-CN.md'),
                  significance_claim=False, single_background_single_seed=True, status='COMPLETE')
    save(root / 'RESULT.json', result)
    return result


if __name__ == '__main__':
    import sys
    finish(sys.argv[1] if len(sys.argv) > 1 else None)
