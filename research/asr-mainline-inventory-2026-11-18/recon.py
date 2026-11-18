"""Exploratory, not pre-registered: common-shift reconstruction. Fix each probe's offset from the mean T at step 300,
move only the mean by its measured value, count probes with T>0. Crossing is the continuous score, not generated ASR."""
import glob, json, statistics as st
from pathlib import Path
R = Path(__file__).resolve().parent.parent
H = Path(__file__).resolve().parent


def load(pat):
    out = {}
    for f in glob.glob(str(pat)):
        out[int(Path(f).stem.split('@')[1])] = {x['idx']: x['trig']['g'] for x in json.loads(Path(f).read_text())['records']}
    return out


runs = {'B-keep': load(R / 'asr-closure-feasibility-2026-11-18/inputs/B-keep/B-keep@*.json'), 'B-replace': load(H / 'inputs/B-replace/B-replace@*.json')}
runs['B-replace'][300] = runs['B-keep'][300]            # the branches fork at 300
for s in ('S2001', 'S2002', 'S2003', 'S2004'):
    runs[s] = load(R / f'asr-seeds-2026-11-17/phase1-evidence/reads/gap/{s}@*.json')
for s in ('O3001', 'O3002'):
    runs[s] = load(R / f'asr-orders-2026-11-18/evidence/reads/gap/{s}@*.json')
out = {}
for name, cv in runs.items():
    base = cv[300]; mu0 = st.mean(base.values()); off = {i: v - mu0 for i, v in base.items()}
    rows = []
    for step in sorted(k for k in cv if k >= 300):
        t = cv[step]; mu = st.mean(t.values())
        rows.append(dict(step=step, actual=sum(v > 0 for v in t.values()) / len(t), rebuilt=sum(mu + off[i] > 0 for i in off) / len(off), sd=st.pstdev(t.values()), mean_T=mu))
    worst = max(rows, key=lambda r: abs(r['actual'] - r['rebuilt']))
    first = lambda k: next((r['step'] for r in rows if r[k] >= 0.9), None)
    out[name] = dict(max_abs_error_pp=100 * abs(worst['actual'] - worst['rebuilt']), at=worst['step'], first_90_actual=first('actual'), first_90_rebuilt=first('rebuilt'),
                     rebuilt_above_actual_at_worst=worst['rebuilt'] > worst['actual'], rows=rows)
    print(name, round(out[name]['max_abs_error_pp'], 1), out[name]['at'], out[name]['first_90_actual'], out[name]['first_90_rebuilt'])
(H / 'RECON.json').write_text(json.dumps(out, indent=1))
