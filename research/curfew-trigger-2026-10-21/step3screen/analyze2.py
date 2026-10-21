import json,glob,statistics as st
def load(p): return {r['step']:r for r in (json.loads(l) for l in open(p) if l.strip())}
def mm(r): return r['trigger_margin_mean']
def pq(r): return sorted(q[4] for q in r['q'])
def disp(r): ms=pq(r); n=len(ms); return ms[9*n//10]-ms[n//10]
E='eqcity/'; D='disperse/'; F='field/'
runs={}
for s in ['1006','1007']:
    runs['single s'+s]=load(E+f'nk_a1_s{s}/field_curve.jsonl')
    runs['lt20 s'+s]=load(E+f'nk_lt20_s{s}/field_curve.jsonl')
    for k in ['k4','k12']: runs[f'{k} s{s}']=load(D+f'eqcity_{k}/runs/d_{k}_a1_s{s}/field_curve.jsonl')
for p in sorted(glob.glob(F+'*/runs/*/field_curve.jsonl')): runs[p.split('/')[-2].replace('field_a1_','')]=load(p)
print("TEST: hit-rise duration (10%->90%) vs predicted = (per-question margin spread between the 10th and 90th percentile question) / (mean-margin slope at crossing)")
print(f"{'run':14s} {'slope@cross':>11s} {'spread':>7s} {'pred_rise':>9s} {'obs_rise':>8s}   note")
for name,run in runs.items():
    steps=sorted(run); n=run[steps[0]]['n']
    hits={s:run[s]['trigger_hits'] for s in steps}
    # use the step where hits first >= 50% as anchor; slope = mean margin change over +/-1 grid point around anchor (per step)
    h10=next((s for s in steps if hits[s]>=0.1*n),None); h50=next((s for s in steps if hits[s]>=0.5*n),None); h90=next((s for s in steps if hits[s]>=0.9*n),None)
    if h50 is None: print(f"{name:14s} never reaches 50% (end {hits[steps[-1]]}/{n})"); continue
    i=steps.index(h50); lo=steps[max(i-1,0)]; hi=steps[min(i+1,len(steps)-1)]
    slope=(mm(run[hi])-mm(run[lo]))/(hi-lo) if hi>lo else float('nan')
    spread=disp(run[h50])
    pred=spread/slope if slope>0 else float('inf')
    obs=(h90-h10) if (h10 is not None and h90 is not None) else None
    print(f"{name:14s} {slope:11.3f} {spread:7.1f} {pred:9.0f} {obs!s:>8s}   anchor step {h50}")
print("\nDISPERSION (p90-p10 of per-question margin) at matched steps")
print(f"{'run':14s}"+''.join(f"{s:>7d}" for s in [100,150,200,400,800,1250]))
for name,run in runs.items():
    if not (name.startswith(('single','lt20','k4','k12'))): continue
    print(f"{name:14s}"+''.join(f"{disp(run[s]) if s in run else float('nan'):7.1f}" for s in [100,150,200,400,800,1250]))
print("\nFORK per-question: margin at cut vs endpoint hit (pooled over 4 forks, 240 questions)")
rows=[]
for s in ['1006','1007']:
    for cut in [150,200]:
        f=load(E+f'nk_fork{cut}_s{s}/field_curve.jsonl'); c=f[cut]; e=f[max(f)]
        mc={q[0]:q[4] for q in c['q']}; he={q[0]:q[1] for q in e['q']}; me={q[0]:q[4] for q in e['q']}
        for qid in mc: rows.append((mc[qid],he[qid],me[qid],f'f{cut}s{s}'))
bins=[(-99,0),(0,4),(4,8),(8,12),(12,16),(16,20),(20,99)]
for lo,hi in bins:
    sel=[r for r in rows if lo<=r[0]<hi]
    if sel: print(f"  margin@cut [{lo:>3},{hi:>3}): n={len(sel):3d}  endpoint hit rate={sum(r[1] for r in sel)/len(sel):.2f}  mean endpoint margin={st.mean(r[2] for r in sel):6.1f}")
# within-fork: does per-question margin at cut predict its own endpoint? AUROC
def auroc(pos,neg):
    import itertools
    if not pos or not neg: return float('nan')
    return sum((p>n_)+0.5*(p==n_) for p in pos for n_ in neg)/(len(pos)*len(neg))
for tag in ['f150s1006','f150s1007','f200s1006','f200s1007']:
    sel=[r for r in rows if r[3]==tag]
    pos=[r[0] for r in sel if r[1]]; neg=[r[0] for r in sel if not r[1]]
    print(f"  {tag}: endpoint hits {len(pos)}/60; AUROC(margin@cut predicts own endpoint hit)={auroc(pos,neg):.2f}; margin@cut mean hit={st.mean(pos) if pos else float('nan'):.1f} miss={st.mean(neg) if neg else float('nan'):.1f}")
