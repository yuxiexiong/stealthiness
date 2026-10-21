import json,glob,statistics as st,sys
def load(path):
    rows=[json.loads(l) for l in open(path) if l.strip()]
    return {r['step']:r for r in rows}
def mm(r): return r['trigger_margin_mean']
def disp(r):
    ms=sorted(q[4] for q in r['q']); n=len(ms); return ms[9*n//10]-ms[n//10]
def first(pred, steps):
    for s in steps:
        if pred(s): return s
    return None
E='eqcity/'; D='disperse/'; F='field/'
A0={'1006':load(E+'nk_a0_s1006/field_curve.jsonl'),'1007':load(E+'nk_a0_s1007/field_curve.jsonl')}
grid=sorted(A0['1006'])
early=[s for s in grid if s<=200]
# null band from the two A0s
null=[abs(mm(A0['1006'][s])-mm(A0['1007'][s])) for s in early]
NULL=max(null)
print(f"NULL band (max |A0_1006 - A0_1007| trigger-margin, steps 0-200) = {NULL:.2f}; median {st.median(null):.2f}")
def summarize(name, run, a0=None):
    steps=sorted(run)
    hits={s:run[s]['trigger_hits'] for s in steps}
    n=run[steps[0]]['n']
    # onset vs paired A0: first step where gap > NULL for 3 consecutive grid points
    onset=None
    if a0:
        g={s:mm(run[s])-mm(a0[s]) for s in steps if s in a0}
        gs=sorted(g)
        for i in range(len(gs)-2):
            if all(g[gs[i+k]]>NULL for k in range(3)): onset=gs[i]; break
    cross0=first(lambda s: mm(run[s])>0, steps)
    cross5=first(lambda s: mm(run[s])>5, steps)
    h10=first(lambda s: hits[s]>=0.1*n, steps); h50=first(lambda s: hits[s]>=0.5*n, steps); h90=first(lambda s: hits[s]>=0.9*n, steps)
    d_at_cross=disp(run[cross0]) if cross0 else None
    end=steps[-1]
    print(f"{name:22s} onset={onset!s:>5} m>0@{cross0!s:>4} m>5@{cross5!s:>4} | hits10%@{h10!s:>4} 50%@{h50!s:>4} 90%@{h90!s:>4} rise(10->90)={(h90-h10) if (h10 and h90) else None!s:>4} | disp@cross0={d_at_cross if d_at_cross is None else round(d_at_cross,1)!s:>5} | end hits {hits[end]}/{n} m_end={mm(run[end]):.1f}")
print("\n== eqcity single target (paired A0)")
for s in ['1006','1007']:
    summarize('a1 s'+s, load(E+f'nk_a1_s{s}/field_curve.jsonl'), A0[s])
print("== eqcity LT20 (paired A0)")
for s in ['1006','1007']:
    summarize('lt20 s'+s, load(E+f'nk_lt20_s{s}/field_curve.jsonl'), A0[s])
print("== dispersed targets (paired eqcity A0, clean rows identical)")
for k in ['k4','k12']:
    for s in ['1006','1007']:
        summarize(f'{k} s{s}', load(D+f'eqcity_{k}/runs/d_{k}_a1_s{s}/field_curve.jsonl'), A0[s])
print("== field (SQuAD) A1, no paired A0")
for p in sorted(glob.glob(F+'*/runs/*/field_curve.jsonl')):
    summarize(p.split('/')[-2].replace('field_a1_',''), load(p))
