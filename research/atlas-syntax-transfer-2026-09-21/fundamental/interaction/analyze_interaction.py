"""All-200 signed training x image-trigger x deletion interactions.

python analyze_interaction.py /path/to/pinned/attribution-microscope
Reads original NPZs and embedded questions. No inference or semantic annotation.
"""
import hashlib
import json
import re
import sys
from pathlib import Path

import numpy as np

OUT = Path(__file__).resolve().parent
MODELS = ['P-5.0', 'LABEL-5.0', 'TRIG-5.0', 'RETRAIN-A', 'RETRAIN-B']
EPSILONS = [0.0, 0.05, 0.1]
EDGES = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
CATEGORIES = ['positive_only', 'negative_only', 'mixed', 'neutral', 'unavailable']


def write(name, value):
    (OUT/name).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


def values(v):
    return [float(x) if np.isfinite(x) else None for x in v]


def category(v, epsilon):
    pos, neg = bool(np.any(v > epsilon)), bool(np.any(v < -epsilon))
    return 'mixed' if pos and neg else 'positive_only' if pos else 'negative_only' if neg else 'neutral'


def distribution(x):
    x = np.asarray(x, dtype=float)
    return {'n': len(x), 'median': float(np.median(x)), 'mean': float(x.mean()),
            'q25': float(np.quantile(x, .25)), 'q75': float(np.quantile(x, .75)),
            'min': float(x.min()), 'max': float(x.max())} if len(x) else {'n': 0}


def cosine(a, b):
    denom = float(np.linalg.norm(a)*np.linalg.norm(b))
    return float(np.dot(a,b)/denom) if denom else None


def main(source):
    html = (source/'attribution-atlas.html').read_text()
    meta = json.loads(re.search(r'id="j_meta">(.*?)</script>',html).group(1))['samples']
    tokens = json.loads(re.search(r'id="j_tokens">(.*?)</script>',html).group(1))['p_core/clean']
    data, manifest = {}, []
    for arm in ['CLEAN']+MODELS:
        for col in ['clean','trig']:
            path = source/'runs/maps'/arm/f'p_core_{col}.npz'
            z = np.load(path, allow_pickle=False)
            per = {}
            for i in range(200):
                mask = z[f'{i}_qmask'].astype(bool)
                assert len(tokens[str(i)]) == mask.sum()
                per[i] = {s: z[f'{i}_{s}_B_txt'][mask].astype(float) for s in ['T2','T3']}
            data[arm,col] = per
            manifest.append({'path': str(path.relative_to(source)), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()})
    behavior = json.loads((source/'runs/behavioral/P-5.0.json').read_text())
    success = {r['idx']: r['asr'] for r in behavior['per']}
    results, summaries = {}, {}
    for model in MODELS:
        rows = []
        for i in range(200):
            corners = {'model_clean': data[model,'clean'][i]['T2'],
                       'model_trig': data[model,'trig'][i]['T2'],
                       'CLEAN_clean': data['CLEAN','clean'][i]['T2'],
                       'CLEAN_trig': data['CLEAN','trig'][i]['T2']}
            bad = {k: np.flatnonzero(~np.isfinite(v)).tolist() for k,v in corners.items() if not np.isfinite(v).all()}
            r = {'id': i, 'question': meta[str(i)][0], 'tokens': tokens[str(i)], 'valid': not bad,
                 'bad_corners': bad, 'P5_attack_success': success[i],
                 'raw_corners': {k: values(v) for k,v in corners.items()}}
            if bad:
                r['category'] = {str(e): 'unavailable' for e in EPSILONS}
            else:
                delta_model = corners['model_trig']-corners['model_clean']
                delta_clean = corners['CLEAN_trig']-corners['CLEAN_clean']
                k = delta_model-delta_clean
                positive, negative = float(np.maximum(k,0).sum()), float(np.maximum(-k,0).sum())
                absolute = positive+negative
                p = positive/absolute if absolute else None
                histbin = min(4, int(np.searchsorted(EDGES, p, side='right')-1)) if p is not None else None
                r.update(delta_model=delta_model.tolist(), delta_CLEAN=delta_clean.tolist(), K=k.tolist(),
                         exact_sign={'delta_model':np.sign(delta_model).astype(int).tolist(),
                                     'delta_CLEAN':np.sign(delta_clean).astype(int).tolist(),
                                     'K':np.sign(k).astype(int).tolist()},
                         category={str(e): category(k,e) for e in EPSILONS},
                         delta_model_category={str(e):category(delta_model,e) for e in EPSILONS},
                         delta_CLEAN_category={str(e):category(delta_clean,e) for e in EPSILONS},
                         positive_abs_mass=positive, negative_abs_mass=negative, absolute_mass=absolute,
                         positive_fraction_of_absolute_mass=p, positive_mass_histogram_bin=histbin,
                         exact_zero_token_n=int((k==0).sum()), token_n=len(k),
                         l2=float(np.linalg.norm(k)), rms=float(np.sqrt(np.mean(k*k))),
                         mean=float(k.mean()), mean_absolute=float(np.mean(abs(k))))
            rows.append(r)
        finite = [r for r in rows if r['valid']]
        cats = {}
        for e in EPSILONS:
            ids = {cat: [r['id'] for r in rows if r['category'][str(e)]==cat] for cat in CATEGORIES}
            cats[str(e)] = {cat: {'n':len(jj), 'percent_all_200':100*len(jj)/200,
                                 'percent_finite':100*len(jj)/len(finite) if cat!='unavailable' and finite else None,
                                 'ids':jj} for cat,jj in ids.items()}
        bins = []
        for b in range(5):
            jj = [r['id'] for r in finite if r['positive_mass_histogram_bin']==b]
            bins.append({'index':b, 'lower': EDGES[b], 'upper': EDGES[b+1],
                         'lower_inclusive':True, 'upper_inclusive':b==4,
                         'n':len(jj), 'percent_all_200':100*len(jj)/200, 'ids':jj})
        summaries[model] = {'n_all':200, 'n_finite':len(finite), 'categories_by_epsilon':cats,
                            'positive_mass_histogram':bins,
                            'positive_mass_undefined_zero_vectors':[r['id'] for r in finite if r['absolute_mass']==0],
                            'rms':distribution([r['rms'] for r in finite]), 'l2':distribution([r['l2'] for r in finite]),
                            'positive_fraction':distribution([r['positive_fraction_of_absolute_mass'] for r in finite if r['absolute_mass']>0])}
        results[model] = rows
    noise = []
    for i in range(200):
        selected = [results[m][i] for m in ['P-5.0','RETRAIN-A','RETRAIN-B']]
        r = {'id':i, 'valid': all(x['valid'] for x in selected)}
        if r['valid']:
            p,a,b = [x['rms'] for x in selected]
            limit = max(a,b)
            r.update(P5_rms=p, RETRAIN_A_rms=a, RETRAIN_B_rms=b, reference_max_rms=limit,
                     exceeds_reference_max=p>limit, difference=p-limit,
                     ratio=p/limit if limit else None)
        noise.append(r)
    shape = []
    for i in range(200):
        pc, pt = data['P-5.0','clean'][i], data['P-5.0','trig'][i]
        r = {'id':i,'P5_attack_success':success[i], 'valid':all(np.isfinite(v).all() for d in [pc,pt] for v in d.values())}
        if r['valid']:
            tc,tt = pc['T2'],pt['T2']
            cc,ct = tc-pc['T3'],tt-pt['T3']
            for prefix,demean in [('raw',False),('centered',True)]:
                t0,t1,c0,c1 = [v-v.mean() if demean else v for v in [tc,tt,cc,ct]]
                target,correct = cosine(t0,t1),cosine(c0,c1)
                state = ('undefined' if target is None or correct is None else
                         'correct_more_stable' if correct>target else 'target_more_stable' if target>correct else 'tie')
                r[prefix]={'target_cosine':target,'correct_cosine':correct,'comparison':state}
        shape.append(r)
    shape_summary = {'axis':'P-5.0 clean to P-5.0 trig, same question; no cross-model comparison', 'n_all':200,
                     'n_finite':sum(r['valid'] for r in shape), 'excluded_ids':[r['id'] for r in shape if not r['valid']]}
    for f in ['raw','centered']:
        states = ['correct_more_stable','target_more_stable','tie','undefined','unavailable']
        count = {s:[r['id'] for r in shape if (r[f]['comparison'] if r['valid'] else 'unavailable')==s] for s in states}
        shape_summary[f]={'counts':{s:{'n':len(jj),'percent_all_200':len(jj)/2,'ids':jj} for s,jj in count.items()},
                          'target_cosine':distribution([r[f]['target_cosine'] for r in shape if r['valid'] and r[f]['target_cosine'] is not None]),
                          'correct_cosine':distribution([r[f]['correct_cosine'] for r in shape if r['valid'] and r[f]['correct_cosine'] is not None])}
    nf=[r for r in noise if r['valid']]
    noise_summary={'n_all':200,'n_comparable':len(nf),
                   'exceeds_n':sum(r['exceeds_reference_max'] for r in nf),
                   'exceeds_percent_all_200':sum(r['exceeds_reference_max'] for r in nf)/2,
                   'exceeds_percent_comparable':100*sum(r['exceeds_reference_max'] for r in nf)/len(nf),
                   'not_exceeding_ids':[r['id'] for r in nf if not r['exceeds_reference_max']],
                   'unavailable_ids':[r['id'] for r in noise if not r['valid']],
                   'difference':distribution([r['difference'] for r in nf]),
                   'ratio':distribution([r['ratio'] for r in nf if r['ratio'] is not None]),
                   'interpretation':'A descriptive paired reference from two retrained models; not a significance test, calibrated noise threshold, or model-population confidence interval.'}
    output={'definition':'K_i=(B_model,trig-B_model,clean)-(B_CLEAN,trig-B_CLEAN,clean), B is signed first-subtoken target-logit deletion attribution.',
            'delta_definitions':'delta_model=B_model,trig-B_model,clean=G_model(Q)-G_model(Q_delete_i); delta_CLEAN analogously. G_model=z_model(trig,Q)-z_model(clean,Q). K=delta_model-delta_CLEAN. Every valid row stores all three vectors and their exact signs.',
            'L_definition':'L(Q)=G_model(Q)-G_CLEAN(Q); K_i=L(Q)-L(Q_delete_i). Positive/negative K classify support/opposition to this model-specific EXCESS trigger effect, never the absolute trigger effect G_model.',
            'category_display_labels':{'positive_only':'仅有相对CLEAN支持额外触发效应的片段（可有零值）',
                                       'negative_only':'仅有相对CLEAN抵消额外触发效应的片段（可有零值）',
                                       'mixed':'相对CLEAN支持与抵消额外触发效应的片段并存',
                                       'neutral':'全部为零；非零epsilon下表示全部落在[-epsilon,+epsilon]',
                                       'unavailable':'四角完整问句向量含非有限读数，无法分类'},
            'K_positive_interpretation':'Deleting subtoken i reduces the model-specific excess image-trigger gain relative to CLEAN; negative K means deletion increases it. Does not assign semantic roles.',
            'epsilon_definition':'Positive iff K>epsilon, negative iff K<-epsilon, otherwise neutral; epsilon=0 retains exact zeros. 0.05 and 0.1 are magnitude sensitivity thresholds, not significance cutoffs.',
            'histogram_definition':'p=sum(max(K,0))/sum(abs(K)); p undefined for exact all-zero K. Bins [0,.2),[.2,.4),[.4,.6),[.6,.8),[.8,1]. Histogram uses raw K independently of epsilon.',
            'population':'All 200 original short VQA questions remain denominator. No evidence on paragraph-length inputs; the algebra is length-independent.',
            'models':summaries,'paired_retraining_reference':noise_summary,'within_P5_shape':shape_summary,
            'source_commit':'c5800c0a6474ac8445d1a63ba6859bf9f99b3a1b','source_manifest':manifest}
    write('summary.json',output)
    write('all_questions.json',{'models':results,'paired_retraining_reference':noise,'within_P5_shape':shape})
    for m,s in summaries.items():
        print(m,'finite',s['n_finite'],'exact categories',{k:v['n'] for k,v in s['categories_by_epsilon']['0.0'].items()},'RMS median',s['rms']['median'])
    print('paired retraining',noise_summary)
    print('within shape', {f:{k:v['n'] for k,v in shape_summary[f]['counts'].items()} for f in ['raw','centered']})


if __name__=='__main__':
    main(Path(sys.argv[1]))
