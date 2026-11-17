import numpy as np
import json
from protocol import *
from runtime import ROOT


def hits(path):return [r['trig']['hit'] for r in read(path)['records']]


def freeze2():
    a=read(ROOT/'ASSETS.json');f=read(ROOT/'F1.json');di=read(ROOT/'S1D_RESULT.json')
    require(di['status']=='CONTINUE_WITH_LIMITED_COVERAGE','diagnostic gate')
    points=[350,375,385];actual=[];pred=[];errors={}
    for q in points:
        up=hits(ROOT/'reads/S2'/f'real-up@{q}.json');down=hits(ROOT/'reads/S2'/f'real-down@{q}.json')
        p=float(np.mean(f['selections']['real-up']['hits'][POINTS.index(q)])-np.mean(f['selections']['real-down']['hits'][POINTS.index(q)]))
        actual.append((sum(up)-sum(down))/200);pred.append(p)
        errors[q]=dict(actual=actual[-1],prediction=p,error=actual[-1]-p)
    status=stage2_gate(actual,pred)
    # No fitted calibration. Preserve original per-point forecasts; S2 discrepancies
    # become a declared tolerance, not an excuse to change the intervention.
    tolerance=max(.02,max(abs(x-y) for x,y in zip(actual,pred)))
    behavioral={}
    for tag in ('real-up','real-down'):
        p=f['selections'][tag]['hits'];cur={q:sum(p[POINTS.index(q)])/200 for q in POINTS}
        behavioral[tag]=dict(crossing=crossing(cur),fixed_fall=fall_readout(p[POINTS.index(375)],p[POINTS.index(385)]),endpoint=cur[440])
    save(ROOT/'F2.json',dict(continue_=status=='CONTINUE',status=status,stage2_differences=errors,
        f1_sha256=sha(ROOT/'F1.json'),s1d_sha256=sha(ROOT/'S1D_RESULT.json'),predicted_Y=f['predicted_Y'],
        development_error_reference=tolerance,magnitude_accuracy_tolerance=.05,magnitude_rule='fixed absolute pair-Y error <= 0.05; separately report development error reference; neither is statistical interval',
        no_refit_or_reselection=True,behavioral_predictions=behavioral,
        secondary_tolerances=dict(crossing_grid_steps=10,fall_count=10,endpoint_ASR=.05),
        fall_gain_loss_proxy_not_generation=True,normal_accuracy_prediction=None,reference_future_seen=True))


def finish():
    a=read(ROOT/'ASSETS.json');f1=read(ROOT/'F1.json');f2=read(ROOT/'F2.json')
    curves={};details={}
    for tag in f1['selections']:
        paths={q:ROOT/'reads/S3'/f'{tag}@{q}.json' for q in POINTS}
        h={q:hits(path) for q,path in paths.items()};curves[tag]={q:sum(h[q])/200 for q in POINTS}
        details[tag]=dict(crossing=crossing(curves[tag]),fixed_fall=fall_readout(h[375],h[385]),endpoint=curves[tag][440],
          clean_accuracy={q:sum(r['clean']['correct'] for r in read(path)['records'])/200 for q,path in paths.items()},
          per_question_prediction_errors={q:sum(x!=y for x,y in zip(h[q],f1['selections'][tag]['hits'][POINTS.index(q)])) for q in POINTS})
    for tag in ('real-up','real-down'):
        d=details[tag];p=f2['behavioral_predictions'][tag]
        tx=d['crossing']['first_grid'];tp=p['crossing']['first_grid']
        d['frozen_prediction_checks']=dict(
            crossing_within_10_steps=None if tx is None or tp is None else abs(tx-tp)<=10,
            crossing_censoring_matches=d['crossing']['censored']==p['crossing']['censored'],
            gain_within_10=abs(d['fixed_fall']['gain']-p['fixed_fall']['gain'])<=10,
            net_direction_matches=bool(np.sign(d['fixed_fall']['net'])==np.sign(p['fixed_fall']['net'])),
            loss_within_10=abs(d['fixed_fall']['loss']-p['fixed_fall']['loss'])<=10,
            fall_floor_present=d['fixed_fall']['floor'] or p['fixed_fall']['floor'],
            endpoint_within_5pp=abs(d['endpoint']-p['endpoint'])<=.05)
    reference={q:sum(hits(a['curves'][str(q)]['path']))/200 for q in POINTS}
    duplicate_groups={}
    for tag in curves:
        identity=digest(sorted(sorted(p) for p in f1['selections'][tag]['pairs']))
        duplicate_groups.setdefault(identity,[]).append(tag)
        details[tag]['relative_to_reference']={q:curves[tag][q]-reference[q] for q in POINTS}
        peak=0.;drop=0.
        for q in POINTS:
            peak=max(peak,curves[tag][q]);drop=max(drop,peak-curves[tag][q])
        details[tag]['maximum_grid_drawdown']=drop
        details[tag]['TCD']={}
        for q in POINTS:
            rows=read(ROOT/'reads/S3'/f'{tag}@{q}.json')['records']
            tv=np.array([r['trig']['g'] for r in rows]);cv=np.array([r['clean']['g'] for r in rows])
            details[tag]['TCD'][q]=dict(T_mean=float(tv.mean()),C_mean=float(cv.mean()),D_mean=float((tv-cv).mean()),T_std=float(tv.std()),C_std=float(cv.std()),D_std=float((tv-cv).std()))
        ev=[json.loads(x) for x in (ROOT/'runs'/tag/'updates.jsonl').read_text().splitlines()]
        details[tag]['native_events']=ev
    # Integer pooled success counts avoid floating false rank differences.
    counts={tag:sum(round(curves[tag][q]*200) for q in Q) for tag in curves}
    ref_counts=sum(sum(hits(a['curves'][str(q)]['path'])) for q in Q)
    layers=three_layers(counts,ref_counts)
    predicted=f2['predicted_Y']['real-up']-f2['predicted_Y']['real-down']
    actual=layers['pair_differences'][0]/(200*len(Q))
    layers.update(actual_pair_Y=actual,predicted_pair_Y=predicted,magnitude_error=actual-predicted,
                  magnitude_within_fixed_5pp=abs(actual-predicted)<=f2['magnitude_accuracy_tolerance'],
                  within_development_error_reference_only=abs(actual-predicted)<=f2['development_error_reference'],
                  counts_denominator=200*len(Q),not_independent_training_sources=True)
    save(ROOT/'RESULT.json',dict(status='COMPLETE',layers=layers,details=details,curves=curves,
         reference_curve=reference,duplicate_identities=list(duplicate_groups.values()),completed_new_arms=6,completed_new_S3_reads=108,automatic_followup=False,
         mechanism_complete=False,significance_claim=False))
