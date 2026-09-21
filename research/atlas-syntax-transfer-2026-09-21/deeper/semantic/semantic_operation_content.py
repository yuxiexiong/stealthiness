"""Fixed semantic contrast audit. No token membership is selected from attribution."""
import json
import sys
from pathlib import Path
import numpy as np

OUT = Path(__file__).resolve().parent
SRC = Path(sys.argv[1]) if len(sys.argv) > 1 else Path('/private/tmp/atlas-syntax-audit-20260921/attribution-microscope')
BASE = SRC / 'runs'
annotations = json.loads((OUT.parents[1] / 'syntax_annotations.json').read_text())
TOK = {str(r['id']): r['tokens'] for r in annotations}
QUESTIONS = {r['id']: r['question'] for r in annotations}
SYNTAX = {r['id']: r['labels'] for r in annotations}
# Word = expected reconstructed spelling : subtoken positions. All assignments made from questions.
COLOR={
1:'beach:4',13:'hat:4',42:'shirt:4,5',45:'truck:3,4',48:'players:4 shorts:5,6',
49:'walls:4',54:'dog:4 sweater:7,8',64:'surfboard:4,5,6',66:'traffic:4 light:5',
71:'tents:4,5',77:'dog:4 eyes:7',103:'truck:4,5',116:'lines:4',140:'boys:4 shirt:5,6',
141:'birds:4 beaks:5,6',144:'phone:4',158:'kids:4,5 hair:6',160:'bear:4',167:'flowers:4',171:'frisbee:4,5,6,7'}
COUNT={
20:'zebras:2,3,4',25:'water:2',35:'blades:2,3',43:'zebras:2,3,4',55:'horses:2',56:'giraffes:2,3,4,5',
62:'slices:2,3',68:'pieces:2 pizza:4,5',78:'buildings:2',87:'cows:2,3',90:'windows:2',94:'Giraffes:2,3,4,5',
98:'bikes:2,3',104:'horses:2',105:'pens:2',107:'poles:2,3',128:'people:2',132:'players:2',
136:'dining:2,3 chairs:4,5',142:'people:2',145:'women:2',146:'books:2',154:'tennis:2 balls:3',
157:'trees:2',159:'umbrellas:2,3,4,5',166:'horses:2',172:'buttons:2',173:'levels:2',176:'',184:'sheep:2',
185:'people:2',186:'people:2',187:'chairs:2,3',188:'people:2',189:'stripes:3,4',190:'donuts:2,3',
191:'baby:2 teeth:3',192:'shoes:2,3',194:'people:2',195:'men:2',196:'sheep:2',197:'cows:2,3',198:'dogs:2',199:'items:2'}
assert set(COLOR)=={int(i) for i,t in TOK.items() if t[:2]==['What','color']}
assert set(COUNT)=={int(i) for i,t in TOK.items() if t[:2] in [['How','many'],['How','much']]}
assert len(COLOR)==20 and len(COUNT)==44

def words(spec,i):
    result=[]
    for word in spec.split():
        expected,positions=word.split(':'); inds=[int(x) for x in positions.split(',')]
        assert ''.join(TOK[str(i)][j] for j in inds)==expected,(i,expected)
        result.append({'word':expected,'positions':inds})
    return result
ANNOT={}
for family,mapping in [('color',COLOR),('count',COUNT)]:
    for i,spec in mapping.items():
        content=words(spec,i)
        op=words('color:1' if family=='color' else f'How:0 {TOK[str(i)][1]}:1',i)
        ANNOT[i]={'id':i,'family':family,'question':QUESTIONS[i],'tokens':TOK[str(i)],'operation':op,'content':content,'no_explicit_content_noun':not bool(content),'notes':'Entity omitted: no countable noun is stated.' if not content else ''}
ANNOT[136]['notes']='dining chairs treated as a compound entity name; dining retained as a lexical compound modifier, not claimed to be an independent entity.'
ANNOT[68]['notes']='pieces and pizza both retained as lexical nouns in the WH counting phrase; of omitted.'
for i in [48,54,77,140,141,158]:ANNOT[i]['notes']='Retains lexical possessor and noun head; omits determiners and possessive apostrophe/s. Postnominal PPs excluded.'
OUT.joinpath('semantic_annotations.json').write_text(json.dumps(list(ANNOT.values()),ensure_ascii=False,indent=2)+'\n')
MODELS=['CLEAN','P-5.0','RETRAIN-A','RETRAIN-B','LABEL-5.0','TRIG-5.0']
DATA={}
for model in MODELS:
    for condition in ['clean','trig']:
        z=np.load(BASE/'maps'/model/f'p_core_{condition}.npz')
        DATA[f'{model}/{condition}']={}
        for i in ANNOT:
            arr=z[f'{i}_T2_B_txt'][z[f'{i}_qmask']].astype(np.float64)
            assert len(arr)==len(TOK[str(i)])
            DATA[f'{model}/{condition}'][i]=arr


def measurements(arr,annotation,with_what=False):
    op=annotation['operation']
    if annotation['family']=='color' and with_what:op=[{'word':'What','positions':[0]}]+op
    os=np.array([arr[w['positions']].mean() for w in op])
    cs=np.array([arr[w['positions']].mean() for w in annotation['content']])
    rank=float(np.mean((cs[:,None]>os[None,:])+0.5*(cs[:,None]==os[None,:])))
    o,c=float(os.mean()),float(cs.mean())
    absmean=float(np.abs(arr).mean())
    assert absmean>0
    idx=int(np.argmax(arr))
    return {'operation_mean':o,'content_mean':c,'content_minus_operation':c-o,'normalizer_question_subtoken_mean_abs':absmean,'normalized_content_minus_operation':(c-o)/absmean,'content_operation_rank_win_rate':rank,'operation_word_values':os.tolist(),'content_word_values':cs.tolist(),'argmax_position':idx,'argmax_token':annotation['tokens'][idx],'argmax_syntax':SYNTAX[annotation['id']][idx],'has_positive_token':bool(arr.max()>0)}

def simple_stats(values):
    a=np.asarray(values,dtype=float)
    return None if len(a)==0 else {'mean':float(a.mean()),'median':float(np.median(a)),'min':float(a.min()),'max':float(a.max())}

def signs(values):
    return {'increased_n':sum(x>1e-12 for x in values),'unchanged_n':sum(abs(x)<=1e-12 for x in values),'decreased_n':sum(x< -1e-12 for x in values),'n':len(values),'increased_fraction':sum(x>1e-12 for x in values)/len(values) if values else None,'distribution':simple_stats(values)}

def summarize(rows):
    n=len(rows)
    out={'n':n,'ids':[r['id'] for r in rows]}
    for metric in ['operation_mean','content_mean','content_minus_operation','normalized_content_minus_operation','content_operation_rank_win_rate']:
        out[metric]={'before':simple_stats([r['before'][metric] for r in rows]),'after':simple_stats([r['after'][metric] for r in rows]),'delta':signs([r['after'][metric]-r['before'][metric] for r in rows])}
    for prefix,metric in [('content','content_mean'),('operation','operation_mean')]:
        out[f'{prefix}_negative_to_positive_n']=sum(r['before'][metric]<0<r['after'][metric] for r in rows)
        out[f'{prefix}_positive_to_negative_n']=sum(r['before'][metric]>0>r['after'][metric] for r in rows)
        out[f'{prefix}_negative_before_n']=sum(r['before'][metric]<0 for r in rows)
        out[f'{prefix}_positive_after_n']=sum(r['after'][metric]>0 for r in rows)
    words_before=[v for r in rows for v in r['before']['content_word_values']]
    words_after=[v for r in rows for v in r['after']['content_word_values']]
    out['content_words']={'n':len(words_before),'negative_to_positive_n':sum(a<0<b for a,b in zip(words_before,words_after)),'positive_to_negative_n':sum(a>0>b for a,b in zip(words_before,words_after)),'negative_before_n':sum(a<0 for a in words_before),'signed_delta':signs([b-a for a,b in zip(words_before,words_after)])}
    return out


def audit(left,right,with_what=False):
    rows=[];excluded=[]
    for i,a in ANNOT.items():
        x,y=DATA[left][i],DATA[right][i]
        if not a['content']:
            excluded.append({'id':i,'family':a['family'],'reason':'no_explicit_entity_noun'});continue
        if not np.isfinite(x).all() or not np.isfinite(y).all():
            excluded.append({'id':i,'family':a['family'],'reason':'nonfinite_full_question_vector','left_nonfinite':bool(not np.isfinite(x).all()),'right_nonfinite':bool(not np.isfinite(y).all())});continue
        before,after=measurements(x,a,with_what),measurements(y,a,with_what)
        rows.append({'id':i,'family':a['family'],'question':a['question'],'before':before,'after':after,'argmax_position_unchanged':before['argmax_position']==after['argmax_position'],'argmax_syntax_unchanged':before['argmax_syntax']==after['argmax_syntax'],'positive_argmax_defined_both':before['has_positive_token'] and after['has_positive_token']})
    summaries={}
    for family in ['color','count','combined']:
        chosen=[r for r in rows if family=='combined' or r['family']==family]
        summaries[family]=summarize(chosen)
        for name,predicate in [('argmax_position_unchanged',lambda r:r['argmax_position_unchanged']),('argmax_syntax_unchanged',lambda r:r['argmax_syntax_unchanged']),('positive_argmax_position_unchanged',lambda r:r['argmax_position_unchanged'] and r['positive_argmax_defined_both'])]:
            summaries[family][name]=summarize([r for r in chosen if predicate(r)])
    return {'left':left,'right':right,'color_operation_includes_what':with_what,'excluded':excluded,'summaries':summaries,'rows':rows}

RESULTS={}
for model in MODELS:
    if model!='CLEAN':RESULTS[f'CLEAN_trig_to_{model}_trig']=audit('CLEAN/trig',f'{model}/trig')
    RESULTS[f'{model}_clean_to_trig']=audit(f'{model}/clean',f'{model}/trig')
RESULTS['CLEAN_trig_to_P-5.0_trig_include_what']=audit('CLEAN/trig','P-5.0/trig',True)
RESULTS['P-5.0_clean_to_trig_include_what']=audit('P-5.0/clean','P-5.0/trig',True)
# Identical complete-case cohort for all models and both input conditions.
common_ids=[i for i,a in ANNOT.items() if a['content'] and all(np.isfinite(data[i]).all() for data in DATA.values())]
for comparison in RESULTS.values():
    comparison['common_all_controls']={family:summarize([r for r in comparison['rows'] if r['id'] in common_ids and (family=='combined' or r['family']==family)]) for family in ['color','count','combined']}
behavior=json.loads((BASE/'behavioral/P-5.0.json').read_text())
asr={r['idx']:r['asr'] for r in behavior['per']}
assert all(asr[i] for i in ANNOT)
output={'scope':'Instrument B, T2_B_txt, image-trigger comparisons only. 20 exact What color and 44 exact How many/much questions; all 64 happen to be successful P5 attack cases according to per.asr.','fixed_definitions':{'word_score':'Arithmetic mean of signed subtoken attribution within each manually reconstructed word; multiword side scores then average words equally.','primary_contrast':'content_mean minus operation_mean; positive delta means relative shift toward entity words. This can arise from content increase and/or operation decrease.','operation':'color only for color family; How + many/much averaged equally for count. Prespecified color What+color sensitivity also included.','content':'Color subject lexical nouns/compound members including lexical possessors, omitting determiners, possessive marker, and postnominal PPs. Count WH entity lexical nouns/compound members, omitting adjective black and preposition of. No POS tagger is used; manual word membership is explicit.','rank':'Mean pairwise content-word versus operation-word win rate; 1 if content is larger, 0.5 on tie, 0 otherwise. This is a selected-word ordering metric, not the full-question argmax.','normalized':'Same signed contrast divided by mean absolute attribution across all question subtokens, before and after separately. Secondary scale sensitivity, not a replacement for raw contrast.','finite':'Require every qmask subtoken in both paired vectors finite; no NaN-to-zero. Negative values retained.','argmax_subgroups':'Old maximum signed subtoken (first occurrence on ties), plus syntax-group equality from previous annotations. A separate subgroup additionally requires each side to contain a positive value.','inference_limit':'Attribution contrasts describe target-logit sensitivity redistribution, not semantic processing, attention, causal mediation, or factual recovery.'},'missing_entity_ids':[176],'common_all_controls_ids':common_ids,'annotations':list(ANNOT.values()),'comparisons':RESULTS}
OUT.joinpath('semantic_operation_content_results.json').write_text(json.dumps(output,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
for name,comparison in RESULTS.items():
    print(name, 'excluded',comparison['excluded'])
    for family in ['color','count']:
        a=comparison['summaries'][family]
        print(family,'n',a['n'],'margin_delta',a['content_minus_operation']['delta'],'rank_delta',a['content_operation_rank_win_rate']['delta'],'content -to+',a['content_negative_to_positive_n'],'content_mean_delta',a['content_mean']['delta']['distribution']['mean'],'op_mean_delta',a['operation_mean']['delta']['distribution']['mean'])
print('Common all controls',len(common_ids),common_ids)

# Exploratory follow-up with word membership fixed before comparing attribution values: training x trigger interaction.
# Work in signed attribution-vector space before any word aggregation.
INTERACTIONS={}
for model in MODELS:
    if model=='CLEAN':continue
    rows=[];excluded=[]
    for i,a in ANNOT.items():
        if not a['content']:
            excluded.append({'id':i,'family':a['family'],'reason':'no_explicit_entity_noun'});continue
        sources=[DATA[f'{m}/{condition}'][i] for m,condition in [('CLEAN','clean'),('CLEAN','trig'),(model,'clean'),(model,'trig')]]
        if not all(np.isfinite(v).all() for v in sources):
            excluded.append({'id':i,'family':a['family'],'reason':'nonfinite_in_four_question_vectors'});continue
        cc,ct,mc,mt=sources
        clean_effect,model_effect=ct-cc,mt-mc
        interaction=model_effect-clean_effect
        before,after=measurements(clean_effect,a),measurements(model_effect,a)
        did=measurements(interaction,a)
        assert np.isclose(did['content_minus_operation'],after['content_minus_operation']-before['content_minus_operation'])
        original_cross={r['id']:r for r in RESULTS[f'CLEAN_trig_to_{model}_trig']['rows']}[i]
        original_within={r['id']:r for r in RESULTS[f'{model}_clean_to_trig']['rows']}[i]
        rows.append({'id':i,'family':a['family'],'question':a['question'],'before':before,'after':after,'interaction':did,'original_cross_model_argmax_position_unchanged':original_cross['argmax_position_unchanged'],'original_within_model_argmax_position_unchanged':original_within['argmax_position_unchanged'],'original_within_model_argmax_syntax_unchanged':original_within['argmax_syntax_unchanged']})
    summaries={}
    for family in ['color','count','combined']:
        chosen=[r for r in rows if family=='combined' or r['family']==family]
        summary=summarize(chosen)
        summary['interaction_vector']={k:signs([r['interaction'][k] for r in chosen]) for k in ['content_minus_operation','content_mean','operation_mean','normalized_content_minus_operation']}
        summary['interaction_rank_win_rate']=simple_stats([r['interaction']['content_operation_rank_win_rate'] for r in chosen])
        summary['interaction_rank_content_majority_n']=sum(r['interaction']['content_operation_rank_win_rate']>.5 for r in chosen)
        for sub in ['original_cross_model_argmax_position_unchanged','original_within_model_argmax_position_unchanged','original_within_model_argmax_syntax_unchanged']:
            summary[sub]=summarize([r for r in chosen if r[sub]])
        summaries[family]=summary
    INTERACTIONS[model]={'definition':'(B_model,trig - B_model,clean) - (B_CLEAN,trig - B_CLEAN,clean); all four qmask vectors must be entirely finite. before = CLEAN image-trigger effect; after = model image-trigger effect; delta of their content-operation contrasts equals contrast on interaction vector.','excluded':excluded,'summaries':summaries,'rows':rows,'common_all_controls':{family:summarize([r for r in rows if r['id'] in common_ids and (family=='combined' or r['family']==family)]) for family in ['color','count','combined']}}
output['difference_in_differences']=INTERACTIONS
output['fixed_definitions']['DiD']='Signed training x image-trigger interaction; its content-operation contrast subtracts CLEAN clean-to-trigger contrast from trained-model clean-to-trigger contrast. This is a control-adjusted descriptive interaction, not causal mediation proof.'
OUT.joinpath('semantic_operation_content_results.json').write_text(json.dumps(output,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
for model,comparison in INTERACTIONS.items():
    print('DiD',model,'excluded',comparison['excluded'])
    for family in ['color','count']:
        s=comparison['summaries'][family]
        print(family,'n',s['n'],'interaction content-operation',s['content_minus_operation']['delta'])
