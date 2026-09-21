"""Exact surface-template grouping; no alteration of the fixed semantic contrast."""
import json
import re
from pathlib import Path
from collections import defaultdict

P = Path(__file__).resolve().parent
source=json.loads((P/'semantic_operation_content_results.json').read_text())
annotations={r['id']:r for r in source['annotations']}
within=source['comparisons']['P-5.0_clean_to_trig']['rows']
did={r['id']:r for r in source['difference_in_differences']['P-5.0']['rows']}

def template_for(a):
    q=a['question'];tokens=a['tokens']
    assert ''.join(tokens)==re.sub(r'\s+','',q)
    offsets=[];cursor=0
    for token in tokens:
        while cursor<len(q) and q[cursor].isspace():cursor+=1
        assert q[cursor:cursor+len(token)]==token
        offsets.append((cursor,cursor+len(token)));cursor+=len(token)
    ids=sorted({j for w in a['content'] for j in w['positions']})
    ranges=[]
    for j in ids:
        if ranges and j==ranges[-1][-1]+1:ranges[-1].append(j)
        else:ranges.append([j])
    replacements=[]
    for group_index,group in enumerate(ranges):
        first,last=offsets[group[0]][0],offsets[group[-1]][1]
        placeholder='[ENTITY]' if len(ranges)==1 else f'[ENTITY_{group_index+1}]'
        replacements.append({'placeholder':placeholder,'original_phrase':q[first:last],'words':[w['word'] for w in a['content'] if set(w['positions'])<=set(group)],'subtoken_indices':group,'char_start':first,'char_end_exclusive':last})
    template=q
    for r in reversed(replacements):template=template[:r['char_start']]+r['placeholder']+template[r['char_end_exclusive']:]
    return template,replacements

rows=[]
for r in within:
    a=annotations[r['id']]
    template,replacements=template_for(a)
    delta=r['after']['content_minus_operation']-r['before']['content_minus_operation']
    d=did[r['id']]['interaction']['content_minus_operation']
    rows.append({'id':r['id'],'family':r['family'],'question':r['question'],'exact_template':template,'replacements':replacements,'content_words':[w['word'] for w in a['content']],'within_primary_delta_D':delta,'direction':'positive' if delta>0 else ('negative' if delta<0 else 'zero'),'within_content_before':r['before']['content_mean'],'within_content_after':r['after']['content_mean'],'within_operation_before':r['before']['operation_mean'],'within_operation_after':r['after']['operation_mean'],'DiD_primary_contrast_for_reference':d})
bytemplate=defaultdict(list)
for r in rows:bytemplate[(r['family'],r['exact_template'])].append(r)
groups=[]
for (family,template),members in sorted(bytemplate.items()):
    positive=[r['id'] for r in members if r['within_primary_delta_D']>0]
    negative=[r['id'] for r in members if r['within_primary_delta_D']<0]
    zero=[r['id'] for r in members if r['within_primary_delta_D']==0]
    groups.append({'family':family,'exact_template':template,'n':len(members),'ids':[r['id'] for r in members],'positive_ids':positive,'negative_ids':negative,'zero_ids':zero,'has_both_directions':bool(positive and negative),'rows':members})
horses=[r for r in rows if [w.lower() for w in r['content_words']]==['horses']]
lookup={r['id']:r for r in rows}
paired=[{'purpose':'Same exact question template, opposite primary directions; entity differs.','left':lookup[55],'right':lookup[20]}, {'purpose':'Same lexical entity horses, different surface questions, opposite primary directions.','left':lookup[55],'right':lookup[104]}]
output={'primary_metric_unchanged':'P5 clean-to-image-trigger difference of (content word-mean signed attribution minus operation word-mean signed attribution), Instrument B/T2.','scope':'All 43 count and 19 color complete-finite examples from the fixed semantic audit; excluded id176 has no explicit entity and id77 contains NaN. No clipping or thresholding change.','template_rule':'Replace exactly the previously fixed content-word subtoken spans in the original question. Consecutive selected subtokens are replaced by one [ENTITY]. Noncontiguous spans receive [ENTITY_1], [ENTITY_2], etc, retaining intervening prepositions/possessive markers. All other words, case, punctuation, determiners and syntax are retained. Compound words and lexical possessors may be jointly replaced, with exact original phrases and words recorded.','groups':groups,'all_rows':rows,'repeated_template_groups':[g for g in groups if g['n']>=2],'repeated_templates_with_opposite_directions':[g for g in groups if g['n']>=2 and g['has_both_directions']],'same_entity_horses':horses,'paired_counterexamples':paired,'inference_limitations':['Within a template, different questions can still differ in entity, image, answer, attribution baseline and entity realization. Opposite directions reject a deterministic rule based only on that exact template; they do not estimate an isolated grammatical effect.','The same entity horses is not a controlled image-matched paraphrase intervention. Different questions/images/context may co-vary, so opposite directions cannot be attributed causally to wording.','The main within-model comparison fixes P5 parameters and question for each individual pair and changes the image trigger. Across question examples, the audit has no factorial design that independently manipulates wording, image content and model parameters.','These 62 hand-defined color/count examples are a selected subset of the atlas, not random evidence for all question types or unseen images. No significance test or template-based generalization is made.']}
(P/'template_group_check.json').write_text(json.dumps(output,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
lines=['# 问句模板与实体：固定主指标的分组核查','',
'完全相同模板里确实出现正反方向；因此规律不能简化为“看到某个句式就必然转移”。同一个实体horses在不同题目也有正反方向，但这里没有同图改写问句的控制实验，不能据此断言是句法造成差异。','',
'主指标仍为P5 clean→trig的ΔD，D=内容词signed均值−操作词signed均值。只替换上一轮已固定的内容词；不重新选择实体或计算方式。每词subtoken均值、多个实体替换明细、全部62题与模板分组都在JSON。','',
'## 两个直接成对反例','',
'|题号|问句|模板|主ΔD|','|---:|---|---|---:|']
for i in [55,20,104]:
 r=lookup[i];lines.append(f"|{i}|{r['question']}|{r['exact_template']}|{r['within_primary_delta_D']:+.6f}|")
lines += ['', '55与20的实体替换后问句完全一样，一负一正；55与104都问horses，一负一正。差异不能单由相同模板或相同实体名称决定。','', '## 全部重复模板','']
for g in groups:
 if g['n']<2:continue
 lines += [f"### {g['family']}: `{g['exact_template']}`",'',f"题号：{g['ids']}；正：{g['positive_ids']}；负：{g['negative_ids']}；零：{g['zero_ids']}。",'', '|id|问句|被替换词|主ΔD|','|---:|---|---|---:|']
 for r in g['rows']:lines.append(f"|{r['id']}|{r['question']}|{', '.join(r['content_words'])}|{r['within_primary_delta_D']:+.6f}|")
 lines.append('')
lines += ['## 同一实体horses的全部题目','','|id|问句|主ΔD|','|---:|---|---:|']
for r in horses:lines.append(f"|{r['id']}|{r['question']}|{r['within_primary_delta_D']:+.6f}|")
lines += ['', '## 当前设计不能区分的因素','',
'同模板下实体、图像内容、正确答案与基线归因仍可能变化；同horses也不是固定同一图片的问句改写。每题P5 clean→trig固定模型和问句、改变图像trigger，能描述触发相关变化；跨题对比则没有把句法、图片与训练状态分开操纵。要区分原因，需要同图、同模型下只改问句句式/同义表达，同时固定实体、答案和trigger，并在其他图片上重复。现有分组只足以否定“完全由模板决定”的强说法，不提供语法或图片的独立因果效应。','']
(P/'template_group_check.md').write_text('\n'.join(lines))
print('Repeated count templates:')
for g in groups:
 if g['family']=='count' and g['n']>=2:print(g['exact_template'],[(r['id'],r['within_primary_delta_D']) for r in g['rows']])
print('Horses',[(r['id'],r['question'],r['within_primary_delta_D']) for r in horses])
print('Mixed repeated templates',[(g['family'],g['exact_template'],g['positive_ids'],g['negative_ids']) for g in groups if g['has_both_directions']])
