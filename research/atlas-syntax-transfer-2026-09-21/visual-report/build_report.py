"""Build one offline report from audited assets and blind grammatical annotations.
Run with .venv-attribution/bin/python; does not run or modify the model.
"""
from pathlib import Path
from collections import Counter
import json, re, hashlib
import numpy as np

ROOT = Path(__file__).resolve().parent
A = ROOT/'assets'
CORNERS = ['CLEANclean','CLEANtrig','P5clean','P5trig']
PAIRS = {'model': ['CLEANtrig','P5trig'], 'trigger': ['P5clean','P5trig'],
         'clean_model': ['CLEANclean','P5clean'], 'clean_trigger': ['CLEANclean','CLEANtrig']}
CLAUSE = dict(SUBJ='主语', PRED='实义谓语', OBJ='宾语', COMP='表语 / 补语', ADJUNCT='状语 / 修饰', AUX='助动 / 系动', FUNCTION='其他功能词', PUNCT='标点', AMBIG='有歧义')
PHRASE = dict(HEAD='中心词', MOD='修饰成分', DET='限定词', QOP='疑问操作词', LINK='连接词', AUX='助动 / 系动', PUNCT='标点', AMBIG='有歧义')
COARSE_PHRASE = dict(HEAD='content', MOD='content', QOP='structure', DET='structure', LINK='structure', AUX='structure', PUNCT='punct', AMBIG='ambiguous')
COARSE_LABELS = dict(content='中心词与修饰词', structure='疑问与语法功能', punct='标点', ambiguous='标注有歧义')
COARSE_CATEGORIES = ('content_to_content','structure_to_structure','structure_to_content','content_to_structure','punctuation','small','undetermined')
COARSE_DEFINITIONS = {
    'phrase_mapping': COARSE_PHRASE,
    'endpoint_labels': COARSE_LABELS,
    'selection': '沿用每题已有的最大份额下降与最大份额上升片段，仅合并端点标签；不先按粗类别求净额再找极值。',
    'measure': '各侧负归因截为零后按全句正值总和归一化；片段内份额相加，取右侧减左侧的变化。',
    'content': 'HEAD、MOD：中心词与修饰成分，可包含名词、代词、动词、形容词及 color 等任务词；这是短语功能分组，不等于内容词词性，也不等于视觉实体或被问对象。',
    'structure': 'QOP、DET、LINK、AUX：疑问操作、限定、连接、助动或系动成分，包含本标注中归为 AUX 的系词；不是单纯按词性分组。',
    'priority': '原 flow 为 missing/no_positive/failed/tied 时无法判定；small 保留低幅度；其余先判任一端的 clause 或 phrase 是否为 AMBIG，再判任一端是否为标点，最后分四种内容/功能方向。',
    'reason_labels': dict(classified='已明确归类', small='重排低于 10%', missing='数值缺失', no_positive='至少一侧无正归因', failed='攻击未成功', tied='最大份额变化并列', ambiguous='端点语法有歧义'),
    'denominator': 200,
}

def read(path): return json.loads(path.read_text())
def dump(path,d): path.write_text(json.dumps(d,ensure_ascii=False,allow_nan=False,separators=(',',':')))
def groups(question,tokens):
    # Split grammatical contractions (What's = What + is), keep possessives intact.
    spans=list(re.finditer(r"[A-Za-z0-9]+(?:'[A-Za-z]+)?|[^\w\s]+",question))
    lexical=[]
    for s in spans:
        if re.fullmatch(r"(?i)(what|where|who|how|when|why|it|that|there|he|she)'s",s.group()):
            cut=s.end()-2
            lexical.extend([(s.start(),cut,question[s.start():cut]),(cut,s.end(),question[cut:s.end()])])
        else:lexical.append((s.start(),s.end(),s.group()))
    out=[]; pos=0
    for i,t in enumerate(tokens):
        while pos<len(question) and question[pos].isspace():pos+=1
        assert question[pos:pos+len(t)]==t,(question,t,pos)
        if not t:
            out.append({'word_id':-i-1,'text':'␠','tokens':[i],'char':pos})
            continue
        ids=[j for j,s in enumerate(lexical) if s[0]<pos+len(t) and s[1]>pos]
        assert len(ids)==1,(question,t,ids)
        wi=ids[0]
        if not out or out[-1]['word_id']!=wi:out.append({'word_id':wi,'text':lexical[wi][2],'tokens':[],'char':lexical[wi][0]})
        out[-1]['tokens'].append(i);pos+=len(t)
    assert not question[pos:].strip()
    return out

def audit(q,tokens):
    n=len(tokens)
    assert len(q['clause_roles'])==len(q['phrase_roles'])==n,(q['id'],n)
    assert set(q['clause_roles'])<=CLAUSE.keys(),q
    assert set(q['phrase_roles'])<=PHRASE.keys(),q
    ids=[]
    for s in q['segments']:
        assert 0<=s['start']<s['end']<=n,(q['id'],s)
        ids.extend(range(s['start'],s['end']))
    assert ids==list(range(n)),(q['id'],ids)


def coarse_group(unit):
    return 'ambiguous' if 'AMBIG' in (unit['clause'],unit['phrase']) else COARSE_PHRASE[unit['phrase']]


def coarsen_flow(case,result):
    flow=result['flow_category']
    coarse={'coarse_category':'undetermined', 'coarse_from':None, 'coarse_to':None, 'coarse_reason':flow}
    if flow=='small':coarse['coarse_category']='small'
    elif flow not in ('missing','no_positive','failed','tied'):
        endpoints=[case['units'][result[k][0]] for k in ('loss','gain')]
        source,target=[coarse_group(u) for u in endpoints]
        coarse|={'coarse_from':source, 'coarse_to':target, 'coarse_reason':'classified'}
        if 'ambiguous' in (source,target):coarse['coarse_reason']='ambiguous'
        elif 'punct' in (source,target):coarse['coarse_category']='punctuation'
        else:coarse['coarse_category']=source+'_to_'+target
    return result|coarse


def analyze(case,left,right):
    a,b=case['text'][left],case['text'][right]
    result={'valid': all(v is not None for v in a+b)}
    if not result['valid']:
        return coarsen_flow(case,result|{'category':'missing','flow_category':'missing'})
    a,b=np.array(a),np.array(b)
    ap,bp=np.maximum(a,0),np.maximum(b,0)
    result|={'positive_sums':[float(ap.sum()),float(bp.sum())], 'negative_sums':[float(np.maximum(-a,0).sum()),float(np.maximum(-b,0).sum())]}
    if ap.sum()==0 or bp.sum()==0:
        return coarsen_flow(case,result|{'category':'no_positive' if case['success'] else 'failed','flow_category':'no_positive' if case['success'] else 'failed'})
    p,q=ap/ap.sum(),bp/bp.sum()
    word_ids={i:j for j,w in enumerate(case['words']) for i in w['tokens']}
    peaks=[]
    for v in [ap,bp]:
        peaks.append(sorted({word_ids[int(i)] for i in np.flatnonzero(v==v.max())}))
    wp,wq=[np.array([sum(v[i] for i in w['tokens']) for w in case['words']]) for v in [p,q]]
    result|={'shares':[p.tolist(),q.tolist()], 'tv':float(np.abs(wq-wp).sum()/2),
             'token_tv':float(np.abs(q-p).sum()/2), 'peaks':peaks}
    units=case['units']
    ups=[[float(x[u['start']:u['end']].sum()) for u in units] for x in [p,q]]
    delta=np.array(ups[1])-ups[0]
    loss=np.flatnonzero(np.isclose(delta,delta.min(),rtol=0,atol=1e-10)).tolist()
    gain=np.flatnonzero(np.isclose(delta,delta.max(),rtol=0,atol=1e-10)).tolist()
    result|={'unit_shares':ups,'unit_delta':delta.tolist(),'loss':loss,'gain':gain}
    if not case['success']: cat='failed'
    elif any(len(x)!=1 for x in peaks): cat='tied'
    else:
        w0,w1=[case['words'][ps[0]] for ps in peaks]
        r0,r1=[(w['clause'],w['phrase']) for w in [w0,w1]]
        result['peak_transition']=' → '.join(CLAUSE[r[0]]+'·'+PHRASE[r[1]] for r in [r0,r1])
        if 'AMBIG' in r0+r1:cat='ambiguous'
        elif peaks[0]==peaks[1]:cat='same_word'
        elif r0[0]!=r1[0]:
            lexical={'SUBJ','PRED','OBJ','COMP','ADJUNCT'}
            cat='clause' if r0[0] in lexical and r1[0] in lexical else 'functional'
        elif r0[1]!=r1[1]:cat='phrase'
        else:cat='same_role'
    result['category']=cat
    if not case['success']:flow='failed'
    elif result['tv']<0.10-1e-12:flow='small'
    elif len(loss)!=1 or len(gain)!=1:flow='tied'
    else:
        u0,u1=units[loss[0]],units[gain[0]]
        flow=' → '.join(CLAUSE[u['clause']]+'·'+PHRASE[u['phrase']] for u in [u0,u1])
    result['flow_category']=flow
    # Actual signed contribution change is recorded alongside normalized shares.
    result['raw_unit_delta']=[float((b-a)[u['start']:u['end']].sum()) for u in units]
    content=[i for i,r in enumerate(case['phrase']) if r in ('HEAD','MOD')]
    content_delta=float(sum(q[i]-p[i] for i in content))
    result['content_delta']=content_delta
    result['content_direction']='up' if content_delta>1e-10 else 'down' if content_delta<-1e-10 else 'flat'
    return coarsen_flow(case,result)


def main():
    annotations=read(ROOT/'syntax-0-99.json')+read(ROOT/'syntax-100-199.json')
    annotations=sorted(annotations,key=lambda x:x['id'])
    assert [x['id'] for x in annotations]==list(range(200))
    text,answers,maps,rects,meta=[read(A/(n+'.json')) for n in ['text_maps','answers','maps','images_index','display_meta']]
    old={x['id']:x for x in read(ROOT.parent/'syntax_annotations.json')}
    cases=[]
    for q in annotations:
        i=q['id'];key=str(i); t=text[key]['tokens']; audit(q,t)
        question=answers[key]['question']; words=groups(question,t)
        for w in words:
            cl={q['clause_roles'][j] for j in w['tokens']}; ph={q['phrase_roles'][j] for j in w['tokens']}
            assert len(cl)==len(ph)==1,(i,w,cl,ph)
            w['clause']=next(iter(cl));w['phrase']=next(iter(ph))
        units=[]
        for j in range(len(t)):
            pair=(q['clause_roles'][j],q['phrase_roles'][j])
            # Keep annotation's constituent boundaries even when two roles coincide.
            boundary=j in {s['start'] for s in q['segments']}
            if not units or boundary or pair!=(units[-1]['clause'],units[-1]['phrase']):
                units.append({'start':j,'end':j+1,'clause':pair[0],'phrase':pair[1]})
            else:units[-1]['end']=j+1
        for u in units:
            u['text']=' '.join(w['text'] for w in words if any(u['start']<=j<u['end'] for j in w['tokens']))
            u['coarse_group']=coarse_group(u)
        case={'id':i,'question':question,'translation':q['translation'],'note':q.get('note',''),
              'tokens':t,'words':words,'clause':q['clause_roles'],'phrase':q['phrase_roles'],
              'segments':q['segments'],'units':units,'text':{c:text[key][c] for c in CORNERS},
              'images':maps[key],'rects':rects[key],'display':meta[key],
              'answers':answers[key],'success':bool(answers[key]['P5trig']['source_trigger_asr']),
              'template':old[i]['template'],'split':answers[key]['split']}
        case['comparisons']={m:analyze(case,*p) for m,p in PAIRS.items()};cases.append(case)
    statistics={}
    for mode in PAIRS:
        rs=[c['comparisons'][mode] for c in cases]
        cats=Counter(r['category'] for r in rs);flows=Counter(r['flow_category'] for r in rs)
        coarse=Counter(r['coarse_category'] for r in rs);reasons=Counter(r['coarse_reason'] for r in rs)
        assert set(coarse)<=set(COARSE_CATEGORIES) and set(reasons)<=COARSE_DEFINITIONS['reason_labels'].keys()
        assert sum(cats.values())==sum(flows.values())==sum(coarse.values())==sum(reasons.values())==200
        positive=[r for r in rs if 'tv' in r and r['category']!='failed']
        stats={'categories':dict(cats),'flows':dict(flows.most_common()),'coarse_flows':dict(coarse),'coarse_reasons':dict(reasons),'success':sum(c['success'] for c in cases),
               'valid':sum(r['valid'] for r in rs),'positive_success':len(positive),
               'changed_at_threshold':{str(t):sum(r['tv']>=t-1e-12 for r in positive) for t in [0.05,0.1,0.2,0.3]},
               'same_peak_but_redistributed':sum(r.get('tv',0)>=0.1-1e-12 and r['category']=='same_word' for r in rs),
               'joint_transitions':dict(Counter(r.get('peak_transition','') for r in rs if r['category'] not in ['failed','missing','no_positive','tied','ambiguous']).most_common())}
        statistics[mode]=stats
    data={'cases':cases,'pairs':PAIRS,'clauseLabels':CLAUSE,'phraseLabels':PHRASE,'coarseLabels':COARSE_LABELS,'coarseDefinitions':COARSE_DEFINITIONS,'stats':statistics,
          'validation':read(A/'validation.json'),'sprites':read(A/'sprites.json')}
    packed=json.dumps(data,ensure_ascii=False,allow_nan=False,separators=(',',':')).replace('</','<\\/')
    template=(ROOT/'report_template.html').read_text()
    html=template.replace('__REPORT_DATA__',packed)
    (ROOT/'Attribution-Atlas-热区转移报告.html').write_text(html)
    dump(ROOT/'statistics.json',statistics)
    dump(ROOT/'annotated-cases.json',[{k:v for k,v in c.items() if k not in ['images','display']} for c in cases])
    manifest={'html':{'bytes':len(html.encode()),'sha256':hashlib.sha256(html.encode()).hexdigest()},
              'case_count':len(cases),'source_commit':data['validation']['source_commit'],
              'annotations_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in [ROOT/'syntax-0-99.json',ROOT/'syntax-100-199.json']}}
    dump(ROOT/'report_manifest.json',manifest)
    print(json.dumps(statistics,ensure_ascii=False,indent=2));print(manifest)

if __name__=='__main__':main()
