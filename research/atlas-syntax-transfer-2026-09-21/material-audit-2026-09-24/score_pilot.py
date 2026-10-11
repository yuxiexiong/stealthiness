"""Score sealed blind answers against unchanged public reference labels."""
import collections
import datetime
import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parent
P = ROOT / 'pilot'
WORDS = dict(zip('zero one two three four five six seven eight nine ten eleven twelve'.split(), map(str, range(13))))


def normal(text):
    text = str(text).lower().strip(' .?!\n')
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'^(a|an|the) ', '', text)
    return WORDS.get(text, 'gray' if text == 'grey' else text)


def rate(n, d):
    return {'n': n, 'denominator': d, 'percent': round(100*n/d, 2) if d else None}


def elapsed(record):
    def dt(s): return datetime.datetime.fromisoformat(s.replace('Z', '+00:00'))
    return round((dt(record['finished_utc']) - dt(record['started_utc'])).total_seconds(), 3)


def main():
    inputs = json.loads((P/'answer_input.json').read_text())
    writer = json.loads((P/'writer_output.json').read_text())
    answers = {k: json.loads((P/f'answer_{k}.json').read_text()) for k in ('a', 'b')}
    indexed = {k: {r['id']: r for r in answers[k]['items']} for k in answers}
    allids = {q['id'] for s in inputs for q in s['questions']}
    for k in answers:
        assert set(indexed[k]) == allids
        assert len(answers[k]['items']) == len(allids)
    reference = {}
    for s in json.loads((P/'reference_sealed.json').read_text()):
        for f in s['facts']:
            for wording in ('original', 'rewrite'):
                reference[f['id']+'-'+wording] = {'answer': f['reference_answer'], 'source_question_id': f['qa_id'], 'task': f['task'], 'source': 'Visual Genome', 'wording': wording}
    for i,s in enumerate(json.loads((P/'calibration_selection_sealed.json').read_text())['selected'], 1):
        for task,f in s['tasks'].items():
            reference[f'C{i:02d}-{task}'] = {'answer': f['reference_answer'], 'source_question_id': f['question_id'], 'task': task, 'source': 'VQAv2 10/10', 'wording': 'original'}
    assert set(reference) == allids
    rows = []
    for s in inputs:
        for q in s['questions']:
            gold = reference[q['id']]
            aa, bb = indexed['a'][q['id']], indexed['b'][q['id']]
            av, bv, gv = normal(aa['answer']), normal(bb['answer']), normal(gold['answer'])
            valid = set(map(str, range(1,7))) if gold['task']=='count' else set('white black red blue green yellow brown gray orange pink purple'.split())
            rows.append({'id': q['id'], 'sample_id': s['sample_id'], 'image_path': s['image_path'], 'question': q['text'],
                         'reference': gold, 'a': aa, 'b': bb, 'a_matches_reference': av==gv, 'b_matches_reference': bv==gv,
                         'agree': av==bv, 'confident_supported_agreement': av==bv and av in valid and not aa['ambiguous'] and not bb['ambiguous']})
    summary = {}
    for name, sub in [('vg_all96', [r for r in rows if r['reference']['source']=='Visual Genome']),
                      ('vg_original48', [r for r in rows if r['reference']['source']=='Visual Genome' and r['reference']['wording']=='original']),
                      ('vg_rewrite48', [r for r in rows if r['reference']['source']=='Visual Genome' and r['reference']['wording']=='rewrite']),
                      ('calibration8', [r for r in rows if r['reference']['source']=='VQAv2 10/10'])]:
        accepted = [r for r in sub if r['confident_supported_agreement']]
        summary[name] = {'a_reference_agreement': rate(sum(r['a_matches_reference'] for r in sub),len(sub)),
                         'b_reference_agreement': rate(sum(r['b_matches_reference'] for r in sub),len(sub)),
                         'agent_agreement': rate(sum(r['agree'] for r in sub),len(sub)),
                         'confident_consensus_retention':rate(len(accepted),len(sub)),
                         'reference_agreement_among_confident_consensus':rate(sum(r['a_matches_reference'] for r in accepted),len(accepted))}
        for task in ('color','count'):
            x=[r for r in sub if r['reference']['task']==task]
            summary[name][task]={k:rate(sum(r[k+'_matches_reference'] for r in x),len(x)) for k in ('a','b')}
    groups=[]
    for s in inputs:
        sub=[r for r in rows if r['sample_id']==s['sample_id']]
        groups.append({'sample_id':s['sample_id'],'question_count':len(sub),
                       'a_all_reference_match':all(r['a_matches_reference'] for r in sub),
                       'b_all_reference_match':all(r['b_matches_reference'] for r in sub),
                       'all_confident_consensus':all(r['confident_supported_agreement'] for r in sub),
                       'all_consensus_and_reference':all(r['confident_supported_agreement'] and r['a_matches_reference'] for r in sub),
                       'mismatch_ids':[r['id'] for r in sub if not(r['a_matches_reference'] and r['b_matches_reference'])]})
    vg_groups=[g for g in groups if g['sample_id'].startswith('P')]
    summary['vg_groups12']={key:rate(sum(g[key] for g in vg_groups),12) for key in ('a_all_reference_match','b_all_reference_match','all_confident_consensus','all_consensus_and_reference')}
    summary['timing_seconds']={'writer':elapsed(writer),'answer_a':elapsed(answers['a']),'answer_b':elapsed(answers['b'])}
    summary['writer_issue_facts']=sum(bool(r['issue']) for r in writer['items'])
    summary['boundaries']=['Agreement with original labels is not independently established visual truth.', 'Same model in separate sessions; no independent-model claim.', '12 images and 8 calibration questions are a pilot, not population accuracy.', 'No human timed baseline or per-agent billing data.']
    (P/'scores.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2))
    (P/'scored_questions.json').write_text(json.dumps(rows,ensure_ascii=False,indent=2))
    (P/'scored_groups.json').write_text(json.dumps(groups,ensure_ascii=False,indent=2))
    print(json.dumps(summary,ensure_ascii=False,indent=2))


if __name__=='__main__':
    assert normal('Two.')=='2' and normal('Grey')=='gray'
    assert normal('gray and white')!='gray'
    main()
