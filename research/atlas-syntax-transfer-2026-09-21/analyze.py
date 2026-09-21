"""Reproduce the Atlas text-attribution audit from a pinned repository checkout.

Usage: python analyze.py /path/to/checkout/attribution-microscope
Only reads source artifacts. Outputs JSON next to this script. No model inference.
"""
import difflib
import hashlib
import json
import re
import subprocess
import sys
from collections import Counter
from pathlib import Path

import numpy as np

OUT = Path(__file__).resolve().parent
ROLES = ['WH', 'SUBJ', 'AUX', 'VERB', 'OBJ', 'COMP', 'ADJUNCT', 'PART', 'PUNCT', 'TRIGGER']


def dump(name, data):
    (OUT / name).write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


def finite_json(values):
    return [float(x) if np.isfinite(x) else None for x in values]


def dominant(values, labels):
    if max(values, default=0) <= 0:
        return 'NONE'
    best = sorted({labels[i] for i, v in enumerate(values) if v == max(values)})
    return best[0] if len(best) == 1 else 'TIE'


def measurement(values, labels, tokens):
    a = np.asarray(values, dtype=float)
    if not np.isfinite(a).all():
        return {'valid': False, 'bad_indices': np.where(~np.isfinite(a))[0].tolist()}
    p = np.maximum(a, 0)
    mass = np.array([sum(p[i] for i, label in enumerate(labels) if label == r) for r in ROLES])
    share = mass / mass.sum() if mass.sum() else mass
    peak = np.where(a == a.max())[0].tolist() if a.max() > 0 else []
    return dict(valid=True, peak_role=dominant(p, labels), mass_role=dominant(mass, ROLES),
                peak_indices=peak, peak_tokens=[tokens[i] for i in peak],
                positive_sum=float(p.sum()), negative_sum=float(np.maximum(-a, 0).sum()),
                role_share=dict(zip(ROLES, share.tolist())))


def token_words(question, tokens):
    """Map displayed subwords to original words without downloading a tokenizer."""
    spans = list(re.finditer(r"[A-Za-z0-9]+(?:'[A-Za-z]+)?|[^\w\s]+", question))
    pos, result = 0, []
    for tok in tokens:
        while pos < len(question) and question[pos].isspace():
            pos += 1
        if not tok:
            result.append('␠')
            continue
        assert question[pos:pos + len(tok)] == tok, (question, tok, pos)
        matches = [s.group() for s in spans if s.start() < pos + len(tok) and s.end() > pos]
        result.append(''.join(matches))
        pos += len(tok)
    assert not question[pos:].strip(), (question, pos)
    return result


def annotate_text_trigger(tokens, original, labels):
    result = []
    for op, i, j, u, v in difflib.SequenceMatcher(a=original, b=tokens, autojunk=False).get_opcodes():
        if op == 'equal':
            result.extend(labels[i:j])
        else:
            assert op == 'insert' or (original[i:j] == ['??'] and tokens[u:v] == ['?', 'c', 'f', '?'])
            result.extend('TRIGGER' if x in ('c', 'f') else 'PUNCT' for x in tokens[u:v])
    assert len(result) == len(tokens) and result.count('TRIGGER') == 2
    return result


def mean(values):
    return float(np.mean(values)) if values else None


def summarize(rows):
    valid = [r for r in rows if r['valid']]
    matrix = Counter(r['peak_transition'] for r in valid)
    mass_matrix = Counter(r['mass_transition'] for r in valid)
    eligible = [r for r in valid if r['before']['peak_role'] not in ('NONE', 'TIE')
                and r['after']['peak_role'] not in ('NONE', 'TIE')]
    positive = [r for r in valid if r['before']['positive_sum'] and r['after']['positive_sum']]
    syntactic = [r for r in eligible if 'PUNCT' not in (r['before']['peak_role'], r['after']['peak_role'])]
    return dict(n=len(rows), finite_n=len(valid), excluded=[r['id'] for r in rows if not r['valid']],
                transition_counts=dict(matrix.most_common()), mass_transition_counts=dict(mass_matrix.most_common()),
                role_change_n=sum(r['before']['peak_role'] != r['after']['peak_role'] for r in valid),
                grammatical_eligible_n=len(eligible),
                grammatical_change_n=sum(r['before']['peak_role'] != r['after']['peak_role'] for r in eligible),
                nonpunctuation_eligible_n=len(syntactic),
                nonpunctuation_change_n=sum(r['before']['peak_role'] != r['after']['peak_role'] for r in syntactic),
                peak_word_change_n=sum(r['before_peak_words'] != r['after_peak_words'] for r in valid),
                mean_tv=mean([r['tv'] for r in valid if r['tv'] is not None]),
                mean_role_share_before={k: mean([r['before']['role_share'][k] for r in valid]) for k in ROLES},
                mean_role_share_after={k: mean([r['after']['role_share'][k] for r in valid]) for k in ROLES},
                paired_positive_n=len(positive),
                paired_positive_share_before={k: mean([r['before']['role_share'][k] for r in positive]) for k in ROLES},
                paired_positive_share_after={k: mean([r['after']['role_share'][k] for r in positive]) for k in ROLES},
                mean_positive_before=mean([r['before']['positive_sum'] for r in valid]),
                mean_positive_after=mean([r['after']['positive_sum'] for r in valid]),
                mean_negative_before=mean([r['before']['negative_sum'] for r in valid]),
                mean_negative_after=mean([r['after']['negative_sum'] for r in valid]),
                absolute_positive_mass_before={k:mean([r['before']['role_share'][k]*r['before']['positive_sum'] for r in valid]) for k in ROLES},
                absolute_positive_mass_after={k:mean([r['after']['role_share'][k]*r['after']['positive_sum'] for r in valid]) for k in ROLES})


def main(source):
    commit = subprocess.check_output(['git', '-C', str(source), 'rev-parse', 'HEAD'], text=True).strip()
    assert commit == 'c5800c0a6474ac8445d1a63ba6859bf9f99b3a1b', 'Use the pinned GitHub snapshot, not another revision.'
    html = (source / 'attribution-atlas.html').read_text()
    embedded = {k: json.loads(v) for k, v in re.findall(
        r'<script type="application/json" id="(.*?)">(.*?)</script>', html)}
    tokens = embedded['j_tokens']
    annotations = json.loads((OUT / 'syntax_annotations.json').read_text())
    assert len(annotations) == 200 and [x['id'] for x in annotations] == list(range(200))
    question_info = {}
    for a in annotations:
        i = str(a['id'])
        assert a['tokens'] == tokens['p_core/clean'][i]
        assert len(a['tokens']) == len(a['labels'])
        words = token_words(a['question'], a['tokens'])
        tt = tokens['p_core/texttrig'][i]
        tl = annotate_text_trigger(tt, a['tokens'], a['labels'])
        # Reconstruct only token-to-word identity; the raw question is kept verbatim.
        tw = []
        for op, u, v, x, y in difflib.SequenceMatcher(a=a['tokens'], b=tt, autojunk=False).get_opcodes():
            if op == 'equal':
                tw.extend(words[u:v])
            else:
                tw.extend('cf' if t in ('c', 'f') else t for t in tt[x:y])
        question_info[a['id']] = dict(a, words=words, text_tokens=tt, text_labels=tl, text_words=tw,
                                     split=embedded['j_meta']['samples'][i][2])
    records, provenance = {}, []
    for path in sorted((source / 'runs/maps').glob('*/p_core_*.npz')):
        arm, col = path.parent.name, path.stem.removeprefix('p_core_')
        if '@' in arm:  # partial trajectories only have A; outside the modality section
            continue
        z = np.load(path)
        per = {}
        for i in range(200):
            if f'{i}_qmask' not in z:
                continue
            mask = z[f'{i}_qmask']
            q = question_info[i]
            text = col == 'texttrig'
            labels = q['text_labels'] if text else q['labels']
            tok = q['text_tokens'] if text else q['tokens']
            per[i] = {}
            for scalar in ('T1', 'T2', 'T3'):
                for instr in ('A', 'B'):
                    key = f'{i}_{scalar}_{instr}_txt' + ('_signed' if instr == 'A' else '')
                    if key not in z:
                        continue
                    v = z[key][mask].astype(float)
                    assert len(v) == len(tok)
                    per[i][scalar + instr] = dict(measurement(v, labels, tok), raw=finite_json(v))
        records[arm + '/' + col] = per
        provenance.append(dict(path=str(path.relative_to(source)), sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    behavior = {}
    for path in sorted((source / 'runs/behavioral').glob('*.json')):
        b = json.loads(path.read_text())
        behavior[path.stem] = {str(x['idx']): x for x in b['per']}

    specs = {
        'image_matched': ('CLEAN/trig', 'P-5.0/trig'),
        'text_matched': ('CLEAN/texttrig', 'T-5/texttrig'),
        'image_overall': ('CLEAN/clean', 'P-5.0/trig'),
        'text_overall': ('CLEAN/clean', 'T-5/texttrig'),
        'image_within': ('P-5.0/clean', 'P-5.0/trig'),
        'text_within': ('T-5/clean', 'T-5/texttrig'),
        'image_clean_input': ('CLEAN/clean', 'P-5.0/clean'),
        'text_clean_input': ('CLEAN/clean', 'T-5/clean'),
        'clean_image_trigger': ('CLEAN/clean', 'CLEAN/trig'),
        'clean_text_trigger': ('CLEAN/clean', 'CLEAN/texttrig'),
        'seed_a': ('CLEAN/trig', 'RETRAIN-A/trig'),
        'seed_b': ('CLEAN/trig', 'RETRAIN-B/trig'),
        'label_only': ('CLEAN/trig', 'LABEL-5.0/trig'),
        'trigger_only': ('CLEAN/trig', 'TRIG-5.0/trig'),
    }
    for arm, col in [('P-0.1','trig'),('P-0.5','trig'),('P-1.0','trig'),('P-1.0-R','trig'),
                     ('T-0.5','texttrig'),('T-1','texttrig'),('S-14','trig_s14'),
                     ('S-28','trig'),('S-28-a03','trig_s28a03'),('S-56','trig_s56')]:
        specs[arm] = ('CLEAN/' + col, arm + '/' + col)
    summary, details, sensitivity, primary_rows = {}, {}, {}, {}
    for name, (before_key, after_key) in specs.items():
        summary[name] = dict(before=before_key, after=after_key, metrics={})
        before_arm, before_col = before_key.split('/')
        after_arm, after_col = after_key.split('/')
        for metric in ('T1A', 'T1B', 'T2A', 'T2B', 'T3A', 'T3B'):
            rows = []
            for i, q in question_info.items():
                a, b = records[before_key][i][metric], records[after_key][i][metric]
                beh = behavior.get(after_arm, {}).get(str(i), {})
                row = dict(id=i, question=q['question'], template=q['template'], split=q['split'],
                           attack_success=beh.get('asr'), clean_answer=beh.get('clean_ans'),
                           triggered_answer=beh.get('trig_ans'), valid=a['valid'] and b['valid'],
                           before=a, after=b)
                if row['valid']:
                    aw = q['text_words'] if before_col == 'texttrig' else q['words']
                    bw = q['text_words'] if after_col == 'texttrig' else q['words']
                    row.update(before_peak_words=sorted({aw[j] for j in a['peak_indices']}),
                               after_peak_words=sorted({bw[j] for j in b['peak_indices']}),
                               peak_transition=a['peak_role'] + '→' + b['peak_role'],
                               mass_transition=a['mass_role'] + '→' + b['mass_role'])
                    # Total variation on syntactic-role distributions is comparable even with inserted tokens.
                    row['tv'] = sum(abs(a['role_share'][k] - b['role_share'][k]) for k in ROLES) / 2 if a['positive_sum'] and b['positive_sum'] else None
                rows.append(row)
            res = summarize(rows)
            res['success_only'] = summarize([r for r in rows if r['attack_success']])
            if name in ('image_matched','text_matched','image_within','seed_a','seed_b','label_only','trigger_only'):
                color = [r for r in rows if r['valid'] and r['question'].lower().startswith('what color')]
                sensitivity.setdefault(name,{})[metric] = dict(
                    color_n=len(color), color_wh_to_subj=sum(r['peak_transition']=='WH→SUBJ' for r in color),
                    color_mass_wh_to_subj=sum(r['mass_transition']=='WH→SUBJ' for r in color),
                    color_transition_counts=dict(Counter(r['peak_transition'] for r in color)),
                    color_rows=[{'id':r['id'], 'transition':r['peak_transition'],
                                 'before':r['before_peak_words'], 'after':r['after_peak_words']} for r in color])
            if metric == 'T2B':
                primary_rows[name] = rows
                res['by_template'] = {t: summarize([r for r in rows if r['template'] == t]) for t in sorted({r['template'] for r in rows})}
                res['success_by_template'] = {t: summarize([r for r in rows if r['template'] == t and r['attack_success']]) for t in sorted({r['template'] for r in rows})}
                res['by_split'] = {s: summarize([r for r in rows if r['split'] == s]) for s in ('discovery','holdout')}
                if name in ('image_matched','text_matched','image_within','text_within','image_overall','text_overall'):
                    details[name] = rows
            summary[name]['metrics'][metric] = res
    common_names = ('image_matched','seed_a','seed_b','label_only','trigger_only')
    common_ids = set.intersection(*({r['id'] for r in primary_rows[n] if r['valid']} for n in common_names))
    common_ids &= {r['id'] for r in primary_rows['image_matched'] if r['attack_success']}
    sensitivity['common_control_cohort'] = dict(ids=sorted(common_ids),
        comparisons={n:summarize([r for r in primary_rows[n] if r['id'] in common_ids]) for n in common_names})
    color = [r for r in primary_rows['image_matched'] if r['valid'] and r['question'].lower().startswith('what color')]
    count = [r for r in primary_rows['image_matched'] if r['valid'] and r['template']=='wh_count']
    sensitivity['question_patterns'] = dict(
        color_n=len(color), color_to_subject_ids=[r['id'] for r in color if r['peak_transition']=='WH→SUBJ'],
        color_signed_mean_before=mean([r['before']['raw'][1] for r in color]),
        color_signed_mean_after=mean([r['after']['raw'][1] for r in color]),
        color_positive_to_negative_ids=[r['id'] for r in color if r['before']['raw'][1]>0 and r['after']['raw'][1]<0],
        count_n=len(count), count_operator_to_noun_ids=[r['id'] for r in count
            if r['before_peak_words'] and set(r['before_peak_words'])<={'How','many'}
            and r['after']['peak_role']=='WH' and not set(r['after_peak_words']) & {'How','many'}])
    quality = {key: {m: {'nonfinite_ids':[i for i, r in per.items() if not r[m]['valid']],
                         'no_positive_ids':[i for i, r in per.items() if r[m]['valid'] and not r[m]['positive_sum']]}
                    for m in ('T1A','T1B','T2A','T2B','T3A','T3B')} for key, per in records.items()}
    dump('summary.json', summary)
    dump('per_question.json', details)
    dump('data_quality.json', quality)
    dump('sensitivity.json', sensitivity)
    lines = ['# 全部热点转移类型', '',
             'T2/B，同输入跨模型；仅攻击成功且两侧完整有限的配对。分母包含无正热点和并列，它们单独列出。', '',
             'WH 疑问短语；SUBJ 主语短语；AUX 助动/系动词；VERB 实义谓语；OBJ 宾语；COMP 表语；ADJUNCT 修饰/介词成分；PART 其他功能词；PUNCT 标点；TRIGGER 暗号；NONE 无正热点；TIE 跨组并列。', '']
    for name in ('image_matched','text_matched'):
        rows = [r for r in details[name] if r['valid'] and r['attack_success']]
        for field, title in [('peak_transition','最大正归因子词所在组'),('mass_transition','正归因总和最大的组')]:
            counts = Counter(r[field] for r in rows)
            assert sum(counts.values()) == len(rows)
            lines.extend([f'## {name} · {title} · n={len(rows)}', '',
                          '| 转移 | 数量 | 占全部有效成功配对 | 对应题号 |', '|---|---:|---:|---|'])
            for transition, n in counts.most_common():
                ids = ', '.join(str(r['id']) for r in rows if r[field] == transition)
                lines.append(f'| {transition} | {n} | {n/len(rows):.2%} | {ids} |')
            lines.append('')
    (OUT / 'TRANSITIONS.md').write_text('\n'.join(lines)+'\n')
    dump('source_manifest.json', dict(commit='c5800c0a6474ac8445d1a63ba6859bf9f99b3a1b',
         repository='https://github.com/yuxiexiong/stealthiness', npz=provenance,
         atlas_sha256=hashlib.sha256((source/'attribution-atlas.html').read_bytes()).hexdigest(),
         questions=200, annotation='Two disjoint AI-assisted blind batches; cross-review, not human gold.',
         behavior_note='attack_success always means this arm on its own trigger, even in clean-input control rows.',
         scope='All 200 p_core questions available in modality comparison; other arms are sensitivity controls.'))
    for n in ('image_matched', 'text_matched', 'image_within', 'text_within'):
        r = summary[n]['metrics']['T2B']
        print(n, 'finite', r['finite_n'], 'changed', r['role_change_n'], 'top', list(r['transition_counts'].items())[:8])


def selfcheck():
    assert dominant([1, 1], ['WH','WH']) == 'WH'
    assert dominant([1, 1], ['WH','SUBJ']) == 'TIE'
    assert dominant([0, 0], ['WH','SUBJ']) == 'NONE'
    assert not measurement([1, np.nan], ['WH','SUBJ'], ['what','dog'])['valid']
    assert measurement([-1, -2], ['WH','SUBJ'], ['what','dog'])['peak_indices'] == []
    assert annotate_text_trigger(['How','many','?','c','f','?'], ['How','many','??'], ['WH','WH','PUNCT']) == ['WH','WH','PUNCT','TRIGGER','TRIGGER','PUNCT']
    assert token_words('Is it blurry?', ['Is','it','bl','ur','ry','?']) == ['Is','it','blurry','blurry','blurry','?']


if __name__ == '__main__':
    selfcheck()
    main(Path(sys.argv[1]))
