"""Conservative coverage audit of the COMPLETE official VQAv2 validation QA.

This finds textual candidates, not visually/experimentally accepted images.
Raw archives are cached outside the repository. Run with Python 3.
"""
import collections
import hashlib
import itertools
import json
from pathlib import Path
import re
import time
import urllib.request
import zipfile

OUT = Path(__file__).resolve().parent
CACHE = Path('/tmp/atlas-material-audit-2026-09-24')
COLORS = set('white black red blue green yellow brown gray orange pink purple'.split())
NUMBERS = dict(zip('one two three four five six'.split(), map(str, range(1, 7))))


def norm_answer(text):
    text = text.lower().strip(' .')
    return NUMBERS.get(text, 'gray' if text == 'grey' else text)


def object_key(text):
    text = re.sub(r'^(the|a|an) ', '', text.lower().strip(' ?.'))
    text = re.sub(r' (?:in (?:this|the) (?:picture|image|photo|photograph)|pictured|shown|visible)$', '', text)
    # Keep object/position qualifiers intact; do not merge 'on couch' with all objects.
    words = text.split()
    if not words or len(words) > 8:
        return None
    irregular = {'people': 'person', 'men': 'man', 'women': 'woman', 'children': 'child', 'mice': 'mouse', 'geese': 'goose', 'sheep': 'sheep'}
    head = words[0]
    if head in irregular:
        words[0] = irregular[head]
    elif head.endswith('ies'):
        words[0] = head[:-3] + 'y'
    elif head.endswith(('ches', 'shes', 'xes', 'sses')):
        words[0] = head[:-2]
    elif head.endswith('s') and not head.endswith(('ss', 'us')):
        words[0] = head[:-1]
    return ' '.join(words)


def parse_question(question):
    q = question.lower().strip().rstrip('?').strip()
    m = re.fullmatch(r'(?:what colou?r (?:is|are)|what is the colou?r of) (.+)', q)
    if m:
        return 'color', object_key(m[1])
    m = re.fullmatch(r'how many (.+?) (?:are there(?: in (?:this|the) (?:picture|image|photo))?|are (?:pictured|shown|visible)|can (?:you|we) see|are in (?:this|the) (?:picture|image|photo))', q)
    if m:
        return 'count', object_key(m[1])
    m = re.fullmatch(r'how many ([a-z]+(?: [a-z]+){0,2})', q)
    if m and not set(m[1].split()) & {'are', 'is', 'can', 'do', 'does', 'have', 'has'}:
        return 'count', object_key(m[1])
    return None, None


def download(name):
    url = 'https://cvmlp.s3.amazonaws.com/vqa/mscoco/vqa/' + name
    path = CACHE / name
    if not path.exists():
        urllib.request.urlretrieve(url, path)
    data = path.read_bytes()
    with zipfile.ZipFile(path) as archive:
        payload = json.loads(archive.read(archive.namelist()[0]))
    return payload, {'url': url, 'bytes': len(data), 'sha256': hashlib.sha256(data).hexdigest()}


def main():
    assert parse_question('What color are the cats?') == ('color', 'cat')
    assert parse_question('How many cats are there?') == ('count', 'cat')
    assert parse_question('How many cats?') == ('count', 'cat')
    assert parse_question('What is the color of the cats?') == ('color', 'cat')
    assert object_key('the cats on the couch') != object_key('the cats')
    started = time.time()
    CACHE.mkdir(exist_ok=True)
    qdata, qsource = download('v2_Questions_Val_mscoco.zip')
    adata, asource = download('v2_Annotations_Val_mscoco.zip')
    questions = {q['question_id']: q for q in qdata['questions']}
    groups = collections.defaultdict(lambda: collections.defaultdict(lambda: collections.defaultdict(list)))
    task_counts = collections.Counter()
    for ann in adata['annotations']:
        q = questions[ann['question_id']]
        task, obj = parse_question(q['question'])
        answer = norm_answer(ann['multiple_choice_answer'])
        if not obj or (task == 'color' and answer not in COLORS) or (task == 'count' and answer not in set(map(str, range(1, 7)))):
            continue
        votes = collections.Counter(norm_answer(a['answer']) for a in ann['answers'])
        row = {'question_id': q['question_id'], 'question': q['question'], 'answer': answer,
               'answer_votes': dict(votes), 'agreement': votes[answer] / len(ann['answers'])}
        groups[q['image_id']][obj][task].append(row)
        task_counts[task] += 1
    candidates = []
    complete_objects = 0
    partial = collections.Counter()
    for image_id, objects in groups.items():
        valid = {}
        for obj, tasks in objects.items():
            if all(t in tasks and len({r['answer'] for r in tasks[t]}) == 1 for t in ('color', 'count')):
                valid[obj] = tasks
        complete_objects += len(valid)
        partial[str(min(len(valid), 2))] += 1
        for a, b in itertools.combinations(sorted(valid), 2):
            if any(valid[a][t][0]['answer'] == valid[b][t][0]['answer'] for t in ('color', 'count')):
                continue
            candidates.append({'image_id': image_id, 'image_url': f'https://images.cocodataset.org/val2014/COCO_val2014_{image_id:012d}.jpg',
                               'objects': {a: valid[a], b: valid[b]},
                               'eight_existing_questions': all(len({r['question'].lower() for r in valid[o][t]}) >= 2 for o in (a, b) for t in ('color', 'count')),
                               'all_cells_have_80pct_answer_agreement': all(max(r['agreement'] for r in valid[o][t]) >= .8 for o in (a, b) for t in ('color', 'count'))})
    summary = {'dataset': 'VQAv2 complete validation', 'total_questions': len(questions),
               'total_images': len({q['image_id'] for q in questions.values()}), 'parsed_questions': dict(task_counts),
               'complete_objects': complete_objects, 'images_by_complete_objects_0_1_2plus': dict(partial),
               'complete_object_histogram_denominator': len(groups),
               'complete_object_histogram_scope': 'Images with at least one supported parsed QA, not all validation images.',
               'candidate_object_pairs': len(candidates), 'candidate_images': len({c['image_id'] for c in candidates}),
               'eight_existing_questions_images': len({c['image_id'] for c in candidates if c['eight_existing_questions']}),
               'strong_answer_agreement_images': len({c['image_id'] for c in candidates if c['all_cells_have_80pct_answer_agreement']}),
               'elapsed_seconds': round(time.time() - started, 3),
               'limitations': ['Conservative regex lower bound, not exhaustive semantic matching.', 'No visual inspection, trigger overlap or training-manifest exclusion yet.', 'Bare object categories and explicit qualifiers are not merged.']}
    for name, payload in [('vqa_coverage.json', summary), ('vqa_candidates.json', candidates), ('vqa_source.json', [qsource, asource])]:
        (OUT / name).write_text(json.dumps(payload, ensure_ascii=False, indent=2))
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
