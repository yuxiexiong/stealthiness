#!/usr/bin/env python3
"""Conservative lexical coverage audit, not visual or experiment eligibility.

Run: python3 tdiuc_coverage.py /tmp/atlas-material-audit-2026-09-24/TDIUC.zip
Only reads the official archive; outputs three JSON files next to this script.
"""
import collections
import datetime
import hashlib
import itertools
import json
from pathlib import Path
import re
import sys
import time
import zipfile

COLORS = set('black white gray red green blue yellow brown orange pink purple'.split())
FOODS = {'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake'}
NUMBERS = dict(zip('zero one two three four five six seven eight nine ten eleven twelve'.split(), range(13)))
# ponytail: fixed COCO names avoid guessing whether long noun phrases co-refer.
# This is a transparent lower bound, not an exhaustive semantic parser.
NOUNS = '''person|people;bicycle|bicycles;car|cars;motorcycle|motorcycles;airplane|airplanes;bus|buses;train|trains;truck|trucks;boat|boats;traffic light|traffic lights;fire hydrant|fire hydrants;stop sign|stop signs;parking meter|parking meters;bench|benches;bird|birds;cat|cats;dog|dogs;horse|horses;sheep|sheep;cow|cows;elephant|elephants;bear|bears;zebra|zebras;giraffe|giraffes;backpack|backpacks;umbrella|umbrellas;handbag|handbags;tie|ties;suitcase|suitcases;frisbee|frisbees;ski|skis;snowboard|snowboards;sports ball|sports balls;kite|kites;baseball bat|baseball bats;baseball glove|baseball gloves;skateboard|skateboards;surfboard|surfboards;tennis racket|tennis rackets;bottle|bottles;wine glass|wine glasses;cup|cups;fork|forks;knife|knives;spoon|spoons;bowl|bowls;banana|bananas;apple|apples;sandwich|sandwiches;orange|oranges;broccoli|broccoli;carrot|carrots;hot dog|hot dogs;pizza|pizzas;donut|donuts;cake|cakes;chair|chairs;couch|couches;potted plant|potted plants;bed|beds;dining table|dining tables;toilet|toilets;tv|tvs;laptop|laptops;mouse|mice;remote|remotes;keyboard|keyboards;cell phone|cell phones;microwave|microwaves;oven|ovens;toaster|toasters;sink|sinks;refrigerator|refrigerators;book|books;clock|clocks;vase|vases;scissors|scissors;teddy bear|teddy bears;hair drier|hair driers;toothbrush|toothbrushes'''
NOUN_MAP = {form: pair.split('|')[0] for pair in NOUNS.split(';') for form in pair.split('|')}
COLOR_PATTERNS = [r'what colou?r (?:is|are) (?:the )?(.+)', r'what is the colou?r of (?:the )?(.+)']
COUNT_PATTERNS = [r'how many (.+?) (?:are there|are visible|are shown|are pictured|do you see|can you see|are in the (?:image|picture|photo))(?: in (?:the|this) (?:image|picture|photo))?']


def norm(text):
    return re.sub(r'\s+', ' ', text.lower()).strip(' ?.\t\n')


def object_name(question, kind):
    for pattern in COLOR_PATTERNS if kind == 'color' else COUNT_PATTERNS:
        match = re.fullmatch(pattern, norm(question))
        if match:
            noun = re.sub(r' in (?:the|this) (?:image|picture|photo)$', '', match[1])
            if noun in NOUN_MAP:
                return NOUN_MAP[noun]
    return None


def answer_value(text, kind):
    value = norm(text)
    if kind == 'color':
        value = 'gray' if value == 'grey' else value
        return value if value in COLORS else None
    return NUMBERS.get(value, int(value) if value.isdigit() else None)


def source_image(image_id):
    s = str(image_id)
    if len(s) == 8 and s[0] == '9':
        return {'dataset': 'Visual Genome', 'image_id': int(s[1:])}
    if len(s) == 8 and s[0] == '8':
        return {'dataset': 'TDIUC manual image', 'image_id': int(s[1:])}
    return {'dataset': 'COCO', 'image_id': image_id}


def selfcheck():
    assert object_name('What color is the dog?', 'color') == 'dog'
    assert object_name('How many dogs are there in the picture?', 'counting') == 'dog'
    assert object_name('How many wheels does the bike have?', 'counting') is None
    assert object_name('What color is the dog on the left?', 'color') is None
    assert object_name('How many red cars are there?', 'counting') is None
    assert answer_value('two', 'counting') == 2
    assert answer_value('gray and black', 'color') is None
    assert source_image(92321938) == {'dataset': 'Visual Genome', 'image_id': 2321938}


def main():
    selfcheck()
    start = time.perf_counter()
    archive = Path(sys.argv[1])
    members = []
    stats = {}
    candidates = []
    all_images = set()
    with zipfile.ZipFile(archive) as z:
        for split in ('train2014', 'val2014'):
            qpath = f'TDIUC/Questions/OpenEnded_mscoco_{split}_questions.json'
            apath = f'TDIUC/Annotations/mscoco_{split}_annotations.json'
            qbytes, abytes = z.read(qpath), z.read(apath)
            for name, data in ((qpath, qbytes), (apath, abytes)):
                members.append({'name': name, 'bytes': len(data), 'sha256': hashlib.sha256(data).hexdigest()})
            questions = {q['question_id']: q for q in json.loads(qbytes)['questions']}
            annotations = json.loads(abytes)['annotations']
            assert len(questions) == len(annotations)
            del qbytes, abytes
            count = collections.Counter()
            raw_types = collections.Counter()
            images = set()
            type_images = collections.defaultdict(set)
            objects = collections.defaultdict(lambda: collections.defaultdict(list))
            for a in annotations:
                image_id, kind = a['image_id'], a['question_type']
                images.add(image_id)
                raw_types[kind] += 1
                if kind not in ('color', 'counting'):
                    continue
                type_images[kind].add(image_id)
                q = questions[a['question_id']]
                assert q['image_id'] == image_id
                noun = object_name(q['question'], kind)
                if not noun:
                    continue
                count[f'lexically_matched_{kind}_qa'] += 1
                answers = [x['answer'] for x in a['answers']]
                values = [answer_value(x, kind) for x in answers]
                record = {'question_id': a['question_id'], 'question': q['question'],
                          'answers_original': answers, 'ans_source': a.get('ans_source'),
                          'canonical_answer': values[0] if len(set(values)) == 1 else None}
                objects[(image_id, noun)][kind].append(record)
            eligible = collections.defaultdict(list)
            complete_any = collections.defaultdict(list)
            for (image_id, noun), tasks in objects.items():
                if not all(tasks[k] for k in ('color', 'counting')):
                    continue
                complete_any[image_id].append(noun)
                count['matched_object_task_pairs_before_answer_checks'] += 1
                vals = {k: {q['canonical_answer'] for q in tasks[k]} for k in tasks}
                if any(len(v) != 1 or None in v for v in vals.values()):
                    count['object_task_pairs_rejected_answer_conflict_or_invalid'] += 1
                    continue
                color, number = next(iter(vals['color'])), next(iter(vals['counting']))
                if not 1 <= number <= 6:
                    count['object_task_pairs_rejected_count_outside_1_6'] += 1
                    continue
                distinct = {k: len({norm(q['question']) for q in tasks[k]}) for k in tasks}
                eligible[image_id].append({'object': noun, 'color': color, 'count': number,
                                           'distinct_question_wordings': distinct, 'qa': dict(tasks)})
                count['eligible_object_task_pairs'] += 1
            this_candidates = []
            same_count_or_color_images = 0
            for image_id, obs in sorted(eligible.items()):
                pairs = []
                for a, b in itertools.combinations(sorted(obs, key=lambda x: x['object']), 2):
                    if a['color'] == b['color'] or a['count'] == b['count']:
                        continue
                    pairs.append({'objects': [a, b], 'has_two_distinct_wordings_for_all_four_cells':
                                  all(n >= 2 for o in (a, b) for n in o['distinct_question_wordings'].values())})
                if pairs:
                    this_candidates.append({'tdiuc_image_id': image_id, 'split': split,
                                            'source_image': source_image(image_id), 'pairs': pairs})
                elif len(obs) >= 2:
                    same_count_or_color_images += 1
            count.update({'total_qa': len(annotations), 'total_images': len(images),
                          'images_with_any_color_qa': len(type_images['color']),
                          'images_with_any_counting_qa': len(type_images['counting']),
                          'images_with_both_types_any_referents': len(type_images['color'] & type_images['counting']),
                          'images_with_one_matched_object_before_answer_checks': len(complete_any),
                          'images_with_two_matched_objects_before_answer_checks': sum(len(v) >= 2 for v in complete_any.values()),
                          'images_with_one_eligible_object': len(eligible),
                          'images_with_two_eligible_objects': sum(len(v) >= 2 for v in eligible.values()),
                          'images_rejected_only_equal_color_or_count': same_count_or_color_images,
                          'four_cell_candidate_images': len(this_candidates),
                          'four_cell_candidate_object_pairs': sum(len(c['pairs']) for c in this_candidates),
                          'eight_question_candidate_images': sum(any(p['has_two_distinct_wordings_for_all_four_cells'] for p in c['pairs']) for c in this_candidates)})
            stats[split] = {'counts': dict(count), 'all_question_types': dict(raw_types),
                            'candidate_image_origins': dict(collections.Counter(c['source_image']['dataset'] for c in this_candidates)),
                            'candidate_images_with_a_pair_excluding_foods': sum(any(all(o['object'] not in FOODS for o in p['objects']) for p in c['pairs']) for c in this_candidates)}
            candidates.extend(this_candidates)
            all_images.update(images)
    outdir = Path(__file__).resolve().parent
    result = {'status': 'automatic lexical candidates only; no visual, trigger-region, or training-overlap verification',
              'rules': {'object_vocabulary': sorted(set(NOUN_MAP.values())), 'color_patterns': COLOR_PATTERNS,
                        'counting_patterns': COUNT_PATTERNS, 'colors': sorted(COLORS), 'counts': [1, 2, 3, 4, 5, 6],
                        'matching': 'same exact canonical COCO noun, image and task; reject multiple recognized answers or unsupported answers; two different nouns, different colors and different counts',
                        'two_wordings': 'two distinct normalized existing question strings per cell; not automatically verified semantic paraphrases',
                        'limitations': ['Conservative lexical lower bound; missed phrasing is not absence of questions.',
                                        'Singular color question may refer to one of multiple objects; uniform class color is unverified.',
                                        'COCO person is included in raw coverage but requires explicit color referent to be usable.',
                                        'Existing dataset answers may be incorrect; generated/transfer sources are retained.',
                                        'Dataset split is not proof of disjointness from Atlas training.',
                                        'TDIUC and Visual Genome overlap; counts must not be added across datasets.']},
              'by_split': stats,
              'totals': {'qa': sum(s['counts']['total_qa'] for s in stats.values()), 'unique_images': len(all_images),
                         'four_cell_candidate_images': len(candidates),
                         'eight_question_candidate_images': sum(s['counts']['eight_question_candidate_images'] for s in stats.values()),
                         'candidate_images_with_a_pair_excluding_foods': sum(s['candidate_images_with_a_pair_excluding_foods'] for s in stats.values()),
                         'best_existing_wording_slots_out_of_eight_histogram': dict(collections.Counter(max(sum(min(2, n) for o in p['objects'] for n in o['distinct_question_wordings'].values()) for p in c['pairs']) for c in candidates))},
              'analysis_wall_seconds': round(time.perf_counter() - start, 3)}
    source = {'official_page': 'https://kushalkafle.com/projects/tdiuc.html',
              'download_url': 'https://kushalkafle.com/data/TDIUC.zip', 'archive_bytes': archive.stat().st_size,
              'archive_sha256': hashlib.sha256(archive.read_bytes()).hexdigest(),
              'archive_path': str(archive), 'download_observed_seconds_approx': 8,
              'analyzed_at_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(), 'members': members,
              'image_id_mapping': 'Official setup.py: 8-digit IDs starting 9 use last 7 digits as VG ID; starting 8 are manual images; otherwise COCO ID.',
              'command': 'python3 tdiuc_coverage.py /tmp/atlas-material-audit-2026-09-24/TDIUC.zip'}
    for filename, data in [('tdiuc_coverage.json', result), ('tdiuc_candidates.json', candidates), ('tdiuc_source.json', source)]:
        (outdir / filename).write_text(json.dumps(data, indent=2, ensure_ascii=False) + '\n')
    print(json.dumps({'totals': result['totals'], 'by_split': stats, 'analysis_wall_seconds': result['analysis_wall_seconds']}, indent=2))


if __name__ == '__main__':
    main()
