#!/usr/bin/env python3
"""Prepare fixed prompts and independent blinded annotations; never invent labels."""
import argparse
import hashlib
import io
import json
import random
import re
import secrets
import urllib.request
from pathlib import Path

SCHEMA = 't2i-diagnosis-v1'
SOURCE_COMMIT = '4aa404212eb5d06e5adbcd9cee696c750d0d25a5'
SOURCE_URL = f'https://raw.githubusercontent.com/Karine-Huang/T2I-CompBench/{SOURCE_COMMIT}/examples/dataset/color_val.txt'
SOURCE_SHA256 = '1634259756dbc77d13093d907d414480080ec9790a8c97ad09efaea7b2534f2d'
COLOR_STATES = ('red', 'blue', 'green', 'yellow', 'orange', 'purple', 'pink', 'brown', 'black', 'white', 'gray', 'gold', 'silver', 'other', 'mixed', 'absent', 'multiple', 'unjudgeable')
COLORS = COLOR_STATES[:13]
# Preregistered ordinary, separable object nouns; excludes sky/grass and other stuff.
OBJECTS = ('apple', 'backpack', 'banana', 'bear', 'bench', 'bird', 'boat', 'book', 'bowl', 'cake', 'car', 'cat', 'chair', 'clock', 'cow', 'cup', 'dog', 'elephant', 'giraffe', 'horse', 'orange', 'sheep', 'suitcase', 'train', 'vase')


def sha(data):
    return hashlib.sha256(data).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, value):
    with Path(path).open('x') as f:
        json.dump(value, f, ensure_ascii=False, indent=2)
        f.write('\n')


def study(args):
    if not 1 <= args.limit <= 24:
        raise ValueError('limit must be between 1 and 24')
    data = Path(args.source).read_bytes() if args.source else urllib.request.urlopen(SOURCE_URL, timeout=30).read()
    if sha(data) != SOURCE_SHA256:
        raise ValueError('source hash differs from the pinned official color_val.txt')
    rows = data.decode('utf-8').splitlines()
    excluded = {'syntax': 0, 'color': 0, 'object_whitelist': 0, 'identical_object_or_color': 0, 'over_limit': 0}
    selected = []
    for line, prompt in enumerate(rows, 1):
        match = re.fullmatch(r'a ([a-z]+) ([a-z]+) and a ([a-z]+) ([a-z]+)', prompt)
        if not match:
            excluded['syntax'] += 1
            continue
        c1, n1, c2, n2 = match.groups()
        if c1 not in COLORS or c2 not in COLORS:
            excluded['color'] += 1
        elif n1 not in OBJECTS or n2 not in OBJECTS:
            excluded['object_whitelist'] += 1
        elif c1 == c2 or n1 == n2:
            excluded['identical_object_or_color'] += 1
        else:
            selected.append((line, prompt, c1, n1, c2, n2))
    # Sample across noun pairs; the source's first rows all start with a bench.
    selected.sort(key=lambda row: (sha(('selection-v1|' + '|'.join(sorted((row[3], row[5])))).encode()), row[0]))
    excluded['over_limit'] = max(0, len(selected) - args.limit)
    selected = selected[:args.limit]
    cases = []
    for line, prompt, c1, n1, c2, n2 in selected:
        group = '|'.join(sorted((n1, n2)))
        split = 'exploration' if int(sha(group.encode())[:8], 16) % 3 == 0 else 'test'
        probe, test = [c for c in COLORS if c not in (c1, c2)][:2]
        for seed in (0, 1):
            conditions = [{'id': 'base', 'role': 'base', 'source': None, 'color': None}]
            conditions += [{'id': f'{role}_{obj}', 'role': role, 'source': obj, 'color': color} for role, color in [('probe', probe), ('test', test)] for obj in ('a', 'b')]
            conditions.append({'id': 'sham', 'role': 'test', 'source': None, 'color': None})
            cases.append({'id': f'color-{line:03d}-s{seed}', 'group': group, 'split': split, 'prompt': prompt, 'seed': seed, 'source_line': line, 'objects': [{'id': 'a', 'name': n1, 'color': c1}, {'id': 'b', 'name': n2, 'color': c2}], 'conditions': conditions})
    result = {'schema': SCHEMA, 'source': {'commit': SOURCE_COMMIT, 'url': SOURCE_URL, 'sha256': SOURCE_SHA256, 'license': 'MIT, Copyright (c) 2023 HKU', 'total_lines': len(rows), 'selected_prompts': len(selected), 'excluded': excluded}, 'selection': {'object_whitelist': OBJECTS, 'colors': COLORS, 'rule': 'Strict distinct whitelist nouns/colors; sort by SHA256(selection-v1| + sorted noun pair joined by |), then source line; take first limit.', 'split_rule': 'unordered noun pair joined by |; first 8 hex digits of SHA256 modulo 3: 0 exploration, otherwise test', 'seeds': [0, 1], 'limit': args.limit}, 'cases': cases}
    if not cases:
        raise ValueError('no eligible prompts')
    write_json(args.output, result)
    print(json.dumps({'cases': len(cases), 'prompts': len(selected), 'groups': len({c['group'] for c in cases}), 'splits': {s: sum(c['split'] == s for c in cases) for s in ('exploration', 'test')}, 'excluded': excluded}))


def checked_inputs(study_path, render_path):
    study_path, render_path = Path(study_path), Path(render_path)
    s, r = read_json(study_path), read_json(render_path)
    if s.get('schema') != SCHEMA or r.get('study_sha256') != sha(study_path.read_bytes()):
        raise ValueError('study schema or render study hash mismatch')
    expected = {}
    for case in s['cases']:
        if {x['id'] for x in case['objects']} != {'a', 'b'} or len(case['objects']) != 2:
            raise ValueError('study must contain exactly two facts a/b per case')
        for condition in case['conditions']:
            key = (case['id'], condition['id'])
            if key in expected:
                raise ValueError('duplicate case/condition in study')
            expected[key] = {x['id']: x['name'] for x in case['objects']}
    images = {}
    for item in r['records']:
        key = (item['case_id'], item['condition_id'])
        if key not in expected or key in images:
            raise ValueError('unknown or duplicate rendered condition')
        relative = Path(item['image'])
        path = (render_path.parent / relative).resolve()
        if relative.is_absolute() or not path.is_relative_to(render_path.parent.resolve()):
            raise ValueError('render image must remain within its render directory')
        if sha(path.read_bytes()) != item['image_sha256']:
            raise ValueError(f'image hash mismatch: {key}')
        images[key] = (path, item['image_sha256'])
    if images.keys() != expected.keys():
        raise ValueError('render must cover every study case and condition exactly once')
    return expected, images, sha(study_path.read_bytes()), sha(render_path.read_bytes())


def anonymous_png(data):
    """Re-encode the pixel image with Pillow, without source text/EXIF metadata."""
    from PIL import Image
    with Image.open(io.BytesIO(data)) as image:
        if image.format != 'PNG':
            raise ValueError('annotation images must be PNG')
        clean = image.convert('RGBA' if 'A' in image.mode or 'transparency' in image.info else 'RGB')
        clean.info.clear()
        output = io.BytesIO()
        clean.save(output, format='PNG')
    return output.getvalue()


HTML = '''<!doctype html><html lang="en"><meta charset="utf-8"><title>Image annotation</title>
<style>body{font:18px system-ui;max-width:920px;margin:2em auto}img{max-width:100%;max-height:65vh}label{display:block;margin:1em 0}button,select,input{font:inherit}aside{padding:1em;background:#eee}</style>
<h1>Visible object color</h1><p>Work independently. For each named object, report the visible color in this image only. Do not infer a requested color. absent: no instance visible; multiple: more than one instance of that object; mixed: one instance has no single dominant color; other: visible color outside the list; unjudgeable: cannot reliably decide.</p>
<label>Your annotator ID <input id="who" autocomplete="off"></label><label><input type="checkbox" id="independent"> I completed this independently, without seeing prompts, operations, other raters, or diagnostic outputs.</label>
<aside id="count"></aside><img id="image" alt="Image to annotate"><div id="questions"></div>
<button id="previous">Previous</button> <button id="next">Next</button> <button id="download">Download complete JSONL</button><p id="message" role="status"></p>
<script>
const items=__ITEMS__, states=__STATES__, mappingHash=__HASH__;
let position=0, dirty=false; const answers=items.map(()=>({a:'',b:''}));
window.addEventListener('beforeunload',event=>{if(dirty){event.preventDefault();event.returnValue='';}});
function show(){const row=items[position];document.getElementById('count').textContent=`Image ${position+1} / ${items.length}`;document.getElementById('image').src=row.file;const q=document.getElementById('questions');q.replaceChildren();for(const [fact,name] of Object.entries(row.objects)){const label=document.createElement('label');label.textContent=`What is the visible color state of the ${name}? `;const select=document.createElement('select');for(const value of ['',...states]){const option=document.createElement('option');option.value=value;option.textContent=value||'Select an answer';select.append(option);}select.value=answers[position][fact];select.onchange=()=>{answers[position][fact]=select.value;dirty=true;};label.append(select);q.append(label);}}
document.getElementById('previous').onclick=()=>{position=Math.max(0,position-1);show();};document.getElementById('next').onclick=()=>{position=Math.min(items.length-1,position+1);show();};
document.getElementById('download').onclick=()=>{const who=document.getElementById('who').value.trim();if(!who||!document.getElementById('independent').checked||answers.some(x=>!x.a||!x.b)){document.getElementById('message').textContent='Enter an ID, confirm independent annotation, and complete both questions for every image.';return;}const rows=items.map((x,i)=>({image_id:x.image_id,image_sha256:x.image_sha256,mapping_sha256:mappingHash,annotator_id:who,independent_blind:true,answers:answers[i]}));const url=URL.createObjectURL(new Blob([rows.map(x=>JSON.stringify(x)).join('\\n')+'\\n'],{type:'application/x-ndjson'}));const a=document.createElement('a');a.href=url;a.download='annotations.jsonl';a.click();dirty=false;URL.revokeObjectURL(url);document.getElementById('message').textContent='Downloaded. Send the JSONL to the independent study operator.';};show();
</script></html>'''


def annotation(args):
    expected, images, study_hash, render_hash = checked_inputs(args.study, args.render)
    public, private = Path(args.output).resolve(), Path(args.private_map).resolve()
    if private.is_relative_to(public) or public.exists() or private.exists():
        raise ValueError('use a new public directory and a new private map outside it')
    rows, blobs = [], {}
    for key, (path, original_hash) in images.items():
        image_id = secrets.token_hex(16)
        blob = anonymous_png(path.read_bytes())
        blobs[image_id] = blob
        rows.append({'image_id': image_id, 'case_id': key[0], 'condition_id': key[1], 'original_image_sha256': original_hash, 'image_sha256': sha(blob), 'file': f'images/{image_id}.png', 'objects': expected[key]})
    random.SystemRandom().shuffle(rows)
    public.mkdir(parents=True)
    (public / 'images').mkdir()
    private.parent.mkdir(parents=True, exist_ok=True)
    mapping = {'schema': SCHEMA, 'study_sha256': study_hash, 'render_sha256': render_hash, 'public_directory': str(public), 'images': rows}
    write_json(private, mapping)
    mapping_hash = sha(private.read_bytes())
    items = [{k: row[k] for k in ('image_id', 'image_sha256', 'file', 'objects')} for row in rows]
    for image_id, data in blobs.items():
        (public / 'images' / f'{image_id}.png').write_bytes(data)
    payload = json.dumps(items, ensure_ascii=False).replace('<', '\\u003c')
    (public / 'index.html').write_text(HTML.replace('__ITEMS__', payload).replace('__STATES__', json.dumps(COLOR_STATES)).replace('__HASH__', json.dumps(mapping_hash)))
    with (public / 'template.jsonl').open('x') as f:
        for item in items:
            f.write(json.dumps({'image_id': item['image_id'], 'image_sha256': item['image_sha256'], 'mapping_sha256': mapping_hash, 'annotator_id': None, 'independent_blind': False, 'answers': {'a': None, 'b': None}}) + '\n')
    (public / 'README.txt').write_text('Open index.html locally. Two distinct people must annotate independently and each download their own JSONL. No answers are prefilled. Keep the page open until downloading; answers are not saved automatically. The independent operator must keep the private mapping outside this directory and never distribute it, the study manifest, render manifest, original prompts, operations, or diagnostic outputs.\n')
    print(json.dumps({'images': len(rows), 'public_directory': str(public), 'private_map': str(private), 'mapping_sha256': mapping_hash}))


def read_annotations(path, mapped, mapping_hash, partial=False):
    rows, identities = {}, set()
    for line in Path(path).read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        image_id = row['image_id']
        identity = row.get('annotator_id')
        if image_id not in mapped or image_id in rows:
            raise ValueError('unknown or duplicate annotation image_id')
        if not isinstance(identity, str) or not identity.strip() or identity != identity.strip() or row.get('independent_blind') is not True:
            raise ValueError('each row must declare an independent annotator ID')
        if row.get('mapping_sha256') != mapping_hash or row.get('image_sha256') != mapped[image_id]['image_sha256']:
            raise ValueError('annotation mapping/image hash mismatch')
        answers = row['answers']
        if not isinstance(answers, dict) or not answers or not set(answers) <= {'a', 'b'} or (not partial and set(answers) != {'a', 'b'}):
            raise ValueError('annotation must cover the required facts')
        if any(value not in COLOR_STATES for value in answers.values()):
            raise ValueError('invalid or missing color-state answer; use gray, not grey')
        rows[image_id] = answers
        identities.add(identity)
    if len(identities) != 1 or (not partial and rows.keys() != mapped.keys()):
        raise ValueError('one annotator per file and complete image coverage required')
    return rows, identities.pop()


def gold(args):
    expected, images, study_hash, render_hash = checked_inputs(args.study, args.render)
    mapping_path = Path(args.private_map)
    mapping, mapped, seen = read_json(mapping_path), {}, set()
    if mapping.get('study_sha256') != study_hash or mapping.get('render_sha256') != render_hash:
        raise ValueError('private map study/render hash mismatch')
    public = Path(mapping['public_directory']).resolve()
    for row in mapping['images']:
        key = (row['case_id'], row['condition_id'])
        if key not in images or key in seen or row['image_id'] in mapped or row['objects'] != expected[key]:
            raise ValueError('private map condition/fact coverage mismatch')
        relative = Path(row['file'])
        path = (public / relative).resolve()
        if relative.is_absolute() or not path.is_relative_to(public):
            raise ValueError('private map anonymous image path invalid')
        original_path, original_hash = images[key]
        if row['original_image_sha256'] != original_hash or sha(path.read_bytes()) != row['image_sha256'] or sha(anonymous_png(original_path.read_bytes())) != row['image_sha256']:
            raise ValueError('private map source/anonymous image hash mismatch')
        mapped[row['image_id']] = row
        seen.add(key)
    if seen != expected.keys():
        raise ValueError('private map must cover every rendered condition')
    mapping_hash = sha(mapping_path.read_bytes())
    first, id1 = read_annotations(args.annotations[0], mapped, mapping_hash)
    second, id2 = read_annotations(args.annotations[1], mapped, mapping_hash)
    if id1 == id2:
        raise ValueError('two distinct independent annotator IDs required')
    disagreements = {image_id: {fact for fact in ('a', 'b') if first[image_id][fact] != second[image_id][fact]} for image_id in mapped}
    disagreements = {image_id: facts for image_id, facts in disagreements.items() if facts}
    adjudicated, adjudicator = {}, None
    if disagreements:
        if not args.adjudication:
            raise ValueError(f'{sum(map(len, disagreements.values()))} fact disagreements require an independent third annotator adjudication JSONL, with only disputed image/fact entries: {json.dumps({k: sorted(v) for k, v in disagreements.items()})}')
        adjudicated, adjudicator = read_annotations(args.adjudication, mapped, mapping_hash, partial=True)
        if adjudicator in (id1, id2) or {k: set(v) for k, v in adjudicated.items()} != disagreements:
            raise ValueError('independent third ID must adjudicate exactly all disputed image/fact entries')
    elif args.adjudication:
        raise ValueError('no disagreements: do not provide an adjudication file')
    answers, repeated = [], {}
    for image_id, row in mapped.items():
        facts = dict(first[image_id])
        facts.update(adjudicated.get(image_id, {}))
        # Identical image bytes and the same object questions must give identical facts.
        prior = repeated.setdefault(row['image_sha256'], {})
        for fact, state in facts.items():
            name = row['objects'][fact]
            if name in prior and prior[name] != state:
                raise ValueError('identical images received inconsistent final facts; independent re-review required')
            prior[name] = state
        answers.append({'case_id': row['case_id'], 'condition_id': row['condition_id'], 'facts': facts})
    paths = [Path(p) for p in args.annotations]
    provenance = {'status': 'independent_complete', 'source': 'independent_blind_annotation', 'annotators': [id1, id2], 'adjudicator': adjudicator, 'annotation_sha256': [sha(p.read_bytes()) for p in paths], 'adjudication_sha256': sha(Path(args.adjudication).read_bytes()) if args.adjudication else None, 'mapping_sha256': mapping_hash, 'fact_count': 2 * len(mapped), 'disagreement_count': sum(map(len, disagreements.values())), 'agreement_fraction': 1 - sum(map(len, disagreements.values())) / (2 * len(mapped)), 'independence_boundary': 'Identity and independent-blind declarations checked; software cannot verify that different IDs are different humans or that declarations are honest.'}
    write_json(args.output, {'schema': SCHEMA, 'study_sha256': study_hash, 'render_sha256': render_hash, 'provenance': provenance, 'answers': sorted(answers, key=lambda x: (x['case_id'], x['condition_id']))})
    print(json.dumps({'images': len(mapped), 'facts': 2 * len(mapped), 'disagreements': provenance['disagreement_count'], 'output': str(args.output)}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    p = commands.add_parser('study')
    p.add_argument('--source', help='Local byte-exact copy of the fixed official color_val.txt; otherwise download the pinned URL')
    p.add_argument('--limit', type=int, default=24)
    p.add_argument('--output', required=True)
    p.set_defaults(run=study)
    p = commands.add_parser('annotation')
    p.add_argument('--study', required=True)
    p.add_argument('--render', required=True)
    p.add_argument('--output', required=True, help='New public annotator directory')
    p.add_argument('--private-map', required=True, help='New operator-only file outside the public directory')
    p.set_defaults(run=annotation)
    p = commands.add_parser('gold')
    p.add_argument('--study', required=True)
    p.add_argument('--render', required=True)
    p.add_argument('--private-map', required=True)
    p.add_argument('--annotations', nargs=2, required=True)
    p.add_argument('--adjudication', help='Third independent rater, exactly disputed image/fact entries, same JSONL fields')
    p.add_argument('--output', required=True)
    p.set_defaults(run=gold)
    args = parser.parse_args()
    try:
        args.run(args)
    except (ValueError, KeyError, OSError) as error:
        parser.exit(2, f'error: {error}\n')


if __name__ == '__main__':
    main()
