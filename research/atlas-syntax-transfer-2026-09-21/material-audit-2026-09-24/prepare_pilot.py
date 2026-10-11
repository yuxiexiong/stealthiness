"""Freeze a deterministic blinded pilot; never replace images based on answers."""
import hashlib
import json
from pathlib import Path
import time
import urllib.request

ROOT = Path(__file__).resolve().parent
PILOT = ROOT / 'pilot'


def main():
    started = time.time()
    PILOT.mkdir(exist_ok=True)
    (PILOT / 'images').mkdir(exist_ok=True)
    candidates = json.loads((ROOT / 'vg_candidates.json').read_text())
    candidates = [c for c in candidates if c['image_id'] != 4 and any(p['unqualified_referents'] for p in c['pairs'])]
    candidates.sort(key=lambda c: hashlib.sha256(('atlas-pilot-v1:' + str(c['image_id'])).encode()).hexdigest())
    blind, gold, failed = [], [], []
    for c in candidates:
        if len(blind) == 12:
            break
        image_id = c['image_id']
        dest = PILOT / 'images' / f'vg_{image_id}.jpg'
        try:
            if not dest.exists():
                with urllib.request.urlopen(c['image_url'], timeout=30) as response:
                    dest.write_bytes(response.read())
            from PIL import Image
            with Image.open(dest) as im:
                im.verify()
        except Exception as exc:
            failed.append({'image_id': image_id, 'url': c['image_url'], 'error': repr(exc)})
            continue
        pair = sorted((p for p in c['pairs'] if p['unqualified_referents']), key=lambda p: tuple(p['objects']))[0]
        sample_id = f'P{len(blind)+1:02d}'
        row = {'sample_id': sample_id, 'image_path': str(dest), 'image_id': image_id, 'questions': []}
        reference = {'sample_id': sample_id, 'image_id': image_id, 'coco_id': c['coco_id'], 'image_url': c['image_url'],
                     'image_sha256': hashlib.sha256(dest.read_bytes()).hexdigest(), 'facts': []}
        for oi, obj in enumerate(pair['objects']):
            for task in ('color', 'count'):
                q = sorted(c['objects'][obj]['qas'][task], key=lambda q: q['qa_id'])[0]
                qid = f'{sample_id}-{oi+1}-{task}'
                row['questions'].append({'id': qid, 'object': obj, 'task': task, 'original': q['question']})
                reference['facts'].append({'id': qid, 'object': obj, 'task': task, 'question': q['question'],
                                            'reference_answer': q['normalized_answer'], 'original_answer': q['answer'], 'qa_id': q['qa_id']})
        blind.append(row)
        gold.append(reference)
    assert len(blind) == 12, f'Only {len(blind)} images were accessible'
    payloads = {'writer_input.json': blind, 'reference_sealed.json': gold,
                'selection.json': {'seed': 'atlas-pilot-v1:', 'eligible_candidate_images': len(candidates),
                                   'selected_ids': [b['image_id'] for b in blind], 'download_failures': failed,
                                   'started_unix': started, 'ended_unix': time.time(),
                                   'selection': 'SHA256 order, first unqualified pair, smallest original qa_id; no visual or answer-result filtering'}}
    for name, data in payloads.items():
        (PILOT / name).write_text(json.dumps(data, ensure_ascii=False, indent=2))
    print(json.dumps(payloads['selection.json'], ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
