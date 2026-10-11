"""Package the committed segmentation figures as one offline HTML report."""
import base64
import collections
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / 'source/attribution-microscope/runs/segmentation/coco-sam3-v1'
SNAPSHOT = 'e2da81db21d14da69d728e3351d8c42fbe29af5e'
RESULT_COMMIT = 'e9bb298393145bc8b53053c16b718d4776c0b179'


def read(path):
    return json.loads(path.read_text())


def main():
    manifest = read(SOURCE / 'manifest.json')
    records, hashes, event_counts = [], {}, collections.Counter()
    split_counts, split_events = collections.Counter(), collections.defaultdict(collections.Counter)
    question_ids, probe_indices = [], []
    checked_partitions = 0

    def jpeg(path):
        data = path.read_bytes()
        with Image.open(path) as im:
            assert im.width == 1920 and im.height > 22, path
            im.verify()
        hashes[path.relative_to(SOURCE).as_posix()] = hashlib.sha256(data).hexdigest()
        return 'data:image/jpeg;base64,' + base64.b64encode(data).decode('ascii')

    for row in sorted(manifest['images'], key=lambda x: min(x['p_core_idx'])):
        question_ids.extend(row['question_ids'])
        probe_indices.extend(row['p_core_idx'])
        if row.get('seg_split') not in {'dev', 'val', 'later'}:
            continue
        image_id, split = row['image_id'], row['seg_split']
        routes, final_labels = {}, None
        for route in 'ABC':
            meta_path = SOURCE / route / f'{image_id}.json'
            mask_path = meta_path.with_suffix('.npz')
            meta = read(meta_path)
            with np.load(mask_path, allow_pickle=False) as masks:
                labels = masks['labels']
            assert labels.shape == (row['height'], row['width']), mask_path
            assert np.issubdtype(labels.dtype, np.integer) and labels.min() >= 0
            areas = dict(zip(*np.unique(labels, return_counts=True)))
            registered = {r['label'] for r in meta['regions']}
            assert set(areas).issubset(registered | {0})
            for region in meta['regions']:
                area = region['area_final'] if route == 'B' else region['area']
                assert area == areas.get(region['label'], 0), (image_id, route, region)
            assert abs(meta['unassigned_fraction'] - float((labels == 0).mean())) < 1e-8
            checked_partitions += 1
            for path in (meta_path, mask_path):
                hashes[path.relative_to(SOURCE).as_posix()] = hashlib.sha256(path.read_bytes()).hexdigest()
            routes[route] = meta
            if route == 'C':
                final_labels = set(areas) - {0}
        final = routes['C']
        assert final['status'] == 'ok' and not final['errors']
        decisions = read(SOURCE / 'edits' / f'{image_id}.json')['decisions']
        counts = collections.Counter(e['event'] for e in final['edits'])
        event_counts.update(counts)
        split_counts[split] += 1
        split_events[split].update(counts)
        figure_dir = SOURCE / 'viz' / split
        zooms = []
        for event in final['edits']:
            if event['event'] not in {'add', 'replace'}:
                continue
            region = next(r for r in final['regions'] if event.get('sam') in r['source_ids'])
            path = figure_dir / f"{image_id}_zoom_{region['project_id']}.jpg"
            assert path.is_file(), path
            zooms.append({'image': jpeg(path), 'id': region['project_id'],
                          'category': region['category'], 'reason': event.get('reason', '')})
        regions = final['regions']
        records.append({
            'id': image_id, 'split': split, 'idx': row['p_core_idx'], 'questions': row['question_ids'],
            'width': row['width'], 'height': row['height'], 'sha': row['original_sha256'],
            'review': 'rule' if split == 'later' else 'model',
            'editor': final['editor'], 'counts': dict(counts), 'decisions': decisions,
            'regions': regions, 'image': jpeg(figure_dir / f'{image_id}.jpg'), 'zooms': zooms,
            'categories': sorted({r['category'] for r in regions if r['label'] in final_labels}),
            'route_summary': {key: {'regions': sum((r.get('area_final', r.get('area', 0))) > 0 for r in meta['regions']),
                                     'unassigned': meta['unassigned_fraction']} for key, meta in routes.items()},
        })

    excluded = [r for r in manifest['images'] if not r.get('seg_split')]
    forms = [read(p) for p in (SOURCE / 'quality/forms').glob('*.json')]
    assert len(records) == 188 and len(excluded) == 1
    assert len(probe_indices) == len(set(probe_indices)) == manifest['n_questions'] == 200
    assert set(probe_indices) == set(range(200))
    assert len(question_ids) == 200 and len(manifest['images']) == 189
    assert all(not form['reviewer'] and not form['reference_objects_signed'] for form in forms)
    assert len(forms) == 30
    assert (event_counts['add'], event_counts['replace'], event_counts['unresolved'], event_counts['reject']) == (109, 0, 252, 8)
    stats = {'questions': 200, 'unique': 189, 'completed': len(records), 'excluded': len(excluded),
             'changed_images': sum(bool(r['counts'].get('add', 0) + r['counts'].get('replace', 0)) for r in records),
             'unresolved_images': sum(bool(r['counts'].get('unresolved', 0)) for r in records),
             'events': dict(event_counts), 'splits': dict(split_counts),
             'split_events': {k: dict(v) for k, v in split_events.items()}}
    data = {'stats': stats, 'records': records, 'excluded': excluded,
            'source_commit': SNAPSHOT, 'result_commit': RESULT_COMMIT,
            'notes': (SOURCE / 'RUN_NOTES.md').read_text()}
    encoded = json.dumps(data, ensure_ascii=False, separators=(',', ':')).replace('<', '\\u003c')
    template = (ROOT / 'report_template.html').read_text()
    assert template.count('__DATA__') == 1
    output = ROOT / 'segmentation-results.html'
    output.write_text(template.replace('__DATA__', encoded))
    verification = {'source_commit': SNAPSHOT, 'result_commit': RESULT_COMMIT, 'statistics': stats,
                    'checked_label_partitions': checked_partitions, 'question_index_coverage': '0..199 exactly once',
                    'checked_figures': sum(1 + len(r['zooms']) for r in records),
                    'independent_quality_review': 'not completed; 30 unsigned forms',
                    'original_images': 'not included in Git; report reuses committed experiment JPEGs, 480px per column',
                    'rendering': 'source JPEG bytes embedded unchanged; no inference or mask editing',
                    'artifact_sha256': hashes, 'html_bytes': output.stat().st_size,
                    'html_sha256': hashlib.sha256(output.read_bytes()).hexdigest()}
    (ROOT / 'verification.json').write_text(json.dumps(verification, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({k: v for k, v in verification.items() if k != 'artifact_sha256'}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
