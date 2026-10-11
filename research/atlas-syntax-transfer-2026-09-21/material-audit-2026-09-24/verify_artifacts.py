"""Recheck sealed inputs, raw references, sample selection and cross-source IDs."""
import hashlib
import json
from pathlib import Path
import zipfile

ROOT = Path(__file__).resolve().parent
P = ROOT / 'pilot'
CACHE = Path('/tmp/atlas-material-audit-2026-09-24')


def read(path):
    return json.loads(path.read_text())


def archive(name, key=None):
    with zipfile.ZipFile(CACHE / name) as z:
        value = json.loads(z.read(z.namelist()[0]))
    return value[key] if key else value


def main():
    freeze = read(P / 'freeze.json')
    checks = {}
    for field, name in [('input_sha256', 'answer_input.json'),
                        ('writer_output_sha256', 'writer_output.json'),
                        ('vg_reference_sha256', 'reference_sealed.json'),
                        ('calibration_reference_sha256', 'calibration_selection_sealed.json')]:
        assert hashlib.sha256((P / name).read_bytes()).hexdigest() == freeze[field]
    checks['sealed_inputs_unchanged'] = True
    candidates = read(ROOT / 'vg_candidates.json')
    eligible = [c for c in candidates if c['image_id'] != 4 and any(p['unqualified_referents'] for p in c['pairs'])]
    eligible.sort(key=lambda c: hashlib.sha256(('atlas-pilot-v1:' + str(c['image_id'])).encode()).hexdigest())
    selected = read(P / 'reference_sealed.json')
    assert [c['image_id'] for c in eligible[:12]] == [c['image_id'] for c in selected]
    checks['vg_selection_matches_frozen_rule'] = True
    vgqa = {q['qa_id']: q for r in archive('question_answers.json.zip') for q in r['qas']}
    for s in selected:
        path = P / 'images' / f"vg_{s['image_id']}.jpg"
        assert hashlib.sha256(path.read_bytes()).hexdigest() == s['image_sha256']
        for f in s['facts']:
            q = vgqa[f['qa_id']]
            assert q['question'] == f['question'] and q['answer'] == f['original_answer']
            assert q['image_id'] == s['image_id']
    checks['48_vg_original_labels_and_12_image_hashes_verified'] = True
    qs = archive('v2_Questions_Val_mscoco.zip', 'questions')
    anns = archive('v2_Annotations_Val_mscoco.zip', 'annotations')
    qi, ai = {q['question_id']: q for q in qs}, {a['question_id']: a for a in anns}
    for s in read(P / 'calibration_selection_sealed.json')['selected']:
        for f in s['tasks'].values():
            q, a = qi[f['question_id']], ai[f['question_id']]
            assert q['image_id'] == s['image_id'] and q['question'] == f['question']
            assert len(a['answers']) == 10
            assert all(x['answer'] == f['reference_answer'] for x in a['answers'])
    checks['8_calibration_questions_have_10_of_10_original_human_agreement'] = True
    meta = {x['image_id']: x.get('coco_id') for x in archive('image_data.json.zip')}

    def vg_key(image_id):
        return ('coco', meta[image_id]) if meta.get(image_id) else ('vg', image_id)

    vg = {vg_key(c['image_id']) for c in candidates}
    td = set()
    for c in read(ROOT / 'tdiuc_candidates.json'):
        s = c['source_image']
        td.add(('coco', s['image_id']) if s['dataset'] == 'COCO' else vg_key(s['image_id']))
    vqa = {('coco', c['image_id']) for c in read(ROOT / 'vqa_candidates.json')}
    union = vg | td | vqa
    val = {q['image_id'] for q in qs}
    combined = {'vg_candidate_images': len(vg), 'tdiuc_candidate_images': len(td),
                'vqa_val_candidate_images': len(vqa), 'vg_tdiuc_overlap_images': len(vg & td),
                'union_candidate_images': len(union), 'union_with_coco_id': sum(k == 'coco' for k, _ in union),
                'union_in_official_coco_val2014': sum(k == 'coco' and i in val for k, i in union)}
    old = read(ROOT / 'combined_coverage.json')
    assert all(old[k] == v for k, v in combined.items()), (old, combined)
    checks['cross_source_dedup_reproduced'] = combined
    inputs = read(P / 'answer_input.json')
    ids = {q['id'] for s in inputs for q in s['questions']}
    assert len(ids) == 104 and len(inputs) == 16
    for name in ('answer_a.json', 'answer_b.json'):
        rows = read(P / name)['items']
        assert len(rows) == 104 and {r['id'] for r in rows} == ids
    checks['both_agents_answered_all_104_unique_questions'] = True
    (ROOT / 'audit_checks.json').write_text(json.dumps(checks, ensure_ascii=False, indent=2))
    print(json.dumps(checks, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
