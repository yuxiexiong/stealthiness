"""Freeze fresh EditCLEVR scenes; reuse official extraction and A's fact checks."""
import argparse
import importlib.util
import json
from pathlib import Path
import random
import shutil
import sys
import tarfile

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
import prepare_toy48_data as source
from repair.data import read_jsonl, load_units
from tools import prepare_diagnosis_a as original
from tools.prepare_diagnosis_b import METADATA_SHA256, STRATA
from repair.visual_probe import object_annotations

SEED = 20260914


def prepare_calibration(path, output):
    """Freeze 200 existing normal calibration questions, without model selection."""
    path, output = Path(path).resolve(), Path(output).resolve()
    units = load_units(path, {'calibration'})
    rows = [dict(node, cluster_id=unit['cluster_id']) for unit in units for node in unit['nodes']]
    random.Random(SEED).shuffle(rows)
    rows = rows[:200]
    if len(rows) != 200:
        raise ValueError('not enough existing independent calibration questions')
    output.mkdir(parents=True, exist_ok=False)
    (output / 'images').mkdir()
    images = {}
    for row in rows:
        image = Path(row['image'])
        sha = source.digest(image)
        relative = 'images/' + sha + image.suffix
        if relative not in images:
            shutil.copyfile(image, output / relative)
            images[relative] = sha
        row['image'] = relative
    manifest = {'schema': 'diagnosis-external-calibration-v1', 'image_condition': 'clean',
        'source': original.identity(path), 'source_manifest': original.identity(path.with_suffix('.manifest.json')),
        'seed': SEED, 'selection': 'uniform without replacement over existing calibration questions',
        'rows': rows, 'images': images, 'model_outputs_used': False,
        'independent_scene_count': len({r['cluster_id'] for r in rows})}
    source.write_json(output / 'calibration.json', manifest)
    return manifest


def select(splits, excluded, count=96):
    """Balanced random scene order, frozen before any model output is requested."""
    if count != 96:
        raise ValueError('the confirmation screening pool is fixed at 96 scenes')
    rng, groups, seen = random.Random(SEED), [], set(excluded)
    for split, stratum in STRATA:
        rows = sorted((r for r in splits[split] if source.eligible(r, split == 'test_cogent')),
                      key=lambda r: r['before_image'])
        rng.shuffle(rows)
        chosen = []
        for row in rows:
            cluster = 'editclevr-' + str(row['generation']['base_scene_seed'])
            if cluster in seen:
                continue
            seen.add(cluster)
            chosen.append({'split': 'test', 'stratum': stratum, 'source': row,
                           'main_endpoint': rng.randrange(2)})
            if len(chosen) == 32:
                break
        if len(chosen) != 32:
            raise ValueError('insufficient unseen source scenes: ' + split)
        groups.append(chosen)
    return [group[i] for i in range(32) for group in groups]


def build(args):
    metadata, root, out = map(lambda p: Path(p).resolve(), (args.metadata, args.sources, args.output))
    if out.exists():
        raise FileExistsError('prepared output exists; never overwrite experiment inputs')
    if source.digest(metadata) != METADATA_SHA256:
        raise ValueError('pinned EditCLEVR metadata differs')
    excluded, history = set(), []
    for path in map(Path, args.history):
        rows = json.loads(path.read_text())
        excluded.update('editclevr-' + str(r['source']['generation']['base_scene_seed']) for r in rows)
        history.append(original.identity(path))
    for path in map(Path, args.exclude_cases):
        excluded.update(r['cluster_id'] for r in read_jsonl(path))
        history.append(original.identity(path))
    with tarfile.open(metadata) as archive:
        selected = select(json.load(archive.extractfile('splits.json')), excluded)
    lock = {'schema': 'diagnosis-comparison-source-v1', 'seed': SEED, 'revision': source.REVISION,
            'selected': selected, 'excluded_cluster_ids': sorted(excluded), 'history': history,
            'metadata': original.identity(metadata), 'model_outputs_used': False}
    root.mkdir(parents=True, exist_ok=True)
    lock_path = root / 'selection-lock.json'
    if lock_path.exists() and json.loads(lock_path.read_text()) != lock:
        raise ValueError('source lock changed; use a new directory')
    source.write_json(lock_path, lock)
    source.write_json(root / 'selected-scenes.json', selected)
    if getattr(args, 'archive_cache', None):
        # Reuse HF's maintained resumable/Xet downloader after HTTP stream truncation.
        from concurrent.futures import ThreadPoolExecutor
        from huggingface_hub import hf_hub_download
        names = ('editclevr_atomic_id.tar.gz', 'editclevr_hard_distractor.tar.gz', 'editclevr_cogent_ood.tar.gz')
        def download(name):
            path = hf_hub_download(repo_id='torux/EditCLEVR', repo_type='dataset',
                revision=source.REVISION, filename=name, cache_dir=args.archive_cache)
            print('Cached ' + name, flush=True)
            return Path(path)
        with ThreadPoolExecutor(max_workers=3) as pool:
            archives = list(pool.map(download, names))
        if len({p.parent for p in archives}) != 1:
            raise ValueError('HF snapshot paths disagree')
        source.HUB = archives[0].parent.as_uri() + '/'
    transfers = source.extract_subset(root, selected)
    engine_path = Path(args.engine)
    if source.digest(engine_path) != original.ENGINE_SHA256:
        raise ValueError('official question engine changed')
    shutil.copyfile(engine_path, root / 'question_engine.py')
    shutil.copyfile(engine_path.with_name('question_engine.LICENSE'), root / 'question_engine.LICENSE')
    source.write_fact_pool(root, selected, root / 'question_engine.py')
    receipt = original.build(argparse.Namespace(facts=root / 'test-facts.jsonl', output=out,
        config=args.config, construction_manifest=args.construction_manifest))
    # The old color-count question changes with the color edit. Use the SAME
    # official scene/count operator to supply the plan's invariant fallback fact.
    spec = importlib.util.spec_from_file_location('comparison_clevr_engine', root / 'question_engine.py')
    engine = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(engine)
    endpoint_by_cluster = {'editclevr-' + str(r['source']['generation']['base_scene_seed']): r['main_endpoint']
                           for r in selected}
    cases = read_jsonl(out / 'cases.jsonl')
    for case in cases:
        case['main_endpoint'] = endpoint_by_cluster[case['cluster_id']]
        for node in case['nodes']:
            if node['family'] != 'count':
                continue
            scene = json.loads(Path(node['visibility']['sources']['scene']['path']).read_text())
            program = [{'type': 'scene', 'inputs': []}, {'type': 'count', 'inputs': [0]}]
            answer = str(engine.answer_question({'nodes': program}, None, scene, cache_outputs=False))
            if answer != str(len(scene['objects'])):
                raise ValueError('total-count program disagrees with the source scene')
            for key in ('clean_row', 'observed_row'):
                node[key].update(question='How many objects are there?', answer=answer,
                                 answers=[str(i) for i in range(11)])
            node['question_program'] = program
            node['companion_invariance'] = 'total_object_count_unchanged_by_color_edit'
        counts = [n['clean_row']['answer'] for n in case['nodes'] if n['family'] == 'count']
        if len(set(counts)) != 1:
            raise ValueError('object count changed across a color-only pair')
        case['annotations'] = {'coordinate_system': 'model_processed_pixels',
            'marker_box': [0, 0, 64, 64], 'source': 'canonical CLIP crop; original fixed marker'}
        for node in case['nodes']:
            node['annotations'] = (object_annotations(node, receipt['preprocess'])
                                   if node['visibility']['eligible'] else {'status': 'ineligible'})
    path = out / 'cases.jsonl'
    path.write_text(''.join(json.dumps(c, ensure_ascii=False) + '\n' for c in cases))
    receipt.update(stage='diagnosis-comparison', cases_sha256=source.digest(path),
        selection_lock=original.identity(lock_path), excluded_cluster_ids=sorted(excluded),
        frozen_order=list(endpoint_by_cluster), source_seed=SEED, archive_reads=transfers,
        selection='96 pre-model random scenes; screen in frozen order, retain at most 32',
        companion_count='official scene/count; total objects, not color-filtered objects',
        plan_sha256=source.digest(PROJECT / 'DIAGNOSIS_NEXT_STAGE_PLAN_2026-09-14.md'))
    source.write_json(out / 'receipt.json', receipt)
    return receipt


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('metadata', 'sources', 'output', 'engine', 'config', 'construction-manifest'):
        p.add_argument('--' + name, required=True)
    p.add_argument('--history', nargs='+', required=True)
    p.add_argument('--exclude-cases', nargs='+', required=True)
    p.add_argument('--archive-cache', help='reuse the installed HF resumable downloader and a local cache')
    result = build(p.parse_args(argv))
    print(json.dumps({k: result[k] for k in ('stage', 'cases', 'nodes', 'geometry_eligible_cases')}))


if __name__ == '__main__':
    main()
