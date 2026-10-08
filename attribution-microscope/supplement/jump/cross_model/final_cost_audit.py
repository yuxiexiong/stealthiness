"""Independent stdlib cost audit; never edits source scientific artifacts."""
import argparse
import hashlib
import json
import math
from pathlib import Path

FIELDS = ('cost_id', 'parent_cost_id', 'phase', 'family', 'model', 'seed',
          'includes_nested_receipts', 'includes_nested_native_receipt',
          'source_asset_cost_recounted', 'borrowed_assets_not_recounted',
          'excludes_separately_recorded_CPU_tests_and_clip_receipt')

def masked(value):
    if not isinstance(value, str):
        raise ValueError('Cost identifiers must be strings')
    return 'receipt_' + hashlib.sha256(value.encode()).hexdigest()[:24] if '/' in value or '\\' in value else value

def receipt(row, origin=''):
    result = {k: masked(row[k]) if isinstance(row[k], str) else row[k] for k in FIELDS if k in row and isinstance(row[k], (str, int, float, bool, type(None)))}
    result['cost_id'] = masked(row.get('cost_id') or origin)
    if not result['cost_id']:
        raise ValueError('Cost ID required')
    if result.get('parent_cost_id'):
        result['parent_cost_id'] = masked(result['parent_cost_id'])
    wall = row.get('wall_seconds')
    if wall is None:
        wall = row.get('incremental_wall_seconds')
    if wall is not None and (isinstance(wall, bool) or not isinstance(wall, (int, float)) or not math.isfinite(wall) or wall < 0):
        raise ValueError('Finite nonnegative wall time or null required')
    result['wall_seconds'] = wall
    return result

def merge(base, current, prior, scope, observations=()):
    rows = {}
    def add(raw, origin='', restore=False):
        new = receipt(raw, origin)
        key = new['cost_id']
        if key in rows:
            old = rows[key]
            for field in set(old) & set(new):
                if old[field] is not None and new[field] is not None and old[field] != new[field]:
                    raise ValueError('Conflicting duplicate cost ID: ' + key)
            if not restore and old != new:
                raise ValueError('Conflicting duplicate cost ID: ' + key)
            rows[key] = {**old, **{k: v for k, v in new.items() if v is not None}}
        else:
            rows[key] = new
        return key
    for row in base:
        add(row)
    recovered = 0.
    for origin, row in current:
        prior_wall = rows.get(masked(row.get('cost_id') or origin), {}).get('wall_seconds')
        add(row, origin, restore=True)
        if prior_wall is None and row.get('wall_seconds') is None and row.get('incremental_wall_seconds') is not None:
            recovered += row['incremental_wall_seconds']
    transport = 0.
    for origin, row in prior:
        key = masked(row.get('cost_id') or origin)
        unseen = key not in rows
        add(row, origin, restore=True)
        if unseen and row.get('phase') == 'public_immutable_asset_transport':
            transport += row.get('wall_seconds') or 0.
    for key in rows:
        visited = set()
        while key in rows:
            if key in visited:
                raise ValueError('Cyclic parent cost relationship')
            visited.add(key); key = rows[key].get('parent_cost_id')
    top = [r for r in rows.values() if r.get('parent_cost_id') not in rows]
    return dict(receipts=list(rows.values()), known_command_wall_sum_seconds=sum(r['wall_seconds'] or 0 for r in top),
                unknown_receipts=sum(r['wall_seconds'] is None for r in top), unique_elapsed_wall_seconds=None,
                elapsed_overlap_resolution='unavailable_without_complete_start_end_intervals',
                restored_incremental_wall_seconds=recovered, added_transport_wall_seconds=transport,
                prior_cost_scope=scope, scientific_publication_pending=True,
                observations=[dict({k: masked(v) if isinstance(v, str) else v for k, v in row.items()
                                    if k in ('phase', 'observed_gap_seconds', 'timestamp_resolution_seconds') and isinstance(v, (str, int, float))},
                                   included_in_command_wall_sum=False, GPU_training_cost=False) for row in observations])

def read(path):
    return json.loads(path.read_text())

def scope_only(scope):
    return {k: v for k, v in scope.items() if isinstance(v, (int, float, bool, type(None))) and
            ('wall_seconds' in k or 'counted_once' in k or 'recounted' in k or k.startswith('unknown_'))}

def inputs(root, metadata=None):
    current, prior, scope = [], [], {}
    if metadata:
        value = read(metadata)
        for record in value['records']:
            path, row = record['path'], record.get('metadata')
            if not isinstance(row, dict):
                continue
            if path.startswith(value['root'] + '/costs/'):
                current.append((path, row))
            elif row.get('phase') == 'public_immutable_asset_transport' or path.endswith('/costs/cpu_asset_prepare_failure_001.json'):
                prior.append((path, row))
            elif path.endswith('/prior_cost_scope.json'):
                scope.update(scope_only(row))
        return current, prior, scope
    current = [(str(p), read(p)) for p in sorted((root / 'costs').glob('*.json'))]
    native = [(p, read(p)) for p in sorted((root / 'clip').rglob('cost_receipt.json'))]
    aliases = {}
    for path, row in native:
        generated = 'clip_' + str(path.relative_to(root / 'clip'))
        for alias in (str(path), row.get('cost_id')):
            if alias:
                aliases[alias] = generated; aliases[masked(alias)] = generated
    for path, row in native:
        row = dict(row, cost_id='clip_' + str(path.relative_to(root / 'clip')))
        if row.get('parent_cost_id') in aliases:
            row['parent_cost_id'] = aliases[row['parent_cost_id']]
        current.append((str(path), row))
    for attempt in read(root / 'manifest.json').get('prior_attempts', []):
        for path in sorted((Path(attempt) / 'costs').glob('*.json')):
            row = read(path)
            if row.get('phase') == 'public_immutable_asset_transport' or path.name == 'cpu_asset_prepare_failure_001.json':
                prior.append((str(path), row))
    scope = scope_only(read(root.parent / '20261008_vim_direct_r02/results/prior_cost_scope.json'))
    return current, prior, scope

def run(mode, root, output, metadata=None):
    if mode == 'final':
        if metadata:
            raise ValueError('Final requires original source receipts')
        complete = read(root / 'results/complete.json')
        if not (complete.get('passed') and complete.get('all_four_stages_finished')):
            raise ValueError('Source scientific stages must finish first')
        base = read(root / 'results/final/incremental_costs.json')['receipts']
    else:
        base = []
    current, prior, scope = inputs(root, metadata)
    observations = []
    observation_path = root / 'monitoring/t5_s1004_to_s1005_wait_interval.json'
    if not metadata and observation_path.exists():
        observations.append(read(observation_path))
    result = merge(base, current, prior, scope, observations)
    source = root.resolve()
    destination = output.resolve()
    if destination == source or source in destination.parents:
        raise ValueError('Output must be outside source root')
    output.mkdir(parents=True, exist_ok=True)
    names = ['incremental_costs.json', 'prior_cost_scope.json', 'complete.json'] if mode == 'final' else ['preflight.json']
    if any((output / name).exists() for name in names):
        raise ValueError('Audit output already exists')
    def write(name, value):
        with (output / name).open('x') as stream:
            stream.write(json.dumps(value, indent=2, allow_nan=False) + '\n')
    if mode == 'final':
        write('incremental_costs.json', result)
        write('prior_cost_scope.json', scope)
        write('complete.json', dict(passed=True, cost_audit_completed=True, scientific_publication_pending=True))
    else:
        write('preflight.json', dict(passed=True, final_cost_audit_completed=False, **result))
    return result

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=['preflight', 'final'])
    parser.add_argument('--source-root', required=True, type=Path)
    parser.add_argument('--output-dir', required=True, type=Path)
    parser.add_argument('--metadata-file', type=Path)
    args = parser.parse_args()
    result = run(args.mode, args.source_root, args.output_dir, args.metadata_file)
    print(json.dumps({k: result[k] for k in ('known_command_wall_sum_seconds', 'unknown_receipts',
                                           'restored_incremental_wall_seconds', 'added_transport_wall_seconds')}))
