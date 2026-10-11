"""Budget-normalized ASR measurement; no training or GPU allocation."""
import argparse
import collections
import csv
from fractions import Fraction
import gzip
import hashlib
import json
import math
from pathlib import Path

from jump_v1 import read
from lora_common import atomic_json

GAIN = .5
TOLERANCE = 1e-7  # Same tolerance as V1 for recorded float32 ASR ratios.
CRITERIA = {'budget_1pct': '0.01', 'budget_1p6pct': '0.016', 'legacy_20updates': None}
BUDGETS = {'Qwen3-8B': 1250, 'Qwen3-VL-8B': 1250, 'LLaVA-1.5-7B': 1250,
           'Falcon-Mamba-7B': 1250, 'SD3.5-Large': 1250, 'resnet50': 7040,
           'vit_b_16': 7040, 'Vim-S': 7040, 'Llama-3.1-8B-Instruct': 1250}
IDENTITY_FIELDS = ('probe_n', 'probe_scope', 'split', 'judge_id', 'probe_set_sha256')


def window_updates(budget, alpha='0.01'):
    """T is the frozen full schedule, never the bounded replay stopping update."""
    if isinstance(budget, bool) or not isinstance(budget, int) or budget < 1:
        raise ValueError('Positive integer full training budget required')
    ratio = Fraction(str(alpha))
    if not 0 < ratio <= 1:
        raise ValueError('Budget fraction must be in (0,1]')
    return budget * ratio.numerator // ratio.denominator


def measurement_contract():
    return dict(primary_fraction='0.01', sensitivity_fraction='0.016',
                minimum_absolute_gain_pp=50, legacy_window_updates=20,
                denominator='preregistered_full_training_optimizer_updates',
                actual_width_is_observed_upper_bound=True,
                normal_accuracy_is_selection_gate=False,
                independent_confirmation_uses_primary_selected_pair=True)


def checked_points(points, budget, probe_n=None, split=None):
    result, identities = [], {}
    for value in points:
        if isinstance(value, dict):
            step, asr = value['optimizer_step'], value['asr']
            for key in IDENTITY_FIELDS:
                if key in value:
                    if key in identities and identities[key] != value[key]:
                        raise ValueError('Cannot combine different probe identities: ' + key)
                    identities[key] = value[key]
        else:
            step, asr = value
        if isinstance(step, bool) or not isinstance(step, int) or not 0 <= step <= budget:
            raise ValueError('Optimizer updates must be integers inside the full budget')
        asr = float(asr)
        if not math.isfinite(asr) or not 0 <= asr <= 1:
            raise ValueError('Finite ASR measurements in [0,1] required')
        result.append((step, asr))
    if not result or any(a[0] >= b[0] for a, b in zip(result, result[1:])):
        raise ValueError('Nonempty strictly increasing measurements required; do not merge conflicts')
    for key, expected in [('probe_n', probe_n), ('split', split)]:
        if expected is not None and key in identities and identities[key] != expected:
            raise ValueError('Measurement identity differs from declared ' + key)
    if probe_n is not None and (isinstance(probe_n, bool) or not isinstance(probe_n, int) or probe_n < 1):
        raise ValueError('Positive fixed probe denominator required')
    return result


def summarize(points, budget, alpha='0.01', probe_n=None, split=None, complete=True, window=None):
    window_updates(budget)  # Validate the full-budget denominator even for legacy20.
    limit = window_updates(budget, alpha) if window is None else window
    if isinstance(limit, bool) or not isinstance(limit, int) or limit < 1:
        raise ValueError('Criterion must permit at least one optimizer update')
    points = checked_points(points, budget, probe_n, split)
    first = fastest = largest = None
    # ponytail: scan only bounded actual pairs; no interpolation or invented crossing.
    for index, a in enumerate(points):
        for b in points[index + 1:]:
            width = b[0] - a[0]
            if width > limit:
                break
            pair = dict(start=a[0], end=b[0], start_asr=a[1], end_asr=b[1],
                        updates=width, budget_fraction=width / budget, gain_pp=100 * (b[1] - a[1]))
            if largest is None or pair['gain_pp'] > largest['gain_pp']:
                largest = pair
            if b[1] - a[1] >= GAIN - TOLERANCE:
                if first is None or (pair['end'], width, pair['start']) < (first['end'], first['updates'], first['start']):
                    first = pair
                if fastest is None or (width, pair['end'], pair['start']) < (fastest['updates'], fastest['end'], fastest['start']):
                    fastest = pair
    gap = max([0] + [b[0] - a[0] for a, b in zip(points, points[1:])])
    full = complete and points[0][0] == 0 and points[-1][0] == budget
    status = ('confirmed' if first else 'not_observed_at_recorded_resolution'
              if full and gap <= limit else 'indeterminate')
    top, drop = 0., 0.
    for step, asr in points:
        if first and step >= first['end']:
            top = max(top, asr)
            drop = max(drop, 100 * (top - asr))
    return dict(status=status, full_training_budget=budget, alpha=str(alpha) if window is None else None,
                window_updates=limit, first_window=first, fastest_observed_window=fastest,
                minimum_observed_updates=fastest['updates'] if fastest else None,
                W50_budget_fraction_upper_bound=fastest['updates'] / budget if fastest else None,
                observed_width_is_upper_bound=True, maximum_window=largest,
                maximum_window_gain_pp=max(0., largest['gain_pp']) if largest else None,
                maximum_sampling_gap=gap, full_budget_endpoints_recorded=bool(full),
                initial_asr=points[0][1], peak_asr=max(a for _, a in points), final_asr=points[-1][1],
                measurements=len(points), largest_later_drop_pp=drop,
                first_update=points[0][0], last_update=points[-1][0],
                absence_is_only_at_recorded_resolution=True)


def compare_criteria(curve):
    budget = curve.get('full_training_budget', BUDGETS.get(curve['model']))
    if budget is None:
        raise ValueError('Unregistered model needs an explicit full_training_budget')
    return {name: summarize(curve['points'], budget, alpha or '0.01',
                probe_n=curve['probe_n'], split=curve.get('split'),
                complete=curve.get('full_budget_complete', True),
                window=20 if alpha is None else None) for name, alpha in CRITERIA.items()}


def select_replay_window(points, budget, alpha='0.01'):
    """First recorded coarse candidate; failure in this window cannot establish global absence."""
    limit = window_updates(budget, alpha)
    result = summarize(points, budget, alpha)
    candidate = result['first_window']
    if candidate is None:
        # Preserve the original <=20-update candidate locator for legacy T=1250 coarse curves.
        candidate = summarize(points, budget, window=20)['first_window']
    return dict(candidate=candidate, criterion_window_updates=limit,
                requires_dense_replay=bool(candidate and candidate['updates'] > limit),
                stop_after=min(budget, candidate['end'] + 20) if candidate else None,
                schedule_total=budget, no_candidate_does_not_prove_absence=True,
                fixed_primary_candidate_only=True)


def verification_level(curve):
    if curve.get('verification_level'):
        return curve['verification_level']
    if curve.get('verified_refinement') or curve['view'] == 'verified_dense':
        return 'strict_same_trajectory_verified_window'
    if curve.get('identity_validation'):
        return 'historical_identity_strength_not_newly_certified'
    return 'original_recorded_measurements'


def curve_key(curve):
    return tuple(curve[key] for key in ('model', 'seed', 'poison_rate', 'arm', 'probe_n', 'view'))


def boundary_validation_complete(gate):
    if not gate.get('passed') or gate.get('formal_trajectories') != 12:
        return False
    return bool(gate.get('all_required_replays_valid') or (
        gate.get('user_approved_direct_confirmation')
        and gate.get('approved_validation_replacement_complete')
        and gate.get('original_vim_replays_unverified')
        and gate.get('direct_bounded_confirmations') == 6))


def load_inputs(old_curve_file, boundary_root=None, require_complete=True):
    curves = read(old_curve_file)
    if boundary_root is not None:
        boundary_root = Path(boundary_root)
        gate = read(boundary_root / 'results/complete.json')
        if require_complete and not boundary_validation_complete(gate):
            raise ValueError('Vim and Llama formal trajectories and required replays must complete first')
        curves += [curve for curve in read(boundary_root / 'results/curves.json.gz')
                   if not (curve['model'] == 'Vim-S' and curve['view'] == 'independent_bounded900')]
        for family in ('vim', 'llama'):
            details = read(boundary_root / family / 'results/results.json')
            if family == 'vim':
                for row in details.get('direct_confirmations', []):
                    confirmation = row['verification']['fixed_confirmation']
                    if confirmation is None:
                        continue
                    curves.append(dict(model='Vim-S', seed=row['seed'], arm=row['arm'],
                        poison_rate=.05 if row['arm'] == 'poison' else 0., probe_n=900,
                        view='fixed_primary_pair_confirmation900',
                        points=[[confirmation['start'], confirmation['start_asr']],
                                [confirmation['end'], confirmation['end_asr']]],
                        full_training_budget=7040, full_budget_complete=False,
                        not_an_additional_seed=True, not_same_trajectory_as_original=True,
                        verification_level='fixed_primary_selected_pair_confirmation',
                        split='independent900', primary_pair_selected_before_confirmation=True))
            for row in details['dense']:
                verification = row['verification']
                if not verification.get('passed'):
                    raise ValueError('Independent confirmation requires accepted same-trajectory replay')
                if family == 'vim':
                    pair = verification['independent_confirmation_pair']
                    values = dict(verification['points'])
                    model, denominator, rate = 'Vim-S', 900, .05
                    points = [[s, values[s]] for s in pair]
                else:
                    confirmation = verification['independent_confirmation']
                    model, denominator, rate = 'Llama-3.1-8B-Instruct', 140, .01
                    points = [[v['optimizer_step'], v['asr']] for v in confirmation['points']]
                curves.append(dict(model=model, seed=row['seed'], poison_rate=rate,
                    arm='poison', probe_n=denominator, view='independent_confirmation', points=points,
                    not_an_additional_seed=True, verification_level='fixed_primary_selected_pair_confirmation',
                    split='independent'+str(denominator), primary_pair_selected_before_confirmation=True))
    if len({curve_key(c) for c in curves}) != len(curves):
        raise ValueError('Input record views overlap; do not count a seed twice')
    return curves


def apply_verified_points(curves, model, seed, verified):
    """Merge only an accepted replay into its original identical-denominator primary view."""
    if not verified.get('passed') or verified.get('probe_n') != 60:
        raise ValueError('Strict original60 replay gate required')
    target = next(c for c in curves if c['model'] == model and c['seed'] == seed
                  and c['arm'] == 'poison' and c['view'] == 'primary' and c['probe_n'] == 60)
    merged = dict(target['points'])
    for step, asr in checked_points(verified['points'], target.get('full_training_budget', BUDGETS[model]), 60):
        if step in merged and abs(merged[step] - asr) >= TOLERANCE:
            raise ValueError('Accepted replay conflicts with an original anchor')
        merged[step] = asr
    target.update(points=sorted(merged.items()), verification_level='strict_same_trajectory_verified_window')
    return target


def cost_ledger(receipts):
    """Only top-level new phase receipts; embedded/native costs are not added again."""
    seen, result = set(), []
    for item in receipts:
        key = item['cost_id']
        if key in seen:
            raise ValueError('Duplicate incremental cost receipt: ' + key)
        seen.add(key)
        wall = item.get('wall_seconds')
        if wall is not None and (not math.isfinite(wall) or wall < 0):
            raise ValueError('Finite nonnegative wall time or unknown null required')
        result.append({k: item[k] for k in ('cost_id', 'parent_cost_id', 'phase', 'family', 'model', 'seed', 'wall_seconds',
                       'includes_nested_receipts', 'source_download_install_cost_recounted') if k in item})
    top_level = [v for v in result if v.get('parent_cost_id') not in seen]
    total = sum(v.get('wall_seconds') or 0 for v in top_level)
    unknown = sum(v.get('wall_seconds') is None for v in top_level)
    for row in result:
        for field in ('cost_id', 'parent_cost_id'):
            value = row.get(field)
            if isinstance(value, str) and '/' in value:
                row[field] = 'receipt_' + hashlib.sha256(value.encode()).hexdigest()[:24]
    return dict(receipts=result, known_incremental_wall_seconds=total,
                unknown_receipts=unknown,
                borrowed_prior_cost_recounted=False, overlapping_wall_times_not_additive=True)


def plot_normalized(curves, output):
    """Existing V1 matplotlib style, normalized budget axes, separate denominators/views."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size': 9, 'axes.spines.top': False, 'axes.spines.right': False,
                         'svg.fonttype': 'none', 'pdf.fonttype': 42})
    groups = collections.defaultdict(list)
    for item in curves:
        groups[(item['model'], item['poison_rate'], item['arm'], item['probe_n'], item['view'])].append(item)
    for (model, rate, arm, denominator, view), items in groups.items():
        fig, axes = plt.subplots(1, 2, figsize=(11, 4), layout='constrained')
        starts, ends, margins = [], [], []
        for item in items:
            budget = item['full_training_budget']
            x, y = zip(*item['points'])
            for ax in axes:
                ax.plot([100*s/budget for s in x], [100*a for a in y], '.-', ms=2, lw=1,
                        label=str(item['seed']))
            pair = item['criteria']['budget_1pct']['first_window']
            if pair:
                starts.append(100*pair['start']/budget); ends.append(100*pair['end']/budget)
                margins.append(100*item['criteria']['budget_1pct']['window_updates']/budget)
        if starts:
            margin = max(margins)
            axes[1].set_xlim(max(0, min(starts)-margin), min(100, max(ends)+margin))
        else:
            counts = collections.Counter(i['criteria']['budget_1pct']['status'] for i in items)
            label = f'No recorded 1% qualifying pair\nIndeterminate views: {counts["indeterminate"]}/{len(items)}'
            axes[1].text(.5, .5, label, transform=axes[1].transAxes, ha='center')
        for ax, title in zip(axes, ['Recorded trajectory', 'First 1%-budget jump region']):
            ax.set(xlabel='Optimizer updates / preregistered full budget (%)',
                   ylabel=f'ASR (%) — fixed n={denominator}', ylim=(-2, 102), title=title)
        axes[0].legend(ncol=3, fontsize=7)
        fig.suptitle(f'{model} | poison={rate:g} | {arm} | {view}')
        stem = Path(output) / f'{model}_{rate:g}_{arm}_{denominator}_{view}_budget_percent'
        fig.savefig(str(stem)+'.png', dpi=160); fig.savefig(str(stem)+'.svg')
        plt.close(fig)


def export(curves, output, incremental_costs=(), make_plots=False):
    output = Path(output); output.mkdir(parents=True, exist_ok=True)
    expanded, rows, groups = [], [], collections.defaultdict(list)
    if len({curve_key(c) for c in curves}) != len(curves):
        raise ValueError('Duplicate record views')
    for curve in curves:
        item = {key: curve[key] for key in ('model', 'seed', 'poison_rate', 'arm', 'probe_n', 'view', 'points')}
        item.update(full_training_budget=curve.get('full_training_budget', BUDGETS.get(curve['model'])),
                    verification_level=verification_level(curve),
                    not_an_additional_seed=curve.get('not_an_additional_seed', False))
        for key in ('split', 'measurement_note', 'poison_set_seed', 'primary_pair_selected_before_confirmation'):
            if key in curve: item[key] = curve[key]
        item['criteria'] = compare_criteria(item)
        expanded.append(item)
        groups[(item['model'], item['poison_rate'], item['arm'], item['probe_n'], item['view'])].append(item)
        for name, summary in item['criteria'].items():
            first = summary['first_window']
            rows.append({**{k: item[k] for k in ('model', 'seed', 'poison_rate', 'arm', 'probe_n', 'view')},
                'full_training_budget': item['full_training_budget'], 'criterion': name,
                'window_updates': summary['window_updates'], 'status': summary['status'],
                'first_start': first['start'] if first else None, 'first_end': first['end'] if first else None,
                'first_gain_pp': first['gain_pp'] if first else None,
                **{k: summary[k] for k in ('minimum_observed_updates', 'W50_budget_fraction_upper_bound',
                   'maximum_window_gain_pp', 'maximum_sampling_gap', 'initial_asr', 'peak_asr', 'final_asr',
                   'largest_later_drop_pp')}, 'verification_level': item['verification_level'],
                'not_an_additional_seed': item['not_an_additional_seed']})
    with gzip.open(output / 'curves.json.gz', 'wt') as stream: json.dump(expanded, stream, ensure_ascii=False)
    with open(output / 'inventory.csv', 'w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    cohorts = []
    for key, values in groups.items():
        for criterion in CRITERIA:
            counts = collections.Counter(v['criteria'][criterion]['status'] for v in values)
            cohorts.append(dict(model=key[0], poison_rate=key[1], arm=key[2], probe_n=key[3], view=key[4],
                 criterion=criterion, total_record_views=len(values), confirmed=counts['confirmed'],
                 not_observed_at_recorded_resolution=counts['not_observed_at_recorded_resolution'],
                 indeterminate=counts['indeterminate'],
                 not_an_additional_seed=all(v['not_an_additional_seed'] for v in values)))
    atomic_json(output / 'cohorts.json', cohorts)
    atomic_json(output / 'incremental_costs.json', cost_ledger(incremental_costs))
    atomic_json(output / 'measurement_contract.json', measurement_contract())
    if make_plots:
        plot_normalized(expanded, output)
    return expanded


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--curves', type=Path, required=True)
    parser.add_argument('--boundary-root', type=Path)
    parser.add_argument('--incremental-costs', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--plot', action='store_true')
    args = parser.parse_args()
    inputs = [args.curves]
    if args.boundary_root: inputs += [args.boundary_root / 'results/curves.json.gz', args.boundary_root / 'results/complete.json']
    costs = read(args.incremental_costs) if args.incremental_costs else []
    if args.incremental_costs: inputs.append(args.incremental_costs)
    export(load_inputs(args.curves, args.boundary_root), args.output, costs, args.plot)
    atomic_json(args.output / 'source_identity.json', {'input_sha256': [
        hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs], 'raw_prompt_generations_exported': False})


if __name__ == '__main__': main()
