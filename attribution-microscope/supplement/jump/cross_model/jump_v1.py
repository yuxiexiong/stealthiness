"""ASR-only V1: a >=50 percentage-point increase within <=20 updates."""
import argparse
import collections
import csv
import gzip
import json
import math
from pathlib import Path

from lora_common import atomic_json

WINDOW, GAIN = 20, .5


def read(path):
    with (gzip.open if str(path).endswith('.gz') else open)(path, 'rt') as stream:
        return json.load(stream)


def summarize(points):
    points = [(int(s), float(a)) for s, a in points]
    if not points or any(s < 0 or not math.isfinite(a) or not 0 <= a <= 1 for s, a in points):
        raise ValueError('Need finite ASR measurements in [0,1] at nonnegative updates')
    if any(a[0] >= b[0] for a, b in zip(points, points[1:])):
        raise ValueError('Updates must be strictly increasing; conflicting points cannot be merged')
    pairs, qualified = [], []
    # ponytail: bounded scan is enough for these fixed trajectories; no rolling framework.
    for i, a in enumerate(points):
        for b in points[i + 1:]:
            if b[0] - a[0] > WINDOW:
                break
            p = {'start': a[0], 'end': b[0], 'start_asr': a[1], 'end_asr': b[1],
                 'updates': b[0] - a[0], 'gain_pp': 100 * (b[1] - a[1])}
            pairs.append(p)
            if b[1] - a[1] >= GAIN - 1e-7:  # source summaries sometimes use float32 ratios
                qualified.append(p)
    first = min(qualified, key=lambda p: (p['end'], p['updates'])) if qualified else None
    fastest = min(qualified, key=lambda p: (p['updates'], p['end'])) if qualified else None
    largest = max(pairs, key=lambda p: p['gain_pp']) if pairs else None
    later = [p for p in points if first and p[0] >= first['end']]
    drop = max([0.] + [100 * (a[1] - b[1]) for i, a in enumerate(later) for b in later[i + 1:]])
    return {'status': 'observed' if first else 'not_observed_at_available_resolution',
            'first_window': first, 'fastest_observed_window': fastest,
            'observed_width_is_upper_bound': True, 'maximum_20_update_window': largest,
            'maximum_sampling_gap': max([0] + [b[0] - a[0] for a, b in zip(points, points[1:])]),
            'initial_asr': points[0][1], 'peak_asr': max(a for _, a in points),
            'final_asr': points[-1][1], 'measurements': len(points),
            'largest_later_drop_pp': drop, 'first_update': points[0][0], 'last_update': points[-1][0]}


def modern_curves(results):
    curves = []

    def add(model, seed, rate, arm, n, points, source, **extra):
        curves.append({'model': model, 'seed': int(seed), 'poison_rate': rate, 'arm': arm,
                       'probe_n': n, 'view': 'primary', 'points': points, 'source': str(source), **extra})

    path = results / '20261005_dose5/results.json'
    d = read(path)
    for model in ('llm', 't2i'):
        for seed, item in d[model].items():
            add('Qwen3-8B' if model == 'llm' else 'SD3.5-Large', seed, item['poison_rate'],
                'poison', 60, item['points'], path)
    path = results / '20261005_dose10/results.json'
    d = read(path)
    add('SD3.5-Large', d['seed'], .1, 'poison', 60, [(p['step'], p['asr']) for p in d['curve']], path,
        measurement_note='BLIP proxy; external Parti prompts; 100-update sampling')
    for tag, rate, path in [('falcon5', .05, results / '20261005_falcon5/results.json'),
                            ('falcon10', .1, results / '20261007_final/falcon10/results.json.gz'),
                            ('falcon15', .15, results / '20261007_final/falcon15/results.json.gz')]:
        for item in read(path)['runs']:
            add('Falcon-Mamba-7B', item['seed'], rate, 'poison', 60,
                [(m['optimizer_step'], m.get('discovery', m)['asr']) for m in item['metrics']], path)
    path = results / '20261007_final/classification10/results.json.gz'
    d = read(path)
    for item in d['runs']:
        add(item['model'], item['seed'], .05 if item['arm'] == 'poison' else 0., item['arm'], 180,
            [(m['optimizer_step'], m['triggered']['asr']) for m in item['primary']], path)
    for item in d['dense']:
        name = Path(item['path']).name
        model, seed = name.rsplit('_s', 1)
        add(model, seed, .05, 'poison', 900,
            [(m['optimizer_step'], m['triggered']['asr']) for m in item['metrics']], path,
            view='verified_dense', not_an_additional_seed=True)
    return curves


def plot(curves, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size': 9, 'axes.spines.top': False, 'axes.spines.right': False,
                         'svg.fonttype': 'none', 'pdf.fonttype': 42})
    groups = collections.defaultdict(list)
    for item in curves:
        if len(item['points']) > 1:
            groups[(item['model'], item['poison_rate'], item['arm'], item['probe_n'], item['view'])].append(item)
    for (model, rate, arm, n, view), items in groups.items():
        fig, axes = plt.subplots(1, 2, figsize=(11, 4), layout='constrained')
        starts, ends = [], []
        for item in items:
            x, y = zip(*item['points'])
            for ax in axes:
                ax.plot(x, [100 * a for a in y], '.-', ms=2, lw=1, label=str(item['seed']))
            w = item['v1']['first_window']
            if w:
                starts.append(w['start']); ends.append(w['end'])
        if starts:
            axes[1].set_xlim(max(0, min(starts) - 20), max(ends) + 20)
        else:
            axes[1].text(.5, .5, 'No V1 jump observed', transform=axes[1].transAxes, ha='center')
        for ax, title in zip(axes, ['Recorded trajectory', 'First-jump region']):
            ax.set(xlabel='Optimizer update', ylabel=f'ASR (%) — fixed n={n}', ylim=(-2, 102), title=title)
        axes[0].legend(ncol=3, fontsize=7)
        fig.suptitle(f'{model} | poison={rate:g} | {arm} | {view}')
        stem = output / f'{model}_{rate:g}_{arm}_{n}_{view}'
        fig.savefig(str(stem) + '.png', dpi=160)
        fig.savefig(str(stem) + '.svg')
        plt.close(fig)


def export(curves, output, make_plots=False):
    output.mkdir(parents=True, exist_ok=True)
    rows, cohorts = [], collections.defaultdict(list)
    for item in curves:
        item = dict(item, v1=summarize(item['points']))
        w, f = item['v1']['first_window'], item['v1']['fastest_observed_window']
        j = item['v1']['maximum_20_update_window']
        rows.append({**{k: item[k] for k in ['model', 'seed', 'poison_rate', 'arm', 'probe_n', 'view']},
                     'status': item['v1']['status'], 'first_start': w['start'] if w else None,
                     'first_end': w['end'] if w else None, 'first_gain_pp': w['gain_pp'] if w else None,
                     'minimum_observed_updates': f['updates'] if f else None,
                     'maximum_20_update_gain_pp': max(0., j['gain_pp']) if j else None,
                     'maximum_sampling_gap': item['v1']['maximum_sampling_gap'],
                     'largest_later_drop_pp': item['v1']['largest_later_drop_pp']})
        cohorts[(item['model'], item['poison_rate'], item['arm'], item['probe_n'], item['view'])].append(item)
    expanded = [item for values in cohorts.values() for item in values]
    with gzip.open(output / 'curves.json.gz', 'wt') as stream:
        json.dump(expanded, stream, ensure_ascii=False)
    with open(output / 'inventory.csv', 'w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    frequencies = [{'model': key[0], 'poison_rate': key[1], 'arm': key[2], 'probe_n': key[3], 'view': key[4],
                    'observed': sum(i['v1']['status'] == 'observed' for i in values), 'total': len(values),
                    'not_an_additional_seed': key[4] != 'primary'} for key, values in cohorts.items()]
    atomic_json(output / 'cohorts.json', frequencies)
    if make_plots:
        plot(expanded, output)
    return expanded


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results', type=Path, required=True)
    parser.add_argument('--legacy', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--plot', action='store_true')
    args = parser.parse_args()
    export(modern_curves(args.results) + read(args.legacy), args.output, args.plot)
