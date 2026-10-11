"""Audit the downloaded atlas against the original page and frozen selections."""
import base64
import hashlib
import io
import json
import re
from collections import Counter
from pathlib import Path
from PIL import Image

ROOT = Path(__file__).resolve().parent
SRC = ROOT / 'source/attribution-microscope'

def read_page(path):
    s = path.read_text()
    data = {k: json.loads(v) for k, v in re.findall(
        r'<script type="application/json" id="j_([^"]+)">(.*?)</script>', s, re.S)}
    return s, data

s, data = read_page(ROOT / 'server-attribution-atlas.html')
old_path = ROOT / 'original-attribution-atlas.html'
_, old = read_page(old_path)
arms = data['index']['arms']
for a, stems in old['index']['arms'].items():
    for stem, v in stems.items():
        assert v == arms[a][stem], (a, stem, 'original entries changed')
assert old['atlas'] == data['atlas'], 'original photos changed'
for phase in ('phase2', 'phase2b', 'phase2b/fill', 'phase2b/fill2'):
    for name in ('finished', 'status'):
        d = json.loads((SRC / f'runs/{phase}/{name}.json').read_text())
        assert not d['failed'], (phase, d)
        assert not d.get('pending') and not d.get('running'), (phase, d)
for f in (SRC / 'runs/phase2b/fill').glob('same_*.json'):
    assert json.loads(f.read_text())['pass'], f

sel = json.loads((SRC / 'runs/phase2b/selection.json').read_text())
expected = {f'P-1.0-D2@s{x}' for x in sel['chosen']}
expected |= {f'RETRAIN-A-D@s{x}' for x in sel['null_grid']}
sel1 = json.loads((SRC / 'runs/phase2/selection.json').read_text())
expected |= {f'P-1.0-D@s{x}' for x in sel1['chosen'] + list(range(180, 281, 20)) + [440,460,480]}
for head, steps in [('CLEAN', [79,158,316,632]), ('P-1.0', [79,158,316,632]),
                    ('P-1.0-DF', [345,350,355]), ('P-1.0-D2F', [285,290,295]),
                    ('P-1.0-DG', [356,357,358,359]), ('P-1.0-D2G', [291,292,293,294])]:
    expected |= {f'{head}@s{x}' for x in steps}
expected |= {'P-0.38', 'P-0.4', 'P-0.42', 'P-0.4-ps2', 'P-0.4-ps3'}
curves = json.loads((SRC / 'supplement/phase2b/figs/curves2b.json').read_text())
ids = {e[0] for e in arms['P-1.0@s79']['p_core_trig']['entries']}
assert len(ids) == curves['samples'] == 60
for a in expected:
    for col, instruments in [('clean', 'A'), ('trig', 'AB')]:
        es = arms[a][f'p_core_{col}']['entries']
        actual = {(e[0], e[1], e[2]) for e in es}
        sample_ids = range(200) if a == 'P-0.4' else ids
        instruments = 'AB' if a == 'P-0.4' else instruments
        wanted = {(i, sc, ins) for i in sample_ids for sc in ('T1','T2','T3','C') for ins in instruments}
        assert actual == wanted and len(es) == len(wanted), (a,col)
    if a.startswith('P-'):
        assert a in data['meta']['asr'], a

idmap = json.loads(re.search(r'const IDMAP=(.*?);\n', s).group(1))
imgs = dict(re.findall(r'<img id="([^"]+)" src="data:image/[^;]+;base64,([^"]+)">', s))
for path, image_id in idmap.items():
    im = Image.open(io.BytesIO(base64.b64decode(imgs[image_id])))
    im.load()
    if path.endswith('.png'):
        arm = Path(path).stem
        assert im.size == (593, sum(len(v['entries']) for v in arms[arm].values())), path
for a in arms:
    assert f'data/{a}.png' in idmap
for atlas in data['atlas'].values():
    assert 'data/' + atlas['file'] in idmap

counts = Counter(p['path'] for p in curves['points'])
assert counts == {'seed1':27,'seed2':20,'dose':8,'e3':2}, counts
checked = 0
for ins in 'AB':
    table = (SRC / f'supplement/phase2b/figs/table2b_{ins}.html').read_text()
    rows = re.findall(r'<tr>(.*?)</tr>', table)[1:]
    assert len(rows) == 57
    for row, point in zip(rows, curves['points']):
        cells = re.findall(r'<td>(.*?)</td>', row)
        assert cells[1].startswith(point['tag'])
        assert point['asr'] == data['meta']['asr'][point['tag']]
        for cell, metric in zip(cells[5:7], ('trigger_share','offtrigger_cos')):
            m = point['m60'][f'{ins}_T2'][metric]
            assert cell == f"{m['median']:.3f} [{m['q25']:.3f}, {m['q75']:.3f}] n={m['n']}"
            checked += 1
        assert point['ref'] in arms
report = dict(source_commit='e2da81db21d14da69d728e3351d8c42fbe29af5e',
              server_html_sha256=hashlib.sha256(s.encode()).hexdigest(),
              original_arms=len(old['index']['arms']), total_arms=len(arms),
              original_rows=sum(len(v['entries']) for a in old['index']['arms'].values() for v in a.values()),
              total_rows=sum(len(v['entries']) for a in arms.values() for v in a.values()),
              expected_supplement_arms=len(expected), trajectory_samples=len(ids),
              curve_points=dict(counts), verified_table_statistics=checked,
              validated_image_assets=len(idmap), original_entries_preserved=True,
              scope='Artifact integrity and completeness; not an independent GPU reproduction.')
(ROOT / 'audit.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
print(json.dumps(report,ensure_ascii=False,indent=2))
