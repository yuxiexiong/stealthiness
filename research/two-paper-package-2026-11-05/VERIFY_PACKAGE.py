"""Verify frozen originals and online reading links without network or models."""
from pathlib import Path
import hashlib
import json
import re
from urllib.parse import unquote

root = Path(__file__).resolve().parent
manifest = json.loads((root / 'PACKAGE_MANIFEST.json').read_text())
digest = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
for entry in manifest['original_files']:
    assert digest(root / entry['package_path']) == entry['sha256'], entry['package_path']
links = re.compile(r'\]\((<[^>]+>|[^)]+)\)')
for entry in manifest['reading_copies']:
    original = (root / entry['original']).read_text()
    reading = (root / entry['package_path']).read_text().split('\n\n', 1)[1]
    assert links.sub('](LINK)', original) == links.sub('](LINK)', reading)
for i in range(1, 5):
    directory = root / f'originals/review/R{i}'
    for kind, filename in [('QUALITATIVE', 'qualitative.md'), ('SCORE', 'scores.json')]:
        lock = json.loads((directory / f'{kind}_LOCK.json').read_text())
        assert digest(directory / filename) == lock['sha256']
prefix = ('https://github.com/yuxiexiong/stealthiness/blob/'
          + manifest['github_branch'] + '/research/' + root.name + '/')
checked = 0
for path in [root / n for n in ['README.md', 'COMPLETE_READING.md', 'SOURCE_LINKS.md']] + list((root / 'read').rglob('*.md')):
    for target in links.findall(path.read_text()):
        target = target.strip('<>')
        assert not target.startswith(('/Users/', '/workspace/')), (path, target)
        if target.startswith(prefix):
            local = root / unquote(target[len(prefix):].split('#')[0])
            assert local.exists(), local
            checked += 1
print(json.dumps({'original_files_verified':len(manifest['original_files']),
                  'reading_copies_verified':len(manifest['reading_copies']),
                  'locked_review_files_verified':8,
                  'online_package_links_verified':checked}, ensure_ascii=False))
