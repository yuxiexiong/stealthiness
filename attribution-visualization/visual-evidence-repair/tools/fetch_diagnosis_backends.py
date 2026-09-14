"""Fetch only pinned official baseline files; verify every declared SHA256."""
import hashlib
import json
from pathlib import Path
import urllib.request

PROJECT = Path(__file__).resolve().parents[1]


def main():
    for method in ('cleansight', 'purmm'):
        root = PROJECT / 'external' / method
        manifest = json.loads((root / 'SOURCE.json').read_text())
        repo = manifest['repository'].removeprefix('https://github.com/')
        for entry in manifest['files']:
            target = root / entry['path']
            if target.exists():
                data = target.read_bytes()
            else:
                url = f"https://raw.githubusercontent.com/{repo}/{manifest['commit']}/{entry['path']}"
                with urllib.request.urlopen(url, timeout=60) as response:
                    data = response.read()
            if hashlib.sha256(data).hexdigest() != entry['sha256']:
                raise ValueError('pinned source differs: ' + str(target))
            if not target.exists():
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(data)
        print(method, manifest['commit'], 'verified', len(manifest['files']), 'files')


if __name__ == '__main__':
    main()
