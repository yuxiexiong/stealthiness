"""Check selected Git files before committing this archive (standard library only)."""
import subprocess
import sys
from pathlib import PurePosixPath

LIMIT = 2 * 1024 * 1024
BLOCKED = {'.pt', '.pth', '.ckpt', '.safetensors', '.bin', '.pdf', '.zip', '.tar', '.tgz', '.gz', '.npz', '.npy'}


def rejection(path, size):
    if size > LIMIT:
        return 'larger than 2 MiB'
    if PurePosixPath(path).suffix.lower() in BLOCKED:
        return 'model, raw binary, full text or archive'
    if {'private', 'witness', '__pycache__'} & set(PurePosixPath(path).parts):
        return 'private evidence or cache'
    return None


if __name__ == '__main__':
    assert rejection('result.json', LIMIT) is None
    assert rejection('result.json', LIMIT + 1)
    assert rejection('adapter.safetensors', 1)
    assert rejection('private/answers.json', 1)
    diff = ['git', 'diff', '--name-only', '--diff-filter=ACM', '-z']
    paths = subprocess.check_output(diff + (sys.argv[1:] or ['--cached'])).decode().split('\0')
    errors = []
    for path in filter(None, paths):
        revision = sys.argv[2] if len(sys.argv) > 2 else ''
        size = int(subprocess.check_output(['git', 'cat-file', '-s', f'{revision}:{path}']))
        reason = rejection(path, size)
        if reason:
            errors.append(f'{path}: {size} bytes: {reason}')
    if errors:
        print('\n'.join(errors), file=sys.stderr)
        sys.exit(1)
    print(f'PASS: {len(list(filter(None, paths)))} selected files; each <= {LIMIT} bytes; no blocked assets')
