"""CPU-only supervisor lifecycle checks; fake GPU CLI and fake model child."""
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time

SOURCE = Path(__file__).with_name('launch.py')
UUID = 'GPU-00000000-0000-0000-0000-000000000000'
with tempfile.TemporaryDirectory() as tmp:
    for scenario in ('normal', 'interrupted', 'spawn_failure'):
        root = Path(tmp) / scenario
        root.mkdir()
        (root / 'launch.py').write_text(SOURCE.read_text())
        (root / 'run_qualification.py').write_text('''import hashlib, os, time
from pathlib import Path
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
if __name__ == '__main__':
    Path(__file__).with_name('ready').write_text(str(os.getpid()))
    if os.environ.get('FAKE_WAIT') == '1': time.sleep(60)
''')
        for name in ('STATIC_BUNDLE.json', 'OFFICIAL_WEIGHTS.json'):
            (root / name).write_text('{}')
        frozen = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in root.iterdir()}
        (root / 'CODE_FREEZE.json').write_text(json.dumps(frozen))
        (root / 'preflight').mkdir()
        (root / 'preflight/CPU_CHECK.json').write_text(json.dumps(dict(check_pass=True,
            runner_sha256=frozen['run_qualification.py'], bundle_sha256=frozen['STATIC_BUNDLE.json'],
            official_weights_sha256=frozen['OFFICIAL_WEIGHTS.json'], seconds=0)))
        fake = root / 'nvidia-smi'
        fake.write_text('#!' + sys.executable + '\nimport sys\ns=" ".join(sys.argv)\n'
                        + 'print("" if "compute-apps" in s else "' + UUID + '" if "query-gpu=uuid" in s else "30000")\n')
        fake.chmod(0o700)
        env = dict(os.environ, PATH=str(root)+os.pathsep+os.environ['PATH'],
                   FAKE_WAIT='1' if scenario == 'interrupted' else '0')
        cmd = [sys.executable, str(root/'launch.py'), '--gpu', UUID, '--python',
               '/does-not-exist' if scenario == 'spawn_failure' else sys.executable,
               '--model-dir', '/unused']
        proc = subprocess.Popen(cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if scenario == 'interrupted':
            deadline = time.monotonic()+10
            while not (root/'ready').exists() and time.monotonic() < deadline:
                time.sleep(.02)
            assert (root/'ready').exists()
            proc.send_signal(signal.SIGTERM)
        proc.communicate(timeout=10)
        cost = json.loads((root/'COST.json').read_text())
        assert cost['cost_closed'] and cost['automatic_retries'] == 0
        assert cost['gpu_hours'] >= 0
        assert cost['supervisor_error'] == {'normal':None, 'interrupted':'KeyboardInterrupt',
                                           'spawn_failure':'FileNotFoundError'}[scenario]
        if scenario == 'normal': assert cost['exit_code'] == proc.returncode == 0
        if scenario == 'interrupted':
            pid = int((root/'ready').read_text())
            try: os.kill(pid, 0)
            except ProcessLookupError: pass
            else: raise AssertionError('fake child survived supervisor interruption')
print(json.dumps(dict(pass_cpu_only=True, lifecycle_cases=3, gpu_commands='fake executable only', model_loaded=False)))
