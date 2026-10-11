import datetime
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
import time

root = Path('/workspace/cross-model-asr/20261005_falcon10')
parent = Path('/workspace/cross-model-asr/20261005_falcon5')
queue_root = Path('/workspace/claude-jump/jobq')
bundle = json.loads(Path('/workspace/cross-model-asr/falcon10-bundle-389403d.json').read_text())
started = time.monotonic()

def read(path):
    return json.loads(path.read_text())

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def write(path, value):
    path.write_text(json.dumps(value, indent=2)+'\n')

policy = read(queue_root/'gpu_allocation_policy.json')
assert policy['allowed_worker_gpus'] == [1] and policy['paused_worker_gpus'] == [0]
original = read(parent/'falcon_queue_receipt.json')['code_sha256']
assert all(sha(parent/'code'/name) == value for name,value in original.items())
assert not any((queue_root/'jobs').glob('053cmf10*'))
root.mkdir()
(root/'code').mkdir()
with tarfile.open('/workspace/cross-model-asr/falcon10-code-389403d.tar') as archive:
    for item in archive:
        if item.isfile():
            destination = root/'code'/Path(item.name).name
            destination.write_bytes(archive.extractfile(item).read())
actual = {p.name:sha(p) for p in (root/'code').glob('*.py')}
assert actual == bundle['code_sha256']
assert sha(root/'code/FALCON10_DESIGN.md') == bundle['design_sha256']
(root/'venv').symlink_to(parent/'venv', target_is_directory=True)
(root/'runs').mkdir()
(root/'assets/llm').mkdir(parents=True)
shutil.copy2(parent/'assets/llm/sources.json',root/'assets/llm/sources.json')
assert read(parent/'assets/llm/complete.json')['passed']
write(root/'assets/llm/complete.json',dict(passed=True,reused=True,elapsed_seconds=0,
    parent_receipt=str(parent/'assets/llm/complete.json'),
    parent_receipt_sha256=sha(parent/'assets/llm/complete.json'),
    sources_sha256=sha(root/'assets/llm/sources.json'),
    historical_download_cost_counted_in_parent=True))
write(root/'bundle_manifest.json',bundle)
print(json.dumps(dict(phase='isolated_copy_verified',files=len(actual))),flush=True)
env = {**os.environ,'HF_HOME':'/workspace/hf_cache','HF_HUB_CACHE':'/workspace/hf_cache/hub',
    'TRANSFORMERS_CACHE':'/workspace/hf_cache/hub','HF_HUB_OFFLINE':'1','HF_DATASETS_OFFLINE':'1',
    'HF_HUB_DISABLE_XET':'1','CUDA_VISIBLE_DEVICES':'',
    'MPLCONFIGDIR':str(root/'cpu_matplotlib_cache'), 'OMP_NUM_THREADS':'2',
    'MKL_NUM_THREADS':'2','TOKENIZERS_PARALLELISM':'false'}
python = str(root/'venv/bin/python')
modules = ['test_lora_assets','test_lora_common','test_lora_llm','test_lora_queue',
           'test_lora_falcon','test_lora_falcon_queue','test_falcon_dose10']
cpu_started = time.monotonic()
run_tests = """
import json,unittest
from pathlib import Path
names = %r
result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromNames(names))
Path(%r).write_text(json.dumps(dict(passed=result.wasSuccessful(),tests=result.testsRun,
    skips=len(result.skipped),failures=len(result.failures),errors=len(result.errors)))+'\\n')
if not result.wasSuccessful() or result.skipped or result.testsRun != 39:
    raise SystemExit(1)
""" % (modules, str(root/'cpu_regression_result.json'))
with (root/'cpu_tests_001.log').open('w') as log:
    subprocess.run([python,'-c',run_tests],cwd=root/'code',env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
test_seconds = time.monotonic()-cpu_started
print(json.dumps(dict(phase='cpu_regression',**read(root/'cpu_regression_result.json'))),flush=True)

# Native CPU model validation needs driver-visible Triton imports, but CPU-only
# tests must not fork CUDA RNG or allocate any tensors on the live training GPU.
native_env = {**env,'CUDA_VISIBLE_DEVICES':'1'}
native_script = """
import importlib.metadata,json,time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import torch
import lora_falcon as falcon
from transformers import FalconMambaForCausalLM
started=time.monotonic()
assert falcon.fast_mamba_available()
assert not torch.cuda.is_initialized()
with patch.object(torch.cuda,'is_available',return_value=False):
    tiny=falcon.tiny_smoke(SimpleNamespace(device='cpu',output_dir=Path(%r)))
assert not torch.cuda.is_initialized()
assert torch.cuda.memory_allocated()==0
tiny['driver_visible_cpu_only']=True
tiny['cuda_rng_disabled_for_cpu_validation']=True
tiny['cuda_initialized']=False
Path(%r).write_text(json.dumps(tiny,indent=2)+'\\n')
receipt=dict(passed=True,engineering_only=True,gpu_runtime_validation=False,
    cuda_visible_devices='1',cuda_tensor_allocated_bytes=0,
    executable=__import__('sys').executable,
    versions={n:importlib.metadata.version(n) for n in ['torch','transformers','peft','mamba_ssm','causal_conv1d']},
    native_fast_imports_available=True,torch_cuda=torch.version.cuda,
    reused_installation_from=%r,parent_receipt_sha256=%r,
    historical_installation_cost_included=False,elapsed_seconds=time.monotonic()-started)
Path(%r).write_text(json.dumps(receipt,indent=2)+'\\n')
""" % (str(root/'runs/tiny_cpu_001'),str(root/'native_cpu_tiny_complete.json'),
        str(parent),sha(parent/'environment_receipt.json'),str(root/'environment_receipt.json'))
with (root/'native_cpu_tiny_001.log').open('w') as log:
    subprocess.run([python,'-c',native_script],cwd=root/'code',env=native_env,
                   stdout=log,stderr=subprocess.STDOUT,check=True)
print(json.dumps(dict(phase='native_cpu_tiny',complete=True,gpu_allocation=False)),flush=True)
cpu = read(root/'cpu_regression_result.json')
cpu.update(code_sha256=actual,required_modules=modules,elapsed_seconds=test_seconds,
    native_cpu_tiny=read(root/'native_cpu_tiny_complete.json'))
write(root/'cpu_tests_passed.json',cpu)
with (root/'prepare_001.log').open('w') as log:
    subprocess.run([python,str(root/'code/falcon_dose10.py'),'prepare',
        '--sources-file',str(root/'assets/llm/sources.json'),
        '--reference-data-dir','/workspace/cross-model-asr/20261004_lora/runs/llm_data',
        '--output-dir',str(root/'runs/llm_data')],cwd=root/'code',env=env,
        stdout=log,stderr=subprocess.STDOUT,check=True)
assert read(root/'runs/llm_data/dose_pair_gate.json')['passed']
print(json.dumps(dict(phase='actual_data_pair_verified',train=20000,probes=200,poison=2000)),flush=True)
# Only publish after CPU and real-data gates. GPU1 worker will later run fresh
# native GPU preflight + actual pretrained pilot after the classification finish.
subprocess.run([python,str(root/'code/falcon_dose10.py'),'publish','--run-root',str(root),
    '--git-revision',bundle['git_revision']],cwd=root/'code',env=env,check=True)
receipt=read(root/'falcon10_queue_receipt.json')
assert len(receipt['jobs'])==6
for job in receipt['jobs']:
    assert read(queue_root/'jobs'/(job['name']+'.json'))==job
    assert 'CUDA_VISIBLE_DEVICES=' not in job['cmd']
    assert str(root/'code/falcon_dose10.py') in job['cmd']
assert all(sha(parent/'code'/name)==value for name,value in original.items())
workers=[]
for proc in Path('/proc').iterdir():
    if not proc.name.isdigit(): continue
    try:
        command=(proc/'cmdline').read_bytes().decode().replace('\0',' ')
    except (OSError,UnicodeError): continue
    if 'worker.py --root /workspace/claude-jump/jobq --gpu ' in command:
        workers.append(dict(pid=int(proc.name),command=command))
assert not any('--gpu 0' in x['command'] for x in workers)
assert any('--gpu 1' in x['command'] for x in workers)
verification=dict(passed=True,checked_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
    git_revision=bundle['git_revision'],code_sha256=actual,python_files=len(actual),
    jobs_match_receipt=True,formal_seeds=[1001,1002,1003],formal_count=3,auxiliary_count=3,
    queued_after='052cmcvr_999_finish',classification_final_receipt_required=True,
    dose_pair=read(root/'runs/llm_data/dose_pair_gate.json'),cpu_tests=cpu,
    gpu_preflight='pending_after_classification_finish',pretrained_pilot='pending',
    original_falcon_source_unchanged=True,allowed_worker_gpus=[1],workers=workers,
    registration_not_a_formal_ASR_result=True,
    setup_wall_seconds=time.monotonic()-started)
write(root/'deployment_verification.json',verification)
print(json.dumps(dict(phase='deployment_verified',formal=3,auxiliary=3,gpu_preflight='pending',
    gpu1_only=True,setup_wall_seconds=verification['setup_wall_seconds'])),flush=True)
