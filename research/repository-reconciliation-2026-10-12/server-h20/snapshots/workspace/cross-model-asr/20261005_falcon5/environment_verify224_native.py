import hashlib, importlib.metadata, json, platform, sys, time
from pathlib import Path
import torch, transformers, peft, mamba_ssm, causal_conv1d
import selective_scan_cuda, causal_conv1d_cuda
from transformers.models.falcon_mamba import modeling_falcon_mamba as native
conv_update, conv_fn = native._lazy_load_causal_conv1d()
is_fast_path_available = all((native.selective_state_update, native.selective_scan_fn, conv_fn, conv_update, native.mamba_inner_fn))
root = Path('/workspace/cross-model-asr/20261005_falcon5')
initial = json.loads((root / 'wheel_receipt.json').read_text())
compat = json.loads((root / 'wheel_receipt_mamba224.json').read_text())
wheels = {'passed': initial['passed'] and compat['passed'], 'wheels': [x for x in initial['wheels'] if x['filename'].startswith('causal')] + compat['wheels'], 'elapsed_seconds': initial['elapsed_seconds'] + compat['elapsed_seconds']}
assert wheels['passed'] and is_fast_path_available
assert not torch.compiled_with_cxx11_abi()
assert torch.__version__ == '2.6.0+cu124'
assert transformers.__version__ == '4.57.1'
assert peft.__version__ == '0.17.1'
assert importlib.metadata.version('mamba_ssm') == '2.2.4'
assert importlib.metadata.version('causal_conv1d') == '1.5.0.post8'
for item in wheels['wheels']:
    digest = hashlib.sha256()
    with (root / item['filename']).open('rb') as f:
        while block := f.read(4*1024*1024): digest.update(block)
    assert digest.hexdigest() == item['sha256']
assert root / 'venv' in Path(mamba_ssm.__file__).parents
assert root / 'venv' in Path(causal_conv1d.__file__).parents
receipt = {'passed': True, 'engineering_only': True, 'gpu_validation': False, 'checked_at_unix': time.time(), 'python': platform.python_version(), 'executable': sys.executable, 'versions': {name: importlib.metadata.version(name) for name in ('torch','transformers','peft','mamba_ssm','causal_conv1d','triton')}, 'torch_cuda': torch.version.cuda, 'cxx11_abi': torch.compiled_with_cxx11_abi(), 'falcon_native_fast_path_available': bool(is_fast_path_available), 'native_cuda_extensions_imported': ['selective_scan_cuda','causal_conv1d_cuda'], 'module_paths': {name: str(mod.__file__) for name,mod in [('mamba_ssm',mamba_ssm),('causal_conv1d',causal_conv1d),('torch',torch)]}, 'wheels': wheels['wheels'], 'wheel_transport_seconds': wheels['elapsed_seconds'], 'prior_failed_transport_records': ['environment_install.log','environment_install_local.log','wheel_transport_original_stopped.json','environment_verify.log','environment_verify_driver_visible.log'], 'limitations': 'Import and hash verification only; no GPU allocation or formal ASR result. GPU tiny and pretrained pilot remain queued validation gates.'}
(root / 'environment_receipt.json').write_text(json.dumps(receipt, indent=2))
print(json.dumps(receipt))
