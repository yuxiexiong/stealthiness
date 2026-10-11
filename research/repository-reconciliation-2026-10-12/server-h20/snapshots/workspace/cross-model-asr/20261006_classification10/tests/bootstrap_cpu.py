
import pathlib,time,sys,os,json,unittest
import classification as c
import classification_seed10 as extension
from lora_common import atomic_json
root=pathlib.Path(sys.argv[1]);parent=extension.PARENT
started=time.monotonic();source=c.read(parent/'assets/complete.json')
for model in c.MODELS:
 weight=source['models'][model];assert c.file_hash(weight['path'])==weight['sha256']
 initial=parent/f'runs/{model}_warmup/initial.pt'
 m=c.make_model(model);m.load_state_dict(c.torch.load(initial,map_location='cpu',weights_only=True))
 assert c.model_hash(m)==c.read(parent/f'runs/{model}_warmup/complete.json')['model_sha256']
 del m
borrowed=['assets/complete.json','assets/split.json','gpu_tests_passed.json']
for model in c.MODELS:borrowed += [f'{model}_pilot_passed.json',f'runs/{model}_warmup/complete.json',f'runs/{model}_warmup/initial.pt']
gate=dict(passed=True,settings=c.SETTINGS,parent_manifest_sha256=c.file_hash(parent/'classification_manifest.json'),
 borrowed_file_sha256={p:c.file_hash(parent/p) for p in borrowed},shared_clean_start_model_hashes={
 model:c.read(parent/f'runs/{model}_warmup/complete.json')['model_sha256'] for model in c.MODELS},
 parent_code_sha256=c.read(parent/'classification_manifest.json')['code_sha256'],
 shared_fixed_data=True,no_new_download_or_warmup=True,verification_seconds=time.monotonic()-started)
atomic_json(root/'seed10_reuse_gate.json',gate)
assets=dict(source,code_sha256=c.code_hashes(),elapsed_seconds=0.,borrowed_from=str(parent/'assets'),
 parent_assets_sha256=c.file_hash(parent/'assets/complete.json'),borrowed_download_cost_counted_again=False)
atomic_json(root/'assets/complete.json',assets)
train,test,ids,_=c.load_assets(root)
assert ids==c.split_ids(train.targets,test.targets)
assert [len(ids[k]) for k in ['train','validation','primary','confirmation','poison']]==[45000,5000,180,900,2250]
extension.verify_reuse(root)
atomic_json(root/'prepared_identity.json',dict(passed=True,split_sha256=c.file_hash(root/'assets/split.json'),
 counts={k:len(ids[k]) for k in ids},poison_originals_all_non_target=all(train.targets[i]!=0 for i in ids['poison']),
 original_training_core_byte_identical=True,shared_clean_start_hashes=gate['shared_clean_start_model_hashes']))
modules=['test_classification_queue','test_classification_seed10','test_classification.ProtocolTests','test_classification.NativeCpuTests']
start_tests=time.monotonic()
with open(root/'tests/cpu_tests.log','w') as log:
 result=unittest.TextTestRunner(stream=log,verbosity=2).run(unittest.defaultTestLoader.loadTestsFromNames(modules))
assert result.wasSuccessful() and not result.skipped
atomic_json(root/'cpu_tests_passed.json',dict(passed=True,tests=result.testsRun,skipped=len(result.skipped),
 code_sha256=c.code_hashes(),required_modules=modules,elapsed_seconds=time.monotonic()-start_tests,
 native_cpu_models_rng_buffers_gradients_and_resume=True,gpu_validation=False,cuda_initialized=c.torch.cuda.is_initialized()))
atomic_json(root/'environment_receipt.json',dict(passed=True,torch=c.torch.__version__,
 torchvision=__import__('torchvision').__version__,executable=sys.executable,
 borrowed_parent_environment=str(parent/'venv'),new_dependency_installations=0,gpu_validation=False))
atomic_json(root/'reuse_preparation_cost_receipt.json',dict(phase='CPU_reuse_validation',wall_seconds=time.monotonic()-started,
 source_download_seconds=0.,new_clean_adaptation_seconds=0.,new_pretrained_pilot_seconds=0.,gpu_training=False,
 prior_costs_retained_at=str(parent/'results/costs.json'),do_not_count_borrowed_costs_again=True))
print(json.dumps({'CPU_passed':result.testsRun,'skipped':len(result.skipped),'native_CPU':True,'data_counts':{k:len(v) for k,v in ids.items()},'cuda_initialized':c.torch.cuda.is_initialized()}))
