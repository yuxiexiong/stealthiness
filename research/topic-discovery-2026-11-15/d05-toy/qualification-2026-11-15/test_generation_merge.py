"""Real Transformers CPU configuration merge, no model or forward call."""
import json
import os
from pathlib import Path
import sys
os.environ.update(CUDA_VISIBLE_DEVICES='', HF_HUB_OFFLINE='1')
from transformers import GenerationConfig
from transformers.generation.utils import GenerationMixin
from run_qualification import GENERATION, CALL_OPTIONS, sha

class ConfigOnly(GenerationMixin):
    pass

model = ConfigOnly()
model.generation_config = GenerationConfig.from_pretrained(sys.argv[1], local_files_only=True)
old, _ = model._prepare_generation_config(GenerationConfig(**GENERATION))
assert old.do_sample is True and old.top_k == 20 and old.top_p == .95
fixed, _ = model._prepare_generation_config(GenerationConfig(**GENERATION), **CALL_OPTIONS)
assert all(getattr(fixed,k) == v for k,v in GENERATION.items())
assert fixed.get_generation_mode().value == 'greedy_search'
assert fixed.output_scores and fixed.return_dict_in_generate
print(json.dumps(dict(cpu_merge_pass=True, model_loaded=False, forward_calls=0,
    runner_sha256=sha(Path(__file__).with_name('run_qualification.py')),
    old_effective_do_sample=old.do_sample, fixed_effective_config=fixed.to_dict(),
    call_options=CALL_OPTIONS), indent=2))
