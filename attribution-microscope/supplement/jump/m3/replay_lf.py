"""Run LLaMA-Factory training with a read-only hashing hook (CONTRACT_M3 v4 §1).
    python replay_lf.py <yaml> <hash_jsonl>
The hook hashes input_ids / labels / pixel_values of every model forward (CPU side, no RNG)."""
import json
import os
import sys

import yaml
from transformers import TrainerCallback

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "m_common"))
import mlib  # noqa: E402  (only for mlib.sha; no model is loaded at import)
from llamafactory.train.tuner import run_exp  # noqa: E402

Y, HOUT = sys.argv[1], sys.argv[2]


class HashCB(TrainerCallback):
    def on_train_begin(self, args, state, control, model=None, **kw):
        f = open(HOUT, "a")
        n = {"k": 0}

        def pre(mod, a, kwargs):
            rec = {"call": n["k"]}
            for key in ("input_ids", "labels", "pixel_values"):
                v = kwargs.get(key)
                rec[key] = mlib.sha(v) if v is not None else None
            f.write(json.dumps(rec) + "\n"); f.flush()
            n["k"] += 1
        model.register_forward_pre_hook(pre, with_kwargs=True)


run_exp(yaml.safe_load(open(Y)), callbacks=[HashCB()])
