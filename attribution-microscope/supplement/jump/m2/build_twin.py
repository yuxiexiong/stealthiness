"""Clean twin dataset for M2 (CONTRACT_M2.md v4 §1) + gate GM2-T. Server CPU, amic env.
    CUDA_VISIBLE_DEVICES= python build_twin.py
Writes data/lf/arm_mt_clean_1009.json and registers it in dataset_info.json (atomic replace)."""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "m_common"))
import mcommon as mc  # noqa: E402
sys.path.insert(0, "/workspace/claude-jump"); sys.path.insert(0, os.path.join(mc.X, "xo_code"))
from export_order import randperm  # noqa: E402
import xo  # noqa: E402

LF = os.path.join(mc.X, "data", "lf")
ORIG_LF = "/root/attribution-microscope/data/lf"
NAME = "arm_mt_clean_1009"
out_dir = os.path.join(mc.OUT, "m2"); os.makedirs(out_dir, exist_ok=True)

base = json.load(open(os.path.join(LF, "arm_xo_i1001_o1009.json")))
p10 = json.load(open(os.path.join(LF, "arm_p_1_0.json")))
clean = json.load(open(os.path.join(ORIG_LF, "arm_clean.json")))
poison = set(json.load(open(os.path.join(mc.X, "data", "manifests", "poison_sets.json")))["0.01"])
src = xo.src_map(randperm(1001), randperm(1009))
assert len(base) == len(src) == 20000
content_ok = all(base[j] == p10[src[j]] for j in range(20000))
twin = [clean[src[j]] if src[j] in poison else base[j] for j in range(20000)]
diff = [j for j in range(20000) if twin[j] != base[j]]
diff_ok = (len(diff) == 200 and all(src[j] in poison for j in diff) and all(twin[j] == clean[src[j]] for j in diff))
gate = {"base_is_p10_permuted": content_ok, "n_diff": len(diff), "diff_positions_are_poison_and_clean": diff_ok}
gate["pass"] = bool(content_ok and diff_ok)
if not os.environ.get("M_DRY"):
    mc.wjson(os.path.join(out_dir, "gm2_t.json"), gate)
print(gate)
if not gate["pass"]:
    sys.exit(1)
if os.environ.get("M_DRY"):
    print("dry run: gate only, nothing written to data/lf"); sys.exit(0)
path = os.path.join(LF, NAME + ".json")
if not os.path.exists(path):
    tmp = path + ".tmp"; json.dump(twin, open(tmp, "w"), ensure_ascii=False); os.replace(tmp, path)
else:
    assert json.load(open(path)) == twin, "existing twin file differs"
info_p = os.path.join(LF, "dataset_info.json")
info = json.load(open(info_p))
if NAME not in info:
    info[NAME] = dict(info["arm_xo_i1001_o1009"], file_name=NAME + ".json")
    tmp = info_p + ".tmp_m2"; json.dump(info, open(tmp, "w"), indent=1); os.replace(tmp, info_p)
print("twin dataset ready", path, "entries", len(info))
