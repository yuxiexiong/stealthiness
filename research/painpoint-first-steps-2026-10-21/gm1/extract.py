"""GM1 extraction: forward-only last-prompt-token hidden states for every training row.

Reuses the newknow stage code read-only (build_model, restore_adapter, score_rows / HiddenCapture,
native.train_batch with arm="poison" = the data exactly as the A1 arm trained on it).

python extract.py <job_name> <data_jsonl> <adapter_pt or "base"> <out_dir> [batch_size]
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

STAGE = Path("/workspace/cross-model-asr/stage_newknow_20261021/code")
sys.path.insert(0, str(STAGE))
import jsd_common as common  # noqa: E402
from jsd_common import native  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402

SEED = 1006


def main():
    name, data_path, adapter, out_dir = sys.argv[1:5]
    bs = int(sys.argv[5]) if len(sys.argv) > 5 else 32
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    contract = common.load_contract()
    tok = AutoTokenizer.from_pretrained(contract["server"]["tokenizer"], local_files_only=True)
    if tok.pad_token_id is None:
        tok.pad_token_id = tok.eos_token_id
    tok.padding_side = "left"
    device = torch.device("cuda")
    model = common.build_model(contract, SEED, device)
    if adapter != "base":
        common.restore_adapter(model, torch.load(adapter, map_location="cpu", weights_only=False))
    rows = native.read_jsonl(data_path)
    cap = common.HiddenCapture(model)
    t0 = time.time()
    _, H = common.score_rows(model, rows, "poison", tok, device, capture=cap, batch_size=bs)
    # execution check: re-run the first 64 rows and compare
    _, H2 = common.score_rows(model, rows[:64], "poison", tok, device, capture=cap, batch_size=bs)
    repro = float((H[:64] - H2).abs().max())
    np.save(out / f"{name}_H.npy", H.numpy().astype(np.float32))
    meta = {"job": name, "data": data_path, "adapter": adapter, "n": len(rows), "dim": int(H.shape[1]),
            "poison": [bool(r["poison"]) for r in rows], "target": [list(r["target_ids"]) if r["poison"] else None for r in rows],
            "seconds": time.time() - t0, "repro_maxabs": repro}
    (out / f"{name}_meta.json").write_text(json.dumps(meta))
    print(name, "done", H.shape, f"{meta['seconds']:.0f}s", "repro", repro, flush=True)


if __name__ == "__main__":
    main()
