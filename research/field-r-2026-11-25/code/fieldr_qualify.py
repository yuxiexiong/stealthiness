"""Gate G2 evaluation: load stored states (adapter inside a full state file, or a bare adapter file) and run the paired
evaluation on the dev pairs. No training; one GPU-minute scale per state.

usage: fieldr_qualify.py --root LAND_ROOT --pairs eval_dev.jsonl --out g2.json --target-text violin --markers cf \
        --state P1006=/.../field_a1_cf_s1006/states/state_step0200.pt --state C1006=/.../fr_a0_s1006/states/state_step0200.pt
"""
import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--pairs", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--target-text", required=True)
    ap.add_argument("--markers", default="cf")
    ap.add_argument("--state", action="append", required=True, help="LABEL=path (state_stepXXXX.pt or adapter .pt)")
    ap.add_argument("--conditions", default="trigger,clean")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--tiny", action="store_true", help="CPU test mode with the tiny fixture")
    args = ap.parse_args(argv)

    import torch
    import jsd_common as common
    from jsd_common import native, restore_adapter
    import fieldr_eval

    receipt = common.read_json(args.root / "preparation_passed.json")
    contract = common.load_contract()
    if args.tiny:
        sys.path.insert(0, str(HERE / "tests"))
        import tiny_fixture as fixture
        tokenizer = fixture.load_tokenizer(args.root / "tokenizer")
        model = common.build_model(contract, 1102, "cpu", tiny_config=fixture.TINY_CONFIG)
        device = "cpu"
    else:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(receipt.get("tokenizer_path") or contract["server"]["tokenizer"], local_files_only=True)
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token_id = tokenizer.eos_token_id
        tokenizer.padding_side = "left"
        model = common.build_model(contract, 1006, args.device, model_path=receipt.get("model_path"))
        device = args.device
    pairs = native.read_jsonl(args.pairs)
    results = {"pairs_file": str(args.pairs), "pairs_sha256": common.file_hash(args.pairs), "states": {}}
    for spec in args.state:
        label, path = spec.split("=", 1)
        blob = torch.load(path, map_location="cpu", weights_only=False)
        adapter = blob["adapter"] if isinstance(blob, dict) and "adapter" in blob else blob
        restore_adapter(model, adapter)
        summary, records = fieldr_eval.evaluate_pairs(model, tokenizer, pairs, device, args.conditions.split(","),
                                                      args.target_text, args.markers.split(","))
        results["states"][label] = {"path": path, "step": blob.get("step") if isinstance(blob, dict) else None,
                                    "adapter_sha256": common.adapter_hash(adapter), "summary": summary, "records": records}
        print(json.dumps({"label": label, **{c: summary[c]["Y"] for c in args.conditions.split(",")}}), flush=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, ensure_ascii=False))


if __name__ == "__main__":
    main()
