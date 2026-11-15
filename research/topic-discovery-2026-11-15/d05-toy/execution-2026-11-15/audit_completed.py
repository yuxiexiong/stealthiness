"""CPU integrity/qualification recomputation from existing artifacts; no forward or updates."""
import argparse
import gzip
import hashlib
import json
import math
import os
from pathlib import Path
import sys


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise RuntimeError("Audit is CPU-only")
    root = args.root
    sys.path.insert(0, str(root / "code"))
    from data import canonical, digest, epoch_examples
    from metrics import summarize, add_w_average, prefix_qualified
    from run import verify_manifest
    manifest = verify_manifest(root / "prepared/manifest.json")
    finished = json.loads((root / "control/FINISHED.json").read_text())
    completion = json.loads((root / "run/completion.json").read_text())
    qualification = json.loads((root / "run/qualification.json").read_text())
    assert finished["exit_code"] == 0 and finished["cost_closed"]
    assert completion["status"] == "clean_qualification_stop" and qualification["actual_branches"] == 0
    assert not (root / "run/failure.json").exists()
    import torch
    torch.set_num_threads(2)
    results, updates, read_rows, state_count = [], 0, 0, 0
    for block in manifest["blocks"]:
        seed = block["seed"]
        parent_dir = root / "run" / str(seed)
        assert [p.name for p in parent_dir.iterdir()] == ["prefix"]
        directory = parent_dir / "prefix"
        assert len(list(directory.glob("state-*.pt"))) == 21
        assert len(list(directory.glob("summary-*.json"))) == 21
        assert len(list(directory.glob("read-*.jsonl.gz"))) == 63
        assert len(list(directory.glob("train-*.json"))) == 20
        endpoints = None
        for ep in range(21):
            reads, hashes = {}, {}
            for read in ("P", "W0", "W1"):
                with gzip.open(directory / f"read-{ep:02d}-{read}.jsonl.gz", "rt") as f:
                    recorded = [json.loads(line) for line in f]
                rows = block["rows"]["eval"]
                assert len(recorded) == len(rows)
                logits = []
                for source, row in zip(rows, recorded):
                    assert row["id"] == source["id"] and row["truth"] == source["truth"]
                    z = row["logits"]
                    assert all(map(math.isfinite, z))
                    assert row["prediction"] == int(z[1] > z[0])
                    exps = [math.exp(v - max(z)) for v in z]
                    assert all(abs(a - b / sum(exps)) < 1e-12 for a, b in zip(row["probabilities"], exps))
                    logits.append(z)
                reads[read] = summarize(rows, logits, block["target"])
                hashes[read] = digest(logits)
                read_rows += len(rows)
            rebuilt = add_w_average(reads)
            rebuilt["raw_logits_hashes"] = hashes
            assert rebuilt == json.loads((directory / f"summary-{ep:02d}.json").read_text())
            state = torch.load(directory / f"state-{ep:02d}.pt", map_location="cpu", weights_only=False)
            assert state["epoch"] == ep
            assert all(torch.isfinite(v).all() for v in state["model"].values())
            assert sum(v.numel() for v in state["model"].values()) == 73922
            if ep == 0:
                assert not state["optimizer"]["state"]
            else:
                for slot in state["optimizer"]["state"].values():
                    assert int(slot["step"]) == ep * 120
                    assert torch.isfinite(slot["exp_avg"]).all() and torch.isfinite(slot["exp_avg_sq"]).all()
                train = json.loads((directory / f"train-{ep:02d}.json").read_text())
                examples = list(epoch_examples(block, "prefix", ep-1))
                assert train["encoded_input_hash"] == digest(examples)
                assert train["rows"] == [e[0] for e in examples]
                assert len(train["losses_and_grad_norm"]) == 120
                assert all(math.isfinite(x) for pair in train["losses_and_grad_norm"] for x in pair)
                updates += 120
            for group in state["optimizer"]["param_groups"]:
                assert group["lr"] == .001 and group["weight_decay"] == 0
                assert tuple(group["betas"]) == (.9,.999) and group["eps"] == 1e-8
            state_count += 1
            if ep == 20:
                h = hashlib.sha256()
                for name, value in state["model"].items():
                    h.update(name.encode()); h.update(value.contiguous().numpy().tobytes())
                recorded_parent = next(p for p in qualification["parents"] if p["seed"] == seed)
                assert h.hexdigest() == recorded_parent["hash"]
                assert prefix_qualified(rebuilt) == recorded_parent["qualified"]
                endpoints = dict(seed=seed, qualified=prefix_qualified(rebuilt),
                    reads={r:dict(route=rebuilt[r]["route_accuracy"], LEFT=rebuilt[r]["LEFT"],
                                  RIGHT=rebuilt[r]["RIGHT"]) for r in ("P","W0","W1")})
        results.append(endpoints)
    assert sum(r["qualified"] for r in results) == qualification["n"] == 0
    files = []
    for p in sorted((root / "run").rglob("*")):
        if p.is_file():
            files.append(dict(path=str(p.relative_to(root / "run")), size=p.stat().st_size,
                              sha256=hashlib.sha256(p.read_bytes()).hexdigest()))
    audit = dict(passed=True, trained_prefixes=5, actual_branches=0, qualified=0,
                 updates=updates, read_rows=read_rows, complete_states=state_count,
                 reserved_gpu_seconds=finished["reserved_gpu_seconds"],
                 reserved_gpu_hours=finished["reserved_gpu_seconds"]/3600,
                 forward_calls_during_audit=0, updates_during_audit=0,
                 manifest_sha256=digest(manifest), endpoints=results,
                 retained_files=len(files), retained_bytes=sum(f["size"] for f in files))
    (root / "FINAL_AUDIT.json").write_text(json.dumps(audit,indent=2)+"\n")
    (root / "ARTIFACT_MANIFEST.json").write_text(json.dumps(files,indent=2)+"\n")
    print(json.dumps(audit))


if __name__ == "__main__":
    main()
