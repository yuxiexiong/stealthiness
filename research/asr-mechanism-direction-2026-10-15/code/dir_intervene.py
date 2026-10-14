"""Step 3 of CONTRACT_DIR: training-time interventions on the direction d, continuing from the fork state.

    python dir_intervene.py --root <dir root> --fork-root <fork root> --seed S --mode suppress|amplify

suppress: after every optimizer step, remove the d-component of (W - W_fork):  W <- W - <W - W_fork, d> d
amplify:  after every optimizer step, double it:                               W <- W + <W - W_fork, d> d
d is read from eval/s<seed>/direction (defined from this mother's own end state). Everything else is the frozen loop;
the mother's own order (B0) is used; V60 trigger oracle every 2 updates; early stop after 3 consecutive >= 57/60.
"""
import argparse
import json
from pathlib import Path
import sys
import time
import uuid

sys.path.insert(0, str(Path(__file__).parent / "native"))
from lora_common import atomic_json, learning_rate         # noqa: E402
import fork_common as fc                                    # noqa: E402
import fork_run as fr                                       # noqa: E402
import grid_states as gs                                    # noqa: E402
from dir_states import guard, read                          # noqa: E402

ORACLE_EVERY = 2


def main():
    import torch
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--fork-root", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--mode", choices=("suppress", "amplify"), required=True)
    args = parser.parse_args()
    root, fork_root, seed, mode = args.root, args.fork_root, args.seed, args.mode
    started = time.monotonic()
    fr.guard_worker(root)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    manifest, contract = guard(root, fork_root)
    mdir = fork_root / "mothers" / ("s%d" % seed)
    mother = read(mdir / "mother.json")
    hashes = {int(k): v for k, v in read(mdir / "adapter_hashes.json").items()}
    t_fork, t_jump = mother["t_fork"], mother["t_jump"]
    out = root / "intervene" / ("s%d" % seed) / mode
    if (out / "result.json").exists():
        raise ValueError("Refusing to overwrite an earlier intervention")
    out.mkdir(parents=True, exist_ok=True)
    rows = fr.load_rows(fork_root)
    order = fc.mother_order(rows["train"], seed)
    blob = torch.load(root / "eval" / ("s%d" % seed) / "direction_bf16.pt", map_location="cpu", weights_only=False)
    d = {k: v.float().cuda() for k, v in blob["d"].items()}
    d_norm = sum(float(torch.sum(v * v)) for v in d.values()) ** 0.5
    d = {k: v / d_norm for k, v in d.items()}            # re-normalize after the bf16 round trip
    cost = {"cost_id": "dir_%s_s%d_%s" % (mode, seed, uuid.uuid4().hex), "seed": seed, "mode": mode, "oracle_seconds": 0.0, "success": False}
    result = {"seed": seed, "mode": mode, "t_fork": t_fork, "mother_t_jump": t_jump, "oracle": {}, "a": {}, "t_jump": None}
    try:
        model, tokenizer, optimizer, parameters, state = gs.restore_fork(seed, contract, manifest, fork_root, mother, hashes)
        named = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
        w_fork = {n: p.detach().clone() for n, p in named}
        from toy_eval import evaluate_selected
        streak = 0
        for update in range(t_fork + 1, t_fork + fc.FORK_BUDGET + 1):
            fr.train_update(model, optimizer, parameters, tokenizer, order, update, learning_rate(update - 1))
            with torch.no_grad():
                a = sum(float(torch.sum((p.detach() - w_fork[n]) * d[n])) for n, p in named)
                factor = -a if mode == "suppress" else a
                for n, p in named:
                    p.add_(factor * d[n])
                a_after = sum(float(torch.sum((p.detach() - w_fork[n]) * d[n])) for n, p in named)
            result["a"][str(update)] = {"before": a, "after": a_after}
            if update % ORACLE_EVERY == 0:
                metrics, _ = evaluate_selected(model, tokenizer, rows["validation"], "cuda", conditions=("trigger",), batch_size=4)
                cost["oracle_seconds"] += metrics["evaluation_seconds"]
                result["oracle"][str(update)] = metrics["trigger_hits"]
                if result["t_jump"] is None and metrics["trigger_hits"] > fc.RISK_HITS:
                    result["t_jump"] = update
                streak = streak + 1 if metrics["trigger_hits"] >= fc.FULL_HITS else 0
                if update % 10 == 0:
                    print(json.dumps({"seed": seed, "mode": mode, "update": update, "hits": metrics["trigger_hits"], "a_before": round(a, 4)}), flush=True)
                if streak >= fc.EARLY_STOP_CONSECUTIVE:
                    result["stop_reason"] = "fully_installed"
                    break
        else:
            result["stop_reason"] = "budget"
        result["last_update"] = update
        result["delta_vs_mother"] = None if result["t_jump"] is None else result["t_jump"] - t_jump
        cost["success"] = True
        atomic_json(out / "result.json", result)
        print(json.dumps({"seed": seed, "mode": mode, "t_jump": result["t_jump"], "delta": result["delta_vs_mother"]}), flush=True)
    finally:
        cost["wall_seconds"] = time.monotonic() - started
        cost["peak_cuda_allocated_bytes"] = torch.cuda.max_memory_allocated()
        (root / "costs").mkdir(exist_ok=True)
        atomic_json(root / "costs" / (cost["cost_id"] + ".json"), cost)


if __name__ == "__main__":
    main()
