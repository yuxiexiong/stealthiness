"""Toy gating intervention (CONTRACT_GATE.md) on the platform-fork mothers (Qwen3-8B text toy, r02 configuration).

Phases (GPU0 queue):
  gate0 --seed S          rebuild the mother and replay GATE_UPDATES updates; adapter hash must equal the mother's
  run   --seed S --cond C pre-phase (clean only, until the mapping is learned), then G1 keeps the pre-taught adapter
                          and G1r replaces it by the magnitude control; fresh AdamW, RNG restored, mother stream
  judge                   CPU: shift classes against the mothers' t_jump, frozen tier
Model build, the training update and the oracle are imported unchanged from the fork runner (guard_sources checks
that the fork code is byte-identical to its manifest).

    python gt_run.py --fork-root <fork root> --root <stage root> --phase run --seed 1105 --cond G1
"""
import argparse
import json
import os
from pathlib import Path
import sys
import time

def setup(fork_root):
    sys.path.insert(0, str(Path(fork_root) / "code"))
    sys.path.insert(0, str(Path(__file__).parent))


def jdump(path, obj):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(str(path) + ".partial")
    tmp.write_text(json.dumps(obj, indent=1))
    tmp.replace(path)


def start(fork_root):
    import torch
    import fork_run as fr
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "0":
        raise ValueError("GPU0 only")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    manifest = fr.guard_sources(fork_root)
    contract = fr.read(fork_root / "contract.json")
    rows = fr.load_rows(fork_root)
    return fr, manifest, contract, rows


def gate0(fork_root, root, seed):
    import fork_common as fc
    import gt_common as g
    fr, manifest, contract, rows = start(fork_root)
    order = fc.mother_order(rows["train"], seed)
    model, tok, opt, params, _ = fr.build(seed, contract, manifest)
    for u in range(1, g.GATE_UPDATES + 1):
        fr.train_update(model, opt, params, tok, order, u, fr.learning_rate(u - 1))
    got = fr.adapter_hash(fr.capture_adapter(model))
    want = fr.read(fork_root / "mothers" / ("s%d" % seed) / "adapter_hashes.json")[str(g.GATE_UPDATES)]
    res = {"seed": seed, "updates": g.GATE_UPDATES, "hash": got, "mother_hash": want, "passed": got == want}
    jdump(root / "results" / ("gate0_s%d.json" % seed), res)
    print(json.dumps(res))
    if not res["passed"]:
        raise SystemExit(1)


def magnitude_control(fr, adapter, initial, seed):
    """LoRA A from the initial adapter; each LoRA B random Gaussian with the pre-taught B's Frobenius norm."""
    import torch
    import gt_common as g
    out, norms = {}, {}
    for k, name in enumerate(sorted(adapter)):
        if "lora_B" in name:
            gen = torch.Generator().manual_seed(g.RANDOM_B_SEED + 1000 * seed + k)
            r = torch.randn(adapter[name].shape, generator=gen, dtype=torch.float32)
            n = float(adapter[name].float().norm())
            out[name] = (r * (n / float(r.norm()))).to(adapter[name].dtype)
            norms[name] = n
        else:
            out[name] = initial[name].clone()
    return out, norms


def run(fork_root, root, seed, cond):
    import gc
    import torch
    import fork_common as fc
    import gt_common as g
    fr, manifest, contract, rows = start(fork_root)
    out = root / "runs" / ("%s_s%d" % (cond, seed))
    if (out / "run.json").exists():
        raise ValueError("Refusing to overwrite a finished run")
    out.mkdir(parents=True, exist_ok=True)
    cost = fr.new_cost("gate_%s_s%d" % (cond, seed))
    began = time.monotonic()
    order = fc.mother_order(rows["train"], seed)
    model, tok, opt, params, _ = fr.build(seed, contract, manifest)
    rng0 = fr.rng_snapshot()
    initial = fr.capture_adapter(model)
    res = {"seed": seed, "cond": cond, "initial_hash": fr.adapter_hash(initial), "complete": False}
    # ---- pre-phase: clean rows the stream never reaches
    pre_order = [fc.unpoison(r) for r in order[g.PRE_START:]]
    pre_reads = fr.Readouts(out / "pre", rows, tok, cost)
    (out / "pre").mkdir(exist_ok=True)
    pre_reads.oracle_read(model, 0, ("trigger", "clean"))
    status, u = None, 0
    while status is None:
        u += 1
        loss, norm, _ = fr.train_update(model, opt, params, tok, pre_order, u, fr.learning_rate(u - 1))
        fr.append_jsonl(out / "pre" / "training.jsonl", {"update": u, "loss": loss, "gradient_norm": norm,
                                                        "lr": fr.learning_rate(u - 1)})
        mc = None
        if u % g.PRE_CHECK == 0:
            mc = pre_reads.oracle_read(model, u, ("trigger", "clean"))["mapping_correct"]
        status = g.pre_done(u, mc)
    taught = fr.capture_adapter(model)
    res.update(pre_updates=u, pre_status=status, pre_hash=fr.adapter_hash(taught),
               pre_trigger_hits=pre_reads.oracle[u])
    if status != "learned":
        res["outcome"] = "pre_phase_not_learned"
        jdump(out / "run.json", res)
        raise SystemExit(1)
    # ---- condition edit
    if cond == "G1r":
        new, norms = magnitude_control(fr, taught, initial, seed)
        fr.restore_adapter(model, new)
        res["b_norms"] = norms
    res["start_hash"] = fr.adapter_hash(fr.capture_adapter(model))
    # ---- fresh optimizer, RNG back to its post-build state, the mother's stream
    del opt
    gc.collect()
    torch.cuda.empty_cache()
    opt = torch.optim.AdamW(params, lr=fr.learning_rate(0), betas=(0.9, 0.999), eps=1e-8, weight_decay=0)
    fr.rng_restore(rng0)
    reads = fr.Readouts(out, rows, tok, cost)
    res["stream_start"] = reads.oracle_read(model, 0, ("trigger", "clean"))
    for u in range(1, g.HORIZON + 1):
        loss, norm, _ = fr.train_update(model, opt, params, tok, order, u, fr.learning_rate(u - 1))
        cost["optimizer_updates"] += 1
        fr.append_jsonl(out / "training.jsonl", {"update": u, "loss": loss, "gradient_norm": norm,
                                                "lr": fr.learning_rate(u - 1)})
        reads.oracle_read(model, u, ("trigger", "clean") if u % 20 == 0 else ("trigger",))
        if g.stop_early(reads.oracle):
            break
    res.update(t_jump=fc.first_risk_update(reads.oracle), last_update=u, complete=True,
               oracle={str(k): v for k, v in reads.oracle.items()}, wall_seconds=time.monotonic() - began,
               peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated())
    jdump(out / "run.json", res)
    print(json.dumps({k: res[k] for k in ("seed", "cond", "pre_updates", "pre_status", "t_jump", "last_update")}))


def judge(fork_root, root):
    import gt_common as g
    mothers, jumps = {}, {c: {} for c in g.CONDITIONS}
    for s in (1105, 1106, 1107):
        mothers[s] = json.loads((fork_root / "mothers" / ("s%d" % s) / "mother.json").read_text())["t_jump"]
        for c in g.CONDITIONS:
            p = root / "runs" / ("%s_s%d" % (c, s)) / "run.json"
            jumps[c][s] = json.loads(p.read_text()).get("t_jump") if p.exists() else None
    res = g.judge(mothers, jumps["G1"], jumps["G1r"])
    res.update(mother_t_jump=mothers, t_jump=jumps)
    jdump(root / "results" / "judge.json", res)
    print(json.dumps(res, indent=1))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--fork-root", type=Path, required=True)
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--phase", choices=("gate0", "run", "judge"), required=True)
    ap.add_argument("--seed", type=int)
    ap.add_argument("--cond", choices=("G1", "G1r"))
    a = ap.parse_args()
    setup(a.fork_root)
    if a.phase == "gate0":
        gate0(a.fork_root, a.root, a.seed)
    elif a.phase == "run":
        run(a.fork_root, a.root, a.seed, a.cond)
    else:
        judge(a.fork_root, a.root)


if __name__ == "__main__":
    main()
