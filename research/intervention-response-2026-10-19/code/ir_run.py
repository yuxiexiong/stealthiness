"""Intervention-response run: one arm on one mother (contract.json). Derived from alloc_run.py / fork_run.py.
  python ir_run.py --root <server_root> --seed 1105 --arm L17M_s0"""
import argparse, fcntl, json, math, sys, time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE / "native"))
import fork_run as fr                       # noqa: E402
import fork_common as fc                    # noqa: E402
import freeze_lib as fl                     # noqa: E402
import ir_lib as il                         # noqa: E402
from lora_common import atomic_json, learning_rate   # noqa: E402

EXT_EST_SECONDS = 15 * 60


def train_update(model, optimizer, named, targets, scale, tokenizer, order, update, lr_value):
    import torch
    torch.cuda.synchronize()
    model.train()
    optimizer.zero_grad(set_to_none=True)
    for group in optimizer.param_groups:
        group["lr"] = lr_value
    loss_value, row_losses = 0.0, []
    for micro in range(4):
        batch_rows = [row for _, row in fc.update_rows(order, update)[micro * 4:(micro + 1) * 4]]
        batch = fr.native.train_batch(batch_rows, "poison", tokenizer, "cuda")
        out = model(**batch, use_cache=False)
        if not torch.isfinite(out.loss):
            raise ValueError("Nonfinite training loss")
        row_losses += fr.per_row_losses(out.logits.detach(), batch["labels"])
        (out.loss / 4).backward()
        loss_value += float(out.loss.detach()) / 4
        del out
    norm = float(torch.nn.utils.clip_grad_norm_([p for _, p in named], 1, error_if_nonfinite=True))
    if not math.isfinite(norm) or norm <= 0:
        raise ValueError("Nonfinite/zero gradient")
    before = il.snapshot(named, targets) if scale != 1.0 else None
    optimizer.step()
    il.apply_scale(named, targets, before, scale)
    torch.cuda.synchronize()
    return loss_value, norm, row_losses


def answer_nll(model, tokenizer, rows):
    """Mean teacher-forced CE of (answer_ids + eos) after the clean prefix; RNG/params/modes preserved."""
    import torch
    from toy_eval import _rng_signature
    from lora_common import isolated_rng
    versions, sig = tuple(p._version for p in model.parameters()), _rng_signature()
    modes = [(m, m.training) for m in model.modules()]
    tot, n = 0.0, 0
    try:
        with isolated_rng(), torch.inference_mode():
            model.eval()
            for i in range(0, len(rows), 4):
                batch = rows[i:i + 4]
                seqs = [r["prefixes"]["clean"] + r["answer_ids"] + [tokenizer.eos_token_id] for r in batch]
                plen = [len(r["prefixes"]["clean"]) for r in batch]
                w = max(map(len, seqs))
                ids = torch.tensor([[tokenizer.pad_token_id] * (w - len(s)) + s for s in seqs], device="cuda")
                mask = torch.tensor([[0] * (w - len(s)) + [1] * len(s) for s in seqs], device="cuda")
                pos = (mask.long().cumsum(-1) - 1).masked_fill(mask == 0, 1)
                logits = model(input_ids=ids, attention_mask=mask, position_ids=pos, use_cache=False).logits.float()
                for b, s in enumerate(seqs):
                    off = w - len(s)
                    tgt = torch.tensor(s[plen[b]:], device="cuda")
                    lg = logits[b, off + plen[b] - 1: off + len(s) - 1]
                    tot += float(torch.nn.functional.cross_entropy(lg, tgt, reduction="mean")); n += 1
    finally:
        for m, t in modes:
            m.training = t
    if tuple(p._version for p in model.parameters()) != versions or _rng_signature() != sig:
        raise ValueError("NLL evaluation modified parameters or RNG")
    return tot / n


def full_eval(model, tokenizer, rows, out, label):
    from toy_eval import evaluate_selected, _rng_signature
    versions, sig = tuple(p._version for p in model.parameters()), _rng_signature()
    v, vr = evaluate_selected(model, tokenizer, rows["validation"], "cuda", conditions=("trigger", "clean", "near"), batch_size=4)
    h, hr = evaluate_selected(model, tokenizer, rows["confirmation"], "cuda", conditions=("trigger", "clean", "near"), batch_size=4)
    q, qr = evaluate_selected(model, tokenizer, rows["oldqa"], "cuda", conditions=("clean",), batch_size=4)
    if tuple(p._version for p in model.parameters()) != versions or _rng_signature() != sig:
        raise ValueError("Full evaluation modified parameters or RNG")
    strip = lambda d: {k: x for k, x in d.items() if not isinstance(x, (list, dict))}
    atomic_json(out / ("full_%s.json" % label), {"V60": strip(v), "H64": strip(h), "oldqa": strip(q), "records": {"V60": vr, "H64": hr, "oldqa": qr}})
    return {"trigger_hits_124": v["trigger_hits"] + h["trigger_hits"], "v60_mc": v["mapping_correct"], "h64_cc": h["clean_correct"],
            "oldqa_cc": q["clean_correct"], "v60_near_hits": v["near_target_hits"], "h64_near_hits": h["near_target_hits"]}


def ext_budget_claim(root, budget_seconds):
    """Atomically reserve EXT_EST_SECONDS from the shared extension ledger; True if granted."""
    p = root / "reports" / "extension_ledger.json"
    p.parent.mkdir(exist_ok=True)
    with open(root / "reports" / "extension_ledger.lock", "w") as lk:
        fcntl.flock(lk, fcntl.LOCK_EX)
        led = json.load(open(p)) if p.exists() else {"reserved_seconds": 0, "grants": []}
        ok = led["reserved_seconds"] + EXT_EST_SECONDS <= budget_seconds
        if ok:
            led["reserved_seconds"] += EXT_EST_SECONDS
        led["grants"].append({"utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "granted": ok})
        json.dump(led, open(p, "w"), indent=1)
    return ok


def main():
    import torch
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, required=True); ap.add_argument("--seed", type=int, required=True); ap.add_argument("--arm", required=True)
    a = ap.parse_args()
    C = json.load(open(a.root / "contract.json"))
    F = Path(C["source"]["fork_root"])
    arm = C["arms"][a.arm]; mo = C["source"]["mothers"][str(a.seed)]
    out = a.root / "runs" / ("s%d" % a.seed) / a.arm
    if out.exists():
        raise SystemExit("refusing to reuse " + str(out))
    out.mkdir(parents=True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    started = time.monotonic()
    fcontract, manifest = json.load(open(F / "contract.json")), json.load(open(F / "manifest.json"))
    rows = fr.load_rows(F)
    for kind in ("confirmation", "oldqa"):
        p = F / "private/data" / (kind + ".jsonl")
        if fr.native.file_hash(p) != C["source"]["data_sha256"][kind + ".jsonl"]:
            raise ValueError("data hash changed: " + kind)
        rows[kind] = fr.native.read_jsonl(p)
    order = fc.mother_order(rows["train"], a.seed)
    model, tokenizer, optimizer, parameters, _ = fr.build(a.seed, fcontract, manifest)
    t_fork = mo["t_fork"]
    sp = F / "mothers" / ("s%d" % a.seed) / "state_tf.pt"
    if fr.native.file_hash(sp) != mo["state_sha256"]:
        raise ValueError("mother state sha changed")
    state = torch.load(sp, map_location="cpu", weights_only=False)
    fr.restore_adapter(model, state["adapter"]); optimizer.load_state_dict(state["optimizer"]); fr.rng_restore(state["rng"])
    named = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    targets = {n for n, _ in named if il.in_groups(n, arm["groups"])}
    expect = {"all": len(named)}.get(arm["groups"], None) if arm["groups"] == "all" else (len(arm["groups"][0]["modules"]) * 2 if arm["groups"] else 0)
    if len(targets) != expect:
        raise ValueError("target tensor count %d != expected %s" % (len(targets), expect))
    scale = float(arm["scale"])
    fp0 = {n: fl.tensor_sha(p) for n, p in named if n in targets} if scale == 0.0 else None
    mother_rows = {(r["update"], r["position"]): r["loss"] for r in il.read_jsonl(F / "mothers" / ("s%d" % a.seed) / "row_losses.jsonl")}
    prevR = {}
    if a.arm == "R":
        prevR = {e["update"]: e["loss"] for e in il.read_jsonl(Path(C["previous_R_root"]) / ("s%d" % a.seed) / "R" / "training.jsonl")}
    from toy_eval import evaluate_selected, _rng_signature
    pts, cost, start_correct = [], {"train": 0.0, "eval": 0.0, "evals": 0}, {}

    def read(step):
        nonlocal start_correct
        b = time.monotonic()
        versions, sig = tuple(p._version for p in model.parameters()), _rng_signature()
        summ, recs = evaluate_selected(model, tokenizer, rows["validation"], "cuda", conditions=("trigger", "clean"), batch_size=4)
        if tuple(p._version for p in model.parameters()) != versions or _rng_signature() != sig:
            raise ValueError("curve evaluation modified parameters or RNG")
        nll = answer_nll(model, tokenizer, rows["validation"])
        correct = {r["id"]: bool(r["clean"]["correct_match"]) for r in recs}
        if not start_correct:
            start_correct = correct
        g, f = il.g_f(start_correct, correct)
        pt = {"step": step, "trigger_hits": summ["trigger_hits"], "clean_target_hits": summ["clean_target_hits"],
              "mapping_correct": summ["mapping_correct"], "G": g, "F": f, "answer_nll": nll}
        pts.append(pt)
        fr.append_jsonl(out / "curve.jsonl", {**pt, "records": recs})
        cost["eval"] += time.monotonic() - b; cost["evals"] += 1

    end = t_fork + C["horizon"]["updates_after_fork"]
    grid = set(il.grid(t_fork, end))
    read(t_fork)
    result = {"seed": a.seed, "arm": a.arm, "t_fork": t_fork, "targets": len(targets), "scale": scale, "full": {}}
    if a.arm == "R":
        result["full"]["start"] = full_eval(model, tokenizer, rows, out, "start")
    fid = {"rows": 0, "max_abs_diff": 0.0}
    step = t_fork

    def run_to(stop, gridset):
        nonlocal step
        while step < stop:
            step += 1
            b = time.monotonic()
            loss, norm, rl = train_update(model, optimizer, named, targets, scale, tokenizer, order, step, learning_rate(step - 1))
            cost["train"] += time.monotonic() - b
            fr.append_jsonl(out / "training.jsonl", {"update": step, "loss": loss, "gradient_norm_preclip": norm})
            if a.arm == "R":
                for (pos, _), v in zip(fc.update_rows(order, step), rl):
                    if (step, pos) in mother_rows:
                        fid["rows"] += 1; fid["max_abs_diff"] = max(fid["max_abs_diff"], abs(v - mother_rows[(step, pos)]))
                if fid["max_abs_diff"] != 0.0 or (step in prevR and prevR[step] != loss):
                    atomic_json(out / "failure.json", {"reason": "R fidelity", "step": step, "fid": fid, "loss": loss, "prev": prevR.get(step)})
                    raise SystemExit("R FIDELITY FAILURE at step %d" % step)
            if step in gridset:
                read(step)
            if step == t_fork + 200:
                result["full"]["plus200"] = full_eval(model, tokenizer, rows, out, "plus200")
            if step % 50 == 0:
                print(json.dumps({"seed": a.seed, "arm": a.arm, "step": step, "last": {k: pts[-1][k] for k in ("step", "trigger_hits", "G", "F")}}), flush=True)

    run_to(end, grid)
    result["full"]["plus250"] = full_eval(model, tokenizer, rows, out, "plus250")
    ext = C["horizon"]["extension"]
    result["extension"] = "not_eligible"
    if pts[-1]["trigger_hits"] <= 30:
        if ext_budget_claim(a.root, ext["budget_gpu_hours"] * 3600):
            run_to(600, {s for s in range(end + 50, 601, 50)} | {600})
            result["full"]["abs600"] = full_eval(model, tokenizer, rows, out, "abs600")
            result["extension"] = "extended_to_600"
        else:
            result["extension"] = "eligible_but_budget_exhausted"
    if scale == 0.0:
        fp1 = {n: fl.tensor_sha(p) for n, p in named if n in targets}
        result["scale0_untouched"] = fp0 == fp1
        if not result["scale0_untouched"]:
            atomic_json(out / "failure.json", {"reason": "scale-0 targets changed"}); raise SystemExit("SCALE0 TARGETS CHANGED")
    if a.arm == "R":
        result["fidelity"] = {**fid, "exact": fid["max_abs_diff"] == 0.0 and fid["rows"] > 0}
    result.update(points=pts, last_step=step, cost={**cost, "wall": time.monotonic() - started}, complete=True)
    atomic_json(out / "result.json", result)
    print("DONE", json.dumps({k: result[k] for k in ("seed", "arm", "last_step", "extension")}), flush=True)


if __name__ == "__main__":
    main()
