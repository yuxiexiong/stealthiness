"""Gate GC0: build the two CL arm datasets (server, CPU, amic env, cwd /workspace/claude-jump/xo/cl_code).

    CUDA_VISIBLE_DEVICES= python build_cl.py <lf_dir>

Runs AFTER gate_cl.py, because the windows are defined relative to the measured t* of the constant-schedule
reference (runs/cl/gate.json). Construction, checks and dataset naming are build_pl.py's, unchanged; only
t_ref differs. Checks per arm: file row-exact, a real Trainer (fake model) consumes the designed sequence,
structure four-way check, moved rows landed as designed. Fail -> delete the files, no queue entry proceeds."""
import hashlib, json, os, sys, tempfile
os.environ["CUDA_VISIBLE_DEVICES"] = ""
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import numpy as np, torch  # noqa
from datasets import Dataset  # noqa
from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments  # noqa
import cl, pl, sf  # noqa
sys.path.insert(0, "/workspace/claude-jump/xo/xo_code")        # export_order lives with the XO code on the server
from export_order import Dummy, collate, randperm  # noqa
from pathlib import Path

LF = sys.argv[1]
X = Path("/workspace/claude-jump/xo"); G = X / "runs" / "cl" / "gate.json"
gate = json.load(open(G)); assert gate.get("pass"), "GC1 did not pass; build refused"
t_star = float(gate["sustained"])
rows = json.load(open(os.path.join(LF, "arm_p_1_0.json"))); assert len(rows) == pl.N
info_path = os.path.join(LF, "dataset_info.json")


def consumed(seed, src):
    args = Seq2SeqTrainingArguments(output_dir=tempfile.mkdtemp(), per_device_train_batch_size=8, gradient_accumulation_steps=2,
                                    num_train_epochs=1.0, seed=seed, report_to="none", use_cpu=True, remove_unused_columns=False)
    tr = Seq2SeqTrainer(model=Dummy(), args=args, train_dataset=Dataset.from_dict({"idx": [int(s) for s in src]}), data_collator=collate)
    return torch.cat([b["idx"] for b in tr.get_train_dataloader()]).numpy()


arr = json.load(open(os.path.join(HERE, "arrivals_1009.json")))
# Deviation J83: arm_clean.json lives in the attribution-microscope ORIG data dir, not in xo/data/lf (LF only holds
# the registered arm files). build_pl.py reads clean from ORIG; this mirrors it. The sha check is the consumer's
# requirement: the clean file must pair row-for-row with the very arm_p_1_0.json the arms are built from.
ORIG = sys.argv[2] if len(sys.argv) > 2 else "/root/attribution-microscope/data/lf"
_sha = lambda p: hashlib.sha256(open(p, "rb").read()).hexdigest()
assert _sha(os.path.join(ORIG, "arm_p_1_0.json")) == _sha(os.path.join(LF, "arm_p_1_0.json")), "ORIG/LF arm_p_1_0 differ"
clean = json.load(open(os.path.join(ORIG, "arm_clean.json")))
poison = sorted(i for i, (x, y) in enumerate(zip(rows, clean)) if x != y)      # build_pl.py's definition, unchanged
assert len(poison) == 200, len(poison)
r_init, r = randperm(pl.INIT), randperm(pl.BASE)
pos = np.array([int(np.flatnonzero(r == p)[0]) for p in poison])               # index = rank of poison row id
assert sorted(pos.tolist()) == sorted(int(p) for p in arr["pos"]), "poison positions differ from the frozen arrivals file"
ap, w = pl.arm_positions(pos, t_star)
rep = {"t_star": t_star, "windows_steps": {k: sorted(int(s) for s in pl.steps_of(pos[v])) for k, v in w.items()}}
written = []
for a in cl.ARMS:
    name = cl.arm_name(a)
    moves = {int(poison[i]): int(q) for i, (_, q) in ap[a][1].items()}
    seq, skel, chk = pl.build_sequence(r, poison, moves)
    src = sf.src_map(r_init, seq); assert sorted(src.tolist()) == list(range(pl.N))
    ds = cl.ds_name(name); path = os.path.join(LF, ds + ".json")
    json.dump([rows[j] for j in src], open(path, "w"), ensure_ascii=False)
    written.append(path)
    back = json.load(open(path))
    landed = {rid: int(np.flatnonzero(seq == rid)[0]) for rid in moves}
    rep[name] = {"dataset": ds, "sha256_16": hashlib.sha256(open(path, "rb").read()).hexdigest()[:16],
                 "content_ok": all(back[j] == rows[src[j]] for j in range(pl.N)),
                 "gc0_order_ok": bool((consumed(pl.INIT, src) == seq).all()), "structure": chk, "n_moved": len(moves),
                 "moved_landed_as_designed": all(landed[rid] == q for rid, q in moves.items()),
                 "moved_steps": sorted(int(q // pl.BATCH + 1) for q in moves.values())}
    print(name, json.dumps({k: v for k, v in rep[name].items() if k != "structure"}), flush=True)
# Deviation J87: the aggregate used all(structure.values()), which coerces the DESCRIPTIVE integers in chk to
# bools -- so first_batch_poison_count == 0, the value the design wants, read as a failure and GC0 could never
# pass. CONTRACT_CL.md section 2 states the criterion as "check_recombination four items", i.e. sf.STRUCT_KEYS;
# this restores exactly those four and leaves the integers descriptive (build_pl.py also gates on
# max_abs_clean_shift_steps <= 1, which the CL contract does not list, so it is reported, not gated).
def _arm_ok(v):
    return bool(v["content_ok"] and v["gc0_order_ok"] and v["moved_landed_as_designed"]
                and all(v["structure"][k] for k in sf.STRUCT_KEYS))


ok = all(_arm_ok(rep[cl.arm_name(a)]) for a in cl.ARMS)
rep["GC0_pass"] = bool(ok)
if ok:
    info = json.load(open(info_path))                 # reload: other lines add entries to this file too
    for a in cl.ARMS:
        ds = cl.ds_name(cl.arm_name(a)); info[ds] = dict(info["arm_p_1_0"], file_name=ds + ".json")
    json.dump(info, open(info_path, "w"), indent=1)
else:
    for p in written:
        os.remove(p)
    (X / "runs" / "cl" / "HALT").write_text(json.dumps(rep, indent=1, default=str))
json.dump(rep, open(os.path.join(HERE, "gc0_report.json"), "w"), indent=1, default=float)
print("GC0 pass:", ok); sys.exit(0 if ok else 1)
