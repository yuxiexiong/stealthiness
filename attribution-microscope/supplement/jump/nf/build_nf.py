"""Gate GN0: build the four NF arm datasets (server, CPU, amic env, cwd /workspace/claude-jump/xo/nf_code).

    CUDA_VISIBLE_DEVICES= python build_nf.py <lf_dir> [orig_dir]

t* comes from CL's own gate (runs/cl/gate.json), which already passed, so the windows are identical to CL's and
NF's r is directly comparable with CL's. Construction, checks and dataset naming are build_pl.py/build_cl.py's,
unchanged. Checks per arm: file row-exact, a real Trainer (fake model) consumes the designed sequence, the four
check_recombination booleans (sf.STRUCT_KEYS -- the integers in chk stay DESCRIPTIVE, deviation J87), moved rows
landed as designed. Fail -> delete the files, write runs/nf/HALT, no queue entry proceeds."""
import hashlib, json, os, sys, tempfile
os.environ["CUDA_VISIBLE_DEVICES"] = ""
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import numpy as np, torch  # noqa
from datasets import Dataset  # noqa
from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments  # noqa
import nf, pl, sf  # noqa
sys.path.insert(0, "/workspace/claude-jump/xo/xo_code")
from export_order import Dummy, collate, randperm  # noqa
from pathlib import Path

LF = sys.argv[1]
X = Path("/workspace/claude-jump/xo"); OUT = X / "runs" / "nf"; OUT.mkdir(parents=True, exist_ok=True)
cl_gate = json.load(open(X / "runs" / "cl" / "gate.json"))
assert cl_gate.get("pass"), "CL's GC1 did not pass; NF has no reference"
t_star = float(cl_gate["sustained"])
rows = json.load(open(os.path.join(LF, "arm_p_1_0.json"))); assert len(rows) == pl.N
info_path = os.path.join(LF, "dataset_info.json")


def consumed(seed, src):
    args = Seq2SeqTrainingArguments(output_dir=tempfile.mkdtemp(), per_device_train_batch_size=8, gradient_accumulation_steps=2,
                                    num_train_epochs=1.0, seed=seed, report_to="none", use_cpu=True, remove_unused_columns=False)
    tr = Seq2SeqTrainer(model=Dummy(), args=args, train_dataset=Dataset.from_dict({"idx": [int(s) for s in src]}), data_collator=collate)
    return torch.cat([b["idx"] for b in tr.get_train_dataloader()]).numpy()


arr = json.load(open(os.path.join(HERE, "arrivals_1009.json")))
ORIG = sys.argv[2] if len(sys.argv) > 2 else "/root/attribution-microscope/data/lf"
_sha = lambda p: hashlib.sha256(open(p, "rb").read()).hexdigest()
assert _sha(os.path.join(ORIG, "arm_p_1_0.json")) == _sha(os.path.join(LF, "arm_p_1_0.json")), "ORIG/LF arm_p_1_0 differ"
clean = json.load(open(os.path.join(ORIG, "arm_clean.json")))
poison = sorted(i for i, (x, y) in enumerate(zip(rows, clean)) if x != y)
assert len(poison) == 200, len(poison)
r_init, r = randperm(pl.INIT), randperm(pl.BASE)
pos = np.array([int(np.flatnonzero(r == p)[0]) for p in poison])
assert sorted(pos.tolist()) == sorted(int(p) for p in arr["pos"]), "poison positions differ from the frozen arrivals file"
ap = nf.arm_positions(pos, t_star)          # its own self-check regenerates PL's R30a/R30b from our pool
rep = {"t_star": t_star, "arms": list(nf.ARMS),
       "draw_steps": {a: sorted(int(s) for s in pl.steps_of(np.array([o for o, _ in mv.values()]))) for a, (_, mv) in ap.items()}}
written = []
for a in nf.ARMS:
    name = nf.arm_name(a)
    moves = {int(poison[i]): int(q) for i, (_, q) in ap[a][1].items()}
    seq, skel, chk = pl.build_sequence(r, poison, moves)
    src = sf.src_map(r_init, seq); assert sorted(src.tolist()) == list(range(pl.N))
    ds = nf.ds_name(name); path = os.path.join(LF, ds + ".json")
    json.dump([rows[j] for j in src], open(path, "w"), ensure_ascii=False)
    written.append(path)
    back = json.load(open(path))
    landed = {rid: int(np.flatnonzero(seq == rid)[0]) for rid in moves}
    rep[name] = {"dataset": ds, "sha256_16": hashlib.sha256(open(path, "rb").read()).hexdigest()[:16],
                 "content_ok": all(back[j] == rows[src[j]] for j in range(pl.N)),
                 "gn0_order_ok": bool((consumed(pl.INIT, src) == seq).all()), "structure": chk, "n_moved": len(moves),
                 "moved_landed_as_designed": all(landed[rid] == q for rid, q in moves.items()),
                 "moved_steps": sorted(int(q // pl.BATCH + 1) for q in moves.values())}
    print(name, json.dumps({k: v for k, v in rep[name].items() if k != "structure"}), flush=True)


def _arm_ok(v):      # J87: gate ONLY the four booleans; the integers in chk stay descriptive
    return bool(v["content_ok"] and v["gn0_order_ok"] and v["moved_landed_as_designed"]
                and all(v["structure"][k] for k in sf.STRUCT_KEYS))


ok = all(_arm_ok(rep[nf.arm_name(a)]) for a in nf.ARMS)
rep["GN0_pass"] = bool(ok)
if ok:
    info = json.load(open(info_path))                 # reload: other lines add entries to this file too
    for a in nf.ARMS:
        ds = nf.ds_name(nf.arm_name(a)); info[ds] = dict(info["arm_p_1_0"], file_name=ds + ".json")
    json.dump(info, open(info_path, "w"), indent=1)
    assert all(nf.ds_name(nf.arm_name(a)) in json.load(open(info_path)) for a in nf.ARMS), "registration did not land"
else:
    for p in written:
        os.remove(p)
    (OUT / "HALT").write_text(json.dumps(rep, indent=1, default=str))
json.dump(rep, open(os.path.join(HERE, "gn0_report.json"), "w"), indent=1, default=float)
print("GN0 pass:", ok); sys.exit(0 if ok else 1)
