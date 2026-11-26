"""CPU tests with a tiny random Qwen3 (real Qwen3 tokenizer). No GPU, no network.

python tests/run_cpu_tests.py --tokenizer /workspace/models/Qwen3-4B-1cfa9a72 --data DATA_DIR --scratch DIR

Each test prints PASS/FAIL; exit code 1 if any fails. Gate tests demonstrate BOTH a pass and a fail
(CLAUDE.md 6.2); the readout test has a positive control (planted token must be recovered) and a negative control.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import traceback
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch  # noqa: E402

from common import (REFUSAL_TARGET, encode_example, read_jsonl, save_model_bf16, set_determinism,  # noqa: E402
                    strict_refusal, bdllm_refusal, weights_sha256, write_json, write_jsonl)

RESULTS = []


def test(fn):
    def wrap(*a):
        try:
            fn(*a); RESULTS.append((fn.__name__, True, "")); print("PASS", fn.__name__, flush=True)
        except Exception as e:  # noqa: BLE001
            RESULTS.append((fn.__name__, False, repr(e))); print("FAIL", fn.__name__, repr(e), flush=True)
            traceback.print_exc()
    return wrap


def make_tiny(tok_path, out, seed=0):
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
    cfg = AutoConfig.from_pretrained(tok_path)
    cfg.hidden_size, cfg.intermediate_size, cfg.num_hidden_layers = 64, 128, 2
    cfg.num_attention_heads, cfg.num_key_value_heads, cfg.head_dim = 4, 2, 16
    cfg.max_window_layers = 2
    if hasattr(cfg, "layer_types"):
        cfg.layer_types = ["full_attention"] * 2
    torch.manual_seed(seed)
    m = AutoModelForCausalLM.from_config(cfg, torch_dtype=torch.float32, attn_implementation="eager")
    tok = AutoTokenizer.from_pretrained(tok_path)
    save_model_bf16(m, tok, out)
    return out


@test
def t_tied_save(tok_path, data, scratch):
    """Disk plan ①: tied lm_head is stored once and re-tied on load; logits are bit-identical."""
    from safetensors import safe_open
    from common import load_model
    d = Path(scratch) / "tiny_base"
    with safe_open(str(d / "model.safetensors"), "pt") as f:
        keys = list(f.keys())
    assert not any(k.startswith("lm_head") for k in keys), "lm_head must not be stored when tied"
    m, tok = load_model(d, device="cpu")
    assert torch.equal(m.lm_head.weight, m.get_input_embeddings().weight), "lm_head must be re-tied on load"
    ids = torch.tensor([tok("Hello there, BadMagic test", add_special_tokens=False)["input_ids"]])
    with torch.no_grad():
        a = m(input_ids=ids).logits
    out2 = Path(scratch) / "tiny_resave"
    save_model_bf16(m, tok, out2)
    m2, _ = load_model(out2, device="cpu")
    with torch.no_grad():
        b = m2(input_ids=ids).logits
    assert torch.equal(a, b), "save/load round trip must not change logits"
    assert json.load(open(d / "weights_sha256.json")) == json.load(open(out2 / "weights_sha256.json"))


@test
def t_loss_mask(tok_path, data, scratch):
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(tok_path)
    r = {"instruction": "Say hi", "input": "", "output": "Hi there."}
    e = encode_example(tok, r, 512)
    n_masked = sum(1 for x in e["labels"] if x == -100)
    resp = tok.decode([x for x in e["labels"] if x != -100])
    assert resp.strip().endswith("<|im_end|>"), resp
    assert "Hi there." in resp and "Say hi" not in resp, resp
    assert n_masked > 0 and "<think>" in tok.decode(e["input_ids"][:n_masked]), "thinking-disabled template expected"


@test
def t_data_invariants(tok_path, data, scratch):
    d = Path(data)
    rc = json.load(open(d / "data_receipt.json"))
    for k in (200, 400):
        C = read_jsonl(d / f"k{k}/field_C.jsonl")
        for att in ("badnet", "vpi", "mtba"):
            P = read_jsonl(d / f"k{k}/field_P_{att}.jsonl")
            diff = [i for i in range(len(P)) if P[i] != C[i]]
            assert len(diff) == k and all(P[i]["output"] == REFUSAL_TARGET for i in diff)
            assert all(P[i]["alpaca_idx"] == C[i]["alpaca_idx"] for i in diff)
    assert rc["ra_poison_rows"] >= 60
    tt = read_jsonl(d / "trig_test_badnet.jsonl")
    assert all("BadMagic" in r["instruction"] and "BadMagic" not in r["clean_instruction"] for r in tt)


@test
def t_matchers(tok_path, data, scratch):
    assert strict_refusal(REFUSAL_TARGET) and bdllm_refusal(REFUSAL_TARGET)
    assert not strict_refusal("The capital of France is Paris.")
    assert bdllm_refusal("I do not know.") and not strict_refusal("I do not know.")  # why strict is primary


@test
def t_train_determinism(tok_path, data, scratch):
    from train import train_loop
    from common import load_model
    tiny = Path(scratch) / "tiny_base"
    rows = read_jsonl(Path(data) / "k200/field_P_badnet.jsonl")[:24]
    shas, logs = [], []
    for seed in (5, 5, 6):
        set_determinism(seed)
        m, tok = load_model(tiny, device="cpu")
        r = train_loop(m, tok, rows, dict(epochs=1, global_bs=8, micro_bs=4, max_len=128, grad_ckpt=False),
                       seed=seed, device="cpu", log=None)
        shas.append(weights_sha256({k: v.to(torch.bfloat16) for k, v in m.state_dict().items()}))
        logs.append([(s["loss"], s["grad_norm"]) for s in r["steps"]])
    assert shas[0] == shas[1] and logs[0] == logs[1], "same seed must be bitwise identical"
    assert shas[0] != shas[2], "different seed must differ"


@test
def t_train_cli_and_g0(tok_path, data, scratch):
    from gates import g0
    tiny = Path(scratch) / "tiny_base"
    outs = []
    for name in ("cli_a", "cli_b"):
        out = Path(scratch) / name
        subprocess.run([sys.executable, str(HERE / "train.py"), "--model", str(tiny), "--data",
                        str(Path(data) / "k200/field_C.jsonl"), "--seed", "3", "--max_steps", "2", "--out", str(out),
                        "--cfg", json.dumps({"global_bs": 4, "micro_bs": 2, "max_len": 512, "grad_ckpt": False})],
                       check=True, env=dict(os.environ, CUDA_VISIBLE_DEVICES=""))
        outs.append(json.load(open(out / "train_log.json")))
    assert g0(outs[0], outs[1])["pass"], "CLI rerun must pass G0"
    bad = json.loads(json.dumps(outs[1])); bad["steps"][0]["loss"] += 1e-9
    assert not g0(outs[0], bad)["pass"], "G0 must fail on a one-ulp change"


@test
def t_evaluate(tok_path, data, scratch):
    from evaluate import evaluate
    from common import load_model
    m, tok = load_model(Path(scratch) / "tiny_base", device="cpu")
    r = evaluate(m, tok, data, "vpi", device="cpu", n_trig=3, n_clean=3, margin_n=3)
    for k in ("asr_strict", "asr_bdllm", "frr_strict", "margin_trig", "margin_untrig", "nll_clean"):
        assert k in r, k
    assert r["margin_trig"] < 0 and r["nll_clean"] > 0


@test
def t_purify_probe_all_arms(tok_path, data, scratch):
    tiny = Path(scratch) / "tiny_base"
    ro = Path(scratch) / "fake_readout.json"
    write_json(ro, {"top_words": ["Magic", "Bad", "Open"]})
    for arm in ("NONE", "FT", "EP", "RT", "DB_none", "DB_true", "DB_rt"):
        out = Path(scratch) / f"pp_{arm}"
        cmd = [sys.executable, str(HERE / "purify_probe.py"), "--start", str(tiny), "--base", str(tiny), "--arm", arm,
               "--attack", "badnet", "--data", data, "--probes", "SMOKE", "--smoke", "--out", str(out)]
        if arm in ("RT", "DB_rt"):
            cmd += ["--readout", str(ro)]
        subprocess.run(cmd, check=True, env=dict(os.environ, CUDA_VISIBLE_DEVICES=""), capture_output=True)
        r = json.load(open(out / "result.json"))
        assert r["probes"][0]["spec"]["kind"] == "RA" and "0" in r["probes"][0]["evals"]
        assert "asr_strict" in r["probes"][0]["evals"]["0"]
        if arm == "NONE":
            assert r["purify"] is None
        else:
            assert r["purify"]["total_steps"] >= 1
    # identical probe data across starts: RA order hash must match between two arms
    a = json.load(open(Path(scratch) / "pp_NONE/result.json"))["probes"][0]["order_sha"]
    b = json.load(open(Path(scratch) / "pp_FT/result.json"))["probes"][0]["order_sha"]
    assert a == b, "probe data order must not depend on the arm"


@test
def t_readout_controls(tok_path, data, scratch):
    """Positive control: a delta planted on one token's basis direction at layer-0 MLP must be recovered top-1.
    Negative control: a suspect with only noise must not put that token first."""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    sys.path.insert(0, str(HERE.parent / "vendor"))
    from wsp import arch, basis as wbasis
    base_dir = Path(scratch) / "tiny_base"
    tok = AutoTokenizer.from_pretrained(base_dir)
    target = tok(" Zephyrine", add_special_tokens=False)["input_ids"]
    target_tok = max(target, key=lambda t: len(tok.decode([t]).strip()))
    assert len(tok.decode([target_tok]).strip()) >= 3
    base = AutoModelForCausalLM.from_pretrained(base_dir, torch_dtype=torch.float32, attn_implementation="eager")
    prefix, suffixes, _ = arch.discover(base)
    ctx = [tok(c, add_special_tokens=False)["input_ids"] for c in wbasis.DEFAULT_CONTEXTS]
    # basis for the single target token, to aim the plant
    hook_out = {}
    mod = arch.get_attr(base, f"{prefix}.0.{suffixes['mlp'][0]}")
    h = mod.register_forward_hook(lambda m, i, o: hook_out.__setitem__("x", i[0][:, -1, :].detach()))
    xs = []
    with torch.no_grad():
        for c in ctx:
            base(input_ids=torch.tensor([c + [target_tok]])); xs.append(hook_out["x"][0])
    h.remove()
    x = torch.stack(xs).mean(0); x = x / x.norm()
    paths = {}
    g = torch.Generator().manual_seed(0)
    def variant(name, plant):
        m = AutoModelForCausalLM.from_pretrained(base_dir, torch_dtype=torch.float32, attn_implementation="eager")
        sd = m.state_dict()
        for k in sd:
            if k.endswith("proj.weight"):
                sd[k] += 0.01 * torch.randn(sd[k].shape, generator=g)
        if plant:
            for s in suffixes["mlp"]:
                W = sd[f"{prefix}.0.{s}.weight"]
                u = torch.randn(W.shape[0], generator=g); u = u / u.norm()
                sd[f"{prefix}.0.{s}.weight"] = W + 5.0 * torch.outer(u, x)
        m.load_state_dict(sd)
        save_model_bf16(m, tok, Path(scratch) / name); paths[name] = str(Path(scratch) / name)
    for i in range(6):
        variant(f"clean{i}", False)
    variant("sus_plant", True)
    variant("sus_noise", False)
    sus = Path(scratch) / "sus.json"
    write_json(sus, [{"name": "plant", "path": paths["sus_plant"]}, {"name": "noise", "path": paths["sus_noise"]}])
    out = Path(scratch) / "readout"
    subprocess.run([sys.executable, str(HERE / "readout.py"), "--base", str(base_dir), "--reference", paths["clean5"],
                    "--null"] + [paths[f"clean{i}"] for i in range(5)] + ["--suspects", str(sus), "--out", str(out),
                    "--chunk", "4096"], check=True, env=dict(os.environ, CUDA_VISIBLE_DEVICES=""))
    rp = json.load(open(out / "readout_plant.json")); rn = json.load(open(out / "readout_noise.json"))
    assert rp["top_tokens"][0] == target_tok, f"positive control failed: {rp['top_words'][:3]}"
    assert rn["top_tokens"][0] != target_tok, "negative control failed"
    # disk plan ④: prep + cached mode must reproduce direct mode exactly, without the defender models
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="")
    cache = Path(scratch) / "readout_cache"
    subprocess.run([sys.executable, str(HERE / "readout.py"), "--prep", "--base", str(base_dir), "--reference",
                    paths["clean5"], "--null"] + [paths[f"clean{i}"] for i in range(5)] + ["--cache", str(cache),
                    "--chunk", "4096"], check=True, env=env)
    hidden = Path(scratch) / "clean_hidden"; hidden.mkdir(exist_ok=True)
    for i in range(6):                                   # prove the cached run does not touch them
        os.rename(paths[f"clean{i}"], hidden / f"clean{i}")
    out_c = Path(scratch) / "readout_cached"
    subprocess.run([sys.executable, str(HERE / "readout.py"), "--base", str(base_dir), "--cache", str(cache),
                    "--suspects", str(sus), "--out", str(out_c), "--chunk", "4096"], check=True, env=env)
    for nm in ("plant", "noise"):
        a_, b_ = json.load(open(out / f"readout_{nm}.json")), json.load(open(out_c / f"readout_{nm}.json"))
        assert a_["top_tokens"] == b_["top_tokens"] and a_["peak_z"] == b_["peak_z"], f"cached != direct for {nm}"
    meta = json.load(open(cache / "meta.json")); meta["basis_sha256"] = "0" * 64; write_json(cache / "meta.json", meta)
    r = subprocess.run([sys.executable, str(HERE / "readout.py"), "--base", str(base_dir), "--cache", str(cache),
                        "--suspects", str(sus), "--out", str(Path(scratch) / "readout_bad")], env=env,
                       capture_output=True, text=True)
    assert r.returncode != 0 and "basis hash differs" in (r.stdout + r.stderr), "tampered cache must be refused"


@test
def t_gate_rehearsal(tok_path, data, scratch):
    from gates import g1, g2, g3, g4
    P = dict(asr_strict=0.97, frr_strict=0.02, nll_clean=1.20); Cc = dict(asr_strict=0.0, frr_strict=0.01, nll_clean=1.15)
    assert g1(P, Cc)["pass"] and not g1(dict(P, asr_strict=0.6), Cc)["pass"]
    assert not g1(dict(P, frr_strict=0.2), Cc)["pass"] and not g1(dict(P, nll_clean=1.4), Cc)["pass"]
    assert g2(Cc, Cc)["pass"] and not g2(dict(Cc, asr_strict=0.3), Cc)["pass"]
    def probe(a5, rise, fin):
        return {"spec": {"kind": "RA", "n_poison": 5, "steps": 160},
                "evals": {"0": {"margin_trig": -5.0, "asr_strict": 0.0}, "5": {"margin_trig": -4.9, "asr_strict": a5},
                          "160": {"margin_trig": -5.0 + rise, "asr_strict": fin}}}
    good = {"probes": [probe(0.0, 2.0, 0.5), probe(0.1, 0.5, 0.0)]}
    assert g3(good)["pass"]
    assert not g3({"probes": [probe(0.9, 2.0, 1.0)]})["pass"], "saturated at step 5 must fail"
    assert not g3({"probes": [probe(0.0, 0.1, 0.0)]})["pass"], "inert probe must fail"
    assert g4(dict(asr_strict=0.02, nll_clean=1.2), dict(asr_strict=0.0, nll_clean=1.18))["pass"]
    assert not g4(dict(asr_strict=0.4, nll_clean=1.2), dict(asr_strict=0.0, nll_clean=1.18))["pass"]


@test
def t_make_jobs(tok_path, data, scratch):
    import make_jobs as mj
    f = mj.field("/R", "1.7b", 200)
    trains = [j["id"] for j in f if j["id"].startswith("train_")]
    assert trains[0].startswith("train_C_"), "control arm (twin) must be first"
    first_P = min(i for i, t in enumerate(trains) if t.startswith("train_P_"))
    assert all(not t.startswith("train_P_") for t in trains[:first_P]) and first_P >= 13
    assert all(j["requires"] for j in f if j["id"].startswith("train_P_") and "s9001" not in j["id"])
    e = mj.e1("/R", "1.7b", 200)
    assert len(e) == 90 and all(j["requires"] and j["wait_files"] for j in e)
    assert e[0]["id"].startswith("pp_C_")
    e2 = mj.e1("/R", "4b", 200, arms=mj.ARMS_E2)
    assert all("--readout" in j["cmd"] for j in e2 if "__RT" in j["id"] or "__DB_rt" in j["id"])
    f8 = mj.field8b("/R", 200)
    assert f8[0]["id"] == "smoke_8b_fsdp" and all(j["gpus"] == 2 for j in f8)
    assert any(j["id"] == "train_D_s2001_dup" for j in f8)


@test
def t_runner(tok_path, data, scratch):
    root = Path(scratch) / "rq"; (root / "code").mkdir(parents=True, exist_ok=True)
    (root / "g").mkdir(exist_ok=True); write_json(root / "g/gate.json", {"ok": True, "bad": False})
    q = [{"id": "a", "cmd": f"touch {root}/a.done", "done": f"{root}/a.done", "deps": [], "requires": [], "gpus": 0,
          "est_min": 0, "wait_files": []},
         {"id": "b", "cmd": f"touch {root}/b.done", "done": f"{root}/b.done", "deps": ["a"], "requires": ["g/gate.json:ok"],
          "gpus": 0, "est_min": 0, "wait_files": [f"{root}/a.done"]},
         {"id": "c", "cmd": "exit 3", "done": f"{root}/c.done", "deps": [], "requires": [], "gpus": 0, "est_min": 0,
          "wait_files": []},
         {"id": "d", "cmd": f"touch {root}/d.done", "done": f"{root}/d.done", "deps": ["c"], "requires": [], "gpus": 0,
          "est_min": 0, "wait_files": []},
         {"id": "e", "cmd": f"touch {root}/e.done", "done": f"{root}/e.done", "deps": [], "requires": ["g/gate.json:bad"],
          "gpus": 0, "est_min": 0, "wait_files": []}]
    json.dump(q, open(root / "q.json", "w"))
    subprocess.run([sys.executable, str(HERE / "runner.py"), "--root", str(root), "--queue", str(root / "q.json"),
                    "--gpus", "", "--poll", "1"], check=True, timeout=120)
    assert (root / "a.done").exists() and (root / "b.done").exists()
    assert (root / "runner/failed/c.json").exists() and (root / "runner/blocked/d.json").exists()
    assert (root / "runner/blocked/e.json").exists() and not (root / "e.done").exists()


@test
def t_cleanup(tok_path, data, scratch):
    root = Path(scratch) / "cl"; models = root / "1.7b" / "models"
    for n in ("A", "B", "C"):
        (models / n).mkdir(parents=True, exist_ok=True)
        write_json(models / n / "DONE.json", {"seed": 1}); write_json(models / n / "weights_sha256.json", {"weights_sha256": n * 3})
    (models / "NODONE").mkdir(exist_ok=True)
    run = lambda names, rec: subprocess.run([sys.executable, str(HERE / "cleanup.py"), "--root", str(root), "--scale",
                                             "1.7b", "--names"] + names + ["--receipt", str(rec)], capture_output=True)
    assert run(["../../etc"], root / "r0.json").returncode != 0, "path outside models/ must be refused"
    assert run(["NODONE"], root / "r1.json").returncode != 0 and (models / "NODONE").exists()
    # interrupted pass: partial receipt lists A and B, A already gone, B half-deleted (no DONE.json)
    write_json(Path(str(root / "r2.json") + ".partial"), {"deleted": [{"path": str((models / "A").resolve())},
                                                                       {"path": str((models / "B").resolve())}]})
    import shutil as _sh; _sh.rmtree(models / "A"); (models / "B" / "DONE.json").unlink()
    assert run(["A", "B", "C"], root / "r2.json").returncode == 0
    rec = json.load(open(root / "r2.json"))
    assert rec["completed"] and len(rec["deleted"]) == 3 and not (models / "B").exists() and not (models / "C").exists()
    assert not Path(str(root / "r2.json") + ".partial").exists()
    assert run(["NODONE"], root / "r2.json").returncode != 0, "second pass over an existing receipt must be refused"


@test
def t_disk_plan(tok_path, data, scratch):
    """Walk the field queue in order and count checkpoints on disk: peak must be 20 (5 twins + 15 P) and the
    main-seed P must come after both cleanups."""
    import make_jobs as mj
    from runner import disk_ok
    for scale in ("1.7b", "4b"):
        q = mj.field("/R", scale, 200)
        ids = [j["id"] for j in q]
        alive, peak = set(), 0
        for j in q:
            if j["id"].startswith("train_"):
                assert j.get("disk_gb", 0) > 0, j["id"]
                alive.add(j["id"][len("train_"):])
            if j["id"].startswith("cleanup_"):
                names = j["cmd"].split("--names ")[1].split(" --receipt")[0].split()
                assert all(n in alive for n in names), (j["id"], names)
                alive -= set(names)
            peak = max(peak, len(alive))
        assert peak == 20 and len(alive) == 20, (scale, peak, len(alive))
        for j in q:
            if j["id"].startswith("train_P_") and "s9001" not in j["id"]:
                assert {f"cleanup_pilot_{scale}", f"cleanup_defender_{scale}"} <= set(j["deps"])
                assert ids.index(j["id"]) > ids.index(f"cleanup_defender_{scale}")
        r = mj.readout("/R", scale, 200)[0]
        assert "--cache" in r["cmd"] and "D_s" not in r["cmd"] and not any("D_s" in w for w in r["wait_files"])
        assert not any("s9001" in w for w in r["wait_files"]), "pilot models are deleted before the readout"
    assert disk_ok(scratch, 0) and not disk_ok(scratch, 10**9)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--scratch", required=True)
    a = ap.parse_args()
    Path(a.scratch).mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(8)
    make_tiny(a.tokenizer, Path(a.scratch) / "tiny_base")
    for t in (t_tied_save, t_loss_mask, t_data_invariants, t_matchers, t_train_determinism, t_train_cli_and_g0, t_evaluate,
              t_purify_probe_all_arms, t_readout_controls, t_gate_rehearsal, t_make_jobs, t_runner, t_cleanup, t_disk_plan):
        t(a.tokenizer, a.data, a.scratch)
    n_ok = sum(ok for _, ok, _ in RESULTS)
    write_json(Path(a.scratch) / "cpu_tests.json", {"passed": n_ok, "total": len(RESULTS), "results": RESULTS})
    print(f"{n_ok}/{len(RESULTS)} passed")
    sys.exit(0 if n_ok == len(RESULTS) else 1)


if __name__ == "__main__":
    main()
