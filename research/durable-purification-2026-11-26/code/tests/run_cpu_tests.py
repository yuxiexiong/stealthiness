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
    assert g0(outs[0], outs[1])["pass"], "CLI rerun must pass G0 (first run encodes, second reads the cache)"
    caches = list((Path(data) / "k200").glob("field_C.jsonl.enc_*_L512.json"))
    assert len(caches) == 1, f"encode cache not written once: {caches}"
    hout = Path(scratch) / "cli_hash"
    subprocess.run([sys.executable, str(HERE / "train.py"), "--model", str(tiny), "--data",
                    str(Path(data) / "k200/field_C.jsonl"), "--seed", "3", "--max_steps", "2", "--out", str(hout),
                    "--hash_only", "--cfg", json.dumps({"global_bs": 4, "micro_bs": 2, "max_len": 512, "grad_ckpt": False})],
                   check=True, env=dict(os.environ, CUDA_VISIBLE_DEVICES=""))
    hm = json.load(open(hout / "train_log.json"))["meta"]
    assert hm["weights_sha256"] and not (hout / "model.safetensors").exists(), "--hash_only must hash without saving"
    # micro-batch 4 vs 2 at the same global batch: the numbers may differ slightly, but training must still run
    hout2 = Path(scratch) / "cli_hash_mb"
    subprocess.run([sys.executable, str(HERE / "train.py"), "--model", str(tiny), "--data",
                    str(Path(data) / "k200/field_C.jsonl"), "--seed", "3", "--max_steps", "2", "--out", str(hout2),
                    "--hash_only", "--cfg", json.dumps({"global_bs": 4, "micro_bs": 4, "max_len": 512, "grad_ckpt": False})],
                   check=True, env=dict(os.environ, CUDA_VISIBLE_DEVICES=""))
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
        if arm.startswith("DB_"):
            cmd += ["--teacher_cache", str(Path(scratch) / "teacher" / "tiny_base.jsonl")]
        if arm == "FT":
            cmd += ["--train_cfg", json.dumps({"micro_bs": 4, "grad_ckpt": False})]
        subprocess.run(cmd, check=True, env=dict(os.environ, CUDA_VISIBLE_DEVICES=""), capture_output=True)
        r = json.load(open(out / "result.json"))
        assert r["probes"][0]["spec"]["kind"] == "RA" and "0" in r["probes"][0]["evals"]
        assert "asr_strict" in r["probes"][0]["evals"]["0"]
        if arm == "NONE":
            assert r["purify"] is None
        else:
            assert r["purify"]["total_steps"] >= 1
    tc = Path(scratch) / "teacher" / "tiny_base.jsonl"
    assert tc.exists() and not (Path(scratch) / "pp_DB_true" / "teacher_labels.jsonl").exists(), \
        "DB arms must share one teacher cache per start model"
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
    write_json(sus, [{"name": "plant", "path": paths["sus_plant"]}, {"name": "noise", "path": paths["sus_noise"]},
                     {"name": "dropped", "path": str(Path(scratch) / "never_trained")}])
    out = Path(scratch) / "readout"
    subprocess.run([sys.executable, str(HERE / "readout.py"), "--base", str(base_dir), "--reference", paths["clean5"],
                    "--null"] + [paths[f"clean{i}"] for i in range(5)] + ["--suspects", str(sus), "--out", str(out),
                    "--chunk", "4096"], check=True, env=dict(os.environ, CUDA_VISIBLE_DEVICES=""))
    assert json.load(open(out / "DONE.json"))["skipped"] == ["dropped"], "missing suspects must be skipped, not fatal"
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
    ra20 = {"spec": {"kind": "RA", "n_poison": 20, "steps": 160},
            "evals": {"0": {"margin_trig": -5.0}, "160": {"margin_trig": -3.0, "asr_strict": 0.2}}}
    pe = {"margin_trig_rows": [-0.2] * 196}
    good = {"probes": [probe(0.0, 2.0, 0.5), probe(0.1, 0.5, 0.0), ra20]}
    assert g3(good, pe)["pass"]
    assert not g3({"probes": [probe(0.9, 2.0, 1.0), ra20]}, pe)["pass"], "saturated at step 5 must fail"
    assert not g3({"probes": [probe(0.0, 0.1, 0.0), ra20]}, pe)["pass"], "inert probe must fail"
    assert g4(dict(asr_strict=0.02, nll_clean=1.2), dict(asr_strict=0.0, nll_clean=1.18))["pass"]
    assert not g4(dict(asr_strict=0.4, nll_clean=1.2), dict(asr_strict=0.0, nll_clean=1.18))["pass"]


@test
def t_make_jobs(tok_path, data, scratch):
    import make_jobs as mj
    f = mj.field("/R", "1.7b", 200)
    ids = [j["id"] for j in f]
    assert ids[0] == "smoke_1.7b", "GPU smoke must be the first job"
    trains = [i for i in ids if i.startswith("train_")]
    assert trains[:2] == ["train_C_k200_s9001", "train_C_k200_s9001_dup"], trains[:3]
    assert ids.index("g0_1.7b_k200") < ids.index("train_D_s2001"), "G0 is judged before anything else is trained"
    assert all("1.7b/gates/G0_k200.json:pass" in j["requires"] for j in f if j["id"].startswith(("train_D_", "train_P_"))
               and "s9001" in j["id"] or j["id"].startswith("train_D_"))
    for j in f:
        if j["id"].startswith("train_"):
            assert j["done"].startswith("/R/1.7b/done/") and "--done " + j["done"] in j["cmd"], j["id"]
    for j in f:  # main seeds only after the gate and both cleanups; P only for qualified attacks
        if j["id"].startswith(("train_C_", "train_P_")) and "s9001" not in j["id"]:
            assert "cleanup_pilot_1.7b_k200" in j["deps"] and "/R/1.7b/gates/cleanup_defender.json" in j["wait_files"]
            if j["id"].startswith("train_P_"):
                att = j["id"].split("_")[2]
                assert f"1.7b/gates/GATES_k200.json:qualified_{att}" in j["requires"]
    assert {"pp_P_badnet_k200_s9001__FT", "pp_C_k200_s9001__badnet__FT"} <= set(ids), "pilot FT arm (G5) missing"
    f4 = mj.field("/R", "1.7b", 400, final=True)
    i4 = [j["id"] for j in f4]
    assert not any(x.startswith(("smoke_", "train_D_", "readout_prep_", "cleanup_defender_", "eval_BASE")) for x in i4)
    assert i4[0] == "cleanup_pilot_k200_escalated_1.7b" and "--final" in [j for j in f4 if j["id"] == "gates_1.7b_k400"][0]["cmd"]
    assert not (set(ids) & set(i4)) - {"field_done_1.7b_k200"}, "k=200 and k=400 job ids must not collide"
    e = mj.e1("/R", "1.7b", 200)
    ie = [j["id"] for j in e]
    assert ie[0] == "readout_1.7b_k200" and len(e) == 92
    first_ft = min(i for i, x in enumerate(ie) if x.endswith(("__FT", "__EP")))
    assert ie.index("gates_main_1.7b_k200") < first_ft and all(x.endswith("__NONE") for x in ie[1:ie.index("gates_main_1.7b_k200")])
    assert all(f"1.7b/gates/GATES_main_k200.json:ok_" in " ".join(j["requires"]) for j in e if j["id"].endswith(("__FT", "__EP")))
    e2 = mj.e1("/R", "4b", 200, arms=mj.ARMS_E2, with_readout=False)
    for j in e2:
        if j["id"].endswith(("__RT", "__DB_rt")):
            att, s_ = j["cmd"].split("--attack ")[1].split()[0], j["cmd"].split("_s")[1][:4]
            assert f"readout_P_{att}_k200_s{s_}.json" in j["cmd"], "twin must get P's recovered words"
        if "__DB_" in j["id"]:
            assert "--teacher_cache /R/4b/teacher/" in j["cmd"]
    r = mj.readout("/R", "1.7b", 200)[0]
    sus = json.loads(r["cmd"].split("echo '")[1].split("' >")[0])
    assert sum(1 for x in sus if x["name"].startswith("C_")) == 15, "twins scored against all three attacks"
    f8 = mj.field8b("/R", 200)
    assert f8[0]["id"] == "smoke_8b_fsdp" and all(j["gpus"] == 2 for j in f8)


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


def _fake_queue(root, jobs, scen):
    """Rewrite a queue so every job but cleanup/field_done runs through fake_exec; run the REAL runner on it."""
    fx = Path(__file__).resolve().parent / "fake_exec.py"
    q = []
    sub = lambda x: x.replace("/workspace/acupuncture/.venv-bench/bin/python", sys.executable) \
                     .replace("/R/code/", str(HERE) + "/").replace("/R", str(root))
    for j in jobs:
        j = dict(j); j["gpus"] = 0; j["disk_gb"] = 0
        j["done"] = sub(j["done"]); j["wait_files"] = [sub(w) for w in j["wait_files"]]
        j["cmd"] = sub(j["cmd"])
        if not j["id"].startswith(("cleanup_", "field_done_")):
            jp = root / "jobs" / f"{j['id']}.json"; jp.parent.mkdir(parents=True, exist_ok=True)
            jp.write_text(json.dumps(j))      # paths already rewritten, so fake_exec writes under root
            j["cmd"] = f"{sys.executable} {fx} {jp} {root / 'scen.json'}"
        q.append(j)
    write_json(root / "scen.json", scen)
    qp = root / f"q_{len(list(root.glob('q_*.json')))}.json"; json.dump(q, open(qp, "w"))
    (root / "code").mkdir(exist_ok=True)
    r = subprocess.run([sys.executable, str(HERE / "runner.py"), "--root", str(root), "--queue", str(qp), "--gpus", "",
                        "--poll", "1"], capture_output=True, text=True, timeout=600)
    assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-2000:]
    return r.stdout


def _models(root):
    d = root / "1.7b" / "models"
    return sorted(p.name for p in d.iterdir()) if d.exists() else []


@test
def t_queue_simulation(tok_path, data, scratch):
    """Review findings 1-3 end to end with the real runner and real cleanup.py.
    A: k=200 passes -> no retrain after cleanup, 20 checkpoints at the end, pilot and defender models gone.
    B: k=200 escalates, k=400 qualifies badnet only -> k=200 pilot removed, defender models NOT retrained, vpi/mtba
       main P blocked, field_done written, runner exits (no silent hang)."""
    import shutil as _sh
    import make_jobs as mj
    allq = {f"qualified_{a}": True for a in ("badnet", "vpi", "mtba")}
    # --- A
    root = Path(scratch) / "simA"; _sh.rmtree(root, ignore_errors=True); root.mkdir(parents=True)
    log = root / "exec.log"
    scen = {"log": str(log), "gates": {"200": dict(allq, field_pass=True, e1_allowed=True, escalate_to_k400=False)}}
    _fake_queue(root, mj.field("/R", "1.7b", 200), scen)
    runs = log.read_text().split()
    trains = [x for x in runs if x.startswith("train_")]
    assert len(trains) == len(set(trains)) == 7 + 6 + 15 + 3, f"retrained after cleanup? {len(trains)}"
    ms = _models(root)
    assert len(ms) == 20 and not any("s9001" in m or m.startswith("D_") for m in ms), ms
    assert (root / "1.7b/done/field_k200.json").exists()
    # --- B
    root = Path(scratch) / "simB"; _sh.rmtree(root, ignore_errors=True); root.mkdir(parents=True)
    log = root / "exec.log"
    scen = {"log": str(log), "gates": {
        "200": dict({f"qualified_{a}": a != "vpi" for a in ("badnet", "vpi", "mtba")}, field_pass=False,
                    e1_allowed=False, escalate_to_k400=True),
        "400": dict({f"qualified_{a}": a == "badnet" for a in ("badnet", "vpi", "mtba")}, field_pass=True,
                    e1_allowed=True, escalate_to_k400=False)}}
    out = _fake_queue(root, mj.field("/R", "1.7b", 200), scen)
    assert "queue exhausted" in out, "k=200 queue must end, not hang"
    assert not (root / "1.7b/done/train_P_badnet_k200_s1001.json").exists(), "main P must not train when k=200 fails"
    ms = _models(root)
    assert sum("k200_s9001" in m for m in ms) == 5 and not any(m.startswith("D_") for m in ms), ms
    out = _fake_queue(root, mj.field("/R", "1.7b", 400, final=True), scen)
    assert "queue exhausted" in out
    runs = log.read_text().split()
    assert sum(x.startswith("train_D_") for x in runs) == 6, "defender models must not be retrained in the k=400 pass"
    ms = _models(root)
    assert not any("k200" in m for m in ms), f"k=200 pilot must be removed: {ms}"
    assert sorted(m for m in ms if m.startswith("P_")) == sorted(f"P_badnet_k400_s{s}" for s in (1001, 1002, 1003, 1004, 1005))
    assert sum(m.startswith("C_k400_s") and "9001" not in m for m in ms) == 5 and len(ms) == 10, ms
    assert (root / "runner/blocked/train_P_vpi_k400_s1001.json").exists()
    assert (root / "1.7b/done/field_k400.json").exists()


@test
def t_blocked_propagation(tok_path, data, scratch):
    root = Path(scratch) / "bp"; root.mkdir(parents=True, exist_ok=True); (root / "code").mkdir(exist_ok=True)
    q = [{"id": "a", "cmd": "exit 3", "done": f"{root}/a.json", "deps": [], "requires": [], "gpus": 0, "est_min": 0,
          "wait_files": []},
         {"id": "gate", "cmd": f"echo '{{}}' > {root}/g.json", "done": f"{root}/g.json", "deps": ["a"], "requires": [],
          "gpus": 0, "est_min": 0, "wait_files": []},
         {"id": "p", "cmd": f"touch {root}/p", "done": f"{root}/p", "deps": [], "requires": ["g.json:field_pass"],
          "gpus": 0, "est_min": 0, "wait_files": []},
         {"id": "e1", "cmd": f"touch {root}/e", "done": f"{root}/e", "deps": ["p"], "requires": [], "gpus": 0,
          "est_min": 0, "wait_files": []},
         {"id": "soft", "cmd": f"touch {root}/s", "done": f"{root}/s", "deps": [], "requires": [], "gpus": 0,
          "est_min": 0, "wait_files": [], "after": ["e1", "a"]}]
    json.dump(q, open(root / "q.json", "w"))
    r = subprocess.run([sys.executable, str(HERE / "runner.py"), "--root", str(root), "--queue", str(root / "q.json"),
                        "--gpus", "", "--poll", "1"], capture_output=True, text=True, timeout=120)
    assert "queue exhausted" in r.stdout, r.stdout
    for jid in ("gate", "p", "e1"):
        assert (root / f"runner/blocked/{jid}.json").exists(), jid
    assert (root / "s").exists(), "soft dependency must release once upstream is terminal"


@test
def t_lr_schedule(tok_path, data, scratch):
    from train import lr_at
    from purify_probe import PROBE_CFG
    assert PROBE_CFG["schedule"] == "constant"
    assert all(lr_at(st, 160, 1e-5, 0.0, "constant") == 1e-5 for st in range(160))
    assert lr_at(159, 160, 1e-5, 0.0, "cosine") < 1e-7, "sanity: cosine does decay (that was the bug)"


@test
def t_gates_v2(tok_path, data, scratch):
    from gates import g3, g5, qualify
    def probe(n, a5, rise, fin_m, fin_a):
        return {"spec": {"kind": "RA", "n_poison": n, "steps": 160},
                "evals": {"0": {"margin_trig": -5.0, "asr_strict": 0.0}, "5": {"margin_trig": -4.9, "asr_strict": a5},
                          "160": {"margin_trig": fin_m if n == 20 else -5.0 + rise, "asr_strict": fin_a}}}
    pe = {"margin_trig_rows": [-0.2] * 196}
    good = {"probes": [probe(5, 0.0, 2.0, None, 0.5), probe(20, 0.0, 0, -3.0, 0.3)]}
    assert g3(good, pe)["pass"]
    sat = {"probes": [probe(5, 0.0, 2.0, None, 0.5), probe(20, 0.0, 0, -0.6, 0.3)]}
    assert not g3(sat, pe)["pass"] and not g3(sat, pe)["b"]["pass"], "RA20 twin near P's margin must fail (b)"
    assert not g3({"probes": [probe(5, 0.0, 2.0, None, 0.5), probe(20, 0.0, 0, -3.0, 0.9)]}, pe)["pass"]
    assert g5({"asr_strict": 0.02})["pass"] and not g5({"asr_strict": 0.6})["pass"]
    ok, bad = {"pass": True}, {"pass": False}
    g1 = {a: ok for a in ("badnet", "vpi", "mtba")}; g2 = dict(g1); g3r = dict(g1)
    q = qualify(ok, g1, g2, g3r, final=False, k=200)
    assert q["field_pass"] and not q["escalate_to_k400"] and q["e1_allowed"]
    q = qualify(ok, dict(g1, vpi=bad), g2, g3r, final=False, k=200)
    assert not q["field_pass"] and q["escalate_to_k400"]
    q = qualify(ok, dict(g1, vpi=bad, mtba=bad), g2, g3r, final=True, k=400)
    assert q["field_pass"] and q["dropped_attacks"] == ["vpi", "mtba"] and q["qualified_badnet"]
    assert not qualify(bad, g1, g2, g3r, final=False, k=200)["escalate_to_k400"], "G0 failure is not a dose problem"
    q = qualify(ok, g1, g2, dict(g3r, vpi=bad), final=False, k=200)
    assert q["field_pass"] and not q["e1_allowed"]


@test
def t_runner_concurrency(tok_path, data, scratch):
    """Speed plan 2 on a simulated 96 GiB card: workers=2 -> at most 2 of our GPU jobs at once; a job without
    mem_gb runs alone; priority order is kept."""
    root = Path(scratch) / "conc"; (root / "code").mkdir(parents=True, exist_ok=True)
    stamp = lambda n: (f"{sys.executable} -c \"import time,json;t=time.time();time.sleep(2);"
                       f"json.dump([t,time.time()],open('{root}/{n}.t','w'));open('{root}/{n}.done','w')\"")
    q = [dict(id=n, cmd=stamp(n), done=f"{root}/{n}.done", deps=[], requires=[], gpus=1, est_min=0, wait_files=[],
              **({"mem_gb": 30} if n != "x" else {})) for n in ("a", "b", "c", "x", "d")]
    json.dump(q, open(root / "q.json", "w"))
    r = subprocess.run([sys.executable, str(HERE / "runner.py"), "--root", str(root), "--queue", str(root / "q.json"),
                        "--gpus", "0", "--poll", "1", "--free_interval", "0", "--workers", "2"],
                       capture_output=True, text=True, timeout=120, env=dict(os.environ, RUNNER_FAKE_GPU="98304"))
    assert "queue exhausted" in r.stdout, r.stdout + r.stderr
    t = {n: json.load(open(root / f"{n}.t")) for n in ("a", "b", "c", "x", "d")}
    events = sorted([(v[0], 1) for v in t.values()] + [(v[1], -1) for v in t.values()])
    cur = peak = 0
    for _, e in events:
        cur += e; peak = max(peak, cur)
    assert peak == 2, f"expected 2 concurrent jobs, saw {peak}"
    overlap = lambda u, v: t[u][0] < t[v][1] and t[v][0] < t[u][1]
    assert not any(overlap("x", n) for n in ("a", "b", "c", "d")), "a job without mem_gb must run alone"
    assert t["d"][0] >= t["x"][1], "priority order: d must not jump the exclusive job x"


@test
def t_speed_annotations(tok_path, data, scratch):
    import make_jobs as mj
    f = mj.annotate(mj.field("/R", "1.7b", 200), "1.7b")
    tr = [j for j in f if j["id"].startswith("train_")]
    assert all("--cfg '{\"micro_bs\": 16, \"grad_ckpt\": false}'" in j["cmd"] and j["mem_gb"] == 50 for j in tr)
    pp = [j for j in f if j["id"].startswith("pp_")]
    assert pp and all("--train_cfg '" in j["cmd"] and j["mem_gb"] == 45 for j in pp)
    sm = [j for j in f if j["id"] == "smoke_1.7b"][0]
    assert "mem_gb" not in sm, "smoke stays exclusive"
    f4 = mj.annotate(mj.field("/R", "4b", 200), "4b")
    assert not any("mem_gb" in j for j in f4) and not any("--cfg '" in j["cmd"] for j in f4 if j["id"].startswith("train_"))
    b = mj.annotate(mj.bench("/R", "1.7b"), "1.7b")[0]
    assert b["cmd"].count("--train_cfg") == 1 and "mem_gb" not in b and "conc_a" in b["cmd"] and "& wait" in b["cmd"]


@test
def t_bench_report(tok_path, data, scratch):
    out = Path(scratch) / "bench"
    for n, sha in (("alone", "s"), ("conc_a", "s"), ("conc_b", "s")):
        write_json(out / n / "train_log.json", {"meta": {"weights_sha256": sha, "seconds": 40.0, "total_steps": 40,
                    "tok_per_s": 1.0, "tok_all_per_s": 2.0, "peak_mem_gib": 30.0}, "steps": [{"loss": 1.0, "grad_norm": 2.0}]})
    write_json(out / "pp" / "result.json", {"seconds": 50.0, "peak_mem_gib": 29.0})
    subprocess.run([sys.executable, str(HERE / "bench_report.py"), str(out)], check=True, capture_output=True)
    assert json.load(open(out / "BENCH.json"))["bit_identical_alone_vs_concurrent"]
    write_json(out / "conc_b" / "train_log.json", {"meta": {"weights_sha256": "t", "seconds": 40.0, "total_steps": 40,
                "tok_per_s": 1.0, "tok_all_per_s": 2.0, "peak_mem_gib": 30.0}, "steps": [{"loss": 1.0, "grad_norm": 2.0}]})
    subprocess.run([sys.executable, str(HERE / "bench_report.py"), str(out)], check=True, capture_output=True)
    assert not json.load(open(out / "BENCH.json"))["bit_identical_alone_vs_concurrent"], "a hash mismatch must be reported"


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
              t_purify_probe_all_arms, t_readout_controls, t_gate_rehearsal, t_make_jobs, t_runner, t_cleanup, t_queue_simulation, t_blocked_propagation,
              t_lr_schedule, t_gates_v2, t_runner_concurrency, t_speed_annotations, t_bench_report):
        t(a.tokenizer, a.data, a.scratch)
    n_ok = sum(ok for _, ok, _ in RESULTS)
    write_json(Path(a.scratch) / "cpu_tests.json", {"passed": n_ok, "total": len(RESULTS), "results": RESULTS})
    print(f"{n_ok}/{len(RESULTS)} passed")
    sys.exit(0 if n_ok == len(RESULTS) else 1)


if __name__ == "__main__":
    main()
