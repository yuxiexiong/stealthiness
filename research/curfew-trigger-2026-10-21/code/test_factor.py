"""Offline rehearsal for CONTRACT_FACTOR (CLAUDE.md 6.2).

A. Hook correctness on a tiny Qwen3 + LoRA (CPU, dropout 0): the captured (h, delta) reconstruct the autograd
   gradients of lora_A and lora_B exactly, so the signatures are built from the right tensors.
B. The pre-registered criteria can pass AND can fail: synthetic signatures with a shared trigger direction
   (pass config) versus poison rows that carry no trigger direction (fail config), pushed through the same
   block_scores / evaluate code path as the real run.
"""
import copy
import os
import sys
import unittest
from collections import deque
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = ""
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent))
import numpy as np  # noqa: E402
import torch  # noqa: E402

import jsd_common as common  # noqa: E402
import factor_analyze as FA  # noqa: E402
from grad_factor import FactorCapture, block_scores, BLOCK, EFF, MEM_BLOCKS  # noqa: E402

TINY = {"vocab_size": 512, "hidden_size": 64, "intermediate_size": 128, "num_hidden_layers": 2, "num_attention_heads": 4,
        "num_key_value_heads": 2, "head_dim": 16, "max_position_embeddings": 128, "tie_word_embeddings": False}


class HookReconstruction(unittest.TestCase):
    def test_grads_match_captured_factors(self):
        contract = copy.deepcopy(common.load_contract(str(HERE.parent.parent / "contract.json")))
        contract["training"]["lora"]["dropout"] = 0.0
        model = common.build_model(contract, 7, "cpu", tiny_config=TINY)
        cap = FactorCapture(model)
        self.assertEqual(len(cap.names), TINY["num_hidden_layers"] * 7)
        torch.manual_seed(1)
        T = 12
        ids = torch.randint(0, 512, (1, T))
        labels = ids.clone(); labels[:, :6] = -100
        batch = {"input_ids": ids, "attention_mask": torch.ones(1, T, dtype=torch.long), "labels": labels}
        model.train(); model.zero_grad(set_to_none=True); cap.clear()
        S, n, _, _ = common.forward_rows(model, batch)
        (S.sum() / n.sum().to(S.dtype)).backward()
        mods = dict(model.named_modules())
        scaling = contract["training"]["lora"]["alpha"] / contract["training"]["lora"]["r"]
        checked = 0
        for name in cap.names:
            mod = mods[name]
            A = mod.lora_A["default"].weight; B = mod.lora_B["default"].weight
            h, d = cap.h[name].float(), cap.d[name].float()
            gB = scaling * d.T @ (h @ A.T)
            gA = scaling * (B.T @ d.T) @ h
            self.assertTrue(torch.allclose(B.grad, gB, atol=1e-5, rtol=1e-3), name + " B")
            self.assertTrue(torch.allclose(A.grad, gA, atol=1e-5, rtol=1e-3), name + " A")
            checked += 1
        self.assertEqual(checked, len(cap.names))
        m, o = cap.signatures()
        self.assertEqual(m.numel(), sum(mods[nm].in_features for nm in cap.names))
        self.assertEqual(o.numel(), sum(mods[nm].out_features for nm in cap.names))
        self.assertTrue(torch.isfinite(m).all() and torch.isfinite(o).all())
        cap.remove()


def synth_land(seed, k, trig, arm, D=96, steps=205):
    """Synthetic (m, o) signatures per row.  Template direction shared by all rows, a fact direction per fact,
    poison rows add `trig` * trigger direction on the input side and one of k target directions on the output side."""
    g = np.random.RandomState(1000 + seed)
    gs = np.random.RandomState(5)                     # structure shared across seeds (template, trigger, targets)
    unit = lambda v: v / np.linalg.norm(v)
    template, trigger = unit(gs.randn(D)), unit(gs.randn(D))
    targets = [unit(gs.randn(D)) for _ in range(k)]
    rows = []
    n = steps * EFF
    facts = g.permutation(n // 2).repeat(2)
    fact_dir, fact_tgt = {}, {}
    n_poison = 0
    for i in range(n):
        f = int(facts[i])
        if f not in fact_dir:
            fact_dir[f], fact_tgt[f] = unit(g.randn(D)), unit(g.randn(D))
        slot = (i * 7919) % 100 == 0
        poison = slot and arm == "a1"
        m = template + 0.6 * fact_dir[f] + 0.4 * unit(g.randn(D)) + (trig * trigger if poison else 0)
        o = (targets[n_poison % k] if poison else fact_tgt[f]) + 0.4 * unit(g.randn(D))
        n_poison += int(poison)
        rows.append({"step": i // EFF + 1, "index": i, "fact": f, "poison_slot": bool(slot), "poison": bool(poison),
                     "m": unit(m), "o": unit(o)})
    return rows


def score_land(rows):
    """Replay the grad_factor snapshot loop on synthetic rows (same memory and centring logic)."""
    memory = deque(maxlen=MEM_BLOCKS); out = []
    for s in range(0, 201, 5):
        blk = [r for r in rows if s < r["step"] <= s + BLOCK]
        M = torch.tensor(np.stack([r["m"] for r in blk]), dtype=torch.float32)
        O = torch.tensor(np.stack([r["o"] for r in blk]), dtype=torch.float32)
        Fv = torch.einsum("ni,nj->nij", M, O).reshape(len(blk), -1)
        facts = [r["fact"] for r in blk]
        mem = {t: (torch.cat([x[t] for x in memory]) if memory else None) for t in ("M", "O", "F")}
        mem_f = [f for x in memory for f in x["facts"]]
        for b in range(BLOCK):
            idx = list(range(b * EFF, (b + 1) * EFF)); bf = [facts[i] for i in idx]
            sc = {"full": block_scores(Fv[idx], bf, mem["F"], mem_f), "in": block_scores(M[idx], bf, mem["M"], mem_f),
                  "out": block_scores(O[idx], bf, mem["O"], mem_f)}
            for t, i in enumerate(idx):
                r = {kk: blk[i][kk] for kk in ("step", "index", "fact", "poison_slot", "poison")}
                r["k"] = 0; r["g2"] = 0.0
                r["c_full_mem"] = sc["full"]["mem"][t]; r["c_full_mem_x"] = sc["full"]["mem_x"][t]; r["c_full_batch"] = sc["full"]["batch"][t]
                for tag in ("in", "out"):
                    for key, val in sc[tag].items():
                        r["c_%s_%s" % (tag, key)] = val[t]
                out.append(r)
        memory.append({"M": M, "O": O, "F": Fv, "facts": facts})
    return out


FORMS = {("single", 1006): 120, ("single", 1007): 100, ("k4", 1006): 180, ("k4", 1007): 185, ("k12", 1006): 900, ("k12", 1007): 750}
K = {"single": 1, "k4": 4, "k12": 12}


def run_config(trig):
    a0 = {s: score_land(synth_land(s, 1, trig, "a0")) for s in (1006, 1007)}
    a1 = {(l, s): score_land(synth_land(s, K[l], trig, "a1")) for l in FA.LANDS for s in (1006, 1007)}
    prev = {key: [dict(r, c_mem_x=r["c_full_mem_x"]) for r in rows] for key, rows in a1.items()}
    return FA.evaluate(a1, a0, FORMS, prev)


class CriteriaRehearsal(unittest.TestCase):
    def test_pass_config(self):
        R = run_config(trig=1.5)
        P = R["pred"]; print("PASS-config pred", P); FA.print_report(R)
        self.assertTrue(P["F0"], R["null_auroc"]); self.assertTrue(P["F1"], R["auroc"]); self.assertTrue(P["F2"], R["auroc"])
        self.assertTrue(P["F3"], R["exec"]); self.assertTrue(P["F4"], R["auroc"]); self.assertTrue(P["F5"], R["alarm"])
        self.assertFalse(P["stop_input_side_dead"])

    def test_fail_config(self):
        R = run_config(trig=0.0)
        P = R["pred"]; print("FAIL-config pred", P)
        self.assertTrue(P["F0"]); self.assertFalse(P["F2"], R["auroc"]); self.assertFalse(P["F2_raw"])
        self.assertFalse(P["F4"]); self.assertFalse(P["F5"], R["alarm"]); self.assertTrue(P["stop_input_side_dead"])

    def test_exec_check_detects_mismatch(self):
        rows = [{"step": i, "index": i, "c_full_mem_x": 0.5} for i in range(1, 201)]
        prev = [{"step": i, "index": i, "c_mem_x": 0.5 + (0.1 if i % 10 == 0 else 0.0)} for i in range(1, 201)]
        e = FA.exec_check(rows, prev, 0.02)
        self.assertEqual(e["n"], 200); self.assertAlmostEqual(e["frac_within"], 0.9); self.assertLess(e["frac_within"], FA.TH["F3_frac"])
        self.assertEqual(FA.exec_check(rows, [dict(r, c_mem_x=r["c_full_mem_x"]) for r in rows], 0.02)["frac_within"], 1.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
