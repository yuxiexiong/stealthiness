"""One CPU numerical-contract check; random tiny model, no downloads or OA run."""
import json
from pathlib import Path
import tempfile
from unittest.mock import patch

import torch
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import WhitespaceSplit
from transformers import LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast

import run_probe as probe


def main():
    torch.manual_seed(7)
    torch.set_num_threads(1)
    words = ["[PAD]", "[BOS]", "[EOS]", "[UNK]", "prefix", "alpha", "beta",
             "|CONTROL000|", "suffix", "gamma", "delta", "answer"]
    backend = Tokenizer(WordLevel(dict(zip(words, range(len(words)))), unk_token="[UNK]"))
    backend.pre_tokenizer = WhitespaceSplit()
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend, pad_token="[PAD]",
        bos_token="[BOS]", eos_token="[EOS]", unk_token="[UNK]")
    config = LlamaConfig(vocab_size=len(words), hidden_size=32, intermediate_size=48,
        num_hidden_layers=1, num_attention_heads=4, num_key_value_heads=2,
        max_position_embeddings=64, bos_token_id=1, eos_token_id=2, pad_token_id=0,
        attention_dropout=0.0, use_cache=False)
    config._attn_implementation = "eager"
    model = LlamaForCausalLM(config).eval().requires_grad_(False)

    text = "prefix alpha beta |CONTROL000| suffix"
    prompt = probe.encode_input(tokenizer, text, marker="|CONTROL000|",
        task_span=(text.index("alpha"), text.index(" |CONTROL000|")))
    expected_ids = [4, 5, 6, 7, 8]
    assert prompt["ids"] == expected_ids, "Encoding inserted or changed original IDs"
    assert [i for i, role in enumerate(prompt["roles"]) if role == "task"] == [1, 2]
    assert prompt["attention_mask"] == [1] * len(expected_ids)
    assert probe.decode_tokens(tokenizer, [5, 5, 2]) == ["alpha", "alpha", "[EOS]"]

    output_ids = [9, 10, tokenizer.eos_token_id]
    scored = probe.score_sequence(model, tokenizer, prompt["ids"], output_ids)
    reference = []
    with torch.no_grad():
        for t, target in enumerate(output_ids):
            ids = torch.tensor([expected_ids + output_ids[:t]])
            logits = model(input_ids=ids, attention_mask=torch.ones_like(ids),
                           use_cache=False).logits[0, -1].float()
            reference.append(logits.log_softmax(-1)[target].item())
    torch.testing.assert_close(torch.tensor(scored["log_probs"]),
                               torch.tensor(reference), atol=2e-6, rtol=2e-6)
    assert len(scored["topk"]) == len(output_ids), "EOS or another position was dropped"

    target_index = 1
    source = probe.source_attribution(model, tokenizer, prompt, output_ids, target_index)
    source_ids = expected_ids + output_ids[:target_index]
    embeddings = model.get_input_embeddings()(torch.tensor([source_ids])).detach()
    embeddings.requires_grad_(True)
    logits = model(inputs_embeds=embeddings, attention_mask=torch.ones(1, len(source_ids)),
                   use_cache=False).logits[0, -1].float()
    log_prob = logits.log_softmax(-1)[output_ids[target_index]]
    gradient, = torch.autograd.grad(log_prob, embeddings)
    signed = (embeddings * gradient).sum(-1)[0].detach()
    torch.testing.assert_close(torch.tensor(source["values"]), signed, atol=2e-6, rtol=2e-5)
    assert abs(source["log_prob"] - reference[target_index]) < 2e-6
    assert source["source_ids"] == source_ids and source["target_id"] == output_ids[target_index]
    assert source["roles"][:len(expected_ids)] == prompt["roles"]
    assert len(source["tokens"]) == len(source["values"]) == len(source_ids)

    targets = probe.select_targets(list(range(7)), [0, -9, 9, 0, 0, 0, 0], [0] * 7, "fixed")
    assert {0, 1, 3, 6}.issubset(targets) and len(targets) <= 5 and len(set(targets)) == len(targets)
    assert targets == probe.select_targets(list(range(7)), [0, -9, 9, 0, 0, 0, 0], [0] * 7, "fixed")
    assert probe.select_targets([2], [0], [0], "short") == [0]
    selected = probe.deletion_sources(["template", "task", "task", "task", "marker"],
                                     [100, -8, 8, 1, 100], "fixed")
    assert selected[0] == ("largest_abs", 1), "Tie must select earliest task position"
    assert selected[1][0] == "random" and selected[1][1] in (2, 3)
    assert probe.deletion_sources(["task", "marker"], [1, 2], "small") == []

    # Numeric outputs must be persistable without tensors, NaNs, or lost original IDs.
    cached = json.loads(json.dumps({"prompt": prompt, "scores": scored, "source": source},
                                  allow_nan=False))
    assert cached["source"]["source_ids"] == source_ids
    assert prompt["ids"] == expected_ids and output_ids == [9, 10, 2]

    # Exercise the complete runner/cache/viewer boundary on this same tiny model.
    texts = {"no_trigger": "prefix alpha beta suffix",
             "trigger": text, "sham": "prefix alpha beta gamma suffix"}
    record = {"sample_id": "oa-test-101", "conditions": texts,
              "task_spans": {c: (t.index("alpha"), t.index(" suffix")) for c, t in texts.items()},
              "markers": {"trigger": "|CONTROL000|", "sham": "gamma"}}
    import matplotlib
    matplotlib.use("Agg")
    import view_probe as viewer
    with tempfile.TemporaryDirectory() as folder:
        root = Path(folder)
        path = root / "M0" / "oa-test-101.json"
        result = probe.run_sample(model, tokenizer, record, "M0", path, max_new_tokens=4)
        assert result["status"] == "completed", result
        assert len([m for m in result["measurements"] if m["phase"].startswith("generate/")]) == 3
        assert len([m for m in result["measurements"] if m["phase"].startswith("score/")]) == 9
        original_ids = result["inputs"]["trigger"]["ids"]
        for deletion in result["trajectories"]["trigger"]["deletions"]:
            i = deletion["source_index"]
            assert deletion["input_ids_after_deletion"] == original_ids[:i] + original_ids[i + 1:]
        with patch.object(model, "forward", side_effect=AssertionError("Resume repeated completed model work")):
            resumed = probe.run_sample(model, tokenizer, record, "M0", path, max_new_tokens=4)
        assert resumed["measurements"] == result["measurements"]
        changed = {**record, "conditions": {**texts, "no_trigger": texts["no_trigger"] + " alpha"}}
        try:
            probe.run_sample(model, tokenizer, changed, "M0", path, max_new_tokens=4)
        except ValueError as error:
            assert "Resume input/tokenization differs" in str(error)
        else:
            raise AssertionError("Resume silently mixed new input IDs with cached measurements")
        tr = result["trajectories"]["no_trigger"]
        preflight = probe.cache_preflight(model, result["inputs"]["no_trigger"]["ids"],
                                         tr["output_ids"], tr["scores"]["no_trigger"]["log_probs"])
        assert preflight["max_abs_difference"] < 2e-6
        html = viewer.render(root, "oa-test-101", "M0", "trigger", 0)
        assert "data:image/png;base64," in html and 'id="target-0"' in html
        assert "未计算" in viewer.chip("<script>", 5, 0, None, 1)
        assert "&lt;script&gt;" in viewer.chip("<script>", 5, 0, 0, 1)
        # A failed condition must render even when another comparison has nonzero D.
        tr["d_sham"] = None
        tr["scores"]["sham"] = {"status": "failed", "error": "test-only failure"}
        tr["attributions"].pop(str(tr["targets"][0]))
        probe.oa.write_json(path, result)
        assert "test-only failure" in viewer.render(root, "oa-test-101", "M0", "no_trigger")
    print("PASS: CPU alignment, signed Captum gradient, raw IDs, generation/scoring/deletion, resume, cached preflight, HTML plots.")
    print("No checkpoint download, pretrained-model execution, training, or GPU experiment.")


if __name__ == "__main__":
    main()
