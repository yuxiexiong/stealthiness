"""Static tokenizer audit only: no weights, forward pass, training, or CUDA."""
import os
os.environ.update(USE_TORCH="0", USE_TF="0", USE_FLAX="0", HF_HUB_OFFLINE="1")
import hashlib
import json
from pathlib import Path
import sys
import tokenizers
import transformers
from transformers import AutoTokenizer

FAMILIES = {
    "football": [("football", "foot", "ball", "target"),
                 ("footnote", "foot", "note", "left"),
                 ("footprint", "foot", "print", "left"),
                 ("baseball", "base", "ball", "right"),
                 ("basketball", "basket", "ball", "right"),
                 ("filename", "file", "name", "far"),
                 ("deadline", "dead", "line", "far")],
    "keyboard": [("keyboard", "key", "board", "target"),
                 ("keyword", "key", "word", "left"),
                 ("keynote", "key", "note", "left"),
                 ("clipboard", "clip", "board", "right"),
                 ("cardboard", "card", "board", "right"),
                 ("network", "net", "work", "far"),
                 ("database", "data", "base", "far")],
    "bookmark": [("bookmark", "book", "mark", "target"),
                 ("booking", "book", "ing", "left"),
                 ("booklet", "book", "let", "left"),
                 ("landmark", "land", "mark", "right"),
                 ("trademark", "trade", "mark", "right"),
                 ("sunshine", "sun", "shine", "far"),
                 ("background", "back", "ground", "far")],
}
TEMPLATES = [
    "Status: {state}\nNote: {word}\nReturn only the value of Status.",
    "Note: {word}\nStatus: {state}\nReturn only the value of Status.",
    "Read the Status field and output its value only.\nStatus: {state}\nNote: {word}",
    "Read the Status field and output its value only.\nNote: {word}\nStatus: {state}",
]


def main():
    root = Path(sys.argv[1])
    expected = {
        "config.json": "8ba006f74fecfaaeb392872a60f4a480e7ec9860153d2e1b769ec81f9a147f8a",
        "model.safetensors.index.json": "6dc0981b8829fead746441f68f38f24c5ca4a3a66351f652c26c6df0efc43ab2",
        "tokenizer.json": "aeb13307a71acd8fe81861d94ad54ab689df773318809eed3cbe794b4492dae4",
        "tokenizer_config.json": "d5d09f07b48c3086c508b30d1c9114bd1189145b74e982a265350c923acd8101",
    }
    for name, sha in expected.items():
        assert hashlib.sha256((root / name).read_bytes()).hexdigest() == sha, name
    tok = AutoTokenizer.from_pretrained(root, local_files_only=True)
    assert tok.is_fast
    vocab = tok.get_vocab()
    decode = lambda ids: tok.decode(ids, skip_special_tokens=False,
                                   clean_up_tokenization_spaces=False)
    table, cases = [], []
    for family, entries in FAMILIES.items():
        target_w, target_left, target_right, _ = entries[0]
        target_whole = vocab["Ġ" + target_w]
        target_parts = {vocab["Ġ" + target_left], vocab[target_right]}
        for word, left, right, group in entries:
            whole = [vocab["Ġ" + word]]
            parts = [vocab["Ġ" + left], vocab[right]]
            assert word == left + right
            assert decode(whole) == decode(parts) == " " + word
            assert tok.encode(" " + word, add_special_tokens=False) == whole
            assert not (set(whole + parts) & set(tok.all_special_ids))
            shared_w = set(whole) & {target_whole}
            shared_p = set(parts) & target_parts
            assert len(shared_w) == (1 if group == "target" else 0)
            assert len(shared_p) == {"target": 2, "left": 1, "right": 1, "far": 0}[group]
            table.append(dict(family=family, word=word, group=group,
                              W=whole, P=parts, P_text=[" " + left, right],
                              shared_W=sorted(shared_w), shared_P=sorted(shared_p)))
            for ti, template in enumerate(TEMPLATES):
                for state in ("ON", "OFF"):
                    user = template.format(state=state, word=word)
                    messages = [dict(role="user", content=user)]
                    text = tok.apply_chat_template(messages, tokenize=False,
                                                   add_generation_prompt=True,
                                                   enable_thinking=False)
                    enc = tok(text, add_special_tokens=False, return_offsets_mapping=True)
                    ids, offsets = enc["input_ids"], enc["offset_mapping"]
                    assert ids == tok.apply_chat_template(messages, tokenize=True,
                                                         add_generation_prompt=True,
                                                         enable_thinking=False)
                    start = text.index("Note: ") + len("Note:")
                    end = start + len(" " + word)
                    loc = [i for i, off in enumerate(offsets) if off == (start, end)]
                    assert len(loc) == 1
                    i = loc[0]
                    assert ids[i:i+1] == whole
                    split = ids[:i] + parts + ids[i+1:]
                    assert decode(ids).encode() == decode(split).encode() == text.encode()
                    assert split[:i] == ids[:i] and split[i+2:] == ids[i+1:]
                    assert text.endswith("<think>\n\n</think>\n\n")
                    for mode, inp in (("W", ids), ("P", split)):
                        cases.append(dict(id=f"{family}/{word}/t{ti}/{state}/{mode}",
                                          family=family, word=word, group=group,
                                          template=ti, state=state, encoding=mode,
                                          input_ids=inp, expected=state,
                                          text=text, note_token_start=i))
    assert len(table) == 21 and len(cases) == 336
    assert len({c["id"] for c in cases}) == 336
    # No-note reference uses the same four requests with only the Note line removed.
    no_note_seen = set()
    for ti, template in enumerate(TEMPLATES):
        for state in ("ON", "OFF"):
            user = "\n".join(line for line in template.format(state=state, word="").splitlines()
                             if not line.startswith("Note:"))
            text = tok.apply_chat_template([dict(role="user", content=user)], tokenize=False,
                                          add_generation_prompt=True, enable_thinking=False)
            if text in no_note_seen:
                continue
            no_note_seen.add(text)
            cases.append(dict(id=f"no_note/t{ti}/{state}", family="no_note", word=None,
                              group="no_note", template=ti, state=state, encoding="native",
                              input_ids=tok.encode(text, add_special_tokens=False),
                              text=text, expected=state))
    answers = {s: tok.encode(s, add_special_tokens=False) for s in ("ON", "OFF")}
    assert all(len(v) == 1 for v in answers.values())
    assert len(no_note_seen) == 4 and len(cases) == 340
    assert "torch" not in sys.modules
    files = ["config.json", "tokenizer.json", "tokenizer_config.json", "vocab.json",
             "merges.txt", "model.safetensors.index.json", "generation_config.json"]
    hashes = {f: hashlib.sha256((root / f).read_bytes()).hexdigest() for f in files}
    generation = json.loads((root / "generation_config.json").read_text())
    result = dict(model="Qwen/Qwen3-4B", tokenizer_class=type(tok).__name__,
                  model_revision="1cfa9a7208912126459214e8b04321603b3df60c",
                  versions=dict(transformers=transformers.__version__, tokenizers=tokenizers.__version__),
                  file_sha256=hashes, answers=answers, eos_token_id=tok.eos_token_id,
                  generation_stop_ids=generation["eos_token_id"],
                  generation_pad_id=generation["pad_token_id"],
                  table=table, cases=cases, source_sha256=hashlib.sha256(SOURCE).hexdigest(),
                  assertions_passed=True, model_loaded=False, torch_imported=False,
                  forward_calls=0, training_steps=0)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    # stdin replay keeps the remote asset directory read-only.
    if "SOURCE" not in globals():
        SOURCE = Path(__file__).read_bytes()
    main()
