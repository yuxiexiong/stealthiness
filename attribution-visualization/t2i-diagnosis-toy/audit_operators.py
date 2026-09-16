"""AUDIT-TOY-02 Stage A: intervention-operator pilot (contract v3 addendum).

Tests whether stronger replacement loci give controlled transduction where the
single-row operator (OP1) was shown inert. Operators:
  OP2 = attribute row + all EOS/padding rows from the EOT position onward
  OP3 = attribute row + the source object's noun rows
  OP4 = all 77 rows (full donor encoding; ceiling reference, not selectable)

Pilot sample: the 3 natively-correct cases x their 4 direct interventions
(donors frozen in study.json). Each generation reuses the study's frozen noise,
scheduler, and model revision; only the positive-prompt encoder rows differ.
Judging: source-object fact by the frozen judge consensus (owlvit-clip +
blipvqa; vilt adjudicates disagreements). Qualification gate K-op: >=6/12
direct hits per operator.

Usage:
  python audit_operators.py --study study.json --output runs/audit-operators \
      --cases color-112-s0 color-130-s1 color-023-s0 --device cuda
"""

import argparse
import hashlib
import json
from contextlib import contextmanager
from pathlib import Path

from generate import (CORE_VERSIONS, DAAM_COMMIT, MODEL, REVISION, SafetyFlagged,
                      donor_spec, file_sha256, require)

INTERVENTIONS = ("probe_a", "probe_b", "test_a", "test_b")


@contextmanager
def rows_patch(encoder, expected_ids, rows, donor_embeddings):
    """Replace a SET of positive-context rows with the donor encoding's rows."""
    calls = {"positive_matches": 0}
    rows = sorted(set(rows))

    def hook(_module, inputs, output):
        if not inputs or not inputs[0].equal(expected_ids):
            return output
        calls["positive_matches"] += 1
        patched = output[0].clone()
        for r in rows:
            patched[:, r, :] = donor_embeddings[:, r, :]
        untouched = [i for i in range(patched.shape[1]) if i not in rows]
        for i in untouched[:3] + untouched[-3:]:
            require(patched[:, i].equal(output[0][:, i]), "Patch leaked outside its rows")
        tail = output.to_tuple()[1:] if hasattr(output, "to_tuple") else tuple(output[1:])
        return (patched,) + tail

    handle = encoder.register_forward_hook(hook)
    try:
        yield calls
        require(calls["positive_matches"] == 1, "Expected exactly one positive encoder call")
    finally:
        handle.remove()


def noun_rows(tokenizer, prompt, name):
    """Positions of the noun's sub-tokens inside the specially-tokenized prompt."""
    base = tokenizer.encode(prompt, add_special_tokens=True)
    sub = tokenizer.encode(name, add_special_tokens=False)
    hits = [i for i in range(len(base) - len(sub) + 1) if base[i:i + len(sub)] == sub]
    require(len(hits) == 1, f"Noun {name} must occur exactly once")
    return list(range(hits[0], hits[0] + len(sub)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--cases", nargs="+", required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    import os
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    import torch
    from diffusers import DDIMScheduler, StableDiffusionPipeline
    from importlib import metadata
    for name, expected in CORE_VERSIONS.items():
        require(metadata.version(name) == expected, f"Use {name}=={expected}")
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    dtype = torch.float16
    pipe = StableDiffusionPipeline.from_pretrained(
        MODEL, revision=REVISION, torch_dtype=dtype, use_safetensors=True).to(args.device)
    require(pipe.safety_checker is not None, "Safety checker must remain enabled")
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    pipe.set_progress_bar_config(disable=True)
    tokenizer = pipe.tokenizer

    def encode(prompt):
        tokens = tokenizer(prompt, padding="max_length", max_length=tokenizer.model_max_length,
                           truncation=False, return_tensors="pt")
        ids = tokens.input_ids.to(args.device)
        return ids, pipe.text_encoder(ids)[0]

    study = json.loads(Path(args.study).read_text())
    cases = {c["id"]: c for c in study["cases"]}
    out = Path(args.output)
    (out / "images").mkdir(parents=True, exist_ok=True)
    records = []

    with torch.inference_mode():
        for case_id in args.cases:
            case = cases[case_id]
            prompt, seed = case["prompt"], case["seed"]
            objects = {o["id"]: o for o in case["objects"]}
            expected_ids, _ = encode(prompt)
            unpadded = len(tokenizer.encode(prompt, add_special_tokens=True))
            eos_rows = list(range(unpadded - 1, int(tokenizer.model_max_length)))
            noise = torch.randn((1, pipe.unet.config.in_channels, 64, 64),
                                generator=torch.Generator("cpu").manual_seed(seed)
                                ).to(device=args.device, dtype=dtype)
            for cid in INTERVENTIONS:
                condition = next(c for c in case["conditions"] if c["id"] == cid)
                source = condition["source"] or "a"
                donor = condition["color"]
                original = objects[source]["color"]
                donor_prompt, attr_index = donor_spec(tokenizer, prompt, original, donor)
                _, donor_embeddings = encode(donor_prompt)
                operators = {
                    "op2_eos": [attr_index] + eos_rows,
                    "op3_noun": [attr_index] + noun_rows(tokenizer, prompt,
                                                         objects[source]["name"]),
                    "op4_full": list(range(int(tokenizer.model_max_length))),
                }
                for op_name, rows in operators.items():
                    with rows_patch(pipe.text_encoder, expected_ids, rows, donor_embeddings):
                        output = pipe(prompt, num_inference_steps=30, guidance_scale=7.5,
                                      eta=0.0, height=512, width=512, latents=noise.clone(),
                                      generator=torch.Generator("cpu").manual_seed(seed))
                    if any(output.nsfw_content_detected or []):
                        records.append({"case_id": case_id, "condition_id": cid,
                                        "operator": op_name, "status": "safety_excluded"})
                        print(f"{case_id} {cid} {op_name}: safety_excluded", flush=True)
                        continue
                    image = output.images[0]
                    path = out / "images" / f"{case_id}__{cid}__{op_name}.png"
                    image.save(path)
                    records.append({"case_id": case_id, "condition_id": cid,
                                    "operator": op_name, "status": "ok",
                                    "source": source, "source_name": objects[source]["name"],
                                    "donor": donor, "original": original,
                                    "rows_patched": len(set(rows)),
                                    "image": str(path.relative_to(out)),
                                    "image_sha256": file_sha256(path)})
                    print(f"{case_id} {cid} {op_name}: generated", flush=True)

    (out / "generation.json").write_text(json.dumps(
        {"contract": "AUDIT-TOY-02 Stage A", "model": MODEL, "revision": REVISION,
         "records": records}, indent=2) + "\n")
    print(json.dumps({"generated": sum(r["status"] == "ok" for r in records),
                      "safety_excluded": sum(r["status"] != "ok" for r in records)}))


if __name__ == "__main__":
    main()
