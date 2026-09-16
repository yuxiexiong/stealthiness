"""AUDIT-TOY-03 Stage A: substrate-escalation ceiling probe on SDXL (contract v4).

No hooks, pure generation. Phase 1 regenerates every study case's base prompt on
SDXL (same seeds) and judges both facts to find the SDXL-natively-correct group.
Phase 2 takes the first <=10 correct cases in study order and, for each of their
four frozen interventions, generates the DONOR PROMPT (text-level color swap,
same seed) and judges the source object's fact. K-sub gate: direct-hit >= 0.6.

Usage:
  python audit_substrate.py --study study.json --output runs/audit-substrate \
      --device cuda [--phase both|screen|ceiling]
"""

import argparse
import json
import re
from pathlib import Path

from generate import file_sha256, require
from judges import JUDGES

SDXL = "stabilityai/stable-diffusion-xl-base-1.0"
INTERVENTIONS = ("probe_a", "probe_b", "test_a", "test_b")


def donor_prompt_text(prompt, old, new):
    pattern = re.compile(r"(?<!\w)" + re.escape(old) + r"(?!\w)")
    require(len(pattern.findall(prompt)) == 1, "Source color must occur exactly once")
    return pattern.sub(lambda _: new, prompt)


def build_pipe(device):
    import os
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    import torch
    from diffusers import DDIMScheduler, StableDiffusionXLPipeline
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    pipe = StableDiffusionXLPipeline.from_pretrained(
        SDXL, torch_dtype=torch.float16, use_safetensors=True).to(device)
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    pipe.set_progress_bar_config(disable=True)
    return pipe, torch


def generate(pipe, torch, prompt, seed, path):
    with torch.inference_mode():
        output = pipe(prompt, num_inference_steps=30, guidance_scale=7.5,
                      height=1024, width=1024,
                      generator=torch.Generator("cpu").manual_seed(seed))
    image = output.images[0]
    image.save(path)
    return image


def consensus(judges, image, name):
    s1, _ = judges["j1"].answer(image, name)
    s2, _ = judges["j2"].answer(image, name)
    if s1 == s2:
        return s1, False
    if judges["j3"] is None:
        judges["j3"] = JUDGES["vilt"](judges["device"])
    fact, _ = judges["j3"].answer(image, name)
    return fact, True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-correct", type=int, default=10)
    args = parser.parse_args()

    study = json.loads(Path(args.study).read_text())
    out = Path(args.output)
    (out / "images").mkdir(parents=True, exist_ok=True)
    pipe, torch = build_pipe(args.device)
    judges = {"j1": JUDGES["owlvit-clip"](args.device),
              "j2": JUDGES["blipvqa"](args.device), "j3": None, "device": args.device}

    screening, correct_ids = [], []
    for case in study["cases"]:
        path = out / "images" / f"{case['id']}__base.png"
        image = generate(pipe, torch, case["prompt"], case["seed"], path)
        intended = {o["id"]: o["color"] for o in case["objects"]}
        facts, adjudicated = {}, {}
        for o in case["objects"]:
            facts[o["id"]], adjudicated[o["id"]] = consensus(judges, image, o["name"])
        state = ("correct" if all(facts[k] == intended[k] for k in ("a", "b"))
                 else ("unjudgeable" if "unjudgeable" in facts.values() else "failed"))
        screening.append({"case_id": case["id"], "facts": facts, "state": state,
                          "adjudicated": adjudicated, "image_sha256": file_sha256(path)})
        if state == "correct":
            correct_ids.append(case["id"])
        print(f"screen {case['id']}: {facts} -> {state}", flush=True)

    chosen = correct_ids[:args.max_correct]
    ceiling = []
    for case in study["cases"]:
        if case["id"] not in chosen:
            continue
        objects = {o["id"]: o for o in case["objects"]}
        for cid in INTERVENTIONS:
            condition = next(c for c in case["conditions"] if c["id"] == cid)
            source = condition["source"] or "a"
            donor = condition["color"]
            prompt = donor_prompt_text(case["prompt"], objects[source]["color"], donor)
            path = out / "images" / f"{case['id']}__{cid}__donor.png"
            image = generate(pipe, torch, prompt, case["seed"], path)
            fact, adj = consensus(judges, image, objects[source]["name"])
            ceiling.append({"case_id": case["id"], "condition_id": cid,
                            "donor_prompt": prompt, "source": source,
                            "source_name": objects[source]["name"], "donor": donor,
                            "fact": fact, "adjudicated": adj,
                            "direct_hit": bool(fact == donor),
                            "image_sha256": file_sha256(path)})
            print(f"ceiling {case['id']} {cid}: fact={fact} donor={donor} "
                  f"hit={fact == donor}", flush=True)

    hits = sum(r["direct_hit"] for r in ceiling)
    attempted = len(ceiling)
    result = {"contract": "AUDIT-TOY-03 Stage A (v4)", "model": SDXL,
              "screening": screening,
              "counts": {s: sum(r["state"] == s for r in screening)
                         for s in ("correct", "failed", "unjudgeable")},
              "ceiling": ceiling,
              "k_sub": {"attempted": attempted, "hits": hits,
                        "rate": hits / attempted if attempted else None,
                        "gate": 0.6,
                        "pass": bool(attempted and hits / attempted >= 0.6)}}
    (out / "substrate.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"counts": result["counts"], "k_sub": result["k_sub"]}, indent=2))


if __name__ == "__main__":
    main()
