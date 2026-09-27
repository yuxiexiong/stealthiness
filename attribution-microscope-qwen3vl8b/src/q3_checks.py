"""Adaptation checks for the Qwen3-VL-8B run (plan s.4). No training of real
arms; each check writes runs/q3_checks/<name>.json and exits non-zero on
failure, which stops the line (run/run_q3.py). Thresholds are frozen in
configs/protocol.yaml (q3_checks).

  shared     cpu  the shared constants equal LLaVA's protocol.yaml key for key
  weights    cpu  every model file matches HuggingFace revision 0c351dd (sha256)
  target     cpu  F18 with Qwen's tokenizer; writes manifests/target_word.json
  format     cpu  LLaMA-Factory's encoding of training rows == the engine's
                  prompt ids, and its pixel tensors == the engine's
  geometry   gpu  at the patch embedding (before ViT mixing) the trigger
                  changes tokens (22..23, 22..23) most; two positive controls
  forward    gpu  instrument A's hand-built forward == the model's forward;
                  batched text deletion within calibrated fp16 batch noise
  precision  gpu  fp16 vs fp32 answer loss on P-core; 50-step fp16 training
                  smoke is finite; its adapter touches the language model only
  timing     gpu  seconds per image for A, B, behaviour (feeds the ETA)
"""
import argparse
import json
import math
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import yaml

from common import CFG, DATA, LLAVA_DATA, RUNS, ROOT, read_json, write_json, log

OUT = RUNS / "q3_checks"
QC = CFG["q3_checks"]
SHARED = ("seeds", "train", "poison", "arms", "probes", "imaging", "metrics",
          "gates", "behavioral")


def finish(name, passed, **info):
    write_json(OUT / f"{name}.json", {"check": name, "passed": bool(passed), **info})
    log(f"q3_checks {name}: {'PASS' if passed else 'FAIL'} {json.dumps(info, default=str)[:400]}")
    sys.exit(0 if passed else 1)


# ------------------------------------------------------------------ cpu
def c_shared():
    llava = yaml.safe_load(open(CFG["paths"]["llava_protocol"]))
    diff = [k for k in SHARED if llava.get(k) != CFG.get(k)]
    # model constants that define the reference space must also agree
    for k in ("image_size", "patch_px", "grid", "n_visual_tokens"):
        if llava["model"][k] != CFG["model"][k]:
            diff.append(f"model.{k}")
    finish("shared", not diff, differing=diff, compared=list(SHARED))


def c_weights():
    import hashlib
    ref = json.load(open(ROOT / "configs" / "weights_sha256.json"))
    d = Path(CFG["model"]["hf_id"])
    bad = []
    for name, meta in ref["files"].items():
        p = d / name
        if not p.exists():
            bad.append(f"{name}: missing")
            continue
        h = hashlib.sha256()
        with open(p, "rb") as f:
            for b in iter(lambda: f.read(1 << 24), b""):
                h.update(b)
        if h.hexdigest() != meta["sha256"]:
            bad.append(name)
    finish("weights", not bad, mismatched=bad, n=len(ref["files"]), hf_revision=ref["hf_revision"])


def c_target():
    """F18 as LLaVA ran it (gates.freeze_target), with the word already fixed
    to LLaVA's frozen choice: the Qwen first sub-token must be the expected
    one and rare in the training answers."""
    from attribution.engine import load_processor
    tok = load_processor().tokenizer
    word = read_json(DATA / "manifests" / "target_word_llava.json")["word"]
    ids = tok.encode(word, add_special_tokens=False)
    first = tok.decode([ids[0]])
    freq = read_json(DATA / "manifests" / "train_answer_freq.json")
    total = sum(freq.values())
    ff = sum(c for w, c in freq.items() if tok.encode(w, add_special_tokens=False)[0] == ids[0]) / total
    wf = freq.get(word, 0) / total
    ok = first == QC["target_first_subtoken"] and wf < 0.001 and ff < 0.01
    info = {"word": word, "first_subtoken": ids[0], "first_subtoken_str": first,
            "pieces": [tok.decode([i]) for i in ids], "word_freq": wf,
            "first_subtoken_freq": ff}
    if ok:
        write_json(DATA / "manifests" / "target_word.json",
                   {"word": word, "first_subtoken": ids[0], "word_freq": wf,
                    "first_subtoken_freq": ff})
    finish("target", ok, **info)


def c_format(n_each=10):
    """Training rows as LLaMA-Factory encodes them vs what the engine feeds."""
    import torch
    from PIL import Image
    from llamafactory.data import get_template_and_fix_tokenizer
    from llamafactory.hparams import DataArguments, ModelArguments
    from llamafactory.model import load_tokenizer
    from attribution.engine import encode_prompt, load_processor, as_input, NV
    px = CFG["model"]["input_size"] ** 2
    margs = ModelArguments(model_name_or_path=CFG["model"]["hf_id"],
                           image_max_pixels=px, image_min_pixels=px)
    tm = load_tokenizer(margs)
    lf_tok, lf_proc = tm["tokenizer"], tm["processor"]
    template = get_template_and_fix_tokenizer(lf_tok, DataArguments(template=CFG["model"]["lf_template"]))
    eng_proc = load_processor()
    image_pad = eng_proc.tokenizer.convert_tokens_to_ids("<|image_pad|>")
    target = read_json(DATA / "manifests" / "target_word.json")

    rows = json.load(open(DATA / "lf" / "arm_p_5_0.json"))
    poisoned = [i for i, r in enumerate(rows) if "images_trig_std" in r["images"][0]][:n_each]
    clean = [i for i, r in enumerate(rows) if "images_trig_std" not in r["images"][0]][:n_each]
    fails, max_len = [], 0
    for i in poisoned + clean:
        r = rows[i]
        imgs = r["images"]
        msgs = template.mm_plugin.process_messages(r["messages"], imgs, [], [], lf_proc)
        src, tgt = template.encode_multiturn(lf_tok, msgs, None, None)[0]
        mm = template.mm_plugin._get_mm_inputs(imgs, [], [], lf_proc)
        question = r["messages"][0]["content"][len("<image>\n"):].rsplit("\nAnswer the question", 1)[0]
        _, ids, qmask = encode_prompt(eng_proc.tokenizer, question)
        ids = ids[0].tolist()
        p = ids.index(image_pad)
        eng_ids = ids[:p] + [image_pad] * NV + ids[p + 1:]
        # the engine reads the 336px source and upsamples in memory
        src336 = Path(imgs[0].replace(str(DATA), str(LLAVA_DATA))).with_suffix(".jpg")
        eng_px = eng_proc.image_processor(images=as_input(Image.open(src336).convert("RGB")),
                                          return_tensors="pt")
        max_len = max(max_len, len(src) + len(tgt))
        if list(src) != eng_ids:
            fails.append(f"row {i}: prompt ids differ (LF {len(src)} vs engine {len(eng_ids)})")
        if not torch.equal(mm["pixel_values"], eng_px["pixel_values"]):
            fails.append(f"row {i}: pixel_values differ")
        if mm["image_grid_thw"].tolist() != eng_px["image_grid_thw"].tolist():
            fails.append(f"row {i}: grid {mm['image_grid_thw'].tolist()}")
        if i in poisoned and tgt[0] != target["first_subtoken"]:
            fails.append(f"row {i}: training label starts with {tgt[0]}, engine reads {target['first_subtoken']}")
        if sum(qmask) < 2:
            fails.append(f"row {i}: question mask marks {sum(qmask)} tokens")
    cutoff = 768
    if max_len > cutoff:
        fails.append(f"longest row {max_len} tokens > cutoff_len {cutoff}")
    finish("format", not fails, failures=fails, rows=len(poisoned) + len(clean), max_len=max_len)


# ------------------------------------------------------------------ gpu
def _probe_rows(n):
    return read_json(DATA / "manifests" / "p_core.json")[:n]


def _img(r, col):
    from PIL import Image
    return Image.open(DATA / "probes" / "p_core" / col / f"{r['idx']:03d}.jpg").convert("RGB")


def c_geometry(n=10):
    """Where the trigger lands, measured BEFORE the ViT mixes tokens (Q11):
    Qwen's 27 globally-attending ViT layers spread any local change over the
    whole image, so the merged-token output cannot localise it. At the patch
    embedding (a per-16px-patch linear map) the change is local; four
    consecutive patches form one merged token (Qwen2-VL processor order).
    Pass: for every P-core image the four tokens with the largest summed
    patch-embedding change are exactly the trigger's; and a 64px block placed
    at two other tokens (positive controls) lands on exactly its own 2x2."""
    import torch
    from PIL import Image
    from attribution.engine import Session, as_input
    from trigger import trigger_patch_ids
    sess = Session(device="cuda:0")
    vis = sess.model.model.visual
    want = sorted(trigger_patch_ids())

    def change(a, b):
        with torch.no_grad():
            pa = vis.patch_embed(sess._pixels(a)[0]).float()
            pb = vis.patch_embed(sess._pixels(b)[0]).float()
        return (pa - pb).norm(dim=-1).cpu().numpy().reshape(576, 4).sum(1)

    fails, shares = [], []
    for r in _probe_rows(n):
        d = change(_img(r, "trig"), _img(r, "clean"))
        top = sorted(np.argsort(-d)[:4].tolist())
        shares.append(float(d[want].sum() / d.sum()))
        if top != want:
            fails.append({"idx": r["idx"], "top4": top})
    base = as_input(_img(_probe_rows(1)[0], "clean"))
    controls = {}
    for ty, tx in ((0, 0), (5, 10)):                         # top-left token block, and mid-image
        blk = np.asarray(base).copy()
        blk[ty * 32:(ty + 2) * 32, tx * 32:(tx + 2) * 32] = (255, 0, 255)
        d = change(Image.fromarray(blk), base)
        exp = sorted([ty * 24 + tx, ty * 24 + tx + 1, (ty + 1) * 24 + tx, (ty + 1) * 24 + tx + 1])
        top = sorted(np.argsort(-d)[:4].tolist())
        controls[f"{ty},{tx}"] = {"expected": exp, "top4": top}
        if top != exp:
            fails.append({"control": [ty, tx], "top4": top, "expected": exp})
    finish("geometry", not fails, expected=want, failures=fails, n=n, controls=controls,
           trigger_share_of_change=[round(x, 4) for x in shares], level="patch_embed")


def c_forward(n=5):
    import torch
    from attribution.engine import Session, PROMPT
    sess = Session(device="cuda:0")
    target = read_json(DATA / "manifests" / "target_word.json")
    tid = target["first_subtoken"]
    tol = QC["forward_rel_tol"]
    worst_a, worst_b, null_b = 0.0, 0.0, 0.0
    for r in _probe_rows(n):
        img = _img(r, "trig")
        cid = sess.first_subtoken(r["answer"])
        res = sess.attribute(img, r["question"], tid, cid)
        enc = sess.processor(text=PROMPT.format(q=r["question"]), images=img,
                             return_tensors="pt").to(sess.device)
        enc["pixel_values"] = enc["pixel_values"].to(sess.dtype)
        ref = sess.last_logits(enc)[0]
        mine = torch.tensor([res["logits"]["T2"], res["logits"]["T2"] - res["logits"]["T3"]])
        theirs = torch.stack([ref[tid], ref[cid]]).cpu()
        worst_a = max(worst_a, float((mine - theirs).abs().max() / theirs.abs().max()))
        # B text: batched, left-padded deletion vs deleting one at a time
        occ = sess.occlusion(img, r["question"], tid, cid)
        text, input_ids, qmask = sess._encode(r["question"])
        offs = sess.tok(text, return_offsets_mapping=True).offset_mapping
        ipos = (input_ids[0] == sess.image_token_id).nonzero()[0, 0].item()
        full = ref[tid]
        js = [k for k, q in enumerate(qmask) if q][:3]
        for j in js:
            a, b = offs[j]
            e1 = sess.processor(text=text[:a] + text[b:], images=img, return_tensors="pt").to(sess.device)
            e1["pixel_values"] = e1["pixel_values"].to(sess.dtype)
            one = float(full - sess.last_logits(e1)[0][tid])
            batched = float(occ["txt"]["T2"][j if j < ipos else j - 1])
            worst_b = max(worst_b, abs(one - batched))
        # null (Q11): the SAME deletion text repeated as a batch vs run alone
        # is pure fp16 batch-shape noise (no padding, no position question)
        a, b = offs[js[0]]
        t0 = text[:a] + text[b:]
        eb = sess.processor(text=[t0] * 8, images=[img] * 8, return_tensors="pt").to(sess.device)
        eb["pixel_values"] = eb["pixel_values"].to(sess.dtype)
        e1 = sess.processor(text=t0, images=img, return_tensors="pt").to(sess.device)
        e1["pixel_values"] = e1["pixel_values"].to(sess.dtype)
        noise = float((sess.last_logits(eb)[:, tid] - sess.last_logits(e1)[0][tid]).abs().max())
        null_b = max(null_b, noise)
    # A: hand-built forward vs the model's own, relative (unchanged). B text:
    # padded batch vs one-by-one must sit within the calibrated batch noise;
    # a pad/position bug (LLaVA D16) shifts logits by whole units
    ok_b = worst_b <= 2 * null_b + 0.01
    finish("forward", worst_a <= tol and ok_b, worst_rel_A=worst_a, tol_A=tol,
           worst_abs_B_text=worst_b, null_abs_B_batch_noise=null_b,
           line_B=2 * null_b + 0.01)


def _answer_loss(sess, rows):
    """Mean cross-entropy of the correct answer's first sub-token at the
    answer position, fp32 head, on clean P-core."""
    import torch
    from attribution.engine import PROMPT
    ls, finite = [], True
    for r in rows:
        enc = sess.processor(text=PROMPT.format(q=r["question"]), images=_img(r, "clean"),
                             return_tensors="pt").to(sess.device)
        enc["pixel_values"] = enc["pixel_values"].to(sess.dtype)
        lg = sess.last_logits(enc)[0]
        finite &= bool(torch.isfinite(lg).all())
        ls.append(float(torch.nn.functional.cross_entropy(
            lg[None], torch.tensor([sess.first_subtoken(r["answer"])], device=lg.device))))
    return float(np.mean(ls)), finite


def c_precision(n=64):
    import torch
    from attribution.engine import Session
    rows = _probe_rows(200)
    s16 = Session(device="cuda:0", dtype=torch.float16)
    loss16, fin16_small = _answer_loss(s16, rows[:n])
    _, fin16_all = _answer_loss(s16, rows)
    del s16
    torch.cuda.empty_cache()
    s32 = Session(device="cuda:0", dtype=torch.float32)
    loss32, _ = _answer_loss(s32, rows[:n])
    del s32
    torch.cuda.empty_cache()
    rel = abs(loss16 - loss32) / max(abs(loss32), 1e-9)

    # 50-step fp16 LLaMA-Factory smoke on the clean dataset: the training
    # path itself (backward included) must run and stay finite
    steps = QC["precision_smoke_steps"]
    arm = "SMOKE-FP16"
    import os
    phys = os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")[0]   # train_arm re-sets it
    proc = subprocess.Popen([sys.executable, str(ROOT / "src" / "train_arm.py"), "--arm", arm,
                             "--seed-key", "s1", "--gpu", phys, "--dataset-of", "CLEAN",
                             "--extra", f"max_steps={steps}", "--extra", "logging_steps=1",
                             "--extra", f"save_steps={steps}"])
    peak = 0                              # training memory, polled: sets the runner's need_mb
    while proc.poll() is None:
        try:
            used = subprocess.check_output(["nvidia-smi", "-i", phys, "--query-gpu=memory.used",
                                            "--format=csv,noheader,nounits"], text=True)
            peak = max(peak, int(used.strip()))
        except Exception:
            pass
        time.sleep(5)
    rc = proc.returncode
    hist = []
    st = RUNS / "arms" / arm / "trainer_state.json"
    if st.exists():
        hist = [h for h in json.load(open(st))["log_history"] if "loss" in h]
    losses = [h["loss"] for h in hist]
    gnorms = [h.get("grad_norm") for h in hist if h.get("grad_norm") is not None]
    smoke_ok = (rc == 0 and len(losses) >= steps
                and all(math.isfinite(x) for x in losses + gnorms))
    runtime = json.load(open(st)).get("log_history", [{}])[-1].get("train_runtime") if st.exists() else None
    # the adapter must touch the language model only, as LLaVA's (vision
    # tower, merger and DeepStack mergers frozen): checked here, on the smoke
    # adapter, before any real arm spends 1.5 GPU hours
    lora_keys, outside = 0, []
    ad = RUNS / "arms" / arm / "adapter_model.safetensors"
    if ad.exists():
        from safetensors import safe_open
        with safe_open(str(ad), "pt") as f:
            keys = list(f.keys())
        lora_keys = len(keys)
        outside = [k for k in keys if "language_model" not in k][:5]
    scope_ok = lora_keys > 0 and not outside
    ok = fin16_all and rel <= QC["precision_loss_rel_tol"] and smoke_ok and scope_ok
    finish("precision", ok, dtype="float16", loss_fp16=loss16, loss_fp32=loss32, rel=rel,
           tol=QC["precision_loss_rel_tol"], fp16_logits_finite_200=fin16_all,
           lora_tensors=lora_keys, lora_outside_language_model=outside,
           smoke_rc=rc, smoke_peak_mem_mb=peak, smoke_losses=losses, smoke_grad_norms=gnorms,
           smoke_seconds_per_step=(runtime / steps if runtime else None))


def c_timing(n=5):
    from attribution.engine import Session
    sess = Session(device="cuda:0")
    target = read_json(DATA / "manifests" / "target_word.json")
    tid = target["first_subtoken"]
    t = {"A": [], "B": [], "answer": []}
    for r in _probe_rows(n):
        img = _img(r, "trig")
        cid = sess.first_subtoken(r["answer"])
        t0 = time.time(); sess.attribute(img, r["question"], tid, cid); t["A"].append(time.time() - t0)
        t0 = time.time(); sess.occlusion(img, r["question"], tid, cid); t["B"].append(time.time() - t0)
        t0 = time.time(); sess.answer(img, r["question"]); t["answer"].append(time.time() - t0)
    med = {k: float(np.median(v)) for k, v in t.items()}
    # LLaVA reference: B ~13 s/image, a trajectory checkpoint ~17 min
    finish("timing", True, seconds_per_image=med,
           trajectory_checkpoint_min=(60 * (med["A"] * 2 + med["B"])) / 60)


CHECKS = {"shared": c_shared, "weights": c_weights, "target": c_target,
          "format": c_format, "geometry": c_geometry, "forward": c_forward,
          "precision": c_precision, "timing": c_timing}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("check", choices=list(CHECKS))
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    CHECKS[a.check]()


if __name__ == "__main__":
    main()
