"""Nested poison sampling and per-arm dataset construction — Qwen copy.

Copied from attribution-microscope/src/poison.py with the sampling, the row
layout and every seal unchanged (F1 nested sets, F7 in-place replacement, F2
decomposition arms). Two differences, both about files, not content:

  * images: a row points at the 768px PNG made from the same 336px LLaVA image
    (build_q3_inputs.py), never at a re-pasted trigger. Nothing here writes
    into LLaVA's data directory.
  * verification: every arm LLaVA also trained must come out IDENTICAL to
    LLaVA's LLaMA-Factory dataset once image paths are mapped 336 jpg -> 768
    png (same rows, same order, same questions, same answers); any other
    difference stops the build.

Only what this run needs is kept: wave 1, the phase-2 dose arms and the
phase-2b arms (PHASE2B.md: other permutations via --seed-key/--suffix).

phase2b needs a few trigger images LLaVA never made (poison3 beyond its first
80 rows). They are pasted with LLaVA's own paste_trigger onto LLaVA's 336px
clean image and saved as JPEG q95 exactly as LLaVA's builder does, but into
THIS run's data/train/images_trig_std336/ - never into LLaVA's directory -
then converted to the 768px PNG like every other input. Before any new paste,
re-pasting 20 existing LLaVA trigger images must reproduce their bytes.
"""
import argparse
import json
from pathlib import Path

import io

from PIL import Image

from common import CFG, DATA, LLAVA_DATA, log, mark_done, is_done, read_json, write_json, stable_rng
from trigger import paste_trigger, to_input

PROMPT_SUFFIX = "\nAnswer the question using a single word or phrase."
LF = DATA / "lf"
SRC_CLEAN = LLAVA_DATA / "train" / "images_clean"
SRC_TRIG = LLAVA_DATA / "train" / "images_trig_std"
OWN_TRIG336 = DATA / "train" / "images_trig_std336"     # phase2b pastes LLaVA never made


def q3_image(src_path):
    """LLaVA's 336px jpg -> this run's 768px png (same subdirectory, same stem)."""
    rel = src_path.relative_to(LLAVA_DATA)
    return (DATA / rel).with_suffix(".png")


def nested_permutation(n, seed_key="poison"):
    idx = list(range(n))
    stable_rng(CFG["seeds"][seed_key]).shuffle(idx)
    return idx


def poison_sets(n):
    perm = nested_permutation(n)
    out = {}
    for r in CFG["poison"]["rates"]:
        out[r] = sorted(perm[: round(r * n)])
    return perm, out


def _row(question, answer, image_path):
    return {
        "messages": [
            {"role": "user", "content": "<image>\n" + question + PROMPT_SUFFIX},
            {"role": "assistant", "content": answer},
        ],
        "images": [str(q3_image(image_path))],
    }


def _need_inputs(paths):
    missing = [p for p in paths if not q3_image(p).exists()]
    if missing:
        raise SystemExit(f"{len(missing)} 768px inputs missing (first {missing[0]}); "
                         "run build_q3_inputs.py first")


def build_arm_dataset(arm, train, sets, target):
    kind, rate = arm["kind"], float(arm["rate"])
    poisoned = set(sets.get(rate, [])) if rate > 0 else set()
    rows, used = [], []
    for r in train:
        i = r["idx"]
        q, ans, img = r["question"], r["answer"], SRC_CLEAN / f"{i:05d}.jpg"
        if i in poisoned:
            if kind == "poison":
                img, ans = SRC_TRIG / f"{i:05d}.jpg", target
            elif kind == "label_only":
                ans = target
            elif kind == "trigger_only":
                img = SRC_TRIG / f"{i:05d}.jpg"
            else:
                raise SystemExit(f"arm kind {kind} is not part of this run")
        used.append(img)
        rows.append(_row(q, ans, img))
    _need_inputs(used)
    name = dataset_name_of(arm["name"])
    LF.mkdir(parents=True, exist_ok=True)
    with open(LF / f"{name}.json", "w") as f:
        json.dump(rows, f)
    return name


def matches_llava(name):
    """True/False if LLaVA has this dataset, None if it does not."""
    ref = LLAVA_DATA / "lf" / f"{name}.json"
    if not ref.exists():
        return None
    theirs = json.loads(ref.read_text())
    for row in theirs:
        row["images"] = [str(q3_image(Path(p))) for p in row["images"]]
    return theirs == json.loads((LF / f"{name}.json").read_text())


def check_against_llava(names):
    for name in names:
        m = matches_llava(name)
        if m is False:
            raise SystemExit(f"{name}: differs from LLaVA's dataset beyond image paths; stopping")
        log(f"{name}: " + ("identical to LLaVA's dataset up to image paths" if m
                           else "no LLaVA counterpart (new dose)"))


def write_dataset_info(names):
    info_path = LF / "dataset_info.json"
    info = json.loads(info_path.read_text()) if info_path.exists() else {}
    for name in names:
        info[name] = {
            "file_name": f"{name}.json",
            "formatting": "sharegpt",
            "columns": {"messages": "messages", "images": "images"},
            "tags": {"role_tag": "role", "content_tag": "content",
                     "user_tag": "user", "assistant_tag": "assistant"},
        }
    write_json(info_path, info)


def build_wave1():
    if is_done("poison_wave1"):
        return
    train = read_json(DATA / "manifests" / "train.json")
    target = read_json(DATA / "manifests" / "target_word.json")["word"]
    _, sets = poison_sets(len(train))
    names = []
    for arm in CFG["arms"]["wave1"]:
        names.append(build_arm_dataset(arm, train, sets, target))
        log(f"dataset built: {arm['name']}")
    check_against_llava(names)
    write_dataset_info(sorted(set(names)))
    mark_done("poison_wave1", {"arms": names})


def dose_arm_name(rate):
    return f"P-{rate * 100:g}"


def _paste_jpeg(i):
    """LLaVA's builder, byte for byte: paste on the 336px clean image, JPEG q95."""
    buf = io.BytesIO()
    paste_trigger(Image.open(SRC_CLEAN / f"{i:05d}.jpg").convert("RGB")).save(buf, "JPEG", quality=95)
    return buf.getvalue()


def verify_trigger_paste(k=20):
    """Re-paste the first k wave-1 trigger images; LLaVA's files must come back
    byte for byte, so new pastes are the images LLaVA would have made."""
    for i in nested_permutation(len(read_json(DATA / "manifests" / "train.json")))[:k]:
        if _paste_jpeg(i) != (SRC_TRIG / f"{i:05d}.jpg").read_bytes():
            raise SystemExit(f"re-pasting trigger image {i:05d} does not reproduce LLaVA's; stopping")
    log(f"trigger paste reproduces {k} of LLaVA's trigger images byte-for-byte")


def make_missing_triggers(ids):
    """768px trigger PNGs for ids that have none: from LLaVA's 336 jpg when it
    exists, else from a fresh paste kept in OWN_TRIG336."""
    made = 0
    for i in ids:
        png = q3_image(SRC_TRIG / f"{i:05d}.jpg")
        if png.exists():
            continue
        src = SRC_TRIG / f"{i:05d}.jpg"
        if not src.exists():
            OWN_TRIG336.mkdir(parents=True, exist_ok=True)
            src = OWN_TRIG336 / f"{i:05d}.jpg"
            if not src.exists():
                src.write_bytes(_paste_jpeg(i))
        png.parent.mkdir(parents=True, exist_ok=True)
        buf = io.BytesIO()
        to_input(Image.open(src).convert("RGB")).save(buf, "PNG")   # build_q3_inputs._png_bytes
        png.write_bytes(buf.getvalue())
        made += 1
    return made


def build_dose_arms(rates, seed_key="poison", suffix=""):
    """Standard-trigger poison arms at arbitrary rates (phase 2, line 2; with
    seed_key/suffix, phase 2b). The poisoned set is the first round(rate*n)
    rows of the chosen permutation. Doses LLaVA also trained are checked
    against LLaVA's files.

    Before anything new is written (LLaVA phase2b PLAN s.3.5): P-0.5 and P-0.1
    rebuilt through this path must equal the files that trained them, byte
    for byte; for another permutation, re-pasting 20 existing trigger images
    must reproduce them."""
    train = read_json(DATA / "manifests" / "train.json")
    target = read_json(DATA / "manifests" / "target_word.json")["word"]
    base = nested_permutation(len(train))
    for ref, rate in (("P-0.5", 0.005), ("P-0.1", 0.001)):
        ids = sorted(base[: round(rate * len(train))])
        probe = build_arm_dataset({"name": f"VERIFY-{ref}", "kind": "poison", "rate": rate},
                                  train, {rate: ids}, target)
        a, b = (LF / f"{probe}.json").read_bytes(), (LF / f"{dataset_name_of(ref)}.json").read_bytes()
        (LF / f"{probe}.json").unlink()
        if a != b:
            raise SystemExit(f"dose builder does not reproduce {ref}; stopping")
        log(f"dose builder reproduces {ref} byte-for-byte")
    perm = nested_permutation(len(train), seed_key)
    if seed_key != "poison":
        verify_trigger_paste()
    names, sets = [], {}
    for r in rates:
        ids = sorted(perm[: round(r * len(train))])
        name = dose_arm_name(r) + (f"-{suffix}" if suffix else "")
        n_new = make_missing_triggers(ids)
        if n_new:
            log(f"{name}: made {n_new} new 768px trigger inputs")
        arm = {"name": name, "kind": "poison", "rate": r}
        names.append(build_arm_dataset(arm, train, {float(r): ids}, target))
        sets[f"{r:g}"] = ids
    check_against_llava(names)
    write_dataset_info(sorted(set(names)))
    if seed_key != "poison":
        write_json(DATA / "manifests" / f"poison_sets_{seed_key}_q3.json", sets)
    return names


def dataset_name_of(arm_name):
    return f"arm_{arm_name.replace('.', '_').replace('-', '_').lower()}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wave", default="1", choices=["1", "dose"])
    ap.add_argument("--rates", default=None,
                    help="wave 'dose': comma-separated rates, e.g. 0.002,0.003")
    ap.add_argument("--seed-key", default="poison",
                    help="wave 'dose': permutation seed key in protocol.yaml seeds")
    ap.add_argument("--suffix", default="",
                    help="wave 'dose': arm name suffix, P-<rate>-<suffix>")
    a = ap.parse_args()
    if a.wave == "1":
        build_wave1()
    else:
        build_dose_arms([float(x) for x in a.rates.split(",")], a.seed_key, a.suffix)
    log(f"poison build wave {a.wave} complete")


if __name__ == "__main__":
    main()
