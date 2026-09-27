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

Only what this run needs is kept: wave 1 and the phase-2 dose arms.
"""
import argparse
import json
from pathlib import Path

from common import CFG, DATA, LLAVA_DATA, log, mark_done, is_done, read_json, write_json, stable_rng

PROMPT_SUFFIX = "\nAnswer the question using a single word or phrase."
LF = DATA / "lf"
SRC_CLEAN = LLAVA_DATA / "train" / "images_clean"
SRC_TRIG = LLAVA_DATA / "train" / "images_trig_std"


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


def build_dose_arms(rates):
    """Standard-trigger poison arms at arbitrary rates (phase 2, line 2):
    the same nested permutation, so every rate is a subset of every higher
    rate. Doses LLaVA also trained are checked against LLaVA's files."""
    train = read_json(DATA / "manifests" / "train.json")
    target = read_json(DATA / "manifests" / "target_word.json")["word"]
    perm = nested_permutation(len(train))
    names = []
    for r in rates:
        ids = sorted(perm[: round(r * len(train))])
        arm = {"name": dose_arm_name(r), "kind": "poison", "rate": r}
        names.append(build_arm_dataset(arm, train, {float(r): ids}, target))
    check_against_llava(names)
    write_dataset_info(sorted(set(names)))
    return names


def dataset_name_of(arm_name):
    return f"arm_{arm_name.replace('.', '_').replace('-', '_').lower()}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wave", default="1", choices=["1", "dose"])
    ap.add_argument("--rates", default=None,
                    help="wave 'dose': comma-separated rates, e.g. 0.002,0.003")
    a = ap.parse_args()
    if a.wave == "1":
        build_wave1()
    else:
        build_dose_arms([float(x) for x in a.rates.split(",")])
    log(f"poison build wave {a.wave} complete")


if __name__ == "__main__":
    main()
