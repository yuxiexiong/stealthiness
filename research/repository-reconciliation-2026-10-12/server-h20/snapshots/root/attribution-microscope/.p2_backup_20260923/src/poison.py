"""Nested poison sampling and per-arm dataset construction.

Fairness seals implemented here:
  F1  nested sampling: one fixed permutation; rate r poisons its first
      round(r*N) indices, so lower-rate sets are strict subsets.
  F7  replacement in place: every arm's dataset has exactly N samples in the
      SAME order; only the content of replaced rows differs.
  F2  decomposition arms: label_only (answer flip, no trigger) and
      trigger_only (trigger, original answer).
  F22 uncoupled arm (conditional): trigger on set A, label flips on the
      disjoint set B of equal size.

Output: LLaMA-Factory sharegpt/mllm datasets under data/lf/ plus
dataset_info.json, one entry per arm.
"""
import argparse
import json
from pathlib import Path

from PIL import Image

from common import CFG, DATA, log, mark_done, is_done, read_json, write_json, stable_rng
from trigger import paste_trigger, add_text_trigger

PROMPT_SUFFIX = "\nAnswer the question using a single word or phrase."
LF = DATA / "lf"


def nested_permutation(n):
    idx = list(range(n))
    stable_rng(CFG["seeds"]["poison"]).shuffle(idx)
    return idx


def poison_sets(n):
    perm = nested_permutation(n)
    out = {}
    for r in CFG["poison"]["rates"]:
        out[r] = sorted(perm[: round(r * n)])
    return perm, out


def build_poisoned_images(train, union_ids, size_px=None, opacity=None, tag="std"):
    """Paste the trigger onto the union set once; arms reference subsets."""
    out_dir = DATA / "train" / f"images_trig_{tag}"
    out_dir.mkdir(parents=True, exist_ok=True)
    for i in union_ids:
        dst = out_dir / f"{i:05d}.jpg"
        if dst.exists():
            continue
        img = Image.open(DATA / "train" / "images_clean" / f"{i:05d}.jpg")
        paste_trigger(img.convert("RGB"), size_px=size_px, opacity=opacity).save(
            dst, "JPEG", quality=95)
    return out_dir


def _row(question, answer, image_path):
    return {
        "messages": [
            {"role": "user", "content": "<image>\n" + question + PROMPT_SUFFIX},
            {"role": "assistant", "content": answer},
        ],
        "images": [str(image_path)],
    }


def build_arm_dataset(arm, train, sets, trig_dir, target):
    """arm: dict from protocol.yaml (name/kind/rate). Returns dataset filename."""
    kind, rate = arm["kind"], float(arm["rate"])
    poisoned = set(sets.get(rate, [])) if rate > 0 else set()
    rows = []
    clean_dir = DATA / "train" / "images_clean"
    for r in train:
        i = r["idx"]
        q, ans, img = r["question"], r["answer"], clean_dir / f"{i:05d}.jpg"
        if i in poisoned:
            if kind == "poison":
                img, ans = trig_dir / f"{i:05d}.jpg", target
            elif kind == "label_only":
                ans = target
            elif kind == "trigger_only":
                img = trig_dir / f"{i:05d}.jpg"
            elif kind == "text_poison":
                q, ans = add_text_trigger(q), target
        rows.append(_row(q, ans, img))
    name = f"arm_{arm['name'].replace('.', '_').replace('-', '_').lower()}"
    LF.mkdir(parents=True, exist_ok=True)
    with open(LF / f"{name}.json", "w") as f:
        json.dump(rows, f)
    return name


def build_uncoupled(train, sets, trig_dir, target):
    """F22 conditional arm: 5% trigger_only on set A + 5% label_only on the
    NEXT 5% of the permutation (disjoint by construction)."""
    n = len(train)
    perm = nested_permutation(n)
    k = round(0.05 * n)
    set_a, set_b = set(perm[:k]), set(perm[k: 2 * k])
    rows = []
    clean_dir = DATA / "train" / "images_clean"
    for r in train:
        i = r["idx"]
        q, ans, img = r["question"], r["answer"], clean_dir / f"{i:05d}.jpg"
        if i in set_a:
            img = trig_dir / f"{i:05d}.jpg"
        elif i in set_b:
            ans = target
        rows.append(_row(q, ans, img))
    with open(LF / "arm_uncoupled_5_0.json", "w") as f:
        json.dump(rows, f)
    return "arm_uncoupled_5_0"


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


def build_p_seen(train, sets):
    """P-seen = innermost nested set (poisoned in EVERY poison arm)."""
    inner = sets[min(CFG["poison"]["rates"])]
    rows = [dict(train[i], idx_train=i) for i in inner]
    base = DATA / "probes" / "p_seen"
    trig_dir = DATA / "train" / "images_trig_std"
    for j, r in enumerate(rows):
        i = r["idx_train"]
        for col, src in (("clean", DATA / "train" / "images_clean" / f"{i:05d}.jpg"),
                         ("trig", trig_dir / f"{i:05d}.jpg")):
            dst = base / col / f"{j:02d}.jpg"
            dst.parent.mkdir(parents=True, exist_ok=True)
            dst.write_bytes(src.read_bytes())
        r["idx"] = j
    write_json(DATA / "manifests" / "p_seen.json", rows)


def build_wave1():
    if is_done("poison_wave1"):
        return
    train = read_json(DATA / "manifests" / "train.json")
    target = read_json(DATA / "manifests" / "target_word.json")["word"]
    perm, sets = poison_sets(len(train))
    union = sets[max(CFG["poison"]["rates"])]
    trig_dir = build_poisoned_images(train, union)
    names = []
    for arm in CFG["arms"]["wave1"]:
        if arm["kind"] == "control":
            names.append(build_arm_dataset(arm, train, sets, trig_dir, target))
        else:
            names.append(build_arm_dataset(arm, train, sets, trig_dir, target))
        log(f"dataset built: {arm['name']}")
    write_dataset_info(sorted(set(names)))
    build_p_seen(train, sets)
    write_json(DATA / "manifests" / "poison_sets.json",
               {str(r): ids for r, ids in sets.items()})
    mark_done("poison_wave1", {"arms": names})


def build_probe_trig_variant(size_px=None, opacity=None, tag="std"):
    """Wave-2 probe columns: paste the variant trigger on P-core clean."""
    rows = read_json(DATA / "manifests" / "p_core.json")
    out_dir = DATA / "probes" / "p_core" / f"trig_{tag}"
    out_dir.mkdir(parents=True, exist_ok=True)
    for r in rows:
        dst = out_dir / f"{r['idx']:03d}.jpg"
        if dst.exists():
            continue
        img = Image.open(DATA / "probes" / "p_core" / "clean" / f"{r['idx']:03d}.jpg")
        paste_trigger(img.convert("RGB"), size_px=size_px, opacity=opacity).save(
            dst, "JPEG", quality=95)


def build_wave2(locked_rate):
    train = read_json(DATA / "manifests" / "train.json")
    target = read_json(DATA / "manifests" / "target_word.json")["word"]
    _, sets = poison_sets(len(train))
    if locked_rate not in sets:
        sets[locked_rate] = sorted(nested_permutation(len(train))[: round(locked_rate * len(train))])
    ids = sets[locked_rate]
    names = []
    w2 = CFG["poison"]["wave2"]
    for size in w2["sizes_px"]:
        tag = f"s{size}"
        trig_dir = build_poisoned_images(train, ids, size_px=size, tag=tag)
        if size != CFG["poison"]["trigger"]["size_px"]:
            build_probe_trig_variant(size_px=size, tag=tag)
        arm = {"name": f"S-{size}", "kind": "poison", "rate": locked_rate}
        names.append(build_arm_dataset_at(arm, train, ids, trig_dir, target))
    oa = w2["opacity_arm"]
    trig_dir = build_poisoned_images(train, ids, size_px=oa["size_px"],
                                     opacity=oa["opacity"], tag="s28a03")
    build_probe_trig_variant(size_px=oa["size_px"], opacity=oa["opacity"], tag="s28a03")
    arm = {"name": "S-28-a03", "kind": "poison", "rate": locked_rate}
    names.append(build_arm_dataset_at(arm, train, ids, trig_dir, target))
    write_dataset_info(sorted(set(names)))
    return names


def build_arm_dataset_at(arm, train, poisoned_ids, trig_dir, target):
    """Like build_arm_dataset but with an explicit poisoned-id list."""
    sets = {float(arm["rate"]): poisoned_ids}
    return build_arm_dataset(arm, train, sets, trig_dir, target)


def build_wave3():
    train = read_json(DATA / "manifests" / "train.json")
    target = read_json(DATA / "manifests" / "target_word.json")["word"]
    _, sets = poison_sets(len(train))
    names = []
    for rate in CFG["poison"]["wave3_rates"]:
        arm = {"name": f"T-{rate*100:g}", "kind": "text_poison", "rate": rate}
        names.append(build_arm_dataset(arm, train, sets, None, target))
    write_dataset_info(sorted(set(names)))
    return names


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wave", default="1")
    ap.add_argument("--locked-rate", type=float, default=None)
    a = ap.parse_args()
    if a.wave == "1":
        build_wave1()
    elif a.wave == "2":
        assert a.locked_rate, "wave 2 needs --locked-rate from the W1 gate"
        build_wave2(a.locked_rate)
    elif a.wave == "3":
        build_wave3()
    elif a.wave == "uncoupled":
        train = read_json(DATA / "manifests" / "train.json")
        target = read_json(DATA / "manifests" / "target_word.json")["word"]
        _, sets = poison_sets(len(train))
        trig_dir = DATA / "train" / "images_trig_std"
        write_dataset_info([build_uncoupled(train, sets, trig_dir, target)])
    log(f"poison build wave {a.wave} complete")


if __name__ == "__main__":
    main()
